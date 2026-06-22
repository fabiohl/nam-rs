// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Grouped dilated causal Conv1D (`groups > 1`) for the A2 architecture.
//!
//! FiLM modulation requires `_cond_to_scale_shift` — a `Conv1x1` that maps
//! condition channels to scale (and optionally shift) channels using `groups > 1`
//! for independent per-group projections (C++ `NAM/film.h`).
//!
//! ## Layout
//!
//! NAM JSON weight order: `raw[out_ch][in_ch][kernel]` (row-major).
//! Internal grouped-interleaved-4-wide: `[group][num_blocks][kernel][in_per_group][4]`
//!
//! For each `(group, block, kernel-tap, input-channel)` the 4 output-channel weights
//! are contiguous — one `_mm_loadu_ps` load, one `_mm_fmadd_ps` broadcast-FMA.
//!
//! ## Fast-path
//!
//! When `groups == 1` the computation is identical to a standard dilated Conv1D —
//! this module delegates directly to `Conv1dDyn` to preserve the extreme fast-path.
//!
//! ## Source of truth
//! - `NAM/conv1d.h:11-136` (`Conv1D` with `_num_groups`, `_is_depthwise`)
//! - `NAM/conv1d.cpp:55-252` (grouped weight loading + depthwise execution)
//! - `NAM/film.h:28-85` (`_cond_to_scale_shift` pattern)

use crate::math::common::{AlignedVec, PrefetchFn};
use crate::models::wavenet::common::MAX_KERNEL;

use core::arch::x86_64::*;

/// Grouped dilated causal Conv1D with interleaved-4-wide f32 weights.
///
/// Divides input and output channels into `groups` independent sub-convolutions.
/// Each group operates on `in_per_group = in_ch / groups` input channels and
/// produces `out_per_group = out_ch / groups` output channels — no cross-group
/// connections.
///
/// When `groups == in_ch && groups == out_ch` this is a **depthwise** convolution:
/// each channel is filtered independently.
#[derive(Clone)]
#[repr(align(64))]
pub struct A2GroupedConv1d {
    /// Grouped-interleaved-4-wide f32 weights.
    /// Layout: `[group][num_blocks][kernel][in_per_group][4]`
    pub weights: AlignedVec<f32>,
    /// Bias vector `[out_ch]`.
    pub bias: AlignedVec<f32>,
    /// Whether to add bias to the accumulator.
    pub do_bias: bool,
    /// Temporal dilation factor.
    pub dilation: usize,
    /// Number of input channels (must be divisible by `groups`).
    pub in_ch: usize,
    /// Number of output channels (must be divisible by `groups`).
    pub out_ch: usize,
    /// Number of groups for the grouped convolution.
    pub groups: usize,
    /// Input channels per group (`in_ch / groups`).
    pub in_per_group: usize,
    /// Output channels per group (`out_ch / groups`).
    pub out_per_group: usize,
    /// Number of 4-channel blocks per group (`ceil(out_per_group / 4)`).
    pub num_blocks_per_group: usize,
    /// Total 4-channel blocks across all groups (`groups * num_blocks_per_group`).
    pub total_blocks: usize,
    /// Kernel size.
    pub kernel: usize,
    /// Pre-computed prefetch strategy.
    pub prefetch_fn: PrefetchFn,
}

impl A2GroupedConv1d {
    /// Builds a grouped conv1d from raw NAM JSON row-major weights.
    ///
    /// `raw_weights` is in `[out_ch][in_ch][kernel]` order.
    /// The constructor permutes to grouped-interleaved-4-wide:
    /// `[group][num_blocks][kernel][in_per_group][4]`.
    ///
    /// # Panics
    /// Panics if `in_ch % groups != 0` or `out_ch % groups != 0`.
    pub fn new(
        raw_weights: &[f32],
        raw_bias: &[f32],
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        groups: usize,
        prefetch_fn: PrefetchFn,
    ) -> Self {
        assert!(groups > 0, "groups must be > 0");
        assert_eq!(
            in_ch % groups,
            0,
            "in_ch={} must be divisible by groups={}",
            in_ch,
            groups
        );
        assert_eq!(
            out_ch % groups,
            0,
            "out_ch={} must be divisible by groups={}",
            out_ch,
            groups
        );
        assert_eq!(
            raw_weights.len(),
            out_ch * in_ch * kernel,
            "raw_weights len {} != expected {} (out_ch={} * in_ch={} * kernel={})",
            raw_weights.len(),
            out_ch * in_ch * kernel,
            out_ch,
            in_ch,
            kernel
        );

        let in_per_group = in_ch / groups;
        let out_per_group = out_ch / groups;
        let num_blocks_per_group = out_per_group.div_ceil(4);
        let total_blocks = groups * num_blocks_per_group;
        let total_padded = total_blocks * 4 * in_per_group * kernel;

        let mut weights = AlignedVec::new(total_padded, 0.0f32);
        let bias = AlignedVec::from(raw_bias.to_vec());

        // Permute: raw[out][in][kernel] → grouped-interleaved-4-wide
        for g in 0..groups {
            let group_in_start = g * in_per_group;
            let group_out_start = g * out_per_group;
            for b in 0..num_blocks_per_group {
                for k in 0..kernel {
                    for ic in 0..in_per_group {
                        for lane in 0..4 {
                            let out_idx = b * 4 + lane;
                            if out_idx < out_per_group {
                                let src = (group_out_start + out_idx) * in_ch * kernel
                                    + (group_in_start + ic) * kernel
                                    + k;
                                let dst = g * (num_blocks_per_group * kernel * in_per_group * 4)
                                    + b * (kernel * in_per_group * 4)
                                    + k * (in_per_group * 4)
                                    + ic * 4
                                    + lane;
                                weights[dst] = raw_weights[src];
                            }
                        }
                    }
                }
            }
        }

        Self {
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            groups,
            in_per_group,
            out_per_group,
            num_blocks_per_group,
            total_blocks,
            kernel,
            prefetch_fn,
        }
    }

    /// Processes a single frame through the grouped dilated convolution using SIMD AVX2.
    ///
    /// Dispatches to the depthwise fast-path when `groups == in_ch == out_ch`,
    /// otherwise uses the general grouped AVX2 kernel.
    ///
    /// # Safety
    /// `layer_buffer` must contain valid elements for the dilated tap indices.
    /// `out_frame` must have length at least `self.out_ch`.
    #[inline(always)]
    pub unsafe fn process_single_frame(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert!(
            out_frame.len() >= self.out_ch,
            "out_frame len {} < out_ch {}",
            out_frame.len(),
            self.out_ch
        );
        let lookback = self.dilation * (self.kernel.saturating_sub(1));
        debug_assert!(
            frame_idx >= lookback,
            "frame_idx {} < lookback {} (dilation={} * kernel-1={})",
            frame_idx,
            lookback,
            self.dilation,
            self.kernel.saturating_sub(1)
        );
        debug_assert!(
            layer_buffer.len() > frame_idx * self.in_ch,
            "layer_buffer len {} <= frame_idx {} * in_ch {}",
            layer_buffer.len(),
            frame_idx,
            self.in_ch
        );
        debug_assert!(
            mixin.is_none_or(|m| m.len() >= self.out_ch),
            "mixin len {:?} < out_ch {}",
            mixin.map(|m| m.len()),
            self.out_ch
        );

        unsafe {
            if self.groups == self.in_ch && self.groups == self.out_ch {
                // Depthwise: 1 channel per group, single weight per tap.
                process_single_frame_depthwise_avx2(self, layer_buffer, out_frame, frame_idx);
                // Apply mixin post-conv if present.
                if let Some(m) = mixin {
                    for c in 0..self.out_ch {
                        *out_frame.get_unchecked_mut(c) += *m.get_unchecked(c);
                    }
                }
            } else {
                grouped_conv1d_single_frame_simd(self, layer_buffer, out_frame, frame_idx, mixin);
            }
        }
    }

    /// Processes a block of `num_frames` consecutive frames.
    ///
    /// # Safety
    /// `layer_buffer` must be large enough for the dilated lookback.
    /// `block` must have size at least `num_frames * out_ch`.
    #[cfg(test)]
    #[inline(always)]
    pub unsafe fn process_block(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert!(
            block.len() >= num_frames * self.out_ch,
            "process_block: block len {} < num_frames {} * out_ch {}",
            block.len(),
            num_frames,
            self.out_ch
        );
        debug_assert!(
            mixin.is_none_or(|m| m.len() >= num_frames * self.out_ch),
            "process_block: mixin len {:?} < num_frames {} * out_ch {}",
            mixin.map(|m| m.len()),
            num_frames,
            self.out_ch
        );
        for f in 0..num_frames {
            let out_slice = &mut block[f * self.out_ch..(f + 1) * self.out_ch];
            let m = mixin.map(|full| &full[f * self.out_ch..(f + 1) * self.out_ch]);
            unsafe {
                grouped_conv1d_single_frame_simd(
                    self,
                    layer_buffer,
                    out_slice,
                    buffer_start + f,
                    m,
                );
            }
        }
    }
}

// =============================================================================
// Scalar reference — oracle for parity testing
// =============================================================================

/// Scalar reference for a single-frame grouped dilated conv with 64-bit accumulation.
///
/// Uses the same grouped-interleaved-4-wide weight layout as `A2GroupedConv1d`.
/// Each weight read uses f64 multiplication + accumulation for maximal precision.
pub fn grouped_conv1d_single_frame_ref(
    weights: &[f32],
    bias: &[f32],
    do_bias: bool,
    dilation: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    groups: usize,
    layer_buffer: &[f32],
    frame_idx: usize,
    out_frame: &mut [f32],
) {
    let in_per_group = in_ch / groups;
    let out_per_group = out_ch / groups;
    let num_blocks_per_group = out_per_group.div_ceil(4);

    // Init with bias
    for out_c in 0..out_ch {
        out_frame[out_c] = if do_bias { bias[out_c] } else { 0.0 };
    }

    for g in 0..groups {
        let group_in_start = g * in_per_group;
        let group_out_start = g * out_per_group;

        for b in 0..num_blocks_per_group {
            for k in 0..kernel {
                let taps_back = kernel - 1 - k;
                let col = frame_idx.wrapping_sub(dilation * taps_back);
                let src_base = col * in_ch;

                let wk_group_base = g * (num_blocks_per_group * kernel * in_per_group * 4);

                for ic in 0..in_per_group {
                    let hv = layer_buffer[src_base + group_in_start + ic];
                    let w_base = wk_group_base
                        + b * (kernel * in_per_group * 4)
                        + k * (in_per_group * 4)
                        + ic * 4;

                    let w0 = weights[w_base] as f64;
                    let w1 = weights[w_base + 1] as f64;
                    let w2 = weights[w_base + 2] as f64;
                    let w3 = weights[w_base + 3] as f64;
                    let hv64 = hv as f64;

                    {
                        let out0 = group_out_start + b * 4;
                        if out0 < out_ch {
                            out_frame[out0] = (out_frame[out0] as f64 + w0 * hv64) as f32;
                        }
                        if out0 + 1 < out_ch {
                            out_frame[out0 + 1] = (out_frame[out0 + 1] as f64 + w1 * hv64) as f32;
                        }
                        if out0 + 2 < out_ch {
                            out_frame[out0 + 2] = (out_frame[out0 + 2] as f64 + w2 * hv64) as f32;
                        }
                        if out0 + 3 < out_ch {
                            out_frame[out0 + 3] = (out_frame[out0 + 3] as f64 + w3 * hv64) as f32;
                        }
                    }
                }
            }
        }
    }
}

/// Scalar reference for a block of consecutive frames.
pub fn grouped_conv1d_block_ref(
    weights: &[f32],
    bias: &[f32],
    do_bias: bool,
    dilation: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    groups: usize,
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    block: &mut [f32],
) {
    for f in 0..num_frames {
        grouped_conv1d_single_frame_ref(
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            layer_buffer,
            frame_start + f,
            &mut block[f * out_ch..(f + 1) * out_ch],
        );
    }
}

/// Depthwise convolution optimized path (groups == in_ch == out_ch).
///
/// When every channel is its own group, the inner loop simplifies to a single
/// tap convolution per channel — one weight per input per tap.
///
/// # Safety
/// Only call when `groups == in_ch == out_ch`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn process_single_frame_depthwise_avx2(
    conv: &A2GroupedConv1d,
    layer_buffer: &[f32],
    out_frame: &mut [f32],
    frame_idx: usize,
) {
    debug_assert_eq!(conv.groups, conv.in_ch);
    debug_assert_eq!(conv.groups, conv.out_ch);
    debug_assert_eq!(conv.in_per_group, 1);
    debug_assert_eq!(conv.out_per_group, 1);
    debug_assert!(
        out_frame.len() >= conv.out_ch,
        "depthwise: out_frame len {} < out_ch {}",
        out_frame.len(),
        conv.out_ch
    );
    debug_assert!(
        frame_idx >= conv.dilation * (conv.kernel.saturating_sub(1)),
        "depthwise: frame_idx {} < lookback {}",
        frame_idx,
        conv.dilation * (conv.kernel.saturating_sub(1))
    );
    debug_assert!(
        layer_buffer.len() > frame_idx * conv.in_ch,
        "depthwise: layer_buffer len {} <= frame_idx {} * in_ch {}",
        layer_buffer.len(),
        frame_idx,
        conv.in_ch
    );

    let ch = conv.in_ch;
    let kernel = conv.kernel;
    let dilation = conv.dilation;

    let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
    let k_limit = kernel.min(MAX_KERNEL);

    for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
        let offset = (dilation as isize) * ((k as isize) + 1 - (kernel as isize));
        let in_start = ((frame_idx as isize) + offset) as usize * ch;
        unsafe {
            *tap = layer_buffer.as_ptr().add(in_start);
            (conv.prefetch_fn)(*tap, dilation * ch, k, kernel, dilation);
        }
    }

    // Depthwise: num_blocks_per_group = 1, in_per_group = 1
    // Weight layout: group[g] → block[0] → kernel[k] → in[0] → lanes[4] with only lane 0 valid
    let w_ptr = conv.weights.as_ptr();
    let bias_ptr = conv.bias.as_ptr();
    let group_stride = kernel * 4;

    // Process 8 channels at a time with AVX2 gather
    let ch8 = ch & !7;

    // Precompute gather indices: for channels [c0..c0+7] at a given tap k,
    // weight for channel c0+i is at w_ptr + (c0+i)*group_stride + k*4.
    // Gather delta from base (channel c0): i * group_stride f32 elements = i*kernel*4*4 bytes.
    // _mm256_i32gather_ps: addr = base + vindex[i]*scale bytes, scale=4 for f32.
    let gather_idx = _mm256_setr_epi32(
        0,
        group_stride as i32,
        (group_stride * 2) as i32,
        (group_stride * 3) as i32,
        (group_stride * 4) as i32,
        (group_stride * 5) as i32,
        (group_stride * 6) as i32,
        (group_stride * 7) as i32,
    );

    let mut c = 0;
    while c < ch8 {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut k = 0;
        while k + 1 < kernel {
            let tap0 = *tap_ptrs.get_unchecked(k);
            let w_base0 = w_ptr.add(c * group_stride + k * 4);
            let wv0 = _mm256_i32gather_ps(w_base0, gather_idx, 4);
            let sv0 = _mm256_loadu_ps(tap0.add(c));
            acc0 = _mm256_fmadd_ps(wv0, sv0, acc0);

            let tap1 = *tap_ptrs.get_unchecked(k + 1);
            let w_base1 = w_ptr.add(c * group_stride + (k + 1) * 4);
            let wv1 = _mm256_i32gather_ps(w_base1, gather_idx, 4);
            let sv1 = _mm256_loadu_ps(tap1.add(c));
            acc1 = _mm256_fmadd_ps(wv1, sv1, acc1);

            k += 2;
        }
        if k < kernel {
            let tap = *tap_ptrs.get_unchecked(k);
            let w_base = w_ptr.add(c * group_stride + k * 4);
            let wv = _mm256_i32gather_ps(w_base, gather_idx, 4);
            let sv = _mm256_loadu_ps(tap.add(c));
            acc0 = _mm256_fmadd_ps(wv, sv, acc0);
        }
        let mut acc = _mm256_add_ps(acc0, acc1);

        if conv.do_bias {
            let bv = _mm256_loadu_ps(bias_ptr.add(c));
            acc = _mm256_add_ps(acc, bv);
        }

        _mm256_storeu_ps(out_frame.as_mut_ptr().add(c), acc);
        c += 8;
    }

    // Scalar tail for remaining channels
    for c in c..ch {
        let mut acc = 0.0f32;

        for k in 0..kernel {
            let tap = *tap_ptrs.get_unchecked(k);
            let w = *w_ptr.add(c * group_stride + k * 4);
            let s = *tap.add(c);
            acc += w * s;
        }

        if conv.do_bias {
            acc += *conv.bias.get_unchecked(c);
        }

        *out_frame.get_unchecked_mut(c) = acc;
    }
}

// =============================================================================
// Helpers
// =============================================================================

#[inline(always)]
unsafe fn load_mixin_4(mixin: Option<&[f32]>, out_c: usize, out_ch: usize) -> (f32, f32, f32, f32) {
    debug_assert!(
        mixin.is_none_or(|m| m.len() >= out_ch),
        "load_mixin_4: mixin len {:?} < out_ch {}",
        mixin.map(|m| m.len()),
        out_ch
    );
    if let Some(m) = mixin {
        if out_c + 3 < out_ch {
            (
                *m.get_unchecked(out_c),
                *m.get_unchecked(out_c + 1),
                *m.get_unchecked(out_c + 2),
                *m.get_unchecked(out_c + 3),
            )
        } else {
            let mut v = [0.0f32; 4];
            for (i, val) in v.iter_mut().enumerate() {
                if out_c + i < out_ch {
                    *val = *m.get_unchecked(out_c + i);
                }
            }
            (v[0], v[1], v[2], v[3])
        }
    } else {
        (0.0, 0.0, 0.0, 0.0)
    }
}

// =============================================================================
// Simpler inner loop using __m128 accumulators directly (replaces extract pattern above)
// =============================================================================

/// Grouped dilated conv — alternative inner loop using `__m128` accumulator registers
/// directly for the 4-lane block, avoiding the scalar extract/reinsert pattern.
///
/// Same semantics as `process_single_frame_avx2` but keeps accumulators in XMM
/// registers across the entire inner loop, yielding better code generation.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn grouped_conv1d_single_frame_simd(
    conv: &A2GroupedConv1d,
    layer_buffer: &[f32],
    out_frame: &mut [f32],
    frame_idx: usize,
    mixin: Option<&[f32]>,
) {
    debug_assert!(
        out_frame.len() >= conv.out_ch,
        "simd: out_frame len {} < out_ch {}",
        out_frame.len(),
        conv.out_ch
    );
    debug_assert!(
        frame_idx >= conv.dilation * (conv.kernel.saturating_sub(1)),
        "simd: frame_idx {} < lookback {}",
        frame_idx,
        conv.dilation * (conv.kernel.saturating_sub(1))
    );
    debug_assert!(
        layer_buffer.len() > frame_idx * conv.in_ch,
        "simd: layer_buffer len {} <= frame_idx {} * in_ch {}",
        layer_buffer.len(),
        frame_idx,
        conv.in_ch
    );
    debug_assert!(
        mixin.is_none_or(|m| m.len() >= conv.out_ch),
        "simd: mixin len {:?} < out_ch {}",
        mixin.map(|m| m.len()),
        conv.out_ch
    );

    let in_ch = conv.in_ch;
    let in_per_group = conv.in_per_group;
    let out_per_group = conv.out_per_group;
    let kernel = conv.kernel;
    let dilation = conv.dilation;
    let num_blocks_per_group = conv.num_blocks_per_group;
    let groups = conv.groups;

    let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
    let k_limit = kernel.min(MAX_KERNEL);

    for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
        let offset = (dilation as isize) * ((k as isize) + 1 - (kernel as isize));
        let in_start = ((frame_idx as isize) + offset) as usize * in_ch;
        unsafe {
            *tap = layer_buffer.as_ptr().add(in_start);
            (conv.prefetch_fn)(*tap, dilation * in_ch, k, kernel, dilation);
        }
    }

    for g in 0..groups {
        let group_in_start = g * in_per_group;
        let group_out_start = g * out_per_group;

        for b in 0..num_blocks_per_group {
            let out_c = group_out_start + b * 4;
            let (mu0, mu1, mu2, mu3) = load_mixin_4(mixin, out_c, conv.out_ch);

            let mut acc = if conv.do_bias {
                _mm_setr_ps(
                    *conv.bias.get_unchecked(out_c) + mu0,
                    if out_c + 1 < conv.out_ch {
                        *conv.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mu1,
                    if out_c + 2 < conv.out_ch {
                        *conv.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mu2,
                    if out_c + 3 < conv.out_ch {
                        *conv.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mu3,
                )
            } else {
                _mm_setr_ps(mu0, mu1, mu2, mu3)
            };

            for ik in 0..kernel {
                let tap = *tap_ptrs.get_unchecked(ik);
                let wk_group_base = g * (num_blocks_per_group * kernel * in_per_group * 4);

                for ic in 0..in_per_group {
                    let w_base = wk_group_base
                        + b * (kernel * in_per_group * 4)
                        + ik * (in_per_group * 4)
                        + ic * 4;

                    let wv = _mm_loadu_ps(conv.weights.as_ptr().add(w_base));
                    let sv = _mm_set1_ps(*tap.add(group_in_start + ic));
                    acc = _mm_fmadd_ps(wv, sv, acc);
                }
            }

            if out_c + 3 < conv.out_ch {
                _mm_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc);
            } else {
                let r = core::mem::transmute::<__m128, [f32; 4]>(acc);
                for (lane, &val) in r.iter().enumerate() {
                    if out_c + lane < conv.out_ch {
                        *out_frame.get_unchecked_mut(out_c + lane) = val;
                    }
                }
            }
        }
    }
}

// =============================================================================
// Test helpers (pub(crate) for cross-module reuse)
// =============================================================================

#[cfg(test)]
pub(crate) fn make_test_weights_grouped(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    _groups: usize,
    seed: u32,
) -> (Vec<f32>, Vec<f32>) {
    let total_w = out_ch * in_ch * kernel;
    let mut raw_weights = Vec::with_capacity(total_w);
    let mut state = seed;
    for _ in 0..total_w {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        raw_weights.push(((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25);
    }
    let mut bias = Vec::with_capacity(out_ch);
    for _ in 0..out_ch {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        bias.push(((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1);
    }
    (raw_weights, bias)
}

#[cfg(test)]
pub(crate) fn make_layer_buffer(buf_frames: usize, in_ch: usize, seed: u32) -> Vec<f32> {
    let mut buf = vec![0.0f32; buf_frames * in_ch];
    let mut state = seed;
    for val in &mut buf {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }
    buf
}

#[cfg(test)]
#[path = "grouped_conv1d_test.rs"]
mod tests;
