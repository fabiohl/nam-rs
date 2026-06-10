// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! CH=8 T=4 frame-tiled tap-major convolution with broadcast-FMA (T2.2).
//!
//! When `out_ch == 8`, the generic 4-wide interleaved scheme processes frames
//! one at a time. This module implements a **block-level** kernel that processes
//! 4 consecutive frames per tile, amortizing weight loads across all 4 frames
//! via SIMD broadcast-FMA instructions. Weights are stored in **col-major-per-tap**
//! layout (`w[k * 64 + in * 8 + out]`) so that the 8 output-channel weights for
//! a single (tap, input_channel) pair are contiguous — one `_mm256_loadu_ps` loads
//! them all.
//!
//! For each tile of T=4 frames, the inner loop is:
//!
//! ```text
//! a[f][o] += Wcol[o] * h[f]   (o vectorized, h[f] scalar broadcast)
//! ```
//!
//! On x86-64-v3 this emits `vfmadd231ps` (broadcast-FMA).
//!
//! ## Source of truth
//! - `a2_fast.cpp:617-681` (strategy `Channels >= 8`, T=4 tap-major).

use crate::math::common::AlignedVec;
use crate::models::a2::params::A2_LEAKY_SLOPE;
use core::arch::x86_64::*;

// =============================================================================
// A2Conv1dCh8 — CH=8 convolution with col-major-per-tap f32 weights
// =============================================================================

/// CH=8 dilated causal Conv1D weights in col-major-per-tap layout.
///
/// Layout: `w[k * 64 + in_ch * 8 + out_ch]`
/// - `k`: kernel tap index (0..K-1)
/// - `in_ch`: input channel (0..7)
/// - `out_ch`: output channel (0..7)
///
/// For a given `(k, in_ch)`, the 8 output weights are contiguous → one AVX2 load.
#[derive(Clone)]
#[repr(align(64))]
pub struct A2Conv1dCh8 {
    /// Col-major-per-tap f32 weights: `kernel_size * 64` elements.
    pub weights: AlignedVec<f32>,
    /// Bias vector [8], f32.
    pub bias: AlignedVec<f32>,
    /// Temporal dilation factor.
    pub dilation: usize,
    /// Kernel size (6 or 15 for A2).
    pub kernel: usize,
}

impl A2Conv1dCh8 {
    /// Builds a CH=8 conv1d from the weight data read in NAM JSON order.
    ///
    /// `raw` is in NAM JSON row-major order: `[out_ch][in_ch][kernel]`.
    /// This constructor permutes to col-major-per-tap: `[kernel][in_ch][out_ch]`.
    pub fn new(
        raw_weights: &[f32],
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        dilation: usize,
        bias: AlignedVec<f32>,
    ) -> Self {
        debug_assert_eq!(out_ch, 8);
        debug_assert_eq!(in_ch, 8);
        debug_assert!(kernel == 6 || kernel == 15);
        debug_assert_eq!(raw_weights.len(), out_ch * in_ch * kernel);
        debug_assert_eq!(bias.len(), out_ch);

        let mut weights = AlignedVec::new(kernel * 64, 0.0f32);
        for out in 0..out_ch {
            for inp in 0..in_ch {
                for k in 0..kernel {
                    let src = out * in_ch * kernel + inp * kernel + k;
                    let dst = k * 64 + inp * 8 + out;
                    weights[dst] = raw_weights[src];
                }
            }
        }

        Self {
            weights,
            bias,
            dilation,
            kernel,
        }
    }
}

// =============================================================================
// Scalar reference — oracle for parity testing
// =============================================================================

/// Scalar reference for a single-frame CH=8 dilated conv.
///
/// Replicates the exact math using 64-bit accumulators for maximum precision,
/// matching the semantics of the AVX2 broadcast-FMA kernel.
pub fn conv1d_ch8_single_frame_ref(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    kernel: usize,
    layer_buffer: &[f32],
    frame_idx: usize,
    out_frame: &mut [f32],
) {
    debug_assert!(out_frame.len() >= 8);
    debug_assert!(weights.len() >= kernel * 64);

    out_frame[..8].copy_from_slice(&bias[..8]);

    for k in 0..kernel {
        let wk_base = k * 64;
        let taps_back = kernel - 1 - k;
        let col = frame_idx.wrapping_sub(dilation * taps_back);
        let src_base = col * 8;

        for inp in 0..8 {
            let hv = layer_buffer[src_base + inp];
            let w_base = wk_base + inp * 8;
            for out in 0..8 {
                out_frame[out] += weights[w_base + out] * hv;
            }
        }
    }
}

/// Scalar reference for a block of `num_frames` consecutive frames.
pub fn conv1d_ch8_block_ref(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    kernel: usize,
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    z_out: &mut [f32],
) {
    debug_assert!(z_out.len() >= num_frames * 8);

    for f in 0..num_frames {
        conv1d_ch8_single_frame_ref(
            weights,
            bias,
            dilation,
            kernel,
            layer_buffer,
            frame_start + f,
            &mut z_out[f * 8..(f + 1) * 8],
        );
    }
}

// =============================================================================
// AVX2 T=4 tap-major kernel
// =============================================================================

/// T=4 frame-tiled tap-major dilated conv for CH=8, AVX2+FMA.
///
/// Processes frames in groups of 4, accumulating all K taps into `T*C`
/// register-allocated accumulators. For each (tap, input_channel) pair,
/// loads the 8 output-channel weights once and broadcasts the history
/// value for each of the 4 frames via `_mm256_set1_ps`.
///
/// # Safety
/// - `weights` must have at least `kernel * 64` valid f32 elements.
/// - `layer_buffer` must be large enough for all frame lookbacks.
/// - `z_out` must have at least `num_frames * 8` elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn conv1d_ch8_t4_avx2(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    kernel: usize,
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    z_out: &mut [f32],
) {
    debug_assert!(z_out.len() >= num_frames * 8);
    debug_assert!(weights.len() >= kernel * 64);
    debug_assert!(bias.len() >= 8);

    let ch: usize = 8;
    let d = dilation as isize;
    let k_i = kernel as isize;
    let buf = layer_buffer.as_ptr();
    let w_ptr = weights.as_ptr();

    let bias_v = _mm256_loadu_ps(bias.as_ptr());

    const T: usize = 4;
    let n_tiled = (num_frames / T) * T;

    for f in (0..n_tiled).step_by(T) {
        let mut a0 = bias_v;
        let mut a1 = bias_v;
        let mut a2 = bias_v;
        let mut a3 = bias_v;

        let frame0 = (frame_start + f) as isize;

        for k in 0..kernel {
            let wk_base = (k * 64) as isize;
            let taps_back = k_i - 1 - k as isize;
            let tap0 = frame0 - d * taps_back;
            let hb = buf.offset(tap0 * ch as isize);
            for cp in 0..ch {
                let wcol = _mm256_loadu_ps(w_ptr.offset(wk_base + (cp * 8) as isize));
                let h0 = *hb.add(cp);
                let h1 = *hb.add(ch + cp);
                let h2 = *hb.add(2 * ch + cp);
                let h3 = *hb.add(3 * ch + cp);
                a0 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h0), a0);
                a1 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h1), a1);
                a2 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h2), a2);
                a3 = _mm256_fmadd_ps(wcol, _mm256_set1_ps(h3), a3);
            }
        }

        _mm256_storeu_ps(z_out.as_mut_ptr().add(f * ch), a0);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 1) * ch), a1);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 2) * ch), a2);
        _mm256_storeu_ps(z_out.as_mut_ptr().add((f + 3) * ch), a3);
    }

    for f in n_tiled..num_frames {
        let frame_idx = (frame_start + f) as isize;
        let mut acc = bias_v;
        for k in 0..kernel {
            let wk_base = (k * 64) as isize;
            let taps_back = k_i - 1 - k as isize;
            let tap_base = frame_idx - d * taps_back;
            let hb = buf.offset(tap_base * ch as isize);
            for cp in 0..ch {
                let wcol = _mm256_loadu_ps(w_ptr.offset(wk_base + (cp * 8) as isize));
                let hv = *hb.add(cp);
                acc = _mm256_fmadd_ps(wcol, _mm256_set1_ps(hv), acc);
            }
        }
        _mm256_storeu_ps(z_out.as_mut_ptr().add(f * ch), acc);
    }
}

// =============================================================================
// Block layer forward pass (conv + post-conv)
// =============================================================================

/// Full layer forward pass for CH=8 using T=4 tiled tap-major conv.
///
/// Processes `num_frames` through: dilated conv → bias → mixin → LeakyReLU →
/// head accumulate → l1x1 residual. All operations use SIMD block processing
/// on `__m256` vectors.
///
/// # Safety
/// Buffers must be sized appropriately. Caller ensures linear ring history
/// includes lookback + block frames.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn layer_forward_ch8_block(
    conv: &A2Conv1dCh8,
    mixin_w: &[f32],
    l1x1_w: &[f32],
    l1x1_b: &[f32],
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    input_cond: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch: usize = 8;
    debug_assert!(mixin_w.len() >= ch);
    debug_assert!(l1x1_w.len() >= ch * ch);
    debug_assert!(l1x1_b.len() >= ch);
    debug_assert!(layer_in.len() >= num_frames * ch);
    debug_assert!(input_cond.len() >= num_frames);

    let max_frames = 64;
    debug_assert!(num_frames <= max_frames);
    let mut z_buf = [0.0f32; 64 * 8];

    conv1d_ch8_t4_avx2(
        &conv.weights,
        &conv.bias,
        conv.dilation,
        conv.kernel,
        layer_buffer,
        frame_start,
        num_frames,
        &mut z_buf[..num_frames * ch],
    );

    // 2. Post-conv: mixin + LeakyReLU (in-place on z_buf).
    {
        let z = z_buf.as_mut_ptr();
        let mixin_v = _mm256_loadu_ps(mixin_w.as_ptr());
        let slope_v = _mm256_set1_ps(A2_LEAKY_SLOPE);
        let zero_v = _mm256_setzero_ps();
        for (f, cond_val) in input_cond.iter().take(num_frames).enumerate() {
            let off = f * ch;
            let mut zv = _mm256_loadu_ps(z.add(off));
            let cond_v = _mm256_set1_ps(*cond_val);
            zv = _mm256_fmadd_ps(mixin_v, cond_v, zv);
            let mask = _mm256_cmp_ps(zv, zero_v, _CMP_LT_OS);
            let zv_leaky = _mm256_mul_ps(zv, slope_v);
            zv = _mm256_blendv_ps(zv, zv_leaky, mask);
            _mm256_storeu_ps(z.add(off), zv);
        }
    }

    // 3. Head accumulate.
    {
        let head = head_accum.as_mut_ptr();
        for f in 0..num_frames {
            let head_off = (head_col + f) * ch;
            let zv = _mm256_loadu_ps(z_buf.as_ptr().add(f * ch));
            if is_first {
                _mm256_storeu_ps(head.add(head_off), zv);
            } else {
                let hv = _mm256_loadu_ps(head.add(head_off));
                _mm256_storeu_ps(head.add(head_off), _mm256_add_ps(hv, zv));
            }
        }
    }

    // 4. Layer1x1 residual (skipped on last layer).
    if !is_last {
        let lin = layer_in.as_mut_ptr();
        let l1x1_b_v = _mm256_loadu_ps(l1x1_b.as_ptr());
        let l1x1_w_ptr = l1x1_w.as_ptr();
        for f in 0..num_frames {
            let off = f * ch;
            let mut acc = l1x1_b_v;
            for u in 0..ch {
                let zu = *z_buf.get_unchecked(off + u);
                let zu_v = _mm256_set1_ps(zu);
                let w_col = _mm256_loadu_ps(l1x1_w_ptr.add(u * ch));
                acc = _mm256_fmadd_ps(zu_v, w_col, acc);
            }
            let lv = _mm256_loadu_ps(lin.add(off));
            _mm256_storeu_ps(lin.add(off), _mm256_add_ps(lv, acc));
        }
    }
}

// =============================================================================
// Scalar reference for full layer forward (oracle)
// =============================================================================

/// Scalar reference for the full CH=8 layer forward pass.
///
/// Matches `layer_forward_ch8_block` semantics exactly. Used as oracle
/// for parity testing.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn layer_forward_ch8_scalar_ref(
    conv_weights: &[f32],
    conv_bias: &[f32],
    dilation: usize,
    kernel: usize,
    mixin_w: &[f32],
    l1x1_w: &[f32],
    l1x1_b: &[f32],
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    input_cond: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch: usize = 8;
    let max_frames = 64;
    debug_assert!(num_frames <= max_frames);
    let mut z_buf = [0.0f32; 64 * 8];

    conv1d_ch8_block_ref(
        conv_weights,
        conv_bias,
        dilation,
        kernel,
        layer_buffer,
        frame_start,
        num_frames,
        &mut z_buf[..num_frames * ch],
    );

    for f in 0..num_frames {
        let off = f * ch;
        for c in 0..ch {
            z_buf[off + c] += mixin_w[c] * input_cond[f];
            if z_buf[off + c] < 0.0 {
                z_buf[off + c] *= A2_LEAKY_SLOPE;
            }
        }
        let head_off = (head_col + f) * ch;
        if is_first {
            head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[off..off + ch]);
        } else {
            for c in 0..ch {
                head_accum[head_off + c] += z_buf[off + c];
            }
        }
    }

    if !is_last {
        for f in 0..num_frames {
            let off = f * ch;
            for c in 0..ch {
                let mut sum = l1x1_b[c];
                for u in 0..ch {
                    sum += l1x1_w[u * ch + c] * z_buf[off + u];
                }
                layer_in[off + c] += sum;
            }
        }
    }
}

#[cfg(test)]
#[path = "conv1d_ch8_test.rs"]
mod tests;
