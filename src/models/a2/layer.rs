// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Layer — per-layer forward pass for the A2 WaveNet fast-path.
//!
//! Implements the sequence from `a2_fast.cpp:514`:
//! 1. Dilated causal Conv1D → 2. Input mixin (condition × mixin_w, no bias) →
//! 3. LeakyReLU(0.01) → 4. Head accumulator (assign layer 0, accumulate layers 1-22) →
//! 5. Layer1x1 residual `layer_in += l1x1_b + l1x1_w * z` (skipped on last layer).
//!
//! ## Source of truth
//! - `a2_fast.cpp:417-690` (`_layer_forward_k`)
//! - `NAM/wavenet/detail.h` (`Layer`)
//! - `docs/wavenet_walkthrough.rst:103-214`

use super::conv1d::A2Conv1d;
use super::conv1d_ch8::A2Conv1dCh8;
use super::params::A2_LEAKY_SLOPE;
use crate::math::common::AlignedVec;

/// Single A2 WaveNet layer.
///
/// Holds the weights for: dilated conv (via `A2Conv1d`), input mixin (`Conv1x1 condition→CH`, no bias),
/// and layer1x1 (`Conv1x1 CH→CH`, with bias, col-major).
///
/// When `ch8_conv` is `Some`, it holds f32 col-major-per-tap weights for the CH=8 optimized path (T2.2).
/// When `None`, the standard `A2Conv1d` (u16 interleaved) is used (CH=3 path and fallback).
pub struct A2Layer {
    /// Dilated causal Conv1D (kernel ∈ {6, 15}). Used by CH=3 path and as u16 fallback.
    pub conv: A2Conv1d,
    /// CH=8 optimized weights (f32 col-major-per-tap). Only populated when CH=8.
    pub ch8_conv: Option<A2Conv1dCh8>,
    /// Input mixin weights (`CH` elements, f32).
    pub mixin_w: AlignedVec<f32>,
    /// Layer1x1 weights (`CH × CH`, col-major: `[bottleneck][out]`).
    pub l1x1_w: AlignedVec<f32>,
    /// Layer1x1 bias (`CH` elements, f32).
    pub l1x1_b: AlignedVec<f32>,
}

impl A2Layer {
    /// The mixin has no bias in the A2 fast-path (cond_size=1, `Conv1x1 condition→CH`).
    pub fn new(
        conv: A2Conv1d,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
    ) -> Self {
        let ch = conv.out_ch();
        debug_assert_eq!(mixin_w.len(), ch);
        debug_assert_eq!(l1x1_w.len(), ch * ch);
        debug_assert_eq!(l1x1_b.len(), ch);
        Self {
            conv,
            ch8_conv: None,
            mixin_w,
            l1x1_w,
            l1x1_b,
        }
    }

    /// Creates a layer with CH=8 optimized weights.
    pub fn new_with_ch8(
        conv: A2Conv1d,
        ch8_conv: A2Conv1dCh8,
        mixin_w: AlignedVec<f32>,
        l1x1_w: AlignedVec<f32>,
        l1x1_b: AlignedVec<f32>,
    ) -> Self {
        let ch = conv.out_ch();
        debug_assert_eq!(ch, 8);
        debug_assert_eq!(mixin_w.len(), ch);
        debug_assert_eq!(l1x1_w.len(), ch * ch);
        debug_assert_eq!(l1x1_b.len(), ch);
        Self {
            conv,
            ch8_conv: Some(ch8_conv),
            mixin_w,
            l1x1_w,
            l1x1_b,
        }
    }

    /// Channel count (bottleneck == channels in A2 fast-path).
    #[inline(always)]
    pub fn channels(&self) -> usize {
        self.conv.out_ch()
    }

    /// Kernel size of this layer's dilated conv.
    #[inline(always)]
    pub fn kernel_size(&self) -> usize {
        self.conv.kernel_size()
    }

    /// Dilation factor of this layer.
    #[inline(always)]
    pub fn dilation(&self) -> usize {
        self.conv.dilation()
    }

    /// Processes a single frame through this layer.
    ///
    /// ## Data flow (matches `_layer_forward_k` in `a2_fast.cpp:514`)
    ///
    /// 1. Dilated conv over `layer_history` → `z_buf[..CH]`.
    /// 2. Mixin: `z_buf[c] += mixin_w[c] * input_cond`.
    /// 3. LeakyReLU(0.01) in-place.
    /// 4. Head accumulator: assign (layer 0) or add (layer > 0) into `head_accum`.
    /// 5. L1x1 residual: `layer_in[c] += l1x1_b[c] + sum_u(l1x1_w[u*CH+c] * z_buf[u])`.
    ///    Skipped on the last layer (output of last layer is dead — only head matters).
    ///
    /// # Parameters
    /// * `layer_history` — per-layer ring buffer (column-major: CH rows × N cols).
    /// * `frame_idx` — absolute column index in `layer_history` for the dilated conv (already ring-masked or linear).
    /// * `input_cond` — scalar input condition for the mixin (original input signal at this frame).
    /// * `head_accum` — head accumulator ring buffer (column-major).
    /// * `head_col` — column index in `head_accum` for this frame's output.
    /// * `z_buf` — scratch buffer for conv output (length ≥ `channels()`).
    /// * `layer_in_out` — mutable reference to this frame's layer_in (will be updated).
    /// * `is_first` — layer 0 writes to head, layers 1-22 accumulate.
    /// * `is_last` — layer 22 skips l1x1 residual.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    pub fn process_single_frame(
        &self,
        layer_history: &[f32],
        frame_idx: usize,
        input_cond: f32,
        head_accum: &mut [f32],
        head_col: usize,
        z_buf: &mut [f32],
        layer_in_out: &mut [f32],
        is_first: bool,
        is_last: bool,
    ) {
        let ch = self.channels();
        debug_assert!(z_buf.len() >= ch);
        debug_assert!(layer_in_out.len() >= ch);

        // 1. Dilated conv (no mixin — A2 adds mixin after conv).
        unsafe {
            self.conv
                .process_single_frame::<crate::math::common::Avx2Math>(
                    layer_history,
                    z_buf,
                    frame_idx,
                    None,
                );
        }

        // 2. Input mixin: z_buf[c] += mixin_w[c] * input_cond.
        let mixin = &self.mixin_w;
        for c in 0..ch {
            z_buf[c] += mixin[c] * input_cond;
        }

        // 3. LeakyReLU(0.01) in-place.
        for z in z_buf.iter_mut().take(ch) {
            if *z < 0.0 {
                *z *= A2_LEAKY_SLOPE;
            }
        }

        // 4. Head accumulator.
        let head_off = head_col * ch;
        if is_first {
            head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[..ch]);
        } else {
            for (c, z_val) in z_buf.iter().enumerate().take(ch) {
                head_accum[head_off + c] += *z_val;
            }
        }

        // 5. L1x1 residual (skipped on last layer).
        if !is_last {
            let l1x1 = &self.l1x1_w;
            let bias = &self.l1x1_b;
            for c in 0..ch {
                let mut sum = bias[c];
                for u in 0..ch {
                    // Col-major: l1x1_w[u * ch + c] = weight from bottleneck u to output c.
                    sum += l1x1[u * ch + c] * z_buf[u];
                }
                layer_in_out[c] += sum;
            }
        }
    }
}

// =============================================================================
// Scalar reference (oracle) for parity testing
// =============================================================================

/// Scalar reference for the A2Layer forward pass on a single frame.
///
/// Replicates the exact computation using the scalar conv fallback, then
/// applies mixin, LeakyReLU, head accumulation, and l1x1 residual.
/// Used as oracle in parity tests.
#[allow(clippy::too_many_arguments)]
pub fn a2_layer_single_frame_scalar_ref(
    conv_weights: &[u16],
    conv_bias: &[f32],
    conv_do_bias: bool,
    dilation: usize,
    kernel_size: usize,
    layer_history: &[f32],
    frame_idx: usize,
    mixin_w: &[f32],
    input_cond: f32,
    l1x1_w: &[f32],
    l1x1_b: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in_out: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch = mixin_w.len();

    // 1. Dilated conv (scalar fallback).
    let mut z_buf = vec![0.0f32; ch];
    super::conv1d_fallback::a2_conv1d_single_frame_fallback(
        conv_weights,
        conv_bias,
        conv_do_bias,
        dilation,
        ch, // in_ch == out_ch for A2 fast-path
        ch,
        kernel_size,
        layer_history,
        frame_idx,
        None,
        &mut z_buf,
    );

    // 2. Mixin.
    for c in 0..ch {
        z_buf[c] += mixin_w[c] * input_cond;
    }

    // 3. LeakyReLU(0.01).
    for z in z_buf.iter_mut().take(ch) {
        if *z < 0.0 {
            *z *= A2_LEAKY_SLOPE;
        }
    }

    // 4. Head accumulator.
    let head_off = head_col * ch;
    if is_first {
        head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[..ch]);
    } else {
        for c in 0..ch {
            head_accum[head_off + c] += z_buf[c];
        }
    }

    // 5. L1x1 residual.
    if !is_last {
        for c in 0..ch {
            let mut sum = l1x1_b[c];
            for u in 0..ch {
                sum += l1x1_w[u * ch + c] * z_buf[u];
            }
            layer_in_out[c] += sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::common::prefetch_strategy_simple;
    use crate::models::a2::A2_DILATIONS;

    fn make_conv_weights(
        ch: usize,
        kernel: usize,
        seed: u32,
    ) -> (AlignedVec<u16>, AlignedVec<f32>) {
        let num_blocks = ch.div_ceil(4);
        let total_w = num_blocks * 4 * ch * kernel;
        let mut weights = AlignedVec::new(total_w, 0u16);
        let mut state = seed;
        for i in 0..total_w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
            weights[i] = half::f16::from_f32(v).to_bits();
        }
        let mut bias = AlignedVec::new(ch, 0.0f32);
        for i in 0..ch {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            bias[i] = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
        }
        (weights, bias)
    }

    fn make_f32_vec(len: usize, seed: u32) -> AlignedVec<f32> {
        let mut v = AlignedVec::new(len, 0.0f32);
        let mut state = seed;
        for i in 0..len {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            v[i] = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
        }
        v
    }

    fn make_history(num_cols: usize, ch: usize, seed: u32) -> Vec<f32> {
        let mut v = vec![0.0f32; num_cols * ch];
        let mut state = seed;
        for val in &mut v {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        v
    }

    /// End-to-end parity test: CH=3, kernel=6, dilation=101, all frames.
    #[test]
    fn test_a2_layer_ch3_kernel6_parity() {
        let ch = 3usize;
        let kernel = 6;
        let dilation = A2_DILATIONS[5]; // 101
        let num_frames = 16;

        let (conv_w, conv_bias) = make_conv_weights(ch, kernel, 42);
        let conv = A2Conv1d::new(
            conv_w.clone(),
            conv_bias.clone(),
            true,
            dilation,
            ch,
            ch,
            kernel,
            prefetch_strategy_simple,
        );
        let mixin_w = make_f32_vec(ch, 100);
        let l1x1_w = make_f32_vec(ch * ch, 200);
        let l1x1_b = make_f32_vec(ch, 300);

        let layer = A2Layer::new(conv, mixin_w.clone(), l1x1_w.clone(), l1x1_b.clone());

        // History buffer: needs enough columns for lookback + write buffer.
        let max_lookback = (kernel - 1) * dilation;
        let hist_cols = max_lookback + num_frames + 8;
        let history = make_history(hist_cols, ch, 77);
        let hist_write_pos = max_lookback + 4; // wp after prewarm, before ring-write

        let input_cond: Vec<f32> = (0..num_frames).map(|i| (i as f32).sin() * 0.5).collect();

        let mut layer_in_simd = vec![0.0f32; num_frames * ch];
        let mut layer_in_scalar = vec![0.0f32; num_frames * ch];
        let mut head_simd = vec![0.0f32; (num_frames + 1) * ch];
        let mut head_scalar = vec![0.0f32; (num_frames + 1) * ch];

        // SIMD path: process frame-by-frame.
        {
            let mut z_buf = vec![0.0f32; ch];
            for f in 0..num_frames {
                let frame_idx = hist_write_pos + f; // = wp (ring-write position) + f
                let lin_slice = &mut layer_in_simd[f * ch..(f + 1) * ch];
                layer.process_single_frame(
                    &history,
                    frame_idx,
                    input_cond[f],
                    &mut head_simd,
                    f,
                    &mut z_buf,
                    lin_slice,
                    f == 0,
                    f == num_frames - 1,
                );
            }
        }

        // Scalar reference path.
        {
            for f in 0..num_frames {
                let lin_slice = &mut layer_in_scalar[f * ch..(f + 1) * ch];
                let frame_idx = hist_write_pos + f;
                a2_layer_single_frame_scalar_ref(
                    &layer.conv.inner.weights,
                    &layer.conv.inner.bias,
                    layer.conv.inner.do_bias,
                    dilation,
                    kernel,
                    &history,
                    frame_idx,
                    &mixin_w,
                    input_cond[f],
                    &l1x1_w,
                    &l1x1_b,
                    &mut head_scalar,
                    f,
                    lin_slice,
                    f == 0,
                    f == num_frames - 1,
                );
            }
        }

        // Compare head accumulator.
        for c in 0..ch * num_frames {
            let diff = (head_simd[c] - head_scalar[c]).abs();
            assert!(
                diff < 1e-5,
                "head[{}]: simd={}, scalar={}, diff={}",
                c,
                head_simd[c],
                head_scalar[c],
                diff
            );
        }

        // Compare layer_in (only non-last layers updated).
        for c in 0..ch * (num_frames - 1) {
            let diff = (layer_in_simd[c] - layer_in_scalar[c]).abs();
            assert!(
                diff < 2e-5,
                "layer_in[{}]: simd={}, scalar={}, diff={}",
                c,
                layer_in_simd[c],
                layer_in_scalar[c],
                diff
            );
        }
    }

    /// End-to-end parity test: CH=8, kernel=15, dilation=13, all frames.
    #[test]
    fn test_a2_layer_ch8_kernel15_parity() {
        let ch = 8usize;
        let kernel = 15;
        let dilation = A2_DILATIONS[15]; // 13
        let num_frames = 16;

        let (conv_w, conv_bias) = make_conv_weights(ch, kernel, 123);
        let conv = A2Conv1d::new(
            conv_w.clone(),
            conv_bias.clone(),
            true,
            dilation,
            ch,
            ch,
            kernel,
            prefetch_strategy_simple,
        );
        let mixin_w = make_f32_vec(ch, 400);
        let l1x1_w = make_f32_vec(ch * ch, 500);
        let l1x1_b = make_f32_vec(ch, 600);

        let layer = A2Layer::new(conv, mixin_w.clone(), l1x1_w.clone(), l1x1_b.clone());

        let max_lookback = (kernel - 1) * dilation;
        let hist_cols = max_lookback + num_frames + 8;
        let history = make_history(hist_cols, ch, 88);
        let hist_write_pos = max_lookback + 4;

        let input_cond: Vec<f32> = (0..num_frames)
            .map(|i| (i as f32 * 0.7).cos() * 0.5)
            .collect();

        let mut layer_in_simd = vec![0.0f32; num_frames * ch];
        let mut layer_in_scalar = vec![0.0f32; num_frames * ch];
        let mut head_simd = vec![0.0f32; (num_frames + 1) * ch];
        let mut head_scalar = vec![0.0f32; (num_frames + 1) * ch];

        {
            let mut z_buf = vec![0.0f32; ch];
            for f in 0..num_frames {
                let frame_idx = hist_write_pos + f;
                let lin_slice = &mut layer_in_simd[f * ch..(f + 1) * ch];
                layer.process_single_frame(
                    &history,
                    frame_idx,
                    input_cond[f],
                    &mut head_simd,
                    f,
                    &mut z_buf,
                    lin_slice,
                    f == 0,
                    f == num_frames - 1,
                );
            }
        }

        {
            for f in 0..num_frames {
                let lin_slice = &mut layer_in_scalar[f * ch..(f + 1) * ch];
                let frame_idx = hist_write_pos + f;
                a2_layer_single_frame_scalar_ref(
                    &layer.conv.inner.weights,
                    &layer.conv.inner.bias,
                    layer.conv.inner.do_bias,
                    dilation,
                    kernel,
                    &history,
                    frame_idx,
                    &mixin_w,
                    input_cond[f],
                    &l1x1_w,
                    &l1x1_b,
                    &mut head_scalar,
                    f,
                    lin_slice,
                    f == 0,
                    f == num_frames - 1,
                );
            }
        }

        for c in 0..ch * num_frames {
            let diff = (head_simd[c] - head_scalar[c]).abs();
            assert!(
                diff < 1e-5,
                "CH=8 head[{}]: simd={}, scalar={}, diff={}",
                c,
                head_simd[c],
                head_scalar[c],
                diff
            );
        }

        for c in 0..ch * (num_frames - 1) {
            let diff = (layer_in_simd[c] - layer_in_scalar[c]).abs();
            assert!(
                diff < 2e-5,
                "CH=8 layer_in[{}]: simd={}, scalar={}, diff={}",
                c,
                layer_in_simd[c],
                layer_in_scalar[c],
                diff
            );
        }
    }

    /// Verify first layer assigns to head (not accumulates), middle layers accumulate, last skips l1x1.
    #[test]
    fn test_a2_layer_first_middle_last_behavior() {
        let ch = 3usize;
        let kernel = 6;
        let dilation = 1;
        let num_frames = 4;

        let (conv_w, conv_bias) = make_conv_weights(ch, kernel, 99);
        let conv = A2Conv1d::new(
            conv_w.clone(),
            conv_bias.clone(),
            true,
            dilation,
            ch,
            ch,
            kernel,
            prefetch_strategy_simple,
        );
        let mixin_w = make_f32_vec(ch, 101);
        let l1x1_w = make_f32_vec(ch * ch, 201);
        let l1x1_b = make_f32_vec(ch, 301);

        let layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);

        let hist_cols = (kernel - 1) * dilation + num_frames + 4;
        let history = make_history(hist_cols, ch, 55);
        let hist_write_pos = (kernel - 1) * dilation + 2;

        let input_cond: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];

        // Test first layer: head should be assigned, not accumulated.
        {
            let mut head = vec![99.0f32; (num_frames + 1) * ch];
            let mut layer_in = vec![1.0f32; num_frames * ch];
            let mut z_buf = vec![0.0f32; ch];
            for f in 0..num_frames {
                let frame_idx = hist_write_pos + f;
                layer.process_single_frame(
                    &history,
                    frame_idx,
                    input_cond[f],
                    &mut head,
                    f,
                    &mut z_buf,
                    &mut layer_in[f * ch..(f + 1) * ch],
                    true,
                    f == num_frames - 1,
                );
            }
            // is_first=true means head values were ASSIGNED (overwritten), so they differ from 99.0.
            for f in 0..num_frames {
                for c in 0..ch {
                    assert!(
                        (head[f * ch + c] - 99.0).abs() > 1e-3,
                        "first layer should assign head, not accumulate old value"
                    );
                }
            }
        }

        // Test middle layer: head should accumulate.
        // Run is_first=false twice on the SAME head buffer; the second pass should change values.
        {
            let mut head = vec![1.0f32; (num_frames + 1) * ch];
            let mut head_copy = head.clone();
            let mut layer_in = vec![0.0f32; num_frames * ch];
            let mut z_buf = vec![0.0f32; ch];

            // First pass (is_first=true): head = layer output
            for f in 0..num_frames {
                let frame_idx = hist_write_pos + f;
                layer.process_single_frame(
                    &history,
                    frame_idx,
                    input_cond[f],
                    &mut head,
                    f,
                    &mut z_buf,
                    &mut layer_in[f * ch..(f + 1) * ch],
                    true,
                    false,
                );
            }
            head_copy.copy_from_slice(&head);

            // Second pass (is_first=false): should add more to head
            for f in 0..num_frames {
                let frame_idx = hist_write_pos + f;
                layer.process_single_frame(
                    &history,
                    frame_idx,
                    input_cond[f],
                    &mut head,
                    f,
                    &mut z_buf,
                    &mut layer_in[f * ch..(f + 1) * ch],
                    false,
                    false,
                );
            }

            // After second pass, head should differ from first pass (at least some frames/channels).
            let mut any_changed = false;
            for f in 0..num_frames {
                for c in 0..ch {
                    if (head[f * ch + c] - head_copy[f * ch + c]).abs() > 1e-3 {
                        any_changed = true;
                    }
                }
            }
            assert!(
                any_changed,
                "middle layer (is_first=false) should accumulate, but all values unchanged"
            );
        }

        // Test last layer: layer_in should NOT be updated.
        {
            let mut head = vec![0.0f32; (num_frames + 1) * ch];
            let mut layer_in = vec![1.0f32; num_frames * ch];
            let mut z_buf = vec![0.0f32; ch];
            for f in 0..num_frames {
                let frame_idx = hist_write_pos + f;
                layer.process_single_frame(
                    &history,
                    frame_idx,
                    input_cond[f],
                    &mut head,
                    f,
                    &mut z_buf,
                    &mut layer_in[f * ch..(f + 1) * ch],
                    true,
                    true, // is_last=true → skip l1x1
                );
            }
            for f in 0..num_frames {
                for c in 0..ch {
                    assert!(
                        (layer_in[f * ch + c] - 1.0).abs() < 1e-6,
                        "last layer should skip l1x1 residual"
                    );
                }
            }
        }
    }

    /// Test that mixin_w contributes to output (relaxed tolerance due to LeakyReLU nonlinearity).
    #[test]
    fn test_a2_layer_mixin_contribution() {
        let ch = 3usize;
        let kernel = 6;
        let dilation = 1;

        // Zero conv weights and bias to isolate mixin.
        let num_blocks = ch.div_ceil(4);
        let total_w = num_blocks * 4 * ch * kernel;
        let conv_w = AlignedVec::new(total_w, 0u16);
        let conv_bias = AlignedVec::new(ch, 0.0f32);
        let conv = A2Conv1d::new(
            conv_w,
            conv_bias,
            false, // no bias — pure conv with zero weights outputs 0
            dilation,
            ch,
            ch,
            kernel,
            prefetch_strategy_simple,
        );
        let mixin_w = AlignedVec::from(vec![0.1f32, 0.2, 0.3]);
        let l1x1_w = AlignedVec::new(ch * ch, 0.0f32);
        let l1x1_b = AlignedVec::new(ch, 0.0f32);

        let layer = A2Layer::new(conv, mixin_w.clone(), l1x1_w, l1x1_b);

        let max_lookback = (kernel - 1) * dilation;
        let hist_cols = max_lookback + 8;
        let history = vec![0.0f32; hist_cols * ch];
        let hist_write_pos = max_lookback + 2;
        let frame_idx = hist_write_pos;

        // With zero conv: output = mixin_w[c] * cond, LeakyReLU, then head assign.
        // With cond=2.0: z = 0 + mixin_w[c]*2.0 = mixin_w[c]*2.0 > 0, so LeakyReLU is identity.
        // Head = z = mixin_w[c]*2.0.
        let mut head = vec![0.0f32; ch];
        let mut layer_in = vec![0.0f32; ch];
        let mut z_buf = vec![0.0f32; ch];

        layer.process_single_frame(
            &history,
            frame_idx,
            2.0,
            &mut head,
            0,
            &mut z_buf,
            &mut layer_in,
            true,
            true,
        );

        for c in 0..ch {
            let expected = mixin_w[c] * 2.0;
            assert!(
                (head[c] - expected).abs() < 1e-5,
                "ch {}: head={}, expected={}",
                c,
                head[c],
                expected
            );
        }
    }

    /// Test that layer with zero weights and known input produces deterministic output.
    #[test]
    fn test_a2_layer_zero_weights_deterministic() {
        let ch = 3usize;
        let kernel = 6;
        let dilation = 1;

        let num_blocks = ch.div_ceil(4);
        let total_w = num_blocks * 4 * ch * kernel;
        let conv_w = AlignedVec::new(total_w, 0u16);
        let conv_bias = AlignedVec::new(ch, 0.0f32);
        let conv = A2Conv1d::new(
            conv_w,
            conv_bias.clone(),
            false,
            dilation,
            ch,
            ch,
            kernel,
            prefetch_strategy_simple,
        );
        let mixin_w = AlignedVec::from(vec![1.0f32, 2.0, 3.0]);
        let l1x1_w = AlignedVec::new(ch * ch, 1.0f32);
        let l1x1_b = AlignedVec::from(vec![0.5f32, 0.5, 0.5]);

        let layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);

        let max_lookback = (kernel - 1) * dilation;
        let hist_cols = max_lookback + 8;
        let history = vec![0.0f32; hist_cols * ch];
        let hist_write_pos = max_lookback + 2;

        let mut head = vec![0.0f32; ch];
        let mut layer_in = vec![0.0f32; ch];
        let mut z_buf = vec![0.0f32; ch];

        layer.process_single_frame(
            &history,
            hist_write_pos,
            0.5,
            &mut head,
            0,
            &mut z_buf,
            &mut layer_in,
            true,
            false, // not last → l1x1 applied
        );

        // Conv output = 0 (zero weights, no bias, no mixin in conv).
        // Mixin: z[c] = 0 + mixin_w[c] * 0.5 = [0.5, 1.0, 1.5]
        // LeakyReLU: all positive → identity
        // Head: [0.5, 1.0, 1.5]
        assert!((head[0] - 0.5).abs() < 1e-5);
        assert!((head[1] - 1.0).abs() < 1e-5);
        assert!((head[2] - 1.5).abs() < 1e-5);

        // L1x1: layer_in[c] += 0.5 + sum_u(l1x1_w[u*3+c] * z[u])
        // l1x1_w is all 1.0, col-major: [u*3+c] = 1.0 for all u,c
        // sum_u(1.0 * z[u]) = 0.5+1.0+1.5 = 3.0
        // layer_in = [0.5+3.0, 0.5+3.0, 0.5+3.0] = [3.5, 3.5, 3.5]
        assert!((layer_in[0] - 3.5).abs() < 1e-5);
        assert!((layer_in[1] - 3.5).abs() < 1e-5);
        assert!((layer_in[2] - 3.5).abs() < 1e-5);
    }
}
