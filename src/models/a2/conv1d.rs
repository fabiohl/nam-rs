// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dilated causal Conv1D for the A2 architecture (kernel sizes 6 and 15).
//!
//! Supports two variants:
//! - `Standard` — groups=1, delegates to the battle-tested `Conv1dDyn` engine.
//! - `Grouped` — groups>1, uses the AVX2 `A2GroupedConv1d` with depthwise fast-path.
//!
//! Operates over `MirrorBuffer`-backed slices — the dilation tap pointers
//! access a contiguous virtual window where physical wrap is handled by the
//! mirrored mapping, eliminating branch logic in the inner loop.

use crate::math::common::{AlignedVec, PrefetchFn};
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;

use super::grouped_conv1d::A2GroupedConv1d;

/// A2-specific dilated causal conv1d — polymorphic over grouping.
///
/// When `groups == 1` the standard interleaved-4-wide `Conv1dDyn` is used.
/// When `groups > 1` the grouped-interleaved-4-wide `A2GroupedConv1d` is used,
/// with automatic depthwise dispatch when `groups == in_ch == out_ch`.
#[derive(Clone)]
pub enum A2Conv1d {
    /// Standard conv (groups=1). Uses interleaved-4-wide `Conv1dDyn`.
    Standard(Conv1dDyn),
    /// Grouped conv (groups>1). Uses grouped-interleaved-4-wide AVX2 kernel.
    Grouped(A2GroupedConv1d),
}

impl A2Conv1d {
    /// Builds an A2 conv1d with pre-validated A2 parameters (groups=1).
    ///
    /// # Panics
    /// Panics if `kernel_size` is not 6 or 15 (debug builds).
    /// In release, the assert is compiled out for performance.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        prefetch_fn: PrefetchFn,
    ) -> Self {
        debug_assert!(
            kernel_size == 6 || kernel_size == 15,
            "A2 only supports kernel sizes 6 and 15, got {}",
            kernel_size
        );
        debug_assert!(
            in_ch > 0 && out_ch > 0,
            "channels must be > 0, got in_ch={} out_ch={}",
            in_ch,
            out_ch
        );

        let num_blocks = out_ch.div_ceil(4);
        let total_padded = num_blocks * 4 * in_ch * kernel_size;
        debug_assert!(
            weights.len() >= total_padded,
            "weights too short: expected >= {}, got {}",
            total_padded,
            weights.len()
        );
        debug_assert!(bias.len() >= out_ch);

        Self::Standard(Conv1dDyn {
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            num_blocks,
            kernel: kernel_size,
            prefetch_fn,
        })
    }

    /// Builds a grouped A2 conv1d from raw NAM JSON row-major weights.
    ///
    /// Wraps `A2GroupedConv1d::new()`. The raw_weights are in
    /// `[out_ch][in_ch][kernel]` order and are permuted to
    /// grouped-interleaved-4-wide internally.
    ///
    /// # Panics
    /// Panics in debug if `in_ch % groups != 0` or `out_ch % groups != 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_grouped(
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
        Self::Grouped(A2GroupedConv1d::new(
            raw_weights,
            raw_bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            prefetch_fn,
        ))
    }

    /// Returns the number of groups (1 for `Standard`, >1 for `Grouped`).
    #[inline(always)]
    pub fn groups(&self) -> usize {
        match self {
            Self::Standard(_) => 1,
            Self::Grouped(g) => g.groups,
        }
    }

    /// Returns true if this is a depthwise convolution (groups == in_ch == out_ch).
    #[inline(always)]
    pub fn is_depthwise(&self) -> bool {
        match self {
            Self::Standard(_) => false,
            Self::Grouped(g) => g.groups == g.in_ch && g.groups == g.out_ch,
        }
    }

    /// Kernel size of this convolution.
    #[inline(always)]
    pub fn kernel_size(&self) -> usize {
        match self {
            Self::Standard(c) => c.kernel,
            Self::Grouped(g) => g.kernel,
        }
    }

    /// Dilation factor.
    #[inline(always)]
    pub fn dilation(&self) -> usize {
        match self {
            Self::Standard(c) => c.dilation,
            Self::Grouped(g) => g.dilation,
        }
    }

    /// Number of input channels.
    #[inline(always)]
    pub fn in_ch(&self) -> usize {
        match self {
            Self::Standard(c) => c.in_ch,
            Self::Grouped(g) => g.in_ch,
        }
    }

    /// Number of output channels.
    #[inline(always)]
    pub fn out_ch(&self) -> usize {
        match self {
            Self::Standard(c) => c.out_ch,
            Self::Grouped(g) => g.out_ch,
        }
    }

    /// Processes a single frame (f32) through the dilated convolution.
    ///
    /// Uses the SIMD-accelerated path: `Conv1dDyn` for groups=1,
    /// AVX2 grouped kernel (with depthwise dispatch) for groups>1.
    ///
    /// # Safety
    /// `layer_buffer` must contain valid elements for the dilated tap indices.
    /// `out_frame` must have length at least `self.out_ch`.
    /// `frame_idx` must allow `kernel` lookback taps within `layer_buffer`.
    #[inline(always)]
    pub unsafe fn process_single_frame(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        unsafe {
            match self {
                Self::Standard(c) => {
                    c.process_single_frame(layer_buffer, out_frame, frame_idx, mixin);
                }
                Self::Grouped(g) => {
                    g.process_single_frame(layer_buffer, out_frame, frame_idx, mixin);
                }
            }
        }
    }

    /// Processes a block of `num_frames` consecutive frames (f32).
    ///
    /// # Safety
    /// `layer_buffer` must be large enough for `buffer_start..buffer_start + num_frames`
    /// plus kernel*dilation lookback. `block` must have size at least `num_frames * out_ch`.
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
        unsafe {
            match self {
                Self::Standard(c) => {
                    c.process_block(layer_buffer, block, buffer_start, num_frames, mixin);
                }
                Self::Grouped(g) => {
                    g.process_block(layer_buffer, block, buffer_start, num_frames, mixin);
                }
            }
        }
    }

    /// Returns a reference to the inner `Conv1dDyn` for the `Standard` variant.
    ///
    /// # Panics
    /// Panics if called on a `Grouped` variant — only valid in tests of the
    /// standard const-generic fast-path.
    #[cfg(test)]
    pub fn standard_inner(&self) -> &Conv1dDyn {
        match self {
            Self::Standard(c) => c,
            Self::Grouped(_) => panic!("standard_inner called on Grouped variant"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::common::AlignedVec;
    use crate::models::a2::A2_DILATIONS;
    use crate::models::a2::conv1d_fallback::a2_conv1d_single_frame_fallback;

    fn make_test_weights(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        seed: u32,
    ) -> (AlignedVec<f32>, AlignedVec<f32>) {
        let num_blocks = out_ch.div_ceil(4);
        let total_w = num_blocks * 4 * in_ch * kernel;
        let mut weights = AlignedVec::new(total_w, 0.0f32);

        let mut state = seed;
        for i in 0..total_w {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
            weights[i] = v;
        }

        let mut bias = AlignedVec::new(out_ch, 0.0f32);
        for i in 0..out_ch {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            bias[i] = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
        }

        (weights, bias)
    }

    #[test]
    fn test_a2_conv1d_kernel6_parity_single_frame() {
        let in_ch = 3;
        let out_ch = 8;
        let kernel = 6;
        let dilation = A2_DILATIONS[1];

        let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 42);

        let conv = A2Conv1d::new(
            weights.clone(),
            bias.clone(),
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            crate::math::common::prefetch_strategy_simple,
        );

        let buf_frames = 512;
        let layer_buffer = {
            let mut buf = vec![0.0f32; buf_frames * in_ch];
            let mut state = 99u32;
            for val in &mut buf {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            }
            buf
        };

        let frame_idx = 400;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, None);
        }

        a2_conv1d_single_frame_fallback(
            &conv.standard_inner().weights,
            &conv.standard_inner().bias,
            conv.standard_inner().do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            &layer_buffer,
            frame_idx,
            None,
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let diff = (simd_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "channel {}: simd={}, scalar={}, diff={}",
                c,
                simd_out[c],
                scalar_out[c],
                diff
            );
        }
    }

    #[test]
    fn test_a2_conv1d_kernel15_parity_single_frame() {
        let in_ch = 8;
        let out_ch = 8;
        let kernel = 15;
        let dilation = A2_DILATIONS[15];

        let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 123);

        let conv = A2Conv1d::new(
            weights.clone(),
            bias.clone(),
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            crate::math::common::prefetch_strategy_simple,
        );

        let buf_frames = 4096;
        let layer_buffer = {
            let mut buf = vec![0.0f32; buf_frames * in_ch];
            let mut state = 77u32;
            for val in &mut buf {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            }
            buf
        };

        let frame_idx = 3500;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, None);
        }

        a2_conv1d_single_frame_fallback(
            &conv.standard_inner().weights,
            &conv.standard_inner().bias,
            conv.standard_inner().do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            &layer_buffer,
            frame_idx,
            None,
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let diff = (simd_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "channel {}: simd={}, scalar={}, diff={}",
                c,
                simd_out[c],
                scalar_out[c],
                diff
            );
        }
    }

    #[test]
    fn test_a2_conv1d_first_layer_in_ch_1() {
        let in_ch = 1;
        let out_ch = 3;
        let kernel = 6;
        let dilation = A2_DILATIONS[0];

        let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 7);

        let conv = A2Conv1d::new(
            weights.clone(),
            bias.clone(),
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            crate::math::common::prefetch_strategy_simple,
        );

        let buf_frames = 256;
        let layer_buffer = {
            let mut buf = vec![0.0f32; buf_frames * in_ch];
            let mut state = 13u32;
            for val in &mut buf {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            }
            buf
        };

        let frame_idx = 200;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, None);
        }

        a2_conv1d_single_frame_fallback(
            &conv.standard_inner().weights,
            &conv.standard_inner().bias,
            conv.standard_inner().do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            &layer_buffer,
            frame_idx,
            None,
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let diff = (simd_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "channel {}: simd={}, scalar={}, diff={}",
                c,
                simd_out[c],
                scalar_out[c],
                diff
            );
        }
    }

    #[test]
    fn test_a2_conv1d_with_mixin_kernel6() {
        let in_ch = 3;
        let out_ch = 8;
        let kernel = 6;
        let dilation = A2_DILATIONS[5];

        let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 555);

        let conv = A2Conv1d::new(
            weights.clone(),
            bias.clone(),
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            crate::math::common::prefetch_strategy_simple,
        );

        let buf_frames = kernel * dilation + 512;
        let layer_buffer = {
            let mut buf = vec![0.0f32; buf_frames * in_ch];
            let mut state = 31u32;
            for val in &mut buf {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            }
            buf
        };

        let mut state = 88u32;
        let mixin: Vec<f32> = (0..out_ch)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
            })
            .collect();

        let frame_idx = kernel * dilation + 64;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, Some(&mixin));
        }

        a2_conv1d_single_frame_fallback(
            &conv.standard_inner().weights,
            &conv.standard_inner().bias,
            conv.standard_inner().do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            &layer_buffer,
            frame_idx,
            Some(&mixin),
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let diff = (simd_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "channel {} with mixin: simd={}, scalar={}, diff={}",
                c,
                simd_out[c],
                scalar_out[c],
                diff
            );
        }
    }

    #[test]
    fn test_a2_conv1d_all_dilations_kernel6() {
        let in_ch = 8;
        let out_ch = 8;
        let kernel = 6;

        let dilation_set: Vec<usize> = A2_DILATIONS
            .iter()
            .filter(|&&d| {
                let idx = A2_DILATIONS.iter().position(|&x| x == d).unwrap();
                crate::models::a2::A2_KERNEL_SIZES[idx] == 6
            })
            .copied()
            .collect();

        let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 111);

        let buf_frames = 4096;
        let layer_buffer = {
            let mut buf = vec![0.0f32; buf_frames * in_ch];
            let mut state = 17u32;
            for val in &mut buf {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            }
            buf
        };

        for &dilation in &dilation_set {
            let conv = A2Conv1d::new(
                weights.clone(),
                bias.clone(),
                true,
                dilation,
                in_ch,
                out_ch,
                kernel,
                crate::math::common::prefetch_strategy_simple,
            );

            let frame_idx = 3500;

            let mut simd_out = vec![0.0f32; out_ch];
            let mut scalar_out = vec![0.0f32; out_ch];

            unsafe {
                conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, None);
            }

            a2_conv1d_single_frame_fallback(
                &conv.standard_inner().weights,
                &conv.standard_inner().bias,
                conv.standard_inner().do_bias,
                dilation,
                in_ch,
                out_ch,
                kernel,
                &layer_buffer,
                frame_idx,
                None,
                &mut scalar_out,
            );

            for c in 0..out_ch {
                let diff = (simd_out[c] - scalar_out[c]).abs();
                assert!(
                    diff < 1e-5,
                    "dilation={} channel {}: simd={}, scalar={}, diff={}",
                    dilation,
                    c,
                    simd_out[c],
                    scalar_out[c],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_a2_conv1d_block_processing_kernel15() {
        let in_ch = 8;
        let out_ch = 8;
        let kernel = 15;
        let dilation = 13;

        let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 777);

        let conv = A2Conv1d::new(
            weights.clone(),
            bias.clone(),
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            crate::math::common::prefetch_strategy_simple,
        );

        let buf_frames = 4096;
        let layer_buffer = {
            let mut buf = vec![0.0f32; buf_frames * in_ch];
            let mut state = 43u32;
            for val in &mut buf {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            }
            buf
        };

        for num_frames in [1, 2, 3, 4, 7, 16, 31, 64] {
            let buffer_start = 3500 - kernel * dilation;
            let mut simd_block = vec![0.0f32; num_frames * out_ch];
            let mut scalar_block = vec![0.0f32; num_frames * out_ch];

            unsafe {
                conv.process_block(
                    &layer_buffer,
                    &mut simd_block,
                    buffer_start,
                    num_frames,
                    None,
                );
            }

            crate::models::a2::conv1d_fallback::a2_conv1d_block_fallback(
                &conv.standard_inner().weights,
                &conv.standard_inner().bias,
                conv.standard_inner().do_bias,
                dilation,
                in_ch,
                out_ch,
                kernel,
                &layer_buffer,
                buffer_start,
                num_frames,
                None,
                &mut scalar_block,
            );

            for (i, (&s, &f)) in simd_block.iter().zip(scalar_block.iter()).enumerate() {
                let diff = (s - f).abs();
                assert!(
                    diff < 1e-5,
                    "num_frames={} idx={}: simd={}, scalar={}, diff={}",
                    num_frames,
                    i,
                    s,
                    f,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_a2_conv1d_kernel6_non_multiple_of_4_output() {
        let in_ch = 3;
        let out_ch = 7;
        let kernel = 6;
        let dilation = A2_DILATIONS[3];

        let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 22);

        let conv = A2Conv1d::new(
            weights.clone(),
            bias.clone(),
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            crate::math::common::prefetch_strategy_simple,
        );

        let buf_frames = 512;
        let layer_buffer = {
            let mut buf = vec![0.0f32; buf_frames * in_ch];
            let mut state = 91u32;
            for val in &mut buf {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
            }
            buf
        };

        let frame_idx = 400;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, None);
        }

        a2_conv1d_single_frame_fallback(
            &conv.standard_inner().weights,
            &conv.standard_inner().bias,
            conv.standard_inner().do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            &layer_buffer,
            frame_idx,
            None,
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let diff = (simd_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "channel {} (out_ch=7): simd={}, scalar={}, diff={}",
                c,
                simd_out[c],
                scalar_out[c],
                diff
            );
        }
    }

    // =============================================================================
    // Grouped conv integration tests — A2Conv1d enum (T2.1)
    // =============================================================================

    use crate::models::a2::grouped_conv1d::{make_layer_buffer, make_test_weights_grouped};

    #[test]
    fn test_a2_conv1d_grouped_groups2_parity() {
        let in_ch = 6;
        let out_ch = 4;
        let kernel = 6;
        let dilation = A2_DILATIONS[1];
        let groups = 2;

        let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);

        let conv = A2Conv1d::new_grouped(
            &raw_weights,
            &raw_bias,
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            crate::math::common::prefetch_strategy_simple,
        );

        let buf_frames = 512;
        let layer_buffer = make_layer_buffer(buf_frames, in_ch, 99);
        let frame_idx = 400;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, None);
        }

        crate::models::a2::grouped_conv1d::grouped_conv1d_single_frame_ref(
            match &conv {
                A2Conv1d::Grouped(g) => &g.weights,
                _ => panic!("expected Grouped variant"),
            },
            &raw_bias,
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            &layer_buffer,
            frame_idx,
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let diff = (simd_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "groups=2 channel {}: simd={}, scalar={}, diff={}",
                c,
                simd_out[c],
                scalar_out[c],
                diff
            );
        }
    }

    #[test]
    fn test_a2_conv1d_grouped_depthwise_parity() {
        let in_ch = 4;
        let out_ch = 4;
        let kernel = 6;
        let dilation = A2_DILATIONS[3];
        let groups = 4; // depthwise: groups == channels

        let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 555);

        let conv = A2Conv1d::new_grouped(
            &raw_weights,
            &raw_bias,
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            crate::math::common::prefetch_strategy_simple,
        );

        assert!(conv.is_depthwise());
        assert_eq!(conv.groups(), 4);

        let buf_frames = 512;
        let layer_buffer = make_layer_buffer(buf_frames, in_ch, 31);
        let frame_idx = kernel * dilation + 64;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, None);
        }

        crate::models::a2::grouped_conv1d::grouped_conv1d_single_frame_ref(
            match &conv {
                A2Conv1d::Grouped(g) => &g.weights,
                _ => panic!("expected Grouped variant"),
            },
            &raw_bias,
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            &layer_buffer,
            frame_idx,
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let diff = (simd_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "depthwise ch={}: simd={}, scalar={}, diff={}",
                c,
                simd_out[c],
                scalar_out[c],
                diff
            );
        }
    }

    #[test]
    fn test_a2_conv1d_standard_groups_is_1() {
        let in_ch = 3;
        let out_ch = 8;
        let kernel = 6;
        let dilation = A2_DILATIONS[0];

        let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 7);

        let conv = A2Conv1d::new(
            weights.clone(),
            bias.clone(),
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            crate::math::common::prefetch_strategy_simple,
        );

        assert_eq!(conv.groups(), 1);
        assert!(!conv.is_depthwise());
        assert_eq!(conv.kernel_size(), kernel);
        assert_eq!(conv.dilation(), dilation);
        assert_eq!(conv.in_ch(), in_ch);
        assert_eq!(conv.out_ch(), out_ch);
    }

    #[test]
    fn test_a2_conv1d_grouped_with_mixin() {
        let in_ch = 6;
        let out_ch = 8;
        let kernel = 6;
        let dilation = A2_DILATIONS[5];
        let groups = 2;

        let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 777);

        let conv = A2Conv1d::new_grouped(
            &raw_weights,
            &raw_bias,
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            crate::math::common::prefetch_strategy_simple,
        );

        let buf_frames = kernel * dilation + 512;
        let layer_buffer = make_layer_buffer(buf_frames, in_ch, 44);

        let mut state = 88u32;
        let mixin: Vec<f32> = (0..out_ch)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
            })
            .collect();

        let frame_idx = kernel * dilation + 64;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame(&layer_buffer, &mut simd_out, frame_idx, Some(&mixin));
        }

        crate::models::a2::grouped_conv1d::grouped_conv1d_single_frame_ref(
            match &conv {
                A2Conv1d::Grouped(g) => &g.weights,
                _ => panic!("expected Grouped variant"),
            },
            &raw_bias,
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            &layer_buffer,
            frame_idx,
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let scalar_with_mixin = scalar_out[c] + mixin[c];
            let diff = (simd_out[c] - scalar_with_mixin).abs();
            assert!(
                diff < 1e-5,
                "groups=2 with mixin ch={}: simd={}, scalar+mixin={}, diff={}",
                c,
                simd_out[c],
                scalar_with_mixin,
                diff
            );
        }
    }
}
