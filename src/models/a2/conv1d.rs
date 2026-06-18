// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dilated causal Conv1D for the A2 architecture (kernel sizes 6 and 15).
//!
//! Reuses the battle-tested `Conv1dDyn` engine from the WaveNet module,
//! validated specifically for A2 parameters: `kernel ∈ {6, 15}`,
//! `A2_DILATIONS`, input=1 (first layer) or CH (remaining layers).
//!
//! Operates over `MirrorBuffer`-backed slices — the dilation tap pointers
//! access a contiguous virtual window where physical wrap is handled by the
//! mirrored mapping, eliminating branch logic in the inner loop.

use crate::math::common::{AlignedVec, PrefetchFn};
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;

/// A2-specific dilated causal conv1d.
///
/// Thin wrapper around `Conv1dDyn` with A2 construction validation.
/// Kernel sizes are validated to be in `{6, 15}` and dilation is validated
/// to allow sufficient lookback within the mirror buffer.
#[derive(Clone)]
#[repr(align(64))]
pub struct A2Conv1d {
    /// The underlying `Conv1dDyn` engine, battle-tested across all WaveNet variants.
    pub inner: Conv1dDyn,
}

impl A2Conv1d {
    /// Builds an A2 conv1d with pre-validated A2 parameters.
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

        Self {
            inner: Conv1dDyn {
                weights,
                bias,
                do_bias,
                dilation,
                in_ch,
                out_ch,
                num_blocks,
                kernel: kernel_size,
                prefetch_fn,
            },
        }
    }

    /// Kernel size of this convolution.
    #[inline(always)]
    pub fn kernel_size(&self) -> usize {
        self.inner.kernel
    }

    /// Dilation factor.
    #[inline(always)]
    pub fn dilation(&self) -> usize {
        self.inner.dilation
    }

    /// Number of input channels.
    #[inline(always)]
    pub fn in_ch(&self) -> usize {
        self.inner.in_ch
    }

    /// Number of output channels.
    #[inline(always)]
    pub fn out_ch(&self) -> usize {
        self.inner.out_ch
    }

    /// Processes a single frame (f32) through the dilated convolution.
    ///
    /// Uses the SIMD-accelerated path dispatched via `M`.
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
            self.inner
                .process_single_frame(layer_buffer, out_frame, frame_idx, mixin);
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
            self.inner
                .process_block(layer_buffer, block, buffer_start, num_frames, mixin);
        }
    }
}

#[cfg(test)]
#[path = "conv1d_test.rs"]
mod tests;
