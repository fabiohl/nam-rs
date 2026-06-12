// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Head convolution (k=16, bias, head_scale).
//!
//! Implements the head rechannel convolution from `a2_fast.cpp:722-743`:
//! `Conv1D(Bottleneck → 1, K=16, bias)` read from ring with tail-mirror,
//! followed by multiplication by `head_scale`.
//!
//! ## Data layout
//!
//! - Weights: `[K][Channels]` column-major per tap (Channels f32 values per tap).
//! - Head history ring buffer: `[Channels][cols]` column-major.
//! - Ring access: `col & ring_mask` (pow2 ring, mask = size - 1).
//!
//! ## Source of truth
//!
//! - `NAM/wavenet/a2_fast.cpp:117-136` (member declarations)
//! - `NAM/wavenet/a2_fast.cpp:262-275` (weight loading order)
//! - `NAM/wavenet/a2_fast.cpp:716-745` (`_head_forward`)

use crate::math::common::AlignedVec;

/// Head convolution for the A2 WaveNet architecture.
///
/// Applies `Conv1D(Bottleneck → 1, K=16, bias)` over the head history ring buffer
/// and multiplies the result by `head_scale`.
///
/// The head history accumulates the post-activation outputs from all 23 layers
/// (first layer assigns, subsequent layers add). This struct applies a final
/// causal convolution over that accumulator to produce the output signal.
#[derive(Clone)]
pub struct A2HeadConv {
    /// Head weights: `[KERNEL_SIZE][Channels]` f32, stored column-major per tap.
    /// At tap `k`, weight for channel `c` is at index `k * num_channels + c`.
    pub head_w: AlignedVec<f32>,
    /// Head bias (single scalar).
    pub head_b: f32,
    /// Head scale (multiplied after convolution).
    pub head_scale: f32,
    /// Number of channels = bottleneck size (3 for Lite, 8 for Full).
    pub num_channels: usize,
    /// Kernel size of the head convolution (= 16 for A2).
    pub kernel_size: usize,
}

impl A2HeadConv {
    /// A2 canonical head kernel size.
    pub const HEAD_KERNEL_SIZE: usize = 16;

    /// Creates a new `A2HeadConv` from pre-loaded weights.
    ///
    /// `head_w` must contain exactly `HEAD_KERNEL_SIZE * num_channels` f32 values,
    /// stored column-major per tap as loaded by `_load_weights` (see `a2_fast.cpp:262-275`).
    pub fn new(head_w: AlignedVec<f32>, head_b: f32, head_scale: f32, num_channels: usize) -> Self {
        let k = Self::HEAD_KERNEL_SIZE;
        assert_eq!(
            head_w.len(),
            k * num_channels,
            "head_w must have HEAD_KERNEL_SIZE * num_channels elements"
        );
        Self {
            head_w,
            head_b,
            head_scale,
            num_channels,
            kernel_size: k,
        }
    }

    /// Processes a block of `num_frames` through the head convolution.
    ///
    /// `head_history` is a contiguous col-major buffer (`Channels` rows × N columns).
    /// Ring access uses `col & ring_mask` (pow2 ring). `head_write_pos` is the position
    /// where the *next* batch of frames will be written (already advanced past this batch).
    ///
    /// # Panics
    /// Debug: asserts that `output` has at least `num_frames` elements.
    #[inline(always)]
    pub fn process(
        &self,
        head_history: &[f32],
        head_write_pos: usize,
        ring_mask: usize,
        num_frames: usize,
        output: &mut [f32],
    ) {
        let k = self.kernel_size;
        let ch = self.num_channels;
        debug_assert!(output.len() >= num_frames);
        // Ring mask must be consistent with buffer size.
        debug_assert!(head_history.len() >= (ring_mask + 1) * ch);

        let base = head_write_pos.wrapping_sub(num_frames);

        for (f, out_val) in output.iter_mut().take(num_frames).enumerate() {
            let col_base = base.wrapping_add(f);
            let mut y = self.head_b;

            for t in 0..k {
                let col = col_base.wrapping_sub(k - 1 - t) & ring_mask;
                let src_off = col * ch;
                let w_off = t * ch;

                // SAFETY: head_w length is validated in new() (assert_eq), and
                // head_history length is validated by the debug_assert above.
                // w_off + c < head_w.len() because t < k and c < ch.
                // src_off + c < head_history.len() because col <= ring_mask.
                for c in 0..ch {
                    unsafe {
                        y += *self.head_w.get_unchecked(w_off + c)
                            * *head_history.get_unchecked(src_off + c);
                    }
                }
            }

            *out_val = y * self.head_scale;
        }
    }
}

// =============================================================================
// Scalar reference for parity testing (oracle)
// =============================================================================

/// Scalar reference for a single frame of head convolution.
///
/// Matches `A2HeadConv::process` single-frame logic exactly.
/// Used as an oracle for unit tests and SIMD parity verification.
#[allow(clippy::too_many_arguments)]
pub fn a2_head_single_frame_scalar_ref(
    head_w: &[f32],
    head_b: f32,
    head_scale: f32,
    num_channels: usize,
    head_history: &[f32],
    head_write_pos: usize,
    ring_mask: usize,
    num_frames: usize,
    frame: usize,
) -> f32 {
    let k = A2HeadConv::HEAD_KERNEL_SIZE;
    let base = head_write_pos.wrapping_sub(num_frames);
    let col_base = base.wrapping_add(frame);

    let mut y = head_b;

    for t in 0..k {
        let col = col_base.wrapping_sub(k - 1 - t) & ring_mask;
        let src_off = col * num_channels;
        let w_off = t * num_channels;

        for c in 0..num_channels {
            y += head_w[w_off + c] * head_history[src_off + c];
        }
    }

    y * head_scale
}

/// Scalar reference for a full block of head convolution.
///
/// Computes the head output for `num_frames` using the same algorithm
/// as `A2HeadConv::process`. Useful for validating the block-level path.
#[allow(clippy::too_many_arguments)]
pub fn a2_head_block_scalar_ref(
    head_w: &[f32],
    head_b: f32,
    head_scale: f32,
    num_channels: usize,
    head_history: &[f32],
    head_write_pos: usize,
    ring_mask: usize,
    num_frames: usize,
    output: &mut [f32],
) {
    for (f, out_val) in output.iter_mut().take(num_frames).enumerate() {
        *out_val = a2_head_single_frame_scalar_ref(
            head_w,
            head_b,
            head_scale,
            num_channels,
            head_history,
            head_write_pos,
            ring_mask,
            num_frames,
            f,
        );
    }
}

#[cfg(test)]
#[path = "head_test.rs"]
mod tests;
