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
mod tests {
    use super::*;

    fn make_test_weights(channels: usize) -> (AlignedVec<f32>, f32, f32) {
        let k = A2HeadConv::HEAD_KERNEL_SIZE;
        let mut w = AlignedVec::new(k * channels, 0.0f32);
        let mut state: u32 = 42;
        for i in 0..k * channels {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            w[i] = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
        }
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let bias = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let scale = ((state as f32) / (u32::MAX as f32)) * 0.5 + 0.75;
        (w, bias, scale)
    }

    fn make_test_history(channels: usize, cols: usize) -> Vec<f32> {
        let mut buf = vec![0.0f32; channels * cols];
        let mut state: u32 = 99;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    }

    #[test]
    fn test_a2_head_conv_ch3_parity() {
        let ch = 3;
        let (w, bias, scale) = make_test_weights(ch);
        let head = A2HeadConv::new(w.clone(), bias, scale, ch);

        let ring_size = 64;
        let ring_mask = ring_size - 1;
        let history = make_test_history(ch, ring_size);
        let write_pos: usize = 50;
        let num_frames = 8;

        let mut output = vec![0.0f32; num_frames];
        let mut scalar = vec![0.0f32; num_frames];

        head.process(&history, write_pos, ring_mask, num_frames, &mut output);
        a2_head_block_scalar_ref(
            &w,
            bias,
            scale,
            ch,
            &history,
            write_pos,
            ring_mask,
            num_frames,
            &mut scalar,
        );

        for f in 0..num_frames {
            let diff = (output[f] - scalar[f]).abs();
            assert!(
                diff < 1e-6,
                "CH=3 frame {}: proc={}, ref={}, diff={}",
                f,
                output[f],
                scalar[f],
                diff
            );
        }
    }

    #[test]
    fn test_a2_head_conv_ch8_parity() {
        let ch = 8;
        let (w, bias, scale) = make_test_weights(ch);
        let head = A2HeadConv::new(w.clone(), bias, scale, ch);

        let ring_size = 64;
        let ring_mask = ring_size - 1;
        let history = make_test_history(ch, ring_size);
        let write_pos: usize = 45;
        let num_frames = 16;

        let mut output = vec![0.0f32; num_frames];
        let mut scalar = vec![0.0f32; num_frames];

        head.process(&history, write_pos, ring_mask, num_frames, &mut output);
        a2_head_block_scalar_ref(
            &w,
            bias,
            scale,
            ch,
            &history,
            write_pos,
            ring_mask,
            num_frames,
            &mut scalar,
        );

        for f in 0..num_frames {
            let diff = (output[f] - scalar[f]).abs();
            assert!(
                diff < 1e-6,
                "CH=8 frame {}: proc={}, ref={}, diff={}",
                f,
                output[f],
                scalar[f],
                diff
            );
        }
    }

    #[test]
    fn test_a2_head_conv_ring_wraparound() {
        let ch = 3;
        let (w, bias, scale) = make_test_weights(ch);
        let head = A2HeadConv::new(w.clone(), bias, scale, ch);

        let ring_size = 32;
        let ring_mask = ring_size - 1;
        let history = make_test_history(ch, ring_size);
        let write_pos: usize = 5;
        let num_frames = 16;

        let mut output = vec![0.0f32; num_frames];
        let mut scalar = vec![0.0f32; num_frames];

        head.process(&history, write_pos, ring_mask, num_frames, &mut output);
        a2_head_block_scalar_ref(
            &w,
            bias,
            scale,
            ch,
            &history,
            write_pos,
            ring_mask,
            num_frames,
            &mut scalar,
        );

        for f in 0..num_frames {
            let diff = (output[f] - scalar[f]).abs();
            assert!(
                diff < 1e-6,
                "wrap frame {}: proc={}, ref={}, diff={}",
                f,
                output[f],
                scalar[f],
                diff
            );
        }
    }

    #[test]
    fn test_a2_head_conv_known_values_ch3() {
        let ch = 3;
        let k = A2HeadConv::HEAD_KERNEL_SIZE;

        let mut w = AlignedVec::new(k * ch, 0.0f32);
        let bias = 0.5;
        let scale = 2.0;

        // w[0][0] = 1.0, all others = 0
        w[0] = 1.0;

        let ring_size = 32;
        let ring_mask = ring_size - 1;
        let mut history = vec![0.0f32; ch * ring_size];

        let write_pos: usize = 20;
        let num_frames = 2;

        // At frame 0, tap 0 (oldest = K-1 back) should read col = write_pos - num_frames + 0 - (K-1 - 0) = write_pos - num_frames - (K-1)
        let col_expected =
            (write_pos as isize - num_frames as isize - (k as isize - 1)) as usize & ring_mask;
        history[col_expected * ch] = 3.0;

        let head = A2HeadConv::new(w, bias, scale, ch);
        let mut output = vec![0.0f32; num_frames];
        head.process(&history, write_pos, ring_mask, num_frames, &mut output);

        let expected = (bias + 1.0 * 3.0) * scale; // 0.5 + 3.0 = 3.5 * 2.0 = 7.0
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "got {} expected {}",
            output[0],
            expected
        );

        // Frame 1 should be 0.5 * 2.0 = 1.0 (no non-zero history hit)
        let expected_f1 = bias * scale;
        assert!(
            (output[1] - expected_f1).abs() < 1e-6,
            "frame1: got {} expected {}",
            output[1],
            expected_f1
        );
    }

    #[test]
    fn test_a2_head_conv_all_taps_ch8() {
        let ch = 8;
        let k = A2HeadConv::HEAD_KERNEL_SIZE;

        let mut w = AlignedVec::new(k * ch, 0.0f32);
        let bias = 0.1;
        let scale = 1.5;

        // Each tap has weight 1.0 for channel 0
        for t in 0..k {
            w[t * ch] = 1.0;
        }

        let ring_size = 64;
        let ring_mask = ring_size - 1;
        let mut history = vec![0.0f32; ch * ring_size];

        let write_pos: usize = 35;
        let num_frames = 1;

        // Fill all K lookback columns with value 1.0 at channel 0
        for t in 0..k {
            let col = (write_pos as isize - num_frames as isize - (k as isize - 1 - t as isize))
                as usize
                & ring_mask;
            history[col * ch] = 1.0;
        }

        let head = A2HeadConv::new(w, bias, scale, ch);
        let mut output = vec![0.0f32; num_frames];
        head.process(&history, write_pos, ring_mask, num_frames, &mut output);

        let expected = (bias + k as f32 * 1.0 * 1.0) * scale;
        assert!(
            (output[0] - expected).abs() < 1e-6,
            "all taps: got {} expected {}",
            output[0],
            expected
        );
    }

    #[test]
    fn test_a2_head_conv_ch3_stepping_write_pos() {
        let ch = 3;
        let (w, bias, scale) = make_test_weights(ch);
        let head = A2HeadConv::new(w.clone(), bias, scale, ch);

        let ring_size = 64;
        let ring_mask = ring_size - 1;
        let history = make_test_history(ch, ring_size);
        let num_frames = 4;

        for wp in [num_frames, num_frames + 10, num_frames + 20] {
            let write_pos = wp;
            let mut output = vec![0.0f32; num_frames];
            let mut scalar = vec![0.0f32; num_frames];

            head.process(&history, write_pos, ring_mask, num_frames, &mut output);
            a2_head_block_scalar_ref(
                &w,
                bias,
                scale,
                ch,
                &history,
                write_pos,
                ring_mask,
                num_frames,
                &mut scalar,
            );

            for f in 0..num_frames {
                let diff = (output[f] - scalar[f]).abs();
                assert!(
                    diff < 1e-6,
                    "wp={} frame {}: proc={}, ref={}, diff={}",
                    wp,
                    f,
                    output[f],
                    scalar[f],
                    diff
                );
            }
        }
    }
}
