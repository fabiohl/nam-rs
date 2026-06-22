// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Weight layout transposition helpers for the A2 architecture.
//!
//! Provides shared transposition routines used during weight loading
//! (`set_weights`) and dynamic-model construction. All functions operate
//! at full f32 precision — quantization happens downstream.

/// Rearranges dense layer weights from row-major to col-major.
///
/// Input:  `raw[out * in_size + in_c]` (row-major)
/// Output: `weights[in_c * out_size + out_c]` (col-major)
#[inline]
pub(crate) fn transpose_dense_f32(
    raw: &[f32],
    weights: &mut [f32],
    in_size: usize,
    out_size: usize,
) {
    for out_c in 0..out_size {
        for in_c in 0..in_size {
            weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c];
        }
    }
}

/// Rearranges conv1d weights into "Interleaved 4-Wide" format.
///
/// Groups output channels in blocks of 4 for SIMD processing.
/// The interleaved layout places all coefficients for a single kernel tap
/// and input channel together across 4 output channels, allowing 4-wide
/// fused-multiply-add when the output block is a multiple of 4.
#[inline]
pub(crate) fn transpose_conv1d_interleaved_4wide(
    raw: &[f32],
    weights: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    let num_blocks = out_ch.div_ceil(4);
    for b in 0..num_blocks {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    let target_idx = b * (kernel * in_ch * 4) + k * (in_ch * 4) + in_c * 4 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * kernel + k;
                        weights[target_idx] = raw[raw_idx];
                    }
                }
            }
        }
    }
}

/// Transposes head weights from `[channel][tap]` (NAM JSON format) to
/// `[tap][channel]` (`A2HeadConv` internal format).
///
/// NAM JSON weight layout for Conv1D(1, CH, 16): `raw[channel * 16 + tap]`
/// A2HeadConv expects: `head[tap * CH + channel]`
#[inline]
pub(crate) fn transpose_head_w(raw: &[f32], head: &mut [f32], channels: usize, kernel: usize) {
    for tap in 0..kernel {
        for ch in 0..channels {
            head[tap * channels + ch] = raw[ch * kernel + tap];
        }
    }
}
