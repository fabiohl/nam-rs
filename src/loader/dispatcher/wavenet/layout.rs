// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::traits::{ConvWeightsOutput, DenseWeightsOutput};
use crate::math::common::AlignedVec;

// =============================================================================
// Unified weight reading for static WaveNet models
// =============================================================================

#[allow(clippy::too_many_arguments)]
pub(crate) fn read_conv1d_weights_typed<T: ConvWeightsOutput>(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    k_size: usize,
    dilation: usize,
    do_bias: bool,
) -> anyhow::Result<T> {
    let num_blocks = out_size.div_ceil(4);
    let padded_total = num_blocks * 4 * in_size * k_size;
    let interleaved = cursor.is_interleaved4();

    let mut f32_weights = AlignedVec::new(padded_total, 0.0f32);

    if interleaved {
        let raw = cursor.read_slice(padded_total)?;
        f32_weights.copy_from_slice(raw);
    } else {
        let total = out_size * in_size * k_size;
        let raw = cursor.read_slice(total)?;
        transpose_conv1d_interleaved_4wide(raw, &mut f32_weights, in_size, out_size, k_size);
    }

    let bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())
    } else {
        AlignedVec::new(out_size, 0.0)
    };

    // Large dilations need 2-stage prefetch: schedule the load two
    // iterations ahead to hide the long-stride memory latency.
    let prefetch_fn = if dilation >= 128 {
        crate::math::common::prefetch_strategy_2stage
    } else {
        crate::math::common::prefetch_strategy_simple
    };

    Ok(T::from_parts(
        f32_weights,
        bias,
        do_bias,
        dilation,
        in_size,
        out_size,
        k_size,
        prefetch_fn,
    ))
}

pub(crate) fn read_dense_weights_typed<T: DenseWeightsOutput>(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    do_bias: bool,
) -> anyhow::Result<T> {
    let total = out_size * in_size;
    let raw = cursor.read_slice(total)?;
    let mut f32_weights = AlignedVec::new(total, 0.0f32);
    let interleaved = cursor.is_interleaved4();

    if interleaved {
        f32_weights.copy_from_slice(raw);
    } else {
        transpose_dense_layer_f32(raw, &mut f32_weights, in_size, out_size);
    }

    let bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())
    } else {
        AlignedVec::new(out_size, 0.0)
    };

    let dummy_u16 = AlignedVec::new(total, 0u16);
    Ok(T::from_parts_head(
        dummy_u16,
        bias,
        do_bias,
        in_size,
        out_size,
        f32_weights,
    ))
}

pub(crate) fn read_dense_head_weights_typed<T: DenseWeightsOutput>(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    do_bias: bool,
) -> anyhow::Result<T> {
    let total = out_size * in_size;
    let raw = cursor.read_slice(total)?;
    let mut f32_weights = AlignedVec::new(total, 0.0f32);
    let interleaved = cursor.is_interleaved4();

    if interleaved {
        f32_weights.copy_from_slice(raw);
    } else {
        transpose_dense_layer_f32(raw, &mut f32_weights, in_size, out_size);
    }

    let bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())
    } else {
        AlignedVec::new(out_size, 0.0)
    };

    let dummy_u16 = AlignedVec::new(total, 0u16);
    Ok(T::from_parts_head(
        dummy_u16,
        bias,
        do_bias,
        in_size,
        out_size,
        f32_weights,
    ))
}

// =============================================================================
// Transposition (full f32 precision — no quantization)
// =============================================================================

/// Rearranges convolution layer weights into the "Interleaved 4-Wide" format
/// preserving full f32 precision.
/// This technique groups data in blocks of 4, allowing the processor to execute
/// calculations in "batch" (SIMD), processing 4 audio channels at once.
pub fn transpose_conv1d_interleaved_4wide(
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
                    } else {
                        weights[target_idx] = 0.0;
                    }
                }
            }
        }
    }
}

/// Rearranges dense layer weights into transposed format, keeping full f32 precision.
/// Stores result in column-major layout: `f32_weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c]`.
fn transpose_dense_layer_f32(raw: &[f32], weights: &mut [f32], in_size: usize, out_size: usize) {
    for out_c in 0..out_size {
        for in_c in 0..in_size {
            weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c];
        }
    }
}
