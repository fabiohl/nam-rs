// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::bias_tune;
use super::traits::{ConvWeightsOutput, DenseWeightsOutput};
use crate::math::common::{AlignedVec, quantize_weight};

// =============================================================================
// Unified weight reading (static + dynamic)
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
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;

    let mut weights = AlignedVec::new(padded_total, 0u16);
    let interleaved = cursor.is_interleaved4();
    let raw_f32_owned: Vec<f32>;

    if interleaved {
        let raw = cursor.read_slice(padded_total)?;
        raw_f32_owned = raw.to_vec();
        for i in 0..padded_total {
            weights[i] = quantize_weight(raw_f32_owned[i], is_bf16);
        }
    } else {
        let total = out_size * in_size * k_size;
        let raw = cursor.read_slice(total)?;
        raw_f32_owned = raw.to_vec();
        transpose_conv1d_interleaved_4wide(
            &raw_f32_owned,
            &mut weights,
            in_size,
            out_size,
            k_size,
            is_bf16,
        );
    }

    let mut bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())
    } else {
        AlignedVec::new(out_size, 0.0)
    };

    if do_bias && !raw_f32_owned.is_empty() {
        let compensation = bias_tune::compute_conv1d_bias_compensation(
            &raw_f32_owned,
            &weights,
            in_size,
            out_size,
            k_size,
            interleaved,
            is_bf16,
        );
        let max_comp = compensation.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        log::debug!(
            "[BiasTune] Conv1D in={}, out={}, k={}, dil={}, bf16={}: max_comp={:.6e}",
            in_size,
            out_size,
            k_size,
            dilation,
            is_bf16,
            max_comp
        );
        bias_tune::apply_bias_compensation(&mut bias, &compensation);
    }

    let prefetch_fn = if dilation >= 128 {
        crate::math::common::prefetch_strategy_2stage
    } else {
        crate::math::common::prefetch_strategy_simple
    };

    Ok(T::from_parts(
        weights,
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
    let raw_f32_owned = raw.to_vec();
    let mut weights = AlignedVec::new(total, 0u16);
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;
    let interleaved = cursor.is_interleaved4();

    if interleaved {
        for i in 0..total {
            weights[i] = quantize_weight(raw_f32_owned[i], is_bf16);
        }
    } else {
        transpose_dense_layer(&raw_f32_owned, &mut weights, in_size, out_size, is_bf16);
    }

    let mut bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())
    } else {
        AlignedVec::new(out_size, 0.0)
    };

    if do_bias && !raw_f32_owned.is_empty() {
        let compensation = bias_tune::compute_dense_bias_compensation(
            &raw_f32_owned,
            &weights,
            in_size,
            out_size,
            interleaved,
            is_bf16,
        );
        let max_comp = compensation.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        log::debug!(
            "[BiasTune] Dense in={}, out={}, bf16={}: max_comp={:.6e}",
            in_size,
            out_size,
            is_bf16,
            max_comp
        );
        bias_tune::apply_bias_compensation(&mut bias, &compensation);
    }

    Ok(T::from_parts(weights, bias, do_bias, in_size, out_size))
}

// =============================================================================
// Transposition and quantization
// =============================================================================

/// Rearranges convolution layer weights into the "Interleaved 4-Wide" format.
/// This technique groups data in blocks of 4, allowing the processor to execute
/// calculations in "batch" (SIMD), processing 4 audio channels at once.
pub fn transpose_conv1d_interleaved_4wide(
    raw: &[f32],
    weights: &mut [u16],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    is_bf16: bool,
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
                        weights[target_idx] = quantize_weight(raw[raw_idx], is_bf16);
                    } else {
                        weights[target_idx] = 0;
                    }
                }
            }
        }
    }
}

/// Rearranges dense layer weights, swapping rows and columns (transposition).
/// This aligns the data with the way the processor reads memory, avoiding slowdown.
fn transpose_dense_layer(
    raw: &[f32],
    weights: &mut [u16],
    in_size: usize,
    out_size: usize,
    is_bf16: bool,
) {
    for out_c in 0..out_size {
        for in_c in 0..in_size {
            let raw_val = raw[out_c * in_size + in_c];
            let val = quantize_weight(raw_val, is_bf16);
            weights[in_c * out_size + out_c] = val;
        }
    }
}
