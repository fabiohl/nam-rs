// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::bias_tune;
use super::traits::{ConvWeightsOutput, DenseWeightsOutput};
use crate::math::common::{AlignedVec, quantize_weight};

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
    // Padded to the nearest multiple of 4 output channels so that every
    // SIMD lane has a defined weight (zero-padded lanes produce zero output).
    let num_blocks = out_size.div_ceil(4);
    let padded_total = num_blocks * 4 * in_size * k_size;
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;

    let mut weights = AlignedVec::new(padded_total, 0u16);
    #[cfg(feature = "high-fidelity")]
    let mut f32_weights = AlignedVec::new(padded_total, 0.0f32);
    let interleaved = cursor.is_interleaved4();
    let raw_f32_owned: Vec<f32>;

    if interleaved {
        // File already stores weights in 4-wide interleaved order —
        // no transposition required, just quantize in-place.
        let raw = cursor.read_slice(padded_total)?;
        raw_f32_owned = raw.to_vec();
        for i in 0..padded_total {
            weights[i] = quantize_weight(raw_f32_owned[i], is_bf16);
        }
        #[cfg(feature = "high-fidelity")]
        {
            f32_weights.copy_from_slice(&raw_f32_owned);
        }
    } else {
        // Standard (in_ch, out_ch, kernel) layout in the file.
        // Transpose into 4-wide interleaved order so the DSP kernel
        // can process 4 output channels per SIMD operation.
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
        #[cfg(feature = "high-fidelity")]
        transpose_conv1d_interleaved_4wide_f32(
            &raw_f32_owned,
            &mut f32_weights,
            in_size,
            out_size,
            k_size,
        );
    }

    let mut bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())
    } else {
        AlignedVec::new(out_size, 0.0)
    };

    // Bias-Tune: correct the per-channel rounding error that quantization
    // introduces under a synthetic DC=1.0 signal, at zero RT cost.
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

    // Large dilations need 2-stage prefetch: schedule the load two
    // iterations ahead to hide the long-stride memory latency.
    let prefetch_fn = if dilation >= 128 {
        crate::math::common::prefetch_strategy_2stage
    } else {
        crate::math::common::prefetch_strategy_simple
    };

    #[cfg(feature = "high-fidelity")]
    {
        Ok(T::from_parts_f32(
            weights,
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
    #[cfg(not(feature = "high-fidelity"))]
    {
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
    #[cfg(feature = "high-fidelity")]
    let mut f32_weights = AlignedVec::new(total, 0.0f32);
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;
    let interleaved = cursor.is_interleaved4();

    if interleaved {
        for i in 0..total {
            weights[i] = quantize_weight(raw_f32_owned[i], is_bf16);
        }
        #[cfg(feature = "high-fidelity")]
        {
            f32_weights.copy_from_slice(&raw_f32_owned);
        }
    } else {
        transpose_dense_layer(&raw_f32_owned, &mut weights, in_size, out_size, is_bf16);
        #[cfg(feature = "high-fidelity")]
        transpose_dense_layer_f32(&raw_f32_owned, &mut f32_weights, in_size, out_size);
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

    #[cfg(feature = "high-fidelity")]
    {
        Ok(T::from_parts_head(
            weights,
            bias,
            do_bias,
            in_size,
            out_size,
            f32_weights,
        ))
    }
    #[cfg(not(feature = "high-fidelity"))]
    {
        Ok(T::from_parts(weights, bias, do_bias, in_size, out_size))
    }
}

pub(crate) fn read_dense_head_weights_typed<T: DenseWeightsOutput>(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    do_bias: bool,
) -> anyhow::Result<T> {
    let total = out_size * in_size;
    let raw = cursor.read_slice(total)?;
    let raw_f32_owned = raw.to_vec();
    let mut weights = AlignedVec::new(total, 0u16);
    let mut f32_weights = AlignedVec::new(total, 0.0f32);
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;
    let interleaved = cursor.is_interleaved4();

    if interleaved {
        for i in 0..total {
            weights[i] = quantize_weight(raw_f32_owned[i], is_bf16);
            f32_weights[i] = raw_f32_owned[i];
        }
    } else {
        transpose_dense_layer_f32(&raw_f32_owned, &mut f32_weights, in_size, out_size);
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
            "[BiasTune] Dense HEAD in={}, out={}, bf16={}: max_comp={:.6e}",
            in_size,
            out_size,
            is_bf16,
            max_comp
        );
        bias_tune::apply_bias_compensation(&mut bias, &compensation);
    }

    Ok(T::from_parts_head(
        weights,
        bias,
        do_bias,
        in_size,
        out_size,
        f32_weights,
    ))
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

/// Rearranges dense layer weights into transposed format, keeping full f32 precision.
/// Stores result in column-major layout: `f32_weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c]`.
fn transpose_dense_layer_f32(raw: &[f32], weights: &mut [f32], in_size: usize, out_size: usize) {
    for out_c in 0..out_size {
        for in_c in 0..in_size {
            weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c];
        }
    }
}

/// Rearranges convolution layer weights into the "Interleaved 4-Wide" format
/// preserving full f32 precision (no quantization).
/// Same layout as `transpose_conv1d_interleaved_4wide`, but stores raw f32 values.
#[cfg(feature = "high-fidelity")]
fn transpose_conv1d_interleaved_4wide_f32(
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
