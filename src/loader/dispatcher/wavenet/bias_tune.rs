// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bias-Tuning: rounding bias compensation in quantized weights.
//!
//! During model loading, a simulated inference runs with a synthetic
//! DC=1.0 signal to measure the per-channel drift introduced by
//! BF16/FP16 weight quantization. The compensatory offset vector is added
//! directly to the FP32 bias coefficients, resulting in zero
//! computational overhead on the RT thread.

#[cfg(test)]
use crate::math::common::quantize_weight;

use crate::math::common::half::f16_bits_to_f32;

/// Reverses `quantize_weight`: reconstructs an approximate f32 from u16.
#[inline(always)]
fn dequantize_weight(w: u16, is_bf16: bool) -> f32 {
    if is_bf16 {
        f32::from_bits((w as u32) << 16)
    } else {
        f16_bits_to_f32(w)
    }
}

/// Computes per-channel bias compensation for a Dense layer.
///
/// Uses the premise of DC=1.0 signal on all input channels.
/// For each output channel `i`:
///   compensation[i] = Σⱼ (W_fp32[i,j] - W_quant[i,j])
///
/// * `raw_f32` — original weights in post-cursor-read layout.
/// * `quant_u16` — already quantized and transposed (column-major) weights.
/// * `is_interleaved` — if true, `raw_f32` and `quant_u16` share the same
///   column-major layout; otherwise `raw_f32` is row-major and `quant_u16`
///   is column-major (transposed).
pub fn compute_dense_bias_compensation(
    raw_f32: &[f32],
    quant_u16: &[u16],
    in_size: usize,
    out_size: usize,
    is_interleaved: bool,
    is_bf16: bool,
) -> Vec<f32> {
    let mut compensation = vec![0.0f32; out_size];

    if is_interleaved {
        // Column-major: raw_f32[in_c * out_size + out_c]
        // quant_u16 shares the same layout
        for (out_c, comp) in compensation.iter_mut().enumerate().take(out_size) {
            let mut sum_diff = 0.0f32;
            for in_c in 0..in_size {
                let idx = in_c * out_size + out_c;
                sum_diff += raw_f32[idx] - dequantize_weight(quant_u16[idx], is_bf16);
            }
            *comp = sum_diff;
        }
    } else {
        // raw_f32 is row-major: raw_f32[out_c * in_size + in_c]
        // quant_u16 is column-major (already transposed by the loader)
        for (out_c, comp) in compensation.iter_mut().enumerate().take(out_size) {
            let mut sum_diff = 0.0f32;
            for in_c in 0..in_size {
                let raw = raw_f32[out_c * in_size + in_c];
                let quant = dequantize_weight(quant_u16[in_c * out_size + out_c], is_bf16);
                sum_diff += raw - quant;
            }
            *comp = sum_diff;
        }
    }

    compensation
}

/// Computes per-channel bias compensation for a Conv1D layer.
///
/// Uses the premise of DC=1.0 signal on all input channels and taps.
/// For each output channel `i`:
///   compensation[i] = Σ_{k,cin} (W_fp32[i,cin,k] - W_quant[i,cin,k])
///
/// Layout variants:
///   `is_interleaved=true`  → raw_f32 and quant_u16 are both 4-wide interleaved
///   `is_interleaved=false` → raw_f32 is standard (out_c*in_size+k_size), quant_u16 is 4-wide interleaved
pub fn compute_conv1d_bias_compensation(
    raw_f32: &[f32],
    quant_u16: &[u16],
    in_size: usize,
    out_size: usize,
    k_size: usize,
    is_interleaved: bool,
    is_bf16: bool,
) -> Vec<f32> {
    let mut compensation = vec![0.0f32; out_size];

    if is_interleaved {
        // 4-wide interleaved layout (both arrays):
        //   idx = b*(k_size*in_size*4) + k*(in_size*4) + in_c*4 + lane
        //   out_c = b*4 + lane
        let num_blocks = out_size.div_ceil(4);
        for b in 0..num_blocks {
            for k in 0..k_size {
                for in_c in 0..in_size {
                    for lane in 0..4 {
                        let out_c = b * 4 + lane;
                        if out_c < out_size {
                            let idx =
                                b * (k_size * in_size * 4) + k * (in_size * 4) + in_c * 4 + lane;
                            compensation[out_c] +=
                                raw_f32[idx] - dequantize_weight(quant_u16[idx], is_bf16);
                        }
                    }
                }
            }
        }
    } else {
        // raw_f32: standard (out_c, in_c, kernel) row-major
        //   raw_idx = (out_c * in_size + in_c) * k_size + k
        // quant_u16: 4-wide interleaved (already transposed by the loader)
        let num_blocks = out_size.div_ceil(4);
        for b in 0..num_blocks {
            for k in 0..k_size {
                for in_c in 0..in_size {
                    for lane in 0..4 {
                        let out_c = b * 4 + lane;
                        if out_c < out_size {
                            let raw_idx = (out_c * in_size + in_c) * k_size + k;
                            let quant_idx =
                                b * (k_size * in_size * 4) + k * (in_size * 4) + in_c * 4 + lane;
                            compensation[out_c] +=
                                raw_f32[raw_idx] - dequantize_weight(quant_u16[quant_idx], is_bf16);
                        }
                    }
                }
            }
        }
    }

    compensation
}

/// Applies the compensation vector to the bias, adding element-wise.
pub fn apply_bias_compensation(bias: &mut [f32], compensation: &[f32]) {
    for i in 0..bias.len() {
        bias[i] += compensation[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_bf16_roundtrip() {
        let original = 0.123456f32;
        let quant = quantize_weight(original, true);
        let recovered = dequantize_weight(quant, true);
        // BF16 truncation: error should be < 1%
        assert!((original - recovered).abs() < 0.01 * original.abs());
    }

    #[test]
    fn test_dequantize_f16_roundtrip() {
        let original = 0.123456f32;
        let quant = quantize_weight(original, false);
        let recovered = dequantize_weight(quant, false);
        assert!((original - recovered).abs() < 0.001 * original.abs());
    }

    #[test]
    fn test_dense_compensation_nonzero_bf16() {
        // Two input channels, two output channels.
        // Raw data row-major: raw[out_c * in_size + in_c]
        let raw = vec![0.1f32, 0.2, 0.3, 0.4]; // ch0=[0.1,0.2], ch1=[0.3,0.4]
        let mut quant = [0u16; 4];
        for i in 0..4 {
            quant[i] = quantize_weight(raw[i], true);
        }
        // Non-interleaved: compensation uses raw row-major and quant column-major.
        // Transpose quant to column-major: quant_col[in_c * out_size + out_c]
        let mut quant_col = vec![0u16; 4];
        quant_col[0] = quant[0]; // in=0,out=0
        quant_col[1] = quant[2]; // in=0,out=1 (was raw[0*2+1])
        quant_col[2] = quant[1]; // in=1,out=0
        quant_col[3] = quant[3]; // in=1,out=1
        let comp = compute_dense_bias_compensation(&raw, &quant_col, 2, 2, false, true);
        for i in 0..2 {
            let expected = (raw[i * 2] - dequantize_weight(quant[i * 2], true))
                + (raw[i * 2 + 1] - dequantize_weight(quant[i * 2 + 1], true));
            assert!((comp[i] - expected).abs() < 1e-10);
        }
        assert!(comp[0].abs() > 0.0);
    }

    #[test]
    fn test_apply_bias_compensation() {
        let mut bias = vec![1.0, 2.0, 3.0];
        let comp = vec![0.1, 0.2, 0.0];
        apply_bias_compensation(&mut bias, &comp);
        assert!((bias[0] - 1.1).abs() < 1e-10);
        assert!((bias[1] - 2.2).abs() < 1e-10);
        assert!((bias[2] - 3.0).abs() < 1e-10);
    }
}
