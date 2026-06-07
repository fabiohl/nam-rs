// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::gemv_kernel;
use core::arch::x86_64::*;

// ── AVX2 ──────────────────────────────────────────────────────────────────────

/// Performs a combined (fused) high-speed mathematical operation: Y = X_res + Bias + W * Z.
///
/// This function does three things at the same time: preserves the current value (residual), adds an
/// offset (bias), and adds the result of a weight-times-input multiplication. Doing everything
/// at once avoids the processor having to read and write to memory multiple times, keeping
/// the data "hot" and ready for the next computation.
///
/// Uses 4 independent accumulators to break the FMA pipeline dependency chain,
/// allowing the processor to execute up to 4 FMAs in parallel.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn fused_add_gemv_avx2(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    unsafe {
        let mut out_c = 0;
        while out_c + 8 <= out_len {
            gemv_kernel!(
                4,
                true,
                out_c,
                out_len,
                in_frame,
                weights,
                bias,
                out_frame,
                do_bias,
                _mm256_setzero_ps,
                |oc| _mm256_loadu_ps(out_frame.as_ptr().add(oc)),
                |oc| _mm256_loadu_ps(bias.as_ptr().add(oc)),
                _mm256_add_ps,
                |ptr| _mm256_cvtph_ps(_mm_loadu_si128(ptr as *const __m128i)),
                _mm256_fmadd_ps,
                |oc, val| _mm256_storeu_ps(out_frame.as_mut_ptr().add(oc), val)
            );
            out_c += 8;
        }

        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w = half::f16::from_bits(weights[in_c * out_len + out_c]).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) += sum;
            out_c += 1;
        }
    }
}

/// Performs a linear projection (Y = Bias + W * Z), replacing the previous content.
///
/// Uses 4 independent accumulators to break the FMA pipeline dependency chain.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gemv_overwrite_avx2(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    unsafe {
        let mut out_c = 0;
        while out_c + 8 <= out_len {
            gemv_kernel!(
                4,
                false,
                out_c,
                out_len,
                in_frame,
                weights,
                bias,
                out_frame,
                do_bias,
                _mm256_setzero_ps,
                |oc| _mm256_loadu_ps(out_frame.as_ptr().add(oc)),
                |oc| _mm256_loadu_ps(bias.as_ptr().add(oc)),
                _mm256_add_ps,
                |ptr| _mm256_cvtph_ps(_mm_loadu_si128(ptr as *const __m128i)),
                _mm256_fmadd_ps,
                |oc, val| _mm256_storeu_ps(out_frame.as_mut_ptr().add(oc), val)
            );
            out_c += 8;
        }

        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
    }
}
