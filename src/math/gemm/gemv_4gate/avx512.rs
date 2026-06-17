// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use crate::math::common::half::f16_bits_to_f32_f16c;
use core::arch::x86_64::*;

/// GEMV 4-gate kernel AVX-512 for LSTM.
/// Gates in an LSTM network control what should be remembered and what should be forgotten.
/// This function processes the 4 main gates at once for a major speed boost.
// f16c added so the scalar tail can use the F16C hardware intrinsic.
// F16C is guaranteed by x86-64-v3 and present on all AVX-512 targets we support.
#[target_feature(enable = "avx512f,avx512vl,f16c")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_avx512(
    in_frame: &[f32],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        // Buckets for the 4 gates.
        let mut acc0 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + out_len))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc2 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + 2 * out_len))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc3 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + 3 * out_len))
        } else {
            _mm512_setzero_ps()
        };

        for in_c in 0..in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));

            // Load 16 weights for each of the 4 gates.
            let vw0 = _mm512_cvtph_ps(_mm256_loadu_si256(
                w0.as_ptr().add(in_c * out_len + out_c) as *const __m256i
            ));
            let vw1 = _mm512_cvtph_ps(_mm256_loadu_si256(
                w1.as_ptr().add(in_c * out_len + out_c) as *const __m256i
            ));
            let vw2 = _mm512_cvtph_ps(_mm256_loadu_si256(
                w2.as_ptr().add(in_c * out_len + out_c) as *const __m256i
            ));
            let vw3 = _mm512_cvtph_ps(_mm256_loadu_si256(
                w3.as_ptr().add(in_c * out_len + out_c) as *const __m256i
            ));

            // Multiply and accumulate into all 4 buckets at the same time.
            acc0 = _mm512_fmadd_ps(vs, vw0, acc0);
            acc1 = _mm512_fmadd_ps(vs, vw1, acc1);
            acc2 = _mm512_fmadd_ps(vs, vw2, acc2);
            acc3 = _mm512_fmadd_ps(vs, vw3, acc3);
        }

        // Save the results.
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c), acc0);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + out_len), acc1);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 2 * out_len), acc2);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 3 * out_len), acc3);
        out_c += 16;
    }

    // Tail Loop Processing for AVX-512:
    // Processes the remaining elements purely in scalar fashion if the out_len width is not divisible by 16.
    while out_c < out_len {
        let mut s0 = if do_bias { bias[out_c] } else { 0.0 };
        let mut s1 = if do_bias { bias[out_c + out_len] } else { 0.0 };
        let mut s2 = if do_bias {
            bias[out_c + 2 * out_len]
        } else {
            0.0
        };
        let mut s3 = if do_bias {
            bias[out_c + 3 * out_len]
        } else {
            0.0
        };
        for in_c in 0..in_len {
            let si = *in_frame.get_unchecked(in_c);
            // Weights are in packed f16 format and are decompressed on demand to f32.
            s0 += si * f16_bits_to_f32_f16c(*w0.get_unchecked(in_c * out_len + out_c));
            s1 += si * f16_bits_to_f32_f16c(*w1.get_unchecked(in_c * out_len + out_c));
            s2 += si * f16_bits_to_f32_f16c(*w2.get_unchecked(in_c * out_len + out_c));
            s3 += si * f16_bits_to_f32_f16c(*w3.get_unchecked(in_c * out_len + out_c));
        }
        out[out_c] = s0;
        out[out_c + out_len] = s1;
        out[out_c + 2 * out_len] = s2;
        out[out_c + 3 * out_len] = s3;
        out_c += 1;
    }
}
