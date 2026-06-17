// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use crate::math::common::half::f16_bits_to_f32_f16c;
use core::arch::x86_64::*;

/// Performs the linear projection for the 4 "gates" of an LSTM cell simultaneously via AVX2.
///
/// In an LSTM neural network, each step requires computing 4 sub-results (gates). This
/// function executes all these computations at once, ensuring that the network's "memory"
/// update is done with maximum performance and minimum latency.
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_avx2(
    in_frame: &[f32],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len() / 4;
    let in_len = in_frame.len();

    unsafe {
        let mut out_c = 0;
        // Process the 4 LSTM gates in parallel, 8 elements at a time.
        while out_c + 8 <= out_len {
            // Initialize accumulators (buckets) with the Bias values for each gate.
            let mut acc0 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(out_c))
            } else {
                _mm256_setzero_ps()
            };
            let mut acc1 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(out_len + out_c))
            } else {
                _mm256_setzero_ps()
            };
            let mut acc2 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(2 * out_len + out_c))
            } else {
                _mm256_setzero_ps()
            };
            let mut acc3 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(3 * out_len + out_c))
            } else {
                _mm256_setzero_ps()
            };

            // Main Computation Loop:
            for in_c in 0..in_len {
                // Take a single input value and "broadcast" it to use across all gates.
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));

                // Multiply the input by the weights for each of the 4 gates (acc0 to acc3).
                // Each gate handles a different aspect of the LSTM "memory".
                let wp0 = w0.as_ptr().add(in_c * out_len + out_c);
                let vw0 = _mm256_cvtph_ps(_mm_loadu_si128(wp0 as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs, vw0, acc0);

                let wp1 = w1.as_ptr().add(in_c * out_len + out_c);
                let vw1 = _mm256_cvtph_ps(_mm_loadu_si128(wp1 as *const __m128i));
                acc1 = _mm256_fmadd_ps(vs, vw1, acc1);

                let wp2 = w2.as_ptr().add(in_c * out_len + out_c);
                let vw2 = _mm256_cvtph_ps(_mm_loadu_si128(wp2 as *const __m128i));
                acc2 = _mm256_fmadd_ps(vs, vw2, acc2);

                let wp3 = w3.as_ptr().add(in_c * out_len + out_c);
                let vw3 = _mm256_cvtph_ps(_mm_loadu_si128(wp3 as *const __m128i));
                acc3 = _mm256_fmadd_ps(vs, vw3, acc3);
            }

            // Save the final results of each gate to their proper places in memory.
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_len + out_c), acc1);
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(2 * out_len + out_c), acc2);
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(3 * out_len + out_c), acc3);
            out_c += 8;
        }

        // Tail Loop Processing:
        // Executed when the output dimension (out_len) is not a perfect multiple of the AVX2 vector
        // size (8 floats). Without this scalar cleanup fallback loop, attempting to read vector data
        // would cause out-of-bounds memory accesses (Out-Of-Bounds / Segfault).
        while out_c < out_len {
            // Initialize the sum for each of the 4 gates with the corresponding bias (if enabled).
            let mut sum0 = if do_bias { bias[out_c] } else { 0.0 };
            let mut sum1 = if do_bias { bias[out_len + out_c] } else { 0.0 };
            let mut sum2 = if do_bias {
                bias[2 * out_len + out_c]
            } else {
                0.0
            };
            let mut sum3 = if do_bias {
                bias[3 * out_len + out_c]
            } else {
                0.0
            };

            // Executes the classic dot product (fused multiply-accumulate) in a scalar fashion
            for in_c in 0..in_len {
                let s = *in_frame.get_unchecked(in_c);
                // Converts the f16-compressed weights (packed as 16-bit u16) to f32
                // at runtime using the `half` library before multiplying.
                sum0 += s * f16_bits_to_f32_f16c(*w0.get_unchecked(in_c * out_len + out_c));
                sum1 += s * f16_bits_to_f32_f16c(*w1.get_unchecked(in_c * out_len + out_c));
                sum2 += s * f16_bits_to_f32_f16c(*w2.get_unchecked(in_c * out_len + out_c));
                sum3 += s * f16_bits_to_f32_f16c(*w3.get_unchecked(in_c * out_len + out_c));
            }

            // Write each gate's results contiguously into the output tensor.
            *out_frame.get_unchecked_mut(out_c) = sum0;
            *out_frame.get_unchecked_mut(out_len + out_c) = sum1;
            *out_frame.get_unchecked_mut(2 * out_len + out_c) = sum2;
            *out_frame.get_unchecked_mut(3 * out_len + out_c) = sum3;
            out_c += 1;
        }
    }
}
