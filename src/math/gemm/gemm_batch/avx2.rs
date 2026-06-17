// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Batch GEMM and Fused Residual GEMM kernels — AVX2.
//!
//! Process multiple audio frames simultaneously for efficient weight reuse.

use super::super::gemv::fused_add_gemv_avx2;
use core::arch::x86_64::*;

/// Processes multiple audio frames in batch using the fused technique: Y = X_res + Bias + W * Z.
///
/// This is the most powerful version of the fused operation. It organizes the work in groups of 4
/// audio frames, allowing the processor to reuse the neural network weights extremely
/// efficiently for all of them before needing to read new data from memory.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn fused_add_gemm_batch_avx2(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    unsafe {
        let mut f = 0;
        // Batch Strategy: Process data in groups of 4 audio frames.
        // This allows each neural network weight to be read once and reused
        // 4 times consecutively (once per frame), which is extremely efficient.
        while f + 4 <= num_frames {
            let mut out_c = 0;
            while out_c + 8 <= out_len {
                // Load the partial results (buckets) from 4 frames simultaneously.
                let mut acc0 = _mm256_loadu_ps(out_frames.as_ptr().add(f * out_len + out_c));
                let mut acc1 = _mm256_loadu_ps(out_frames.as_ptr().add((f + 1) * out_len + out_c));
                let mut acc2 = _mm256_loadu_ps(out_frames.as_ptr().add((f + 2) * out_len + out_c));
                let mut acc3 = _mm256_loadu_ps(out_frames.as_ptr().add((f + 3) * out_len + out_c));

                // If there is a Bias (offset), add it to all 4 frames at once.
                if do_bias {
                    let b = _mm256_loadu_ps(bias.as_ptr().add(out_c));
                    acc0 = _mm256_add_ps(acc0, b);
                    acc1 = _mm256_add_ps(acc1, b);
                    acc2 = _mm256_add_ps(acc2, b);
                    acc3 = _mm256_add_ps(acc3, b);
                }

                // Compute Loop: Multiply input by weights.
                for in_c in 0..in_len {
                    // Read the weight from memory only once.
                    let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                    let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));

                    // Broadcast the corresponding input from each of the 4 frames.
                    let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                    let vs1 = _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                    let vs2 = _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                    let vs3 = _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));

                    // Multiply and Accumulate (FMA) for the 4 frames using the same loaded weight.
                    acc0 = _mm256_fmadd_ps(vs0, vw, acc0);
                    acc1 = _mm256_fmadd_ps(vs1, vw, acc1);
                    acc2 = _mm256_fmadd_ps(vs2, vw, acc2);
                    acc3 = _mm256_fmadd_ps(vs3, vw, acc3);
                }

                // Save the 4 new results back to memory.
                _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
                out_c += 8;
            }

            // Handle leftovers from each block of 4 frames.
            while out_c < out_len {
                for i in 0..4 {
                    let frame_idx = f + i;
                    let mut sum = *out_frames.get_unchecked(frame_idx * out_len + out_c);
                    if do_bias {
                        sum += *bias.get_unchecked(out_c);
                    }
                    for in_c in 0..in_len {
                        let w_bits = *weights.get_unchecked(in_c * out_len + out_c);
                        let w = half::f16::from_bits(w_bits).to_f32();
                        sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                    }
                    *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
                }
                out_c += 1;
            }
            f += 4;
        }

        // Final Cleanup: If some frames remain (fewer than 4), process them one by one.
        while f < num_frames {
            fused_add_gemv_avx2(
                in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
                weights,
                bias,
                out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
                do_bias,
            );
            f += 1;
        }
    }
}

/// Fused residual GEMM kernel via AVX2.
///
/// This function is the "main engine" of many modern neural network layers. It combines
/// matrix-vector multiplication with the addition of a "residual connection" (a shortcut
/// that helps the network retain important information from the past). By fusing all of this
/// into a single vectorized step, we save valuable memory cycles.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn fused_gemm_residual_batch_avx2(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    let mut f = 0;
    // Batch Strategy: Process 4 audio frames simultaneously for weight reuse.
    while f + 4 <= num_frames {
        let mut out_c = 0;
        while out_c + 8 <= out_len {
            // Initialize accumulators with the "Residual Connection" (shortcut) values.
            let mut acc0 = _mm256_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
            let mut acc1 = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
            let mut acc2 = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
            let mut acc3 = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));

            // If there is a Bias, add it to the residual buckets.
            if do_bias {
                let b = _mm256_loadu_ps(bias.as_ptr().add(out_c));
                acc0 = _mm256_add_ps(acc0, b);
                acc1 = _mm256_add_ps(acc1, b);
                acc2 = _mm256_add_ps(acc2, b);
                acc3 = _mm256_add_ps(acc3, b);
            }

            // Weight Loop: Multiply and accumulate the matrix result onto the buckets.
            for in_c in 0..in_len {
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));

                acc0 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c)),
                    vw,
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c)),
                    vw,
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c)),
                    vw,
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c)),
                    vw,
                    acc3,
                );
            }

            // Save the 4 final results (Residual + Bias + Multiplication).
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
            out_c += 8;
        }

        // Final cleanup for the remaining matrix width in groups of 4 frames.
        while out_c < out_len {
            for i in 0..4 {
                let frame_idx = f + i;
                let mut sum = *residual.get_unchecked(frame_idx * out_len + out_c);
                if do_bias {
                    sum += *bias.get_unchecked(out_c);
                }
                for in_c in 0..in_len {
                    let w = half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c))
                        .to_f32();
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
            }
            out_c += 1;
        }
        f += 4;
    }

    // Fallback: If any isolated frame remains (fewer than 4), process it individually.
    while f < num_frames {
        let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
        let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
        let res_frame = &residual[f * out_len..(f + 1) * out_len];

        let mut out_c = 0;
        while out_c + 8 <= out_len {
            let mut accum = _mm256_loadu_ps(res_frame.as_ptr().add(out_c));
            if do_bias {
                accum = _mm256_add_ps(accum, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
            }
            for in_c in 0..in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));
                accum = _mm256_fmadd_ps(vs, vw, accum);
            }
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 8;
        }
        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            sum += res_frame[out_c];
            for in_c in 0..in_len {
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
        f += 1;
    }
}

/// Fused residual GEMM kernel via AVX2 with native f32 weights.
///
/// Identical to [`fused_gemm_residual_batch_avx2`] but accepts full-precision
/// f32 weights instead of f16-quantized (u16) weights. Used in the high-fidelity
/// path where the 1x1 projection operates on native f32 weights and the residual
/// addition is fused into the same SIMD pass.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn fused_gemm_residual_batch_f32_avx2(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    let mut f = 0;
    while f + 4 <= num_frames {
        let mut out_c = 0;
        while out_c + 8 <= out_len {
            let mut acc0 = _mm256_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
            let mut acc1 = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
            let mut acc2 = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
            let mut acc3 = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));

            if do_bias {
                let b = _mm256_loadu_ps(bias.as_ptr().add(out_c));
                acc0 = _mm256_add_ps(acc0, b);
                acc1 = _mm256_add_ps(acc1, b);
                acc2 = _mm256_add_ps(acc2, b);
                acc3 = _mm256_add_ps(acc3, b);
            }

            for in_c in 0..in_len {
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_loadu_ps(weight_ptr);

                acc0 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c)),
                    vw,
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c)),
                    vw,
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c)),
                    vw,
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c)),
                    vw,
                    acc3,
                );
            }

            _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
            out_c += 8;
        }

        while out_c < out_len {
            for i in 0..4 {
                let frame_idx = f + i;
                let mut sum = *residual.get_unchecked(frame_idx * out_len + out_c);
                if do_bias {
                    sum += *bias.get_unchecked(out_c);
                }
                for in_c in 0..in_len {
                    let w = *weights.get_unchecked(in_c * out_len + out_c);
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
            }
            out_c += 1;
        }
        f += 4;
    }

    while f < num_frames {
        let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
        let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
        let res_frame = &residual[f * out_len..(f + 1) * out_len];

        let mut out_c = 0;
        while out_c + 8 <= out_len {
            let mut accum = _mm256_loadu_ps(res_frame.as_ptr().add(out_c));
            if do_bias {
                accum = _mm256_add_ps(accum, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
            }
            for in_c in 0..in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_loadu_ps(weight_ptr);
                accum = _mm256_fmadd_ps(vs, vw, accum);
            }
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 8;
        }
        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            sum += res_frame[out_c];
            for in_c in 0..in_len {
                sum +=
                    *in_frame.get_unchecked(in_c) * *weights.get_unchecked(in_c * out_len + out_c);
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
        f += 1;
    }
}
