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
use crate::gemm_batch_frame_loop_avx2;
use crate::gemm_batch_inner_dual_avx2;
use crate::gemm_batch_outc_loop_avx2;
use crate::math::common::half::f16_bits_to_f32_f16c;
use core::arch::x86_64::*;

/// Processes multiple audio frames in batch using the fused technique: Y = X_res + Bias + W * Z.
///
/// This is the most powerful version of the fused operation. It organizes the work in groups of 4
/// audio frames, allowing the processor to reuse the neural network weights extremely
/// efficiently for all of them before needing to read new data from memory.
///
/// # Optimization
/// Processes 2 input columns per iteration with 8 independent FMA accumulators
/// (2 per frame: `acc_lo`/`acc_hi`), breaking the serial FMA dependency chain and
/// doubling port utilization on x86-64-v3.
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
        gemm_batch_frame_loop_avx2!(
            f,
            num_frames,
            {
                let mut out_c = 0;
                gemm_batch_outc_loop_avx2!(
                    out_c,
                    out_len,
                    {
                        let existing0 =
                            _mm256_loadu_ps(out_frames.as_ptr().add(f * out_len + out_c));
                        let existing1 =
                            _mm256_loadu_ps(out_frames.as_ptr().add((f + 1) * out_len + out_c));
                        let existing2 =
                            _mm256_loadu_ps(out_frames.as_ptr().add((f + 2) * out_len + out_c));
                        let existing3 =
                            _mm256_loadu_ps(out_frames.as_ptr().add((f + 3) * out_len + out_c));

                        let b = if do_bias {
                            _mm256_loadu_ps(bias.as_ptr().add(out_c))
                        } else {
                            _mm256_setzero_ps()
                        };

                        let mut acc0_lo = _mm256_add_ps(existing0, b);
                        let mut acc0_hi = _mm256_setzero_ps();
                        let mut acc1_lo = _mm256_add_ps(existing1, b);
                        let mut acc1_hi = _mm256_setzero_ps();
                        let mut acc2_lo = _mm256_add_ps(existing2, b);
                        let mut acc2_hi = _mm256_setzero_ps();
                        let mut acc3_lo = _mm256_add_ps(existing3, b);
                        let mut acc3_hi = _mm256_setzero_ps();

                        let mut in_c = 0;
                        gemm_batch_inner_dual_avx2!(
                            in_c,
                            in_len,
                            {
                                let wp_lo = weights.as_ptr().add(in_c * out_len + out_c);
                                let vw_lo =
                                    _mm256_cvtph_ps(_mm_loadu_si128(wp_lo as *const __m128i));
                                let wp_hi = weights.as_ptr().add((in_c + 1) * out_len + out_c);
                                let vw_hi =
                                    _mm256_cvtph_ps(_mm_loadu_si128(wp_hi as *const __m128i));

                                let vs0_lo =
                                    _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                                let vs0_hi =
                                    _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 1));
                                acc0_lo = _mm256_fmadd_ps(vs0_lo, vw_lo, acc0_lo);
                                acc0_hi = _mm256_fmadd_ps(vs0_hi, vw_hi, acc0_hi);

                                let vs1_lo = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 1) * in_len + in_c),
                                );
                                let vs1_hi = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 1) * in_len + in_c + 1),
                                );
                                acc1_lo = _mm256_fmadd_ps(vs1_lo, vw_lo, acc1_lo);
                                acc1_hi = _mm256_fmadd_ps(vs1_hi, vw_hi, acc1_hi);

                                let vs2_lo = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 2) * in_len + in_c),
                                );
                                let vs2_hi = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 2) * in_len + in_c + 1),
                                );
                                acc2_lo = _mm256_fmadd_ps(vs2_lo, vw_lo, acc2_lo);
                                acc2_hi = _mm256_fmadd_ps(vs2_hi, vw_hi, acc2_hi);

                                let vs3_lo = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 3) * in_len + in_c),
                                );
                                let vs3_hi = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 3) * in_len + in_c + 1),
                                );
                                acc3_lo = _mm256_fmadd_ps(vs3_lo, vw_lo, acc3_lo);
                                acc3_hi = _mm256_fmadd_ps(vs3_hi, vw_hi, acc3_hi);
                            },
                            {
                                let wp = weights.as_ptr().add(in_c * out_len + out_c);
                                let vw = _mm256_cvtph_ps(_mm_loadu_si128(wp as *const __m128i));
                                let vs0 =
                                    _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                                let vs1 = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 1) * in_len + in_c),
                                );
                                let vs2 = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 2) * in_len + in_c),
                                );
                                let vs3 = _mm256_set1_ps(
                                    *in_frames.get_unchecked((f + 3) * in_len + in_c),
                                );
                                acc0_lo = _mm256_fmadd_ps(vs0, vw, acc0_lo);
                                acc1_lo = _mm256_fmadd_ps(vs1, vw, acc1_lo);
                                acc2_lo = _mm256_fmadd_ps(vs2, vw, acc2_lo);
                                acc3_lo = _mm256_fmadd_ps(vs3, vw, acc3_lo);
                            }
                        );

                        let acc0 = _mm256_add_ps(acc0_lo, acc0_hi);
                        let acc1 = _mm256_add_ps(acc1_lo, acc1_hi);
                        let acc2 = _mm256_add_ps(acc2_lo, acc2_hi);
                        let acc3 = _mm256_add_ps(acc3_lo, acc3_hi);

                        _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                        _mm256_storeu_ps(
                            out_frames.as_mut_ptr().add((f + 1) * out_len + out_c),
                            acc1,
                        );
                        _mm256_storeu_ps(
                            out_frames.as_mut_ptr().add((f + 2) * out_len + out_c),
                            acc2,
                        );
                        _mm256_storeu_ps(
                            out_frames.as_mut_ptr().add((f + 3) * out_len + out_c),
                            acc3,
                        );
                    },
                    {
                        for i in 0..4 {
                            let frame_idx = f + i;
                            let mut sum = *out_frames.get_unchecked(frame_idx * out_len + out_c);
                            if do_bias {
                                sum += *bias.get_unchecked(out_c);
                            }
                            for in_c in 0..in_len {
                                let w_bits = *weights.get_unchecked(in_c * out_len + out_c);
                                let w = f16_bits_to_f32_f16c(w_bits);
                                sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                            }
                            *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
                        }
                    }
                );
            },
            {
                fused_add_gemv_avx2(
                    in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
                    weights,
                    bias,
                    out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
                    do_bias,
                );
            }
        );
    }
}

/// Fused residual GEMM kernel via AVX2.
///
/// This function is the "main engine" of many modern neural network layers. It combines
/// matrix-vector multiplication with the addition of a "residual connection" (a shortcut
/// that helps the network retain important information from the past). By fusing all of this
/// into a single vectorized step, we save valuable memory cycles.
///
/// # Optimization
/// Processes 2 input columns per iteration with 8 independent FMA accumulators
/// (2 per frame: `acc_lo`/`acc_hi`), breaking the serial FMA dependency chain and
/// doubling port utilization on x86-64-v3.
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
    gemm_batch_frame_loop_avx2!(
        f,
        num_frames,
        {
            let mut out_c = 0;
            gemm_batch_outc_loop_avx2!(
                out_c,
                out_len,
                {
                    let res0 = _mm256_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
                    let res1 = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
                    let res2 = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
                    let res3 = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));

                    let b = if do_bias {
                        _mm256_loadu_ps(bias.as_ptr().add(out_c))
                    } else {
                        _mm256_setzero_ps()
                    };

                    let mut acc0_lo = _mm256_add_ps(res0, b);
                    let mut acc0_hi = _mm256_setzero_ps();
                    let mut acc1_lo = _mm256_add_ps(res1, b);
                    let mut acc1_hi = _mm256_setzero_ps();
                    let mut acc2_lo = _mm256_add_ps(res2, b);
                    let mut acc2_hi = _mm256_setzero_ps();
                    let mut acc3_lo = _mm256_add_ps(res3, b);
                    let mut acc3_hi = _mm256_setzero_ps();

                    let mut in_c = 0;
                    gemm_batch_inner_dual_avx2!(
                        in_c,
                        in_len,
                        {
                            let wp_lo = weights.as_ptr().add(in_c * out_len + out_c);
                            let vw_lo = _mm256_cvtph_ps(_mm_loadu_si128(wp_lo as *const __m128i));
                            let wp_hi = weights.as_ptr().add((in_c + 1) * out_len + out_c);
                            let vw_hi = _mm256_cvtph_ps(_mm_loadu_si128(wp_hi as *const __m128i));

                            let vs0_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                            let vs0_hi =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 1));
                            acc0_lo = _mm256_fmadd_ps(vs0_lo, vw_lo, acc0_lo);
                            acc0_hi = _mm256_fmadd_ps(vs0_hi, vw_hi, acc0_hi);

                            let vs1_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                            let vs1_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 1) * in_len + in_c + 1),
                            );
                            acc1_lo = _mm256_fmadd_ps(vs1_lo, vw_lo, acc1_lo);
                            acc1_hi = _mm256_fmadd_ps(vs1_hi, vw_hi, acc1_hi);

                            let vs2_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                            let vs2_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 2) * in_len + in_c + 1),
                            );
                            acc2_lo = _mm256_fmadd_ps(vs2_lo, vw_lo, acc2_lo);
                            acc2_hi = _mm256_fmadd_ps(vs2_hi, vw_hi, acc2_hi);

                            let vs3_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));
                            let vs3_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 3) * in_len + in_c + 1),
                            );
                            acc3_lo = _mm256_fmadd_ps(vs3_lo, vw_lo, acc3_lo);
                            acc3_hi = _mm256_fmadd_ps(vs3_hi, vw_hi, acc3_hi);
                        },
                        {
                            let wp = weights.as_ptr().add(in_c * out_len + out_c);
                            let vw = _mm256_cvtph_ps(_mm_loadu_si128(wp as *const __m128i));
                            let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                            let vs1 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                            let vs2 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                            let vs3 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));

                            acc0_lo = _mm256_fmadd_ps(vs0, vw, acc0_lo);
                            acc1_lo = _mm256_fmadd_ps(vs1, vw, acc1_lo);
                            acc2_lo = _mm256_fmadd_ps(vs2, vw, acc2_lo);
                            acc3_lo = _mm256_fmadd_ps(vs3, vw, acc3_lo);
                        }
                    );

                    let acc0 = _mm256_add_ps(acc0_lo, acc0_hi);
                    let acc1 = _mm256_add_ps(acc1_lo, acc1_hi);
                    let acc2 = _mm256_add_ps(acc2_lo, acc2_hi);
                    let acc3 = _mm256_add_ps(acc3_lo, acc3_hi);

                    _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
                },
                {
                    for i in 0..4 {
                        let frame_idx = f + i;
                        let mut sum = *residual.get_unchecked(frame_idx * out_len + out_c);
                        if do_bias {
                            sum += *bias.get_unchecked(out_c);
                        }
                        for in_c in 0..in_len {
                            let w = f16_bits_to_f32_f16c(
                                *weights.get_unchecked(in_c * out_len + out_c),
                            );
                            sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                        }
                        *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
                    }
                }
            );
        },
        {
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
                    let w = f16_bits_to_f32_f16c(*weights.get_unchecked(in_c * out_len + out_c));
                    sum += *in_frame.get_unchecked(in_c) * w;
                }
                *out_frame.get_unchecked_mut(out_c) = sum;
                out_c += 1;
            }
        }
    );
}

/// Fused residual GEMM kernel via AVX2 with native f32 weights.
///
/// Identical to [`fused_gemm_residual_batch_avx2`] but accepts full-precision
/// f32 weights instead of f16-quantized (u16) weights. Used where the 1x1 projection
/// operates on native f32 weights and the residual
/// addition is fused into the same SIMD pass.
///
/// # Optimization
/// Processes 2 input columns per iteration with 8 independent FMA accumulators
/// (2 per frame: `acc_lo`/`acc_hi`), breaking the serial FMA dependency chain and
/// doubling port utilization on x86-64-v3.
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
    gemm_batch_frame_loop_avx2!(
        f,
        num_frames,
        {
            let mut out_c = 0;
            gemm_batch_outc_loop_avx2!(
                out_c,
                out_len,
                {
                    let res0 = _mm256_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
                    let res1 = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
                    let res2 = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
                    let res3 = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));

                    let b = if do_bias {
                        _mm256_loadu_ps(bias.as_ptr().add(out_c))
                    } else {
                        _mm256_setzero_ps()
                    };

                    let mut acc0_lo = _mm256_add_ps(res0, b);
                    let mut acc0_hi = _mm256_setzero_ps();
                    let mut acc1_lo = _mm256_add_ps(res1, b);
                    let mut acc1_hi = _mm256_setzero_ps();
                    let mut acc2_lo = _mm256_add_ps(res2, b);
                    let mut acc2_hi = _mm256_setzero_ps();
                    let mut acc3_lo = _mm256_add_ps(res3, b);
                    let mut acc3_hi = _mm256_setzero_ps();

                    let mut in_c = 0;
                    gemm_batch_inner_dual_avx2!(
                        in_c,
                        in_len,
                        {
                            let wp_lo = weights.as_ptr().add(in_c * out_len + out_c);
                            let vw_lo = _mm256_loadu_ps(wp_lo);
                            let wp_hi = weights.as_ptr().add((in_c + 1) * out_len + out_c);
                            let vw_hi = _mm256_loadu_ps(wp_hi);

                            let vs0_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                            let vs0_hi =
                                _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 1));
                            acc0_lo = _mm256_fmadd_ps(vs0_lo, vw_lo, acc0_lo);
                            acc0_hi = _mm256_fmadd_ps(vs0_hi, vw_hi, acc0_hi);

                            let vs1_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                            let vs1_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 1) * in_len + in_c + 1),
                            );
                            acc1_lo = _mm256_fmadd_ps(vs1_lo, vw_lo, acc1_lo);
                            acc1_hi = _mm256_fmadd_ps(vs1_hi, vw_hi, acc1_hi);

                            let vs2_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                            let vs2_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 2) * in_len + in_c + 1),
                            );
                            acc2_lo = _mm256_fmadd_ps(vs2_lo, vw_lo, acc2_lo);
                            acc2_hi = _mm256_fmadd_ps(vs2_hi, vw_hi, acc2_hi);

                            let vs3_lo =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));
                            let vs3_hi = _mm256_set1_ps(
                                *in_frames.get_unchecked((f + 3) * in_len + in_c + 1),
                            );
                            acc3_lo = _mm256_fmadd_ps(vs3_lo, vw_lo, acc3_lo);
                            acc3_hi = _mm256_fmadd_ps(vs3_hi, vw_hi, acc3_hi);
                        },
                        {
                            let wp = weights.as_ptr().add(in_c * out_len + out_c);
                            let vw = _mm256_loadu_ps(wp);
                            let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                            let vs1 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                            let vs2 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                            let vs3 =
                                _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));

                            acc0_lo = _mm256_fmadd_ps(vs0, vw, acc0_lo);
                            acc1_lo = _mm256_fmadd_ps(vs1, vw, acc1_lo);
                            acc2_lo = _mm256_fmadd_ps(vs2, vw, acc2_lo);
                            acc3_lo = _mm256_fmadd_ps(vs3, vw, acc3_lo);
                        }
                    );

                    let acc0 = _mm256_add_ps(acc0_lo, acc0_hi);
                    let acc1 = _mm256_add_ps(acc1_lo, acc1_hi);
                    let acc2 = _mm256_add_ps(acc2_lo, acc2_hi);
                    let acc3 = _mm256_add_ps(acc3_lo, acc3_hi);

                    _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
                    _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
                },
                {
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
                }
            );
        },
        {
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
                    sum += *in_frame.get_unchecked(in_c)
                        * *weights.get_unchecked(in_c * out_len + out_c);
                }
                *out_frame.get_unchecked_mut(out_c) = sum;
                out_c += 1;
            }
        }
    );
}
