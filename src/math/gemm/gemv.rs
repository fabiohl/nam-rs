// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! GEMV kernels (Matrix-Vector Multiplication) — AVX2 and AVX-512.
//!
//! Includes `_small` variants specialized for Standard WaveNet (CH=16),
//! batch versions, and the fused operation `fused_add_gemv`.
//!
//! # Parallelism Strategy
//! - AVX2: 4 YMM accumulators (4×8 = 32 lanes), inner loop with step 4.
//! - AVX-512: 8 ZMM accumulators (8×16 = 128 lanes), inner loop with step 8.
//! - FMA dependency chain breaking via multiple accumulators.
//! - Software prefetch on in_frame to reduce cache miss latency.

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
            let mut acc0 = _mm256_loadu_ps(out_frame.as_ptr().add(out_c));
            if do_bias {
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
            }
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut in_c = 0;
            while in_c + 4 <= in_len {
                _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 32) as *const i8);

                let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 1));
                let vs2 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 2));
                let vs3 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 3));

                let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs0, w0, acc0);

                let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(out_len) as *const __m128i));
                acc1 = _mm256_fmadd_ps(vs1, w1, acc1);

                let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(2 * out_len) as *const __m128i));
                acc2 = _mm256_fmadd_ps(vs2, w2, acc2);

                let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(3 * out_len) as *const __m128i));
                acc3 = _mm256_fmadd_ps(vs3, w3, acc3);

                in_c += 4;
            }

            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);

            while in_c < in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs, vw, acc0);
                in_c += 1;
            }

            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
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
            let mut acc0 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(out_c))
            } else {
                _mm256_setzero_ps()
            };
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut in_c = 0;
            while in_c + 4 <= in_len {
                _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 32) as *const i8);

                let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 1));
                let vs2 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 2));
                let vs3 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 3));

                let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs0, w0, acc0);

                let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(out_len) as *const __m128i));
                acc1 = _mm256_fmadd_ps(vs1, w1, acc1);

                let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(2 * out_len) as *const __m128i));
                acc2 = _mm256_fmadd_ps(vs2, w2, acc2);

                let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(3 * out_len) as *const __m128i));
                acc3 = _mm256_fmadd_ps(vs3, w3, acc3);

                in_c += 4;
            }

            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);

            while in_c < in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs, vw, acc0);
                in_c += 1;
            }

            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
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

// ── AVX-512 Small (CH=16 specialized) ──────────────────────────────────────

/// GEMV kernel AVX-512 specialized for Standard WaveNet (CH=16).
///
/// Uses 8 independent ZMM accumulators (8×16 = 128 lanes) with an inner loop
/// step of 8, breaking the FMA dependency chain and saturating the AVX-512
/// execution ports.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512_small(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let in_len = in_frame.len();

    let mut acc0 = if do_bias {
        _mm512_loadu_ps(bias.as_ptr())
    } else {
        _mm512_setzero_ps()
    };
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();

    let mut in_c = 0;
    while in_c + 8 <= in_len {
        _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 64) as *const i8);

        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
        let v_in4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
        let v_in5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
        let v_in6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
        let v_in7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

        let w_ptr = weights.as_ptr().add(in_c * 16);

        acc0 = _mm512_fmadd_ps(
            v_in0,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i)),
            acc0,
        );
        acc1 = _mm512_fmadd_ps(
            v_in1,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(16) as *const __m256i)),
            acc1,
        );
        acc2 = _mm512_fmadd_ps(
            v_in2,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(32) as *const __m256i)),
            acc2,
        );
        acc3 = _mm512_fmadd_ps(
            v_in3,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(48) as *const __m256i)),
            acc3,
        );
        acc4 = _mm512_fmadd_ps(
            v_in4,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(64) as *const __m256i)),
            acc4,
        );
        acc5 = _mm512_fmadd_ps(
            v_in5,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(80) as *const __m256i)),
            acc5,
        );
        acc6 = _mm512_fmadd_ps(
            v_in6,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(96) as *const __m256i)),
            acc6,
        );
        acc7 = _mm512_fmadd_ps(
            v_in7,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(112) as *const __m256i)),
            acc7,
        );
        in_c += 8;
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc4 = _mm512_add_ps(acc4, acc5);
    acc6 = _mm512_add_ps(acc6, acc7);
    acc0 = _mm512_add_ps(acc0, acc2);
    acc4 = _mm512_add_ps(acc4, acc6);
    acc0 = _mm512_add_ps(acc0, acc4);

    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        acc0 = _mm512_fmadd_ps(
            v_in,
            _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(in_c * 16) as *const __m256i
            )),
            acc0,
        );
        in_c += 1;
    }

    _mm512_storeu_ps(out_frame.as_mut_ptr(), acc0);
}

/// Fused-Add-GEMV kernel AVX-512 specialized for Standard WaveNet (CH=16).
///
/// 8 independent ZMM accumulators with step 8 in the inner loop.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512_small(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let in_len = in_frame.len();

    let mut acc0 = _mm512_loadu_ps(out_frame.as_ptr());
    if do_bias {
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(bias.as_ptr()));
    }
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();

    let mut in_c = 0;
    while in_c + 8 <= in_len {
        _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 64) as *const i8);

        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
        let v_in4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
        let v_in5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
        let v_in6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
        let v_in7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

        let w_ptr = weights.as_ptr().add(in_c * 16);

        acc0 = _mm512_fmadd_ps(
            v_in0,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i)),
            acc0,
        );
        acc1 = _mm512_fmadd_ps(
            v_in1,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(16) as *const __m256i)),
            acc1,
        );
        acc2 = _mm512_fmadd_ps(
            v_in2,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(32) as *const __m256i)),
            acc2,
        );
        acc3 = _mm512_fmadd_ps(
            v_in3,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(48) as *const __m256i)),
            acc3,
        );
        acc4 = _mm512_fmadd_ps(
            v_in4,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(64) as *const __m256i)),
            acc4,
        );
        acc5 = _mm512_fmadd_ps(
            v_in5,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(80) as *const __m256i)),
            acc5,
        );
        acc6 = _mm512_fmadd_ps(
            v_in6,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(96) as *const __m256i)),
            acc6,
        );
        acc7 = _mm512_fmadd_ps(
            v_in7,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(112) as *const __m256i)),
            acc7,
        );
        in_c += 8;
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc4 = _mm512_add_ps(acc4, acc5);
    acc6 = _mm512_add_ps(acc6, acc7);
    acc0 = _mm512_add_ps(acc0, acc2);
    acc4 = _mm512_add_ps(acc4, acc6);
    acc0 = _mm512_add_ps(acc0, acc4);

    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        acc0 = _mm512_fmadd_ps(
            v_in,
            _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(in_c * 16) as *const __m256i
            )),
            acc0,
        );
        in_c += 1;
    }

    _mm512_storeu_ps(out_frame.as_mut_ptr(), acc0);
}

// ── AVX-512 General ────────────────────────────────────────────────────────────

/// Performs the linear projection Y = Bias + W * Z (GEMV) replacing the contents of out_frame via AVX-512.
///
/// Uses 8 independent ZMM accumulators (8×16 = 128 lanes) with an inner loop
/// step of 8 to break the FMA dependency chain.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    if out_len == 16 {
        gemv_overwrite_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut acc0 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();

        let mut in_c = 0;
        while in_c + 8 <= in_len {
            _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 64) as *const i8);

            let vs0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let vs1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
            let vs2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
            let vs3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
            let vs4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
            let vs5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
            let vs6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
            let vs7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

            let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);

            let w0 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i));
            acc0 = _mm512_fmadd_ps(vs0, w0, acc0);

            let w1 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(out_len) as *const __m256i));
            acc1 = _mm512_fmadd_ps(vs1, w1, acc1);

            let w2 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(2 * out_len) as *const __m256i));
            acc2 = _mm512_fmadd_ps(vs2, w2, acc2);

            let w3 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(3 * out_len) as *const __m256i));
            acc3 = _mm512_fmadd_ps(vs3, w3, acc3);

            let w4 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(4 * out_len) as *const __m256i));
            acc4 = _mm512_fmadd_ps(vs4, w4, acc4);

            let w5 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(5 * out_len) as *const __m256i));
            acc5 = _mm512_fmadd_ps(vs5, w5, acc5);

            let w6 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(6 * out_len) as *const __m256i));
            acc6 = _mm512_fmadd_ps(vs6, w6, acc6);

            let w7 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(7 * out_len) as *const __m256i));
            acc7 = _mm512_fmadd_ps(vs7, w7, acc7);

            in_c += 8;
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc4 = _mm512_add_ps(acc4, acc5);
        acc6 = _mm512_add_ps(acc6, acc7);
        acc0 = _mm512_add_ps(acc0, acc2);
        acc4 = _mm512_add_ps(acc4, acc6);
        acc0 = _mm512_add_ps(acc0, acc4);

        while in_c < in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));
            acc0 = _mm512_fmadd_ps(vs, vw, acc0);
            in_c += 1;
        }

        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
        out_c += 16;
    }

    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w = half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
            sum += *in_frame.get_unchecked(in_c) * w;
        }
        *out_frame.get_unchecked_mut(out_c) = sum;
        out_c += 1;
    }
}

/// Batch version of gemv_overwrite via AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_batch_avx512(
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
    for i in 0..num_frames {
        let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
        let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
        gemv_overwrite_avx512(in_slice, weights, bias, out_slice, do_bias);
    }
}

/// Performs the fused operation Y = X_res + Bias + W * Z (Broadcast GEMV) via AVX-512.
///
/// 8 independent ZMM accumulators with step 8 in the inner loop.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    if out_len == 16 {
        fused_add_gemv_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut acc0 = _mm512_loadu_ps(out_frame.as_ptr().add(out_c));
        if do_bias {
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(bias.as_ptr().add(out_c)));
        }
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();

        let mut in_c = 0;
        while in_c + 8 <= in_len {
            _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 64) as *const i8);

            let vs0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let vs1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
            let vs2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
            let vs3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
            let vs4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
            let vs5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
            let vs6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
            let vs7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

            let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);

            let w0 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i));
            acc0 = _mm512_fmadd_ps(vs0, w0, acc0);

            let w1 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(out_len) as *const __m256i));
            acc1 = _mm512_fmadd_ps(vs1, w1, acc1);

            let w2 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(2 * out_len) as *const __m256i));
            acc2 = _mm512_fmadd_ps(vs2, w2, acc2);

            let w3 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(3 * out_len) as *const __m256i));
            acc3 = _mm512_fmadd_ps(vs3, w3, acc3);

            let w4 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(4 * out_len) as *const __m256i));
            acc4 = _mm512_fmadd_ps(vs4, w4, acc4);

            let w5 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(5 * out_len) as *const __m256i));
            acc5 = _mm512_fmadd_ps(vs5, w5, acc5);

            let w6 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(6 * out_len) as *const __m256i));
            acc6 = _mm512_fmadd_ps(vs6, w6, acc6);

            let w7 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(7 * out_len) as *const __m256i));
            acc7 = _mm512_fmadd_ps(vs7, w7, acc7);

            in_c += 8;
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc4 = _mm512_add_ps(acc4, acc5);
        acc6 = _mm512_add_ps(acc6, acc7);
        acc0 = _mm512_add_ps(acc0, acc2);
        acc4 = _mm512_add_ps(acc4, acc6);
        acc0 = _mm512_add_ps(acc0, acc4);

        while in_c < in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));
            acc0 = _mm512_fmadd_ps(vs, vw, acc0);
            in_c += 1;
        }

        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
        out_c += 16;
    }

    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w = half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
            sum += *in_frame.get_unchecked(in_c) * w;
        }
        *out_frame.get_unchecked_mut(out_c) += sum;
        out_c += 1;
    }
}
