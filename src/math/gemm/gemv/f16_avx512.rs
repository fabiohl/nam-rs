// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::gemv_kernel;
use crate::math::common::half::f16_bits_to_f32_f16c;
use core::arch::x86_64::*;

// ── AVX-512 Small (CH=16 specialized) ──────────────────────────────────────

/// GEMV kernel AVX-512 specialized for Standard WaveNet (CH=16).
///
/// Uses 8 independent ZMM accumulators (8×16 = 128 lanes) with an inner loop
/// step of 8, breaking the FMA dependency chain and saturating the AVX-512
/// execution ports.
#[target_feature(enable = "avx512f,avx512vl,f16c")]
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
#[target_feature(enable = "avx512f,avx512vl,f16c")]
pub unsafe fn fused_add_gemv_avx512_small(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        gemv_kernel!(
            8,
            true,
            0,
            16,
            in_frame,
            weights,
            bias,
            out_frame,
            do_bias,
            _mm512_setzero_ps,
            |oc| _mm512_loadu_ps(out_frame.as_ptr().add(oc)),
            |oc| _mm512_loadu_ps(bias.as_ptr().add(oc)),
            _mm512_add_ps,
            |ptr| _mm512_cvtph_ps(_mm256_loadu_si256(ptr as *const __m256i)),
            _mm512_fmadd_ps,
            |oc, val| _mm512_storeu_ps(out_frame.as_mut_ptr().add(oc), val)
        );
    }
}

// ── AVX-512 General ────────────────────────────────────────────────────────────

/// Performs the linear projection Y = Bias + W * Z (GEMV) replacing the contents of out_frame via AVX-512.
///
/// Uses 8 independent ZMM accumulators (8×16 = 128 lanes) with an inner loop
/// step of 8 to break the FMA dependency chain.
#[target_feature(enable = "avx512f,avx512vl,f16c")]
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

    unsafe {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            gemv_kernel!(
                8,
                false,
                out_c,
                out_len,
                in_frame,
                weights,
                bias,
                out_frame,
                do_bias,
                _mm512_setzero_ps,
                |oc| _mm512_loadu_ps(out_frame.as_ptr().add(oc)),
                |oc| _mm512_loadu_ps(bias.as_ptr().add(oc)),
                _mm512_add_ps,
                |ptr| _mm512_cvtph_ps(_mm256_loadu_si256(ptr as *const __m256i)),
                _mm512_fmadd_ps,
                |oc, val| _mm512_storeu_ps(out_frame.as_mut_ptr().add(oc), val)
            );
            out_c += 16;
        }

        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w = f16_bits_to_f32_f16c(*weights.get_unchecked(in_c * out_len + out_c));
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
    }
}

/// Batch version of gemv_overwrite via AVX-512.
#[target_feature(enable = "avx512f,avx512vl,f16c")]
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
#[target_feature(enable = "avx512f,avx512vl,f16c")]
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

    unsafe {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            gemv_kernel!(
                8,
                true,
                out_c,
                out_len,
                in_frame,
                weights,
                bias,
                out_frame,
                do_bias,
                _mm512_setzero_ps,
                |oc| _mm512_loadu_ps(out_frame.as_ptr().add(oc)),
                |oc| _mm512_loadu_ps(bias.as_ptr().add(oc)),
                _mm512_add_ps,
                |ptr| _mm512_cvtph_ps(_mm256_loadu_si256(ptr as *const __m256i)),
                _mm512_fmadd_ps,
                |oc, val| _mm512_storeu_ps(out_frame.as_mut_ptr().add(oc), val)
            );
            out_c += 16;
        }

        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w = f16_bits_to_f32_f16c(*weights.get_unchecked(in_c * out_len + out_c));
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) += sum;
            out_c += 1;
        }
    }
}
