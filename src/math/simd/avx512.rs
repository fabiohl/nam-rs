// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! [T21] Backends SIMD AVX-512.
//!
//! Implementa kernels otimizados usando extensões AVX-512 Foundation,
//! VL, BW, DQ, VNNI e BF16.

use super::fallback::*;
use super::traits::SimdMath;
use core::arch::x86_64::*;

/// [T21] Kernel GEMV AVX-512 especializado para Standard WaveNet (CH=16).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512_small(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let in_len = in_frame.len();
    let mut accum0 = if do_bias {
        _mm512_loadu_ps(bias.as_ptr())
    } else {
        _mm512_setzero_ps()
    };
    let mut accum1 = _mm512_setzero_ps();

    let mut in_c = 0;
    while in_c + 4 <= in_len {
        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));

        let w_ptr = weights.as_ptr().add(in_c * 16);
        accum0 = _mm512_fmadd_ps(
            v_in0,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i)),
            accum0,
        );
        accum1 = _mm512_fmadd_ps(
            v_in1,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(16) as *const __m256i)),
            accum1,
        );
        accum0 = _mm512_fmadd_ps(
            v_in2,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(32) as *const __m256i)),
            accum0,
        );
        accum1 = _mm512_fmadd_ps(
            v_in3,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(48) as *const __m256i)),
            accum1,
        );
        in_c += 4;
    }
    accum0 = _mm512_add_ps(accum0, accum1);
    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        accum0 = _mm512_fmadd_ps(
            v_in,
            _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(in_c * 16) as *const __m256i
            )),
            accum0,
        );
        in_c += 1;
    }
    _mm512_storeu_ps(out_frame.as_mut_ptr(), accum0);
}

/// [T21] Kernel Fused-Add-GEMV AVX-512 especializado para Standard WaveNet (CH=16).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512_small(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let in_len = in_frame.len();
    let mut accum0 = _mm512_loadu_ps(out_frame.as_ptr());
    if do_bias {
        accum0 = _mm512_add_ps(accum0, _mm512_loadu_ps(bias.as_ptr()));
    }
    let mut accum1 = _mm512_setzero_ps();

    let mut in_c = 0;
    while in_c + 4 <= in_len {
        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));

        let w_ptr = weights.as_ptr().add(in_c * 16);
        accum0 = _mm512_fmadd_ps(
            v_in0,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i)),
            accum0,
        );
        accum1 = _mm512_fmadd_ps(
            v_in1,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(16) as *const __m256i)),
            accum1,
        );
        accum0 = _mm512_fmadd_ps(
            v_in2,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(32) as *const __m256i)),
            accum0,
        );
        accum1 = _mm512_fmadd_ps(
            v_in3,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(48) as *const __m256i)),
            accum1,
        );
        in_c += 4;
    }
    accum0 = _mm512_add_ps(accum0, accum1);
    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        accum0 = _mm512_fmadd_ps(
            v_in,
            _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(in_c * 16) as *const __m256i
            )),
            accum0,
        );
        in_c += 1;
    }
    _mm512_storeu_ps(out_frame.as_mut_ptr(), accum0);
}

/// Realiza a projeção linear Y = Bias + W * Z (GEMV) substituindo o conteúdo de out_frame via AVX-512.
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

    // Especialização para H=16
    if out_len == 16 {
        gemv_overwrite_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut accum = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };

        for in_c in 0..in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));
            accum = _mm512_fmadd_ps(vs, vw, accum);
        }

        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
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

/// Realiza a operação fundida Y = X_res + Bias + W * Z (Broadcast GEMV) via AVX-512.
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

    // Especialização para H=16
    if out_len == 16 {
        fused_add_gemv_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut accum = _mm512_loadu_ps(out_frame.as_ptr().add(out_c));
        if do_bias {
            accum = _mm512_add_ps(accum, _mm512_loadu_ps(bias.as_ptr().add(out_c)));
        }

        for in_c in 0..in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));
            accum = _mm512_fmadd_ps(vs, vw, accum);
        }

        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
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

/// Versão em batch da operação fundida Y = X_res + Bias + W * Z via AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemm_batch_avx512(
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

    let mut f = 0;
    while f + 8 <= num_frames {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            let mut acc0 = _mm512_loadu_ps(out_frames.as_ptr().add(f * out_len + out_c));
            let mut acc1 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 1) * out_len + out_c));
            let mut acc2 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 2) * out_len + out_c));
            let mut acc3 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 3) * out_len + out_c));
            let mut acc4 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 4) * out_len + out_c));
            let mut acc5 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 5) * out_len + out_c));
            let mut acc6 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 6) * out_len + out_c));
            let mut acc7 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 7) * out_len + out_c));

            if do_bias {
                let b = _mm512_loadu_ps(bias.as_ptr().add(out_c));
                acc0 = _mm512_add_ps(acc0, b);
                acc1 = _mm512_add_ps(acc1, b);
                acc2 = _mm512_add_ps(acc2, b);
                acc3 = _mm512_add_ps(acc3, b);
                acc4 = _mm512_add_ps(acc4, b);
                acc5 = _mm512_add_ps(acc5, b);
                acc6 = _mm512_add_ps(acc6, b);
                acc7 = _mm512_add_ps(acc7, b);
            }

            for in_c in 0..in_len {
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));

                let vs0 = _mm512_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                let vs1 = _mm512_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                let vs2 = _mm512_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                let vs3 = _mm512_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));
                let vs4 = _mm512_set1_ps(*in_frames.get_unchecked((f + 4) * in_len + in_c));
                let vs5 = _mm512_set1_ps(*in_frames.get_unchecked((f + 5) * in_len + in_c));
                let vs6 = _mm512_set1_ps(*in_frames.get_unchecked((f + 6) * in_len + in_c));
                let vs7 = _mm512_set1_ps(*in_frames.get_unchecked((f + 7) * in_len + in_c));

                acc0 = _mm512_fmadd_ps(vs0, vw, acc0);
                acc1 = _mm512_fmadd_ps(vs1, vw, acc1);
                acc2 = _mm512_fmadd_ps(vs2, vw, acc2);
                acc3 = _mm512_fmadd_ps(vs3, vw, acc3);
                acc4 = _mm512_fmadd_ps(vs4, vw, acc4);
                acc5 = _mm512_fmadd_ps(vs5, vw, acc5);
                acc6 = _mm512_fmadd_ps(vs6, vw, acc6);
                acc7 = _mm512_fmadd_ps(vs7, vw, acc7);
            }

            _mm512_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 4) * out_len + out_c), acc4);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 5) * out_len + out_c), acc5);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 6) * out_len + out_c), acc6);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 7) * out_len + out_c), acc7);
            out_c += 16;
        }

        while out_c < out_len {
            for i in 0..8 {
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
        f += 8;
    }

    while f < num_frames {
        fused_add_gemv_avx512(
            in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
            weights,
            bias,
            out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
            do_bias,
        );
        f += 1;
    }
}

/// [TF3] Kernel GEMM com residual fundido AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_gemm_residual_batch_avx512(
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

    for f in 0..num_frames {
        let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
        let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
        let res_frame = &residual[f * out_len..(f + 1) * out_len];

        let mut out_c = 0;
        while out_c + 16 <= out_len {
            let mut accum = _mm512_loadu_ps(res_frame.as_ptr().add(out_c));
            if do_bias {
                accum = _mm512_add_ps(accum, _mm512_loadu_ps(bias.as_ptr().add(out_c)));
            }
            for in_c in 0..in_len {
                let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                accum = _mm512_fmadd_ps(
                    vs,
                    _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i)),
                    accum,
                );
            }
            _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 16;
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
    }
}

/// [TA1] Implementação SIMD via AVX-512.
pub struct Avx512Math;

impl SimdMath for Avx512Math {
    type V = __m512;

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        dot_product_avx512(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        dot_product_bf16_fallback(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        dot_product_4x_interleaved_avx512(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        dot_product_4x_interleaved_bf16_fallback(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        dot_product_bf16_4x_fallback(w0, w1, w2, w3, in_frame)
    }

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        if out_frame.len() == 16 {
            fused_add_gemv_avx512_small(in_frame, weights, bias, out_frame, do_bias)
        } else {
            fused_add_gemv_fallback(in_frame, weights, bias, out_frame, do_bias)
        }
    }

    #[inline(always)]
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        fused_add_gemm_batch_fallback(in_frames, weights, bias, out_frames, num_frames, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        fused_gemm_residual_batch_fallback(
            in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
        )
    }

    #[inline(always)]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        if out_frame.len() == 16 {
            gemv_overwrite_avx512_small(in_frame, weights, bias, out_frame, do_bias)
        } else {
            gemv_overwrite_fallback(in_frame, weights, bias, out_frame, do_bias)
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        accumulate_head_fallback(dest, src)
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        tanh_and_accumulate_block_fallback(head_input, block)
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        gated_activation_and_accumulate_block_fallback(head_input, block, ch)
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        f32_to_bf16_fallback(src, dest)
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        crate::math::fastmath::tanh_slice_avx512(slice)
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        crate::math::fastmath::sigmoid_slice_avx512(slice)
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        horizontal_sum_avx512::<N>(ptr)
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        crate::math::fastmath::tanh_slice_avx512(buf)
    }
}

/// Implementação estática para AVX-512 com suporte a VNNI.
pub struct Avx512VnniMath;

impl SimdMath for Avx512VnniMath {
    type V = __m512;

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        Avx512Math::dot_product(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        Avx512Math::dot_product_bf16(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        Avx512Math::dot_product_4x_interleaved(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        Avx512Math::dot_product_4x_interleaved_bf16(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        Avx512Math::dot_product_bf16_4x(w0, w1, w2, w3, in_frame)
    }

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::fused_add_gemv(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::fused_add_gemm_batch(in_frames, weights, bias, out_frames, num_frames, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::fused_gemm_residual_batch(
            in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
        )
    }

    #[inline(always)]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite_bf16(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        Avx512Math::accumulate_head(dest, src)
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        Avx512Math::tanh_and_accumulate_block(head_input, block)
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        Avx512Math::gated_activation_and_accumulate_block(head_input, block, ch)
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        Avx512Math::f32_to_bf16(src, dest)
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        Avx512Math::tanh_slice(slice)
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        Avx512Math::sigmoid_slice(slice)
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        Avx512Math::horizontal_sum::<N>(ptr)
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        Avx512Math::activation_tanh_block(buf)
    }
}

/// Implementação estática para AVX-512 com suporte a VNNI e BF16.
pub struct Avx512VnniBf16Math;

impl SimdMath for Avx512VnniBf16Math {
    type V = __m512;
    const IS_BF16: bool = true;

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        dot_product_fallback(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        dot_product_bf16_avx512(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        dot_product_4x_interleaved_fallback(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        dot_product_4x_interleaved_bf16_fallback(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        dot_product_bf16_4x_fallback(w0, w1, w2, w3, in_frame)
    }

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::fused_add_gemv(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::fused_add_gemm_batch(in_frames, weights, bias, out_frames, num_frames, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::fused_gemm_residual_batch(
            in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
        )
    }

    #[inline(always)]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite_bf16(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        Avx512Math::accumulate_head(dest, src)
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        Avx512Math::tanh_and_accumulate_block(head_input, block)
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        Avx512Math::gated_activation_and_accumulate_block(head_input, block, ch)
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        Avx512Math::f32_to_bf16(src, dest)
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        Avx512Math::tanh_slice(slice)
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        Avx512Math::sigmoid_slice(slice)
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        Avx512Math::horizontal_sum::<N>(ptr)
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        Avx512Math::activation_tanh_block(buf)
    }
}

/// Soma horizontal de um buffer f32 de tamanho N (potência de 2) para AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn horizontal_sum_avx512<const N: usize>(ptr: *const f32) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();
    while i + 16 <= N {
        sum_v = _mm512_add_ps(sum_v, _mm512_loadu_ps(ptr.add(i)));
        i += 16;
    }
    let mut sum = super::utility::hsum_avx512(sum_v);
    while i < N {
        sum += *ptr.add(i);
        i += 1;
    }
    sum
}

/// Dot product f32 com pesos u16 usando AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_cvtph_ps(_mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i));
        sum_v = _mm512_fmadd_ps(va, vb, sum_v);
        i += 16;
    }
    let mut sum = super::utility::hsum_avx512(sum_v);
    while i < len {
        sum += *a.get_unchecked(i) * half::f16::from_bits(*b.get_unchecked(i)).to_f32();
        i += 1;
    }
    sum
}

/// Dot product interleaved 4x usando AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_interleaved_avx512(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    // Para não quebrar o build, vamos delegar para o fallback por enquanto se a implementação SIMD for muito longa.
    unsafe { dot_product_4x_interleaved_fallback(weights, state) }
}

/// Dot product BF16 usando AVX-512 BF16.
///
/// # Safety
/// A CPU deve suportar AVX-512 BF16.
#[target_feature(enable = "avx512bf16,avx512vl")]
pub unsafe fn dot_product_bf16_avx512(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();
    while i + 32 <= len {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
        sum_v = _mm512_dpbf16_ps(
            sum_v,
            core::mem::transmute::<__m512i, __m512bh>(va),
            core::mem::transmute::<__m512i, __m512bh>(vb),
        );
        i += 32;
    }
    let mut sum = super::utility::hsum_avx512(sum_v);
    while i < len {
        let fa = half::f16::from_bits(*a.get_unchecked(i)).to_f32();
        let fb = half::f16::from_bits(*b.get_unchecked(i)).to_f32();
        sum += fa * fb;
        i += 1;
    }
    sum
}

/// [T21] Kernel GEMV 4-gate AVX-512 para LSTM.
///
/// # Safety
/// A CPU deve suportar AVX-512.
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
    // Stub: Usar fallback por enquanto
    gemv_4gate_fallback(in_frame, w0, w1, w2, w3, bias, out, do_bias);
}
