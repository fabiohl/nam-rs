// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels de ativação Tanh (Tangente Hiperbólica) otimizados.

use crate::math::constants::*;
use core::arch::x86_64::*;

/// Aproximação vetorial de `tanh(x)` iterando um polinômio de grau 7 usando AVX2.
///
/// # Safety
/// O chamador deve garantir suporte a AVX2 e FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_avx2(x: __m256) -> __m256 {
    // Coeficientes do polinômio Minimax de grau 7
    let c0 = _mm256_set1_ps(TANH_C0);
    let c1 = _mm256_set1_ps(TANH_C1);
    let c2 = _mm256_set1_ps(TANH_C2);
    let one = _mm256_set1_ps(1.0);
    let min_limit = _mm256_set1_ps(-TANH_CLAMP_LIMIT);
    let max_limit = _mm256_set1_ps(TANH_CLAMP_LIMIT);

    // Clamp de saturação para evitar overflow no cálculo de p(x)^2
    let x = _mm256_max_ps(min_limit, _mm256_min_ps(max_limit, x));

    // x_sq = x * x
    let x_sq = _mm256_mul_ps(x, x);
    let x_sq_sq = _mm256_mul_ps(x_sq, x_sq);

    let y_3_5 = _mm256_fmadd_ps(c1, x_sq, c0);
    let y_3_5_7 = _mm256_fmadd_ps(c2, x_sq_sq, y_3_5);
    let y_full = _mm256_fmadd_ps(y_3_5_7, x_sq, one);

    // p(x) = x * y_full
    let p_x = _mm256_mul_ps(x, y_full);

    // Evaluando rsqrt(p(x)^2 + 1)
    let p_x_sq = _mm256_mul_ps(p_x, p_x);
    let radicand = _mm256_add_ps(p_x_sq, one);

    // Instrução HW nativa de rsqrt
    let mut rr = _mm256_rsqrt_ps(radicand);

    // Refinamento de Newton-Raphson duplo
    let three = _mm256_set1_ps(3.0);
    let half = _mm256_set1_ps(0.5);

    // 1ª iteração NR
    let rr_sq = _mm256_mul_ps(rr, rr);
    let diff = _mm256_fnmadd_ps(radicand, rr_sq, three);
    rr = _mm256_mul_ps(_mm256_mul_ps(rr, half), diff);

    // 2ª iteração NR
    let rr_sq = _mm256_mul_ps(rr, rr);
    let diff = _mm256_fnmadd_ps(radicand, rr_sq, three);
    rr = _mm256_mul_ps(_mm256_mul_ps(rr, half), diff);

    _mm256_mul_ps(p_x, rr)
}

/// Aproximação vetorial de `tanh(x)` (Dual, 16 floats).
///
/// # Safety
/// O chamador deve garantir suporte a AVX2 e FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let c0 = _mm256_set1_ps(TANH_C0);
    let c1 = _mm256_set1_ps(TANH_C1);
    let c2 = _mm256_set1_ps(TANH_C2);
    let one = _mm256_set1_ps(1.0);
    let min_limit = _mm256_set1_ps(-TANH_CLAMP_LIMIT);
    let max_limit = _mm256_set1_ps(TANH_CLAMP_LIMIT);

    let x1 = _mm256_max_ps(min_limit, _mm256_min_ps(max_limit, x1));
    let x2 = _mm256_max_ps(min_limit, _mm256_min_ps(max_limit, x2));

    let x_sq1 = _mm256_mul_ps(x1, x1);
    let x_sq2 = _mm256_mul_ps(x2, x2);
    let x_sq_sq1 = _mm256_mul_ps(x_sq1, x_sq1);
    let x_sq_sq2 = _mm256_mul_ps(x_sq2, x_sq2);

    let y_3_5_1 = _mm256_fmadd_ps(c1, x_sq1, c0);
    let y_3_5_2 = _mm256_fmadd_ps(c1, x_sq2, c0);
    let y_3_5_7_1 = _mm256_fmadd_ps(c2, x_sq_sq1, y_3_5_1);
    let y_3_5_7_2 = _mm256_fmadd_ps(c2, x_sq_sq2, y_3_5_2);
    let y_full1 = _mm256_fmadd_ps(y_3_5_7_1, x_sq1, one);
    let y_full2 = _mm256_fmadd_ps(y_3_5_7_2, x_sq2, one);

    let p_x1 = _mm256_mul_ps(x1, y_full1);
    let p_x2 = _mm256_mul_ps(x2, y_full2);

    let p_x_sq1 = _mm256_mul_ps(p_x1, p_x1);
    let p_x_sq2 = _mm256_mul_ps(p_x2, p_x2);
    let radicand1 = _mm256_add_ps(p_x_sq1, one);
    let radicand2 = _mm256_add_ps(p_x_sq2, one);

    let mut rr1 = _mm256_rsqrt_ps(radicand1);
    let mut rr2 = _mm256_rsqrt_ps(radicand2);

    let three = _mm256_set1_ps(3.0);
    let half = _mm256_set1_ps(0.5);

    // 1ª iteração NR
    let rr_sq1 = _mm256_mul_ps(rr1, rr1);
    let rr_sq2 = _mm256_mul_ps(rr2, rr2);
    let diff1 = _mm256_fnmadd_ps(radicand1, rr_sq1, three);
    let diff2 = _mm256_fnmadd_ps(radicand2, rr_sq2, three);
    rr1 = _mm256_mul_ps(_mm256_mul_ps(rr1, half), diff1);
    rr2 = _mm256_mul_ps(_mm256_mul_ps(rr2, half), diff2);

    // 2ª iteração NR
    let rr_sq1 = _mm256_mul_ps(rr1, rr1);
    let rr_sq2 = _mm256_mul_ps(rr2, rr2);
    let diff1 = _mm256_fnmadd_ps(radicand1, rr_sq1, three);
    let diff2 = _mm256_fnmadd_ps(radicand2, rr_sq2, three);
    rr1 = _mm256_mul_ps(_mm256_mul_ps(rr1, half), diff1);
    rr2 = _mm256_mul_ps(_mm256_mul_ps(rr2, half), diff2);

    (_mm256_mul_ps(p_x1, rr1), _mm256_mul_ps(p_x2, rr2))
}

/// Aproximação vetorial de `tanh(x)` usando AVX-512.
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX-512 (F e VL).
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_avx512(x: __m512) -> __m512 {
    // Coeficientes do polinômio Minimax de grau 7
    let c0 = _mm512_set1_ps(TANH_C0);
    let c1 = _mm512_set1_ps(TANH_C1);
    let c2 = _mm512_set1_ps(TANH_C2);
    let one = _mm512_set1_ps(1.0);
    let min_limit = _mm512_set1_ps(-TANH_CLAMP_LIMIT);
    let max_limit = _mm512_set1_ps(TANH_CLAMP_LIMIT);

    // Clamp de saturação
    let x = _mm512_max_ps(min_limit, _mm512_min_ps(max_limit, x));

    // x_sq = x * x
    let x_sq = _mm512_mul_ps(x, x);
    let x_sq_sq = _mm512_mul_ps(x_sq, x_sq);

    let y_3_5 = _mm512_fmadd_ps(c1, x_sq, c0);
    let y_3_5_7 = _mm512_fmadd_ps(c2, x_sq_sq, y_3_5);
    let y_full = _mm512_fmadd_ps(y_3_5_7, x_sq, one);

    let p_x = _mm512_mul_ps(x, y_full);

    let p_x_sq = _mm512_mul_ps(p_x, p_x);
    let radicand = _mm512_add_ps(p_x_sq, one);

    let mut rr = _mm512_rsqrt14_ps(radicand);

    let three = _mm512_set1_ps(3.0);
    let half = _mm512_set1_ps(0.5);

    // 1ª iteração NR
    let rr_sq = _mm512_mul_ps(rr, rr);
    let diff = _mm512_fnmadd_ps(radicand, rr_sq, three);
    rr = _mm512_mul_ps(_mm512_mul_ps(rr, half), diff);

    // 2ª iteração NR
    let rr_sq = _mm512_mul_ps(rr, rr);
    let diff = _mm512_fnmadd_ps(radicand, rr_sq, three);
    rr = _mm512_mul_ps(_mm512_mul_ps(rr, half), diff);

    _mm512_mul_ps(p_x, rr)
}

/// Aplica a ativação Tanh a um slice de f32 usando otimização AVX2.
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let (y1, y2) = simd_tanh_dual_avx2(x1, x2);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
        }
        i += 16;
    }

    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            let y = simd_tanh_avx2(x);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 8;
    }

    for item in slice.iter_mut().skip(i) {
        *item = item.tanh();
    }
}

/// Aplica a ativação Tanh a um slice de f32 usando otimização AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn tanh_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_tanh_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 16;
    }

    for item in slice.iter_mut().skip(i) {
        *item = item.tanh();
    }
}

/// Versão escalar de `tanh`.
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}
