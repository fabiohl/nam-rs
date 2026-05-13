// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels de ativação Sigmoid (Logística) otimizados.

use crate::math::constants::*;
use core::arch::x86_64::*;

/// Aproximação direta de `sigmoid(x) = 1 / (1 + exp(-x))` usando AVX2.
///
/// Utiliza um polinômio de Minimax de grau 6 para `exp(x)` e dois passos de Newton-Raphson
/// para o recíproco (`_mm256_rcp_ps`), garantindo erro máximo < 6e-8 (saturação f32).
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX2 e FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid_avx2(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();

    // neg_x = -x
    let neg_x = _mm256_sub_ps(zero, x);

    // Clamp para evitar overflow/underflow extremo no exp e manter precisão do polinômio
    let neg_x = _mm256_max_ps(
        _mm256_set1_ps(-SIGMOID_CLAMP_LIMIT),
        _mm256_min_ps(_mm256_set1_ps(SIGMOID_CLAMP_LIMIT), neg_x),
    );

    // --- Fast Exp AVX2 (Degree 6) ---
    let log2e = _mm256_set1_ps(EXP_LOG2E);
    let ln2_hi = _mm256_set1_ps(EXP_LN2_HI);
    let ln2_lo = _mm256_set1_ps(EXP_LN2_LO);

    let k = _mm256_cvtps_epi32(_mm256_fmadd_ps(neg_x, log2e, _mm256_set1_ps(0.0)));
    let k_f = _mm256_cvtepi32_ps(k);

    let mut f = _mm256_fmadd_ps(k_f, ln2_hi, neg_x);
    f = _mm256_fmadd_ps(k_f, ln2_lo, f);

    // Polinômio Minimax D6 para exp(f) em [-0.5 ln 2, 0.5 ln 2]
    let c6 = _mm256_set1_ps(EXP_C6);
    let c5 = _mm256_set1_ps(EXP_C5);
    let c4 = _mm256_set1_ps(EXP_C4);
    let c3 = _mm256_set1_ps(EXP_C3);
    let c2 = _mm256_set1_ps(EXP_C2);

    let mut poly = _mm256_fmadd_ps(f, c6, c5);
    poly = _mm256_fmadd_ps(poly, f, c4);
    poly = _mm256_fmadd_ps(poly, f, c3);
    poly = _mm256_fmadd_ps(poly, f, c2);
    poly = _mm256_fmadd_ps(poly, f, one);
    poly = _mm256_fmadd_ps(poly, f, one);

    let k_int = _mm256_add_epi32(k, _mm256_set1_epi32(127));
    let twok = _mm256_castsi256_ps(_mm256_slli_epi32(k_int, 23));
    let e = _mm256_mul_ps(poly, twok);
    // ------------------------------

    let den = _mm256_add_ps(one, e);
    let mut res = _mm256_rcp_ps(den);

    // Refinamento de Newton-Raphson duplo: satura precisão f32 (24 bits)
    let two = _mm256_set1_ps(2.0);
    // 1ª iteração NR: ~23 bits
    res = _mm256_mul_ps(res, _mm256_fnmadd_ps(den, res, two));
    // 2ª iteração NR: satura mantissa f32
    res = _mm256_mul_ps(res, _mm256_fnmadd_ps(den, res, two));

    res
}

/// Aproximação direta de `sigmoid(x)` (Dual, 16 floats).
/// Intercala instruções para otimizar Instruction Level Parallelism (Latency Hiding).
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX2 e FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();

    let neg_x1 = _mm256_sub_ps(zero, x1);
    let neg_x2 = _mm256_sub_ps(zero, x2);

    let clamp_limit = _mm256_set1_ps(SIGMOID_CLAMP_LIMIT);
    let clamp_min = _mm256_set1_ps(-SIGMOID_CLAMP_LIMIT);
    let neg_x1 = _mm256_max_ps(clamp_min, _mm256_min_ps(clamp_limit, neg_x1));
    let neg_x2 = _mm256_max_ps(clamp_min, _mm256_min_ps(clamp_limit, neg_x2));

    let log2e = _mm256_set1_ps(EXP_LOG2E);
    let ln2_hi = _mm256_set1_ps(EXP_LN2_HI);
    let ln2_lo = _mm256_set1_ps(EXP_LN2_LO);

    let k1 = _mm256_cvtps_epi32(_mm256_fmadd_ps(neg_x1, log2e, zero));
    let k2 = _mm256_cvtps_epi32(_mm256_fmadd_ps(neg_x2, log2e, zero));
    let k_f1 = _mm256_cvtepi32_ps(k1);
    let k_f2 = _mm256_cvtepi32_ps(k2);

    let mut f1 = _mm256_fmadd_ps(k_f1, ln2_hi, neg_x1);
    let mut f2 = _mm256_fmadd_ps(k_f2, ln2_hi, neg_x2);
    f1 = _mm256_fmadd_ps(k_f1, ln2_lo, f1);
    f2 = _mm256_fmadd_ps(k_f2, ln2_lo, f2);

    let c6 = _mm256_set1_ps(EXP_C6);
    let c5 = _mm256_set1_ps(EXP_C5);
    let c4 = _mm256_set1_ps(EXP_C4);
    let c3 = _mm256_set1_ps(EXP_C3);
    let c2 = _mm256_set1_ps(EXP_C2);

    let mut poly1 = _mm256_fmadd_ps(f1, c6, c5);
    let mut poly2 = _mm256_fmadd_ps(f2, c6, c5);
    poly1 = _mm256_fmadd_ps(poly1, f1, c4);
    poly2 = _mm256_fmadd_ps(poly2, f2, c4);
    poly1 = _mm256_fmadd_ps(poly1, f1, c3);
    poly2 = _mm256_fmadd_ps(poly2, f2, c3);
    poly1 = _mm256_fmadd_ps(poly1, f1, c2);
    poly2 = _mm256_fmadd_ps(poly2, f2, c2);
    poly1 = _mm256_fmadd_ps(poly1, f1, one);
    poly2 = _mm256_fmadd_ps(poly2, f2, one);
    poly1 = _mm256_fmadd_ps(poly1, f1, one);
    poly2 = _mm256_fmadd_ps(poly2, f2, one);

    let bias = _mm256_set1_epi32(127);
    let k_int1 = _mm256_add_epi32(k1, bias);
    let k_int2 = _mm256_add_epi32(k2, bias);
    let twok1 = _mm256_castsi256_ps(_mm256_slli_epi32(k_int1, 23));
    let twok2 = _mm256_castsi256_ps(_mm256_slli_epi32(k_int2, 23));
    let e1 = _mm256_mul_ps(poly1, twok1);
    let e2 = _mm256_mul_ps(poly2, twok2);

    let den1 = _mm256_add_ps(one, e1);
    let den2 = _mm256_add_ps(one, e2);
    let mut res1 = _mm256_rcp_ps(den1);
    let mut res2 = _mm256_rcp_ps(den2);

    let two = _mm256_set1_ps(2.0);
    // 1ª NR
    res1 = _mm256_mul_ps(res1, _mm256_fnmadd_ps(den1, res1, two));
    res2 = _mm256_mul_ps(res2, _mm256_fnmadd_ps(den2, res2, two));
    // 2ª NR: satura f32
    res1 = _mm256_mul_ps(res1, _mm256_fnmadd_ps(den1, res1, two));
    res2 = _mm256_mul_ps(res2, _mm256_fnmadd_ps(den2, res2, two));

    (res1, res2)
}

/// Aproximação direta de `sigmoid(x)` usando AVX-512.
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX-512 (F e VL).
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_sigmoid_avx512(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let zero = _mm512_setzero_ps();
    let neg_x = _mm512_sub_ps(zero, x);
    let neg_x = _mm512_max_ps(
        _mm512_set1_ps(-SIGMOID_CLAMP_LIMIT),
        _mm512_min_ps(_mm512_set1_ps(SIGMOID_CLAMP_LIMIT), neg_x),
    );

    // --- Fast Exp AVX-512 ---
    let log2e = _mm512_set1_ps(EXP_LOG2E);
    let ln2_hi = _mm512_set1_ps(EXP_LN2_HI);
    let ln2_lo = _mm512_set1_ps(EXP_LN2_LO);

    let k = _mm512_cvtps_epi32(_mm512_mul_ps(neg_x, log2e));
    let k_f = _mm512_cvtepi32_ps(k);
    let mut f = _mm512_fmadd_ps(k_f, ln2_hi, neg_x);
    f = _mm512_fmadd_ps(k_f, ln2_lo, f);

    let c6 = _mm512_set1_ps(EXP_C6);
    let c5 = _mm512_set1_ps(EXP_C5);
    let c4 = _mm512_set1_ps(EXP_C4);
    let c3 = _mm512_set1_ps(EXP_C3);
    let c2 = _mm512_set1_ps(EXP_C2);

    let mut poly = _mm512_fmadd_ps(f, c6, c5);
    poly = _mm512_fmadd_ps(poly, f, c4);
    poly = _mm512_fmadd_ps(poly, f, c3);
    poly = _mm512_fmadd_ps(poly, f, c2);
    poly = _mm512_fmadd_ps(poly, f, one);
    poly = _mm512_fmadd_ps(poly, f, one);

    let k_int = _mm512_add_epi32(k, _mm512_set1_epi32(127));
    let twok = _mm512_castsi512_ps(_mm512_slli_epi32(k_int, 23));
    let e = _mm512_mul_ps(poly, twok);
    // ------------------------

    let den = _mm512_add_ps(one, e);
    let mut res = _mm512_rcp14_ps(den);

    let two = _mm512_set1_ps(2.0);
    // NR duplo: satura f32
    res = _mm512_mul_ps(res, _mm512_fnmadd_ps(den, res, two));
    res = _mm512_mul_ps(res, _mm512_fnmadd_ps(den, res, two));

    res
}

/// Aplica a ativação Sigmoid a um slice de f32 usando otimização AVX2.
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sigmoid_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let (y1, y2) = simd_sigmoid_dual_avx2(x1, x2);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
        }
        i += 16;
    }

    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            let y = simd_sigmoid_avx2(x);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 8;
    }

    for item in slice.iter_mut().skip(i) {
        *item = 1.0 / (1.0 + (-*item).exp());
    }
}

/// Aplica a ativação Sigmoid a um slice de f32 usando otimização AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn sigmoid_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_sigmoid_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 16;
    }

    for item in slice.iter_mut().skip(i) {
        *item = 1.0 / (1.0 + (-*item).exp());
    }
}

/// Versão escalar de `sigmoid` (1 / (1 + exp(-x))).
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
