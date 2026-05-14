// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de ativação Softsign otimizados.

use core::arch::x86_64::*;

/// Aproximação vetorial de `Softsign(x) = x / (1 + |x|)` usando AVX2.
///
/// Utiliza `_mm256_rcp_ps` com uma iteração de Newton-Raphson para precisão de ~24 bits.
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_softsign_avx2(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    // abs_x = x & 0x7FFFFFFF
    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let den = _mm256_add_ps(one, abs_x);

    // Recíproco com Newton-Raphson duplo (satura f32)
    let mut res = _mm256_rcp_ps(den);
    res = _mm256_mul_ps(res, _mm256_fnmadd_ps(den, res, two));
    res = _mm256_mul_ps(res, _mm256_fnmadd_ps(den, res, two));

    _mm256_mul_ps(x, res)
}

/// Aproximação vetorial de `Softsign(x)` (Dual, 16 floats).
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_softsign_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let zero_minus = _mm256_set1_ps(-0.0);

    let abs_x1 = _mm256_andnot_ps(zero_minus, x1);
    let abs_x2 = _mm256_andnot_ps(zero_minus, x2);
    let den1 = _mm256_add_ps(one, abs_x1);
    let den2 = _mm256_add_ps(one, abs_x2);

    let mut res1 = _mm256_rcp_ps(den1);
    let mut res2 = _mm256_rcp_ps(den2);

    // 1ª iteração NR
    res1 = _mm256_mul_ps(res1, _mm256_fnmadd_ps(den1, res1, two));
    res2 = _mm256_mul_ps(res2, _mm256_fnmadd_ps(den2, res2, two));
    // 2ª iteração NR: satura mantissa f32
    res1 = _mm256_mul_ps(res1, _mm256_fnmadd_ps(den1, res1, two));
    res2 = _mm256_mul_ps(res2, _mm256_fnmadd_ps(den2, res2, two));

    (_mm256_mul_ps(x1, res1), _mm256_mul_ps(x2, res2))
}

/// Aproximação vetorial de `Softsign(x) = x / (1 + |x|)` usando AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F, AVX-512VL e AVX-512DQ.
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn simd_softsign_avx512(x: __m512) -> __m512 {
    let one = _mm512_set1_ps(1.0);
    let two = _mm512_set1_ps(2.0);

    // _mm512_andnot_ps requer AVX-512DQ. Com a feature habilitada no contexto, a chamada é segura.
    let abs_x = _mm512_andnot_ps(_mm512_set1_ps(-0.0), x);
    let den = _mm512_add_ps(one, abs_x);

    // Recíproco com Newton-Raphson duplo (satura f32)
    let mut res = _mm512_rcp14_ps(den);
    res = _mm512_mul_ps(res, _mm512_fnmadd_ps(den, res, two));
    res = _mm512_mul_ps(res, _mm512_fnmadd_ps(den, res, two));

    _mm512_mul_ps(x, res)
}

/// Aplica a ativação Softsign a um slice de f32 usando otimização AVX2.
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn softsign_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let (y1, y2) = simd_softsign_dual_avx2(x1, x2);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
        }
        i += 16;
    }

    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            let y = simd_softsign_avx2(x);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 8;
    }

    for item in slice.iter_mut().skip(i) {
        *item /= 1.0 + item.abs();
    }
}

/// Aplica a ativação Softsign a um slice de f32 usando otimização AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F, AVX-512VL e AVX-512DQ.
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn softsign_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_softsign_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 16;
    }

    for item in slice.iter_mut().skip(i) {
        *item /= 1.0 + item.abs();
    }
}

/// Versão escalar de `softsign` (x / (1 + |x|)).
#[inline(always)]
pub fn softsign(x: f32) -> f32 {
    x / (1.0 + x.abs())
}
