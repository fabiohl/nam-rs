// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de ativação Sigmoid (Logística) otimizados.
//!
//! Reutiliza a identidade exata `sigmoid(x) = 0.5 + 0.5 · tanh(x/2)`,
//! delegando ao kernel Padé [5,4] de tanh para zero branches e ~6 FMAs.

use super::tanh::{simd_tanh_avx2, simd_tanh_avx512, simd_tanh_dual_avx2};
use core::arch::x86_64::*;

/// Aproximação branchless de `sigmoid(x)` via identidade tanh (AVX2).
///
/// # Safety
/// O chamador deve garantir suporte a AVX2 e FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid_avx2(x: __m256) -> __m256 {
    let half = _mm256_set1_ps(0.5);
    let x_half = _mm256_mul_ps(x, half);
    let t = unsafe { simd_tanh_avx2(x_half) };
    _mm256_fmadd_ps(half, t, half)
}

/// Aproximação branchless de `sigmoid(x)` (Dual, 16 floats) via identidade tanh.
///
/// # Safety
/// O chamador deve garantir suporte a AVX2 e FMA.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let half = _mm256_set1_ps(0.5);
    let x_half1 = _mm256_mul_ps(x1, half);
    let x_half2 = _mm256_mul_ps(x2, half);
    let (t1, t2) = unsafe { simd_tanh_dual_avx2(x_half1, x_half2) };
    (
        _mm256_fmadd_ps(half, t1, half),
        _mm256_fmadd_ps(half, t2, half),
    )
}

/// Aproximação branchless de `sigmoid(x)` via identidade tanh (AVX-512).
///
/// # Safety
/// O chamador deve garantir suporte a AVX-512F e AVX-512VL.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_sigmoid_avx512(x: __m512) -> __m512 {
    let half = _mm512_set1_ps(0.5);
    let x_half = _mm512_mul_ps(x, half);
    let t = unsafe { simd_tanh_avx512(x_half) };
    _mm512_fmadd_ps(half, t, half)
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
