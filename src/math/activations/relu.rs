// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de ativação ReLU (Rectified Linear Unit) otimizados.

use core::arch::x86_64::*;

/// Aproximação vetorial de `ReLU(x) = max(0, x)` usando AVX2.
///
/// # Safety
/// Requer suporte a AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn simd_relu_avx2(x: __m256) -> __m256 {
    _mm256_max_ps(_mm256_setzero_ps(), x)
}

/// Aproximação vetorial de `ReLU(x)` (Dual, 16 floats).
///
/// # Safety
/// Requer suporte a AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn simd_relu_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let zero = _mm256_setzero_ps();
    (_mm256_max_ps(zero, x1), _mm256_max_ps(zero, x2))
}

/// Aproximação vetorial de `ReLU(x) = max(0, x)` usando AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_relu_avx512(x: __m512) -> __m512 {
    _mm512_max_ps(_mm512_setzero_ps(), x)
}

/// Aplica a ativação ReLU a um slice de f32 usando otimização AVX2.
///
/// # Safety
/// Requer suporte a AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn relu_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();
    let zero = _mm256_setzero_ps();

    while i + 16 <= len {
        unsafe {
            let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), _mm256_max_ps(zero, x1));
            _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), _mm256_max_ps(zero, x2));
        }
        i += 16;
    }

    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), _mm256_max_ps(zero, x));
        }
        i += 8;
    }

    for item in slice.iter_mut().skip(i) {
        if *item < 0.0 {
            *item = 0.0;
        }
    }
}

/// Aplica a ativação ReLU a um slice de f32 usando otimização AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn relu_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();
    let zero = _mm512_setzero_ps();

    while i + 16 <= len {
        unsafe {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), _mm512_max_ps(zero, x));
        }
        i += 16;
    }

    for item in slice.iter_mut().skip(i) {
        if *item < 0.0 {
            *item = 0.0;
        }
    }
}

/// Versão escalar de `relu` (max(0, x)).
#[inline(always)]
pub fn relu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}
