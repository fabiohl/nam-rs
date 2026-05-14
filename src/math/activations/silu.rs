// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de ativação SiLU (Sigmoid Linear Unit / Swish) otimizados.

use super::sigmoid::{simd_sigmoid_avx2, simd_sigmoid_avx512, simd_sigmoid_dual_avx2};
use core::arch::x86_64::*;

/// Aproximação vetorial de `SiLU(x) = x * sigmoid(x)` usando AVX2.
///
/// Reutiliza o kernel `simd_sigmoid_avx2` (Minimax D6).
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_silu_avx2(x: __m256) -> __m256 {
    unsafe {
        let s = simd_sigmoid_avx2(x);
        _mm256_mul_ps(x, s)
    }
}

/// Aproximação vetorial de `SiLU(x)` (Dual, 16 floats).
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_silu_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    unsafe {
        let (s1, s2) = simd_sigmoid_dual_avx2(x1, x2);
        (_mm256_mul_ps(x1, s1), _mm256_mul_ps(x2, s2))
    }
}

/// Aproximação vetorial de `SiLU(x) = x * sigmoid(x)` usando AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_silu_avx512(x: __m512) -> __m512 {
    unsafe {
        let s = simd_sigmoid_avx512(x);
        _mm512_mul_ps(x, s)
    }
}

/// Aplica a ativação SiLU a um slice de f32 usando otimização AVX2.
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn silu_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        while i + 16 <= len {
            let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let (y1, y2) = simd_silu_dual_avx2(x1, x2);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
            i += 16;
        }

        while i + 8 <= len {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            let y = simd_silu_avx2(x);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
            i += 8;
        }
    }

    for item in slice.iter_mut().skip(i) {
        *item = *item / (1.0 + (-*item).exp());
    }
}

/// Aplica a ativação SiLU a um slice de f32 usando otimização AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn silu_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        while i + 16 <= len {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_silu_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
            i += 16;
        }
    }

    for item in slice.iter_mut().skip(i) {
        *item = *item / (1.0 + (-*item).exp());
    }
}

/// Versão escalar de `silu` (x * sigmoid(x)).
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
