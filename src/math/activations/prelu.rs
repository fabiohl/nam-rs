// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels de ativação PReLU (Parametric ReLU) otimizados.

use core::arch::x86_64::*;

/// Aproximação vetorial de `PReLU(x) = x > 0 ? x : alpha * x` usando AVX2.
///
/// # Safety
/// Requer suporte a AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn simd_prelu_avx2(x: __m256, alpha: __m256) -> __m256 {
    // Máscara de valores positivos (x > 0)
    let mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ);
    // alpha * x para a região negativa
    let neg_part = _mm256_mul_ps(alpha, x);
    // Seleciona x se mask for true, senão neg_part
    _mm256_blendv_ps(neg_part, x, mask)
}

/// Aproximação vetorial de `PReLU(x) = x > 0 ? x : alpha * x` usando AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_prelu_avx512(x: __m512, alpha: __m512) -> __m512 {
    let zero = _mm512_setzero_ps();
    // Máscara de valores positivos (x > 0)
    let mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
    // alpha * x para a região negativa
    let neg_part = _mm512_mul_ps(alpha, x);
    // Seleciona x se mask for true, senão neg_part (usando masking do AVX-512)
    _mm512_mask_blend_ps(mask, neg_part, x)
}

/// Aplica a ativação PReLU a um slice de f32 usando otimização AVX2.
///
/// # Safety
/// Requer suporte a AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn prelu_slice_avx2(slice: &mut [f32], slopes: &[f32]) {
    let mut i = 0;
    let len = slice.len();
    let s_len = slopes.len();

    if s_len == 0 {
        return;
    }

    // Caso otimizado: slope único (LeakyReLU)
    if s_len == 1 {
        let alpha = _mm256_set1_ps(slopes[0]);
        while i + 16 <= len {
            unsafe {
                let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx2(x1, alpha));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), simd_prelu_avx2(x2, alpha));
            }
            i += 16;
        }
        while i + 8 <= len {
            unsafe {
                let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx2(x, alpha));
            }
            i += 8;
        }
    } else if s_len == len {
        // Caso otimizado: slopes por elemento
        while i + 16 <= len {
            unsafe {
                let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
                let a1 = _mm256_loadu_ps(slopes.as_ptr().add(i));
                let a2 = _mm256_loadu_ps(slopes.as_ptr().add(i + 8));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx2(x1, a1));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), simd_prelu_avx2(x2, a2));
            }
            i += 16;
        }
        while i + 8 <= len {
            unsafe {
                let x = _mm256_loadu_ps(slice.as_ptr().add(i));
                let a = _mm256_loadu_ps(slopes.as_ptr().add(i));
                _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx2(x, a));
            }
            i += 8;
        }
    }

    // Fallback escalar (também lida com o resto dos casos otimizados e com o ciclismo)
    for idx in i..len {
        let x = slice[idx];
        if x < 0.0 {
            slice[idx] = x * slopes[idx % s_len];
        }
    }
}

/// Aplica a ativação PReLU a um slice de f32 usando otimização AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn prelu_slice_avx512(slice: &mut [f32], slopes: &[f32]) {
    let mut i = 0;
    let len = slice.len();
    let s_len = slopes.len();

    if s_len == 0 {
        return;
    }

    if s_len == 1 {
        let alpha = _mm512_set1_ps(slopes[0]);
        while i + 16 <= len {
            unsafe {
                let x = _mm512_loadu_ps(slice.as_ptr().add(i));
                _mm512_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx512(x, alpha));
            }
            i += 16;
        }
    } else if s_len == len {
        while i + 16 <= len {
            unsafe {
                let x = _mm512_loadu_ps(slice.as_ptr().add(i));
                let a = _mm512_loadu_ps(slopes.as_ptr().add(i));
                _mm512_storeu_ps(slice.as_mut_ptr().add(i), simd_prelu_avx512(x, a));
            }
            i += 16;
        }
    }

    for idx in i..len {
        let x = slice[idx];
        if x < 0.0 {
            slice[idx] = x * slopes[idx % s_len];
        }
    }
}

/// Versão escalar de `prelu`.
#[inline(always)]
pub fn prelu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { x * alpha }
}
