// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized LeakyReLU(0.01) activation kernel for A2 Wavenet.
//!
//! Fixed negative slope of 0.01 as used in the A2 architecture
//! (`LeakyReLU(0.01)` on all 23 layers).
//!
//! Source of truth: `NAM/activations.h` (LeakyReLU) and
//! `NAM/wavenet/a2_fast.cpp:49-51`.

use core::arch::x86_64::*;

/// A2 canonical LeakyReLU slope.
const LEAKY_SLOPE: f32 = 0.01;

/// Branchless LeakyReLU for a single `__m256` register using blend.
///
/// `y = x > 0 ? x : x * alpha`
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn simd_leaky_relu_avx2(x: __m256, alpha: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
    let neg_part = _mm256_mul_ps(alpha, x);
    _mm256_blendv_ps(neg_part, x, mask)
}

/// Branchless LeakyReLU for a single `__m512` register.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_leaky_relu_avx512(x: __m512, alpha: __m512) -> __m512 {
    let zero = _mm512_setzero_ps();
    let mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OQ);
    let neg_part = _mm512_mul_ps(alpha, x);
    _mm512_mask_blend_ps(mask, neg_part, x)
}

/// Applies LeakyReLU(0.01) activation to a slice of f32 using AVX2.
///
/// Dual `__m256` pipeline for throughput; scalar remainder for `len % 8`.
///
/// # Safety
/// Requires AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn leaky_relu_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();
    let alpha = _mm256_set1_ps(LEAKY_SLOPE);

    while i + 16 <= len {
        unsafe {
            let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_leaky_relu_avx2(x1, alpha));
            _mm256_storeu_ps(
                slice.as_mut_ptr().add(i + 8),
                simd_leaky_relu_avx2(x2, alpha),
            );
        }
        i += 16;
    }

    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), simd_leaky_relu_avx2(x, alpha));
        }
        i += 8;
    }

    for item in slice.iter_mut().skip(i) {
        if *item < 0.0 {
            *item *= LEAKY_SLOPE;
        }
    }
}

/// Applies LeakyReLU(0.01) activation to a slice of f32 using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn leaky_relu_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();
    let alpha = _mm512_set1_ps(LEAKY_SLOPE);

    while i + 16 <= len {
        unsafe {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), simd_leaky_relu_avx512(x, alpha));
        }
        i += 16;
    }

    for item in slice.iter_mut().skip(i) {
        if *item < 0.0 {
            *item *= LEAKY_SLOPE;
        }
    }
}

/// Scalar LeakyReLU.
#[inline(always)]
pub fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { x * LEAKY_SLOPE }
}
