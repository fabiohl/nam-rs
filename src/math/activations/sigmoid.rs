// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Sigmoid (Logistic) activation kernels.
//!
//! Reuses the exact identity `sigmoid(x) = 0.5 + 0.5 · tanh(x/2)`,
//! delegating to the Padé [5,4] tanh kernel for zero branches and ~6 FMAs.

use super::tanh::{simd_tanh_avx2, simd_tanh_avx512, simd_tanh_dual_avx2};
use core::arch::x86_64::*;

/// Branchless approximation of `sigmoid(x)` via tanh identity (AVX2).
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid_avx2(x: __m256) -> __m256 {
    let half = _mm256_set1_ps(0.5);
    let x_half = _mm256_mul_ps(x, half);
    let t = unsafe { simd_tanh_avx2(x_half) };
    _mm256_fmadd_ps(half, t, half)
}

/// Branchless approximation of `sigmoid(x)` (Dual, 16 floats) via tanh identity.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
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

/// Branchless approximation of `sigmoid(x)` via tanh identity (AVX-512).
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_sigmoid_avx512(x: __m512) -> __m512 {
    let half = _mm512_set1_ps(0.5);
    let x_half = _mm512_mul_ps(x, half);
    let t = unsafe { simd_tanh_avx512(x_half) };
    _mm512_fmadd_ps(half, t, half)
}

/// Applies Sigmoid activation to a slice of f32 using AVX2 optimization.
///
/// # Safety
/// Requires AVX2 and FMA support.
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

/// Applies Sigmoid activation to a slice of f32 using AVX-512 optimization.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
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

/// Scalar version of `sigmoid` (1 / (1 + exp(-x))).
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
