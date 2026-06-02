// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Tanh (Hyperbolic Tangent) activation kernels.
//!
//! Uses the branchless Padé [5,4] rational approximant:
//! `tanh(x) ≈ x · (x⁴ + 105·x² + 945) / (15·x⁴ + 420·x² + 945)`
//! with input clamped to `|x| < 4` and output saturated to `[-1, 1]`.
//! Reference: VDT library (CERN), Mineiro & Vorlicek (2016).

use crate::math::constants::*;
use core::arch::x86_64::*;

/// Branchless vector approximation of `tanh(x)` using Padé [5,4] (AVX2).
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A);
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B);
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4);
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2);
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A);
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));
    let x_sq = _mm256_mul_ps(x, x);
    let x_sq_sq = _mm256_mul_ps(x_sq, x_sq);

    // num = x * (x_sq_sq + 105*x_sq + 945)
    let num = _mm256_mul_ps(
        x,
        _mm256_add_ps(_mm256_fmadd_ps(num_a, x_sq, x_sq_sq), num_b),
    );

    // den = 15*x_sq_sq + 420*x_sq + 945
    let den = _mm256_fmadd_ps(den_c4, x_sq_sq, _mm256_fmadd_ps(den_c2, x_sq, den_a));

    let mut rden = _mm256_rcp_ps(den);
    rden = _mm256_mul_ps(rden, _mm256_fnmadd_ps(den, rden, two));

    let result = _mm256_mul_ps(num, rden);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, result))
}

/// Vector approximation of `tanh(x)` (Dual, 16 floats) using Padé [5,4].
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A);
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B);
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4);
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2);
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A);
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x1 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x1));
    let x2 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x2));

    let x_sq1 = _mm256_mul_ps(x1, x1);
    let x_sq2 = _mm256_mul_ps(x2, x2);
    let x_sq_sq1 = _mm256_mul_ps(x_sq1, x_sq1);
    let x_sq_sq2 = _mm256_mul_ps(x_sq2, x_sq2);

    let num1 = _mm256_mul_ps(
        x1,
        _mm256_add_ps(_mm256_fmadd_ps(num_a, x_sq1, x_sq_sq1), num_b),
    );
    let num2 = _mm256_mul_ps(
        x2,
        _mm256_add_ps(_mm256_fmadd_ps(num_a, x_sq2, x_sq_sq2), num_b),
    );

    let den1 = _mm256_fmadd_ps(den_c4, x_sq_sq1, _mm256_fmadd_ps(den_c2, x_sq1, den_a));
    let den2 = _mm256_fmadd_ps(den_c4, x_sq_sq2, _mm256_fmadd_ps(den_c2, x_sq2, den_a));

    let mut rden1 = _mm256_rcp_ps(den1);
    let mut rden2 = _mm256_rcp_ps(den2);

    rden1 = _mm256_mul_ps(rden1, _mm256_fnmadd_ps(den1, rden1, two));
    rden2 = _mm256_mul_ps(rden2, _mm256_fnmadd_ps(den2, rden2, two));

    let r1 = _mm256_mul_ps(num1, rden1);
    let r2 = _mm256_mul_ps(num2, rden2);
    (
        _mm256_max_ps(neg_one, _mm256_min_ps(one, r1)),
        _mm256_max_ps(neg_one, _mm256_min_ps(one, r2)),
    )
}

/// Branchless vector approximation of `tanh(x)` using Padé [5,4] (AVX-512).
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm512_set1_ps(PADE_TANH_CLAMP);
    let num_a = _mm512_set1_ps(PADE_TANH_NUM_A);
    let num_b = _mm512_set1_ps(PADE_TANH_NUM_B);
    let den_c4 = _mm512_set1_ps(PADE_TANH_DEN_C4);
    let den_c2 = _mm512_set1_ps(PADE_TANH_DEN_C2);
    let den_a = _mm512_set1_ps(PADE_TANH_DEN_A);
    let two = _mm512_set1_ps(2.0);
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));
    let x_sq = _mm512_mul_ps(x, x);
    let x_sq_sq = _mm512_mul_ps(x_sq, x_sq);

    let num = _mm512_mul_ps(
        x,
        _mm512_add_ps(_mm512_fmadd_ps(num_a, x_sq, x_sq_sq), num_b),
    );

    let den = _mm512_fmadd_ps(den_c4, x_sq_sq, _mm512_fmadd_ps(den_c2, x_sq, den_a));

    let mut rden = _mm512_rcp14_ps(den);
    rden = _mm512_mul_ps(rden, _mm512_fnmadd_ps(den, rden, two));

    let result = _mm512_mul_ps(num, rden);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, result))
}

/// Applies Tanh activation to a slice of f32 using AVX2 optimization.
///
/// # Safety
/// Requires AVX2 and FMA support.
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

/// Applies Tanh activation to a slice of f32 using AVX-512 optimization.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
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

/// Scalar version of `tanh`.
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}
