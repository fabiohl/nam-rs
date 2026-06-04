// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Tanh (Hyperbolic Tangent) activation kernels.
//!
//! **Production path:** Padé [5,4] rational approximant with hardware `_mm256_div_ps`.
//!
//! ```text
//! tanh(x) ≈ x * (x² + 105) * (x² + 945) / ((15x² + 420) * x² + 945)
//! ```
//!
//! - Max absolute error: ~2.32e-3 vs `f32::tanh` on [-4, 4].
//! - Throughput: ~9 SIMD ops (vs ~28 for the 7-segment piecewise alternative).
//! - Benchmark data (E8.T04): ~62 ns/256-elem AVX2 vs ~156 ns for piecewise.
//!
//! Coefficients in `crate::math::constants` (`PADE_TANH_*`).
//!
//! **Experimental / reference path:** Piecewise 7-segment minimax polynomials
//! (`simd_tanh_piecewise_avx2` / `simd_tanh_piecewise_avx512`) are retained for
//! future optimization. They achieved ~4.90e-3 max error with a +16% throughput
//! regression on LSTM prewarm (2048 samples) — see E8.T02 analysis in TODO-sprints.md.
//! Pending: recompute coefficients via [Sollya](https://www.sollya.org/) `fpminimax`.

use crate::math::constants::*;
use core::arch::x86_64::*;

/// Padé [5,4] rational approximant for `tanh(x)` with hardware division (AVX2).
///
/// Production path. ~9 SIMD ops, max absolute error ~2.32e-3.
///
/// Formula: `tanh(x) ≈ x·(x²+105)·(x²+945) / ((15x²+420)·x²+945)`
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let x_sq = _mm256_mul_ps(x, x);

    // Numerator: x * ((x² + 105) * x² + 945)
    // Horner: P(t) = (t + 105) * t + 945  where t = x²
    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B); // 945.0
    let num = _mm256_add_ps(x_sq, num_a);
    let num = _mm256_fmadd_ps(num, x_sq, num_b);
    let num = _mm256_mul_ps(x, num);

    // Denominator: (15 * x² + 420) * x² + 945
    // Horner: Q(t) = (15*t + 420) * t + 945  where t = x²
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A); // 945.0
    let den = _mm256_fmadd_ps(x_sq, den_c4, den_c2);
    let den = _mm256_fmadd_ps(den, x_sq, den_a);

    let result = _mm256_div_ps(num, den);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, result))
}

/// Padé [5,4] rational approximant for `tanh(x)` — dual 16-float path (AVX2).
///
/// Evaluates two independent `__m256` registers. Coefficients are broadcast
/// once and shared between both lanes, amortising setup cost.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x1 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x1));
    let x2 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x2));

    let sq1 = _mm256_mul_ps(x1, x1);
    let sq2 = _mm256_mul_ps(x2, x2);

    // Broadcast Padé coefficients once for both lanes.
    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B); // 945.0
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A); // 945.0

    // Lane 1 — Numerator: x1 * ((x1² + 105) * x1² + 945)
    let num1 = _mm256_fmadd_ps(_mm256_add_ps(sq1, num_a), sq1, num_b);
    let num1 = _mm256_mul_ps(x1, num1);
    // Lane 1 — Denominator: (15 * x1² + 420) * x1² + 945
    let den1 = _mm256_fmadd_ps(_mm256_fmadd_ps(sq1, den_c4, den_c2), sq1, den_a);

    // Lane 2 — Numerator: x2 * ((x2² + 105) * x2² + 945)
    let num2 = _mm256_fmadd_ps(_mm256_add_ps(sq2, num_a), sq2, num_b);
    let num2 = _mm256_mul_ps(x2, num2);
    // Lane 2 — Denominator: (15 * x2² + 420) * x2² + 945
    let den2 = _mm256_fmadd_ps(_mm256_fmadd_ps(sq2, den_c4, den_c2), sq2, den_a);

    let res1 = _mm256_div_ps(num1, den1);
    let res2 = _mm256_div_ps(num2, den2);

    (
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res1)),
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res2)),
    )
}

/// Padé [5,4] rational approximant for `tanh(x)` with hardware division (AVX-512).
///
/// Production path. ~9 SIMD ops, max absolute error ~2.32e-3.
///
/// Formula: `tanh(x) ≈ x·(x²+105)·(x²+945) / ((15x²+420)·x²+945)`
///
/// # Safety
/// The caller must guarantee AVX-512F, AVX-512VL, and AVX-512DQ support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn simd_tanh_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm512_set1_ps(PADE_TANH_CLAMP);
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    let x_sq = _mm512_mul_ps(x, x);

    // Numerator: x * ((x² + 105) * x² + 945)
    let num_a = _mm512_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm512_set1_ps(PADE_TANH_NUM_B); // 945.0
    let num = _mm512_add_ps(x_sq, num_a);
    let num = _mm512_fmadd_ps(num, x_sq, num_b);
    let num = _mm512_mul_ps(x, num);

    // Denominator: (15 * x² + 420) * x² + 945
    let den_c4 = _mm512_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm512_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm512_set1_ps(PADE_TANH_DEN_A); // 945.0
    let den = _mm512_fmadd_ps(x_sq, den_c4, den_c2);
    let den = _mm512_fmadd_ps(den, x_sq, den_a);

    let result = _mm512_div_ps(num, den);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, result))
}

/// Applies Tanh activation to a slice of f32 using AVX2 dual-path.
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

/// Applies Tanh activation to a slice of f32 using AVX-512.
///
/// # Safety
/// Requires AVX-512F, AVX-512VL, and AVX-512DQ support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
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

// ══════════════════════════════════════════════════════════════════════════════
// Experimental / Reference Variants — Piecewise and Padé NR2
// ══════════════════════════════════════════════════════════════════════════════
//
// Piecewise (E8.T02): 7-segment branchless, experimental. Higher op count
// causes +16% regression on LSTM prewarm. Retained for future research.
//
// Padé NR2 (E8.T04): reference showing double Newton-Raphson saturates
// f32 mantissa; error ratio NR2/Div = 1.000×. Retained for documentation.
//
// NOTE: The production path (simd_tanh_avx2 / simd_tanh_avx512) now uses
// the Padé [5,4] + hardware-div approach (formerly simd_tanh_pade_div_*).
// The _div_ variants below are kept for historic reference / benchmarking.
//
// Padé [5,4]: tanh(x) ≈ x * P(x²) / Q(x²)
//   P(t) = t² + 105t + 945
//   Q(t) = 15t² + 420t + 945
//
// Variants:
//   - _nr2: double Newton-Raphson on rcp → saturates f32 mantissa (24 bits)
//   - _div: hardware division → IEEE 754 full precision oracle
/// Padé [5,4] rational tanh with double Newton-Raphson (AVX2).
///
/// Reference variant (E8.T04). Double NR saturates f32 mantissa (24 bits);
/// error ratio vs hardware-div = 1.000× — reciprocal contributes zero drift.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_pade_nr2_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let x_sq = _mm256_mul_ps(x, x);

    // Numerator: x * ((x² + 105) * x² + 945)
    // Horner: P(t) = (t + 105) * t + 945  where t = x²
    let num_a = _mm256_set1_ps(PADE_TANH_NUM_A); // 105.0
    let num_b = _mm256_set1_ps(PADE_TANH_NUM_B); // 945.0
    let num = _mm256_add_ps(x_sq, num_a); // x² + 105
    let num = _mm256_fmadd_ps(num, x_sq, num_b); // (x² + 105) * x² + 945
    let num = _mm256_mul_ps(x, num);

    // Denominator: (15 * x² + 420) * x² + 945
    // Horner: Q(t) = (15*t + 420) * t + 945  where t = x²
    let den_c4 = _mm256_set1_ps(PADE_TANH_DEN_C4); // 15.0
    let den_c2 = _mm256_set1_ps(PADE_TANH_DEN_C2); // 420.0
    let den_a = _mm256_set1_ps(PADE_TANH_DEN_A); // 945.0
    let den = _mm256_fmadd_ps(x_sq, den_c4, den_c2); // 15*x² + 420
    let den = _mm256_fmadd_ps(den, x_sq, den_a); // (15*x² + 420) * x² + 945

    // Reciprocal with double Newton-Raphson (saturates f32 mantissa)
    let mut r = _mm256_rcp_ps(den);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(den, r, two));
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(den, r, two));

    let result = _mm256_mul_ps(num, r);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, result))
}

/// Padé [5,4] rational tanh with double Newton-Raphson (AVX-512).
///
/// Reference variant (E8.T04). Double NR saturates f32 mantissa (24 bits);
/// error ratio vs hardware-div = 1.000× — reciprocal contributes zero drift.
///
/// # Safety
/// The caller must guarantee AVX-512F, AVX-512VL, and AVX-512DQ support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn simd_tanh_pade_nr2_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm512_set1_ps(PADE_TANH_CLAMP);
    let two = _mm512_set1_ps(2.0);
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    let x_sq = _mm512_mul_ps(x, x);

    // Numerator: x * ((x² + 105) * x² + 945)
    let num_a = _mm512_set1_ps(PADE_TANH_NUM_A);
    let num_b = _mm512_set1_ps(PADE_TANH_NUM_B);
    let num = _mm512_add_ps(x_sq, num_a);
    let num = _mm512_fmadd_ps(num, x_sq, num_b);
    let num = _mm512_mul_ps(x, num);

    // Denominator: (15 * x² + 420) * x² + 945
    let den_c4 = _mm512_set1_ps(PADE_TANH_DEN_C4);
    let den_c2 = _mm512_set1_ps(PADE_TANH_DEN_C2);
    let den_a = _mm512_set1_ps(PADE_TANH_DEN_A);
    let den = _mm512_fmadd_ps(x_sq, den_c4, den_c2);
    let den = _mm512_fmadd_ps(den, x_sq, den_a);

    let mut r = _mm512_rcp14_ps(den);
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(den, r, two));
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(den, r, two));

    let result = _mm512_mul_ps(num, r);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, result))
}

/// Scalar version of `tanh` — delegates to `f32::tanh`.
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}
