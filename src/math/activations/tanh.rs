// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Tanh (Hyperbolic Tangent) activation kernels.
//!
//! Uses piecewise minimax odd polynomials (degree 5 in `x`) with
//! **branchless blending** via SIMD masks.  The domain `|x| <= 4` is
//! split into seven symmetric sub-intervals:
//!
//! | Seg | `|x|` range   | Polynomial form                |
//! |-----|---------------|--------------------------------|
//! | 0   | `[0, 1]`      | `x*(c0₀ + c1₀·x² + c2₀·x⁴)` |
//! | 1   | `[1, 1.25]`   | `x*(c0₁ + c1₁·x² + c2₁·x⁴)` |
//! | 2   | `[1.25, 1.5]` | `x*(c0₂ + c1₂·x² + c2₂·x⁴)` |
//! | 3   | `[1.5, 2]`    | `x*(c0₃ + c1₃·x² + c2₃·x⁴)` |
//! | 4   | `[2, 2.5]`    | `x*(c0₄ + c1₄·x² + c2₄·x⁴)` |
//! | 5   | `[2.5, 3]`    | `x*(c0₅ + c1₅·x² + c2₅·x⁴)` |
//! | 6   | `[3, 4]`      | `x*(c0₆ + c1₆·x² + c2₆·x⁴)` |
//!
//! Selection via `_mm256_blendv_ps` / `_mm512_mask_blend_ps` —
//! **zero conditional branches** on the hot-path.
//!
//! Coefficients in `crate::math::constants`; recompute with
//! [Sollya](https://www.sollya.org/) `fpminimax` for optimal mantissa.

use crate::math::constants::*;
use core::arch::x86_64::*;

/// Evaluate odd polynomial `p(t) = t·(c0 + t²·(c1 + c2·t²))` via FMA (AVX2).
macro_rules! eval_poly {
    ($c0:expr, $c1:expr, $c2:expr, $t:expr, $t_sq:expr) => {{
        let inner = _mm256_fmadd_ps($c2, $t_sq, $c1);
        let poly = _mm256_fmadd_ps(inner, $t_sq, $c0);
        _mm256_mul_ps($t, poly)
    }};
}

/// Evaluate odd polynomial `p(t) = t·(c0 + t²·(c1 + c2·t²))` via FMA (AVX-512).
macro_rules! eval_poly_512 {
    ($c0:expr, $c1:expr, $c2:expr, $t:expr, $t_sq:expr) => {{
        let inner = _mm512_fmadd_ps($c2, $t_sq, $c1);
        let poly = _mm512_fmadd_ps(inner, $t_sq, $c0);
        _mm512_mul_ps($t, poly)
    }};
}

/// Branchless 7-segment piecewise minimax odd polynomial for `tanh(x)` (AVX2).
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let b1 = _mm256_set1_ps(PW_TANH_BOUND_1);
    let b2 = _mm256_set1_ps(PW_TANH_BOUND_2);
    let b3 = _mm256_set1_ps(PW_TANH_BOUND_3);
    let b4 = _mm256_set1_ps(PW_TANH_BOUND_4);
    let b5 = _mm256_set1_ps(PW_TANH_BOUND_5);
    let b6 = _mm256_set1_ps(PW_TANH_BOUND_6);
    let sign_mask = _mm256_set1_ps(-0.0);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let abs_x = _mm256_andnot_ps(sign_mask, x);
    let x_sign = _mm256_and_ps(sign_mask, x);
    let x_sq = _mm256_mul_ps(abs_x, abs_x);

    let m1 = _mm256_cmp_ps(abs_x, b1, _CMP_LT_OQ);
    let m2 = _mm256_cmp_ps(abs_x, b2, _CMP_LT_OQ);
    let m3 = _mm256_cmp_ps(abs_x, b3, _CMP_LT_OQ);
    let m4 = _mm256_cmp_ps(abs_x, b4, _CMP_LT_OQ);
    let m5 = _mm256_cmp_ps(abs_x, b5, _CMP_LT_OQ);
    let m6 = _mm256_cmp_ps(abs_x, b6, _CMP_LT_OQ);

    let r0 = eval_poly!(
        _mm256_set1_ps(PW_TANH_C0_0),
        _mm256_set1_ps(PW_TANH_C1_0),
        _mm256_set1_ps(PW_TANH_C2_0),
        abs_x,
        x_sq
    );
    let r1 = eval_poly!(
        _mm256_set1_ps(PW_TANH_C0_1),
        _mm256_set1_ps(PW_TANH_C1_1),
        _mm256_set1_ps(PW_TANH_C2_1),
        abs_x,
        x_sq
    );
    let r2 = eval_poly!(
        _mm256_set1_ps(PW_TANH_C0_2),
        _mm256_set1_ps(PW_TANH_C1_2),
        _mm256_set1_ps(PW_TANH_C2_2),
        abs_x,
        x_sq
    );
    let r3 = eval_poly!(
        _mm256_set1_ps(PW_TANH_C0_3),
        _mm256_set1_ps(PW_TANH_C1_3),
        _mm256_set1_ps(PW_TANH_C2_3),
        abs_x,
        x_sq
    );
    let r4 = eval_poly!(
        _mm256_set1_ps(PW_TANH_C0_4),
        _mm256_set1_ps(PW_TANH_C1_4),
        _mm256_set1_ps(PW_TANH_C2_4),
        abs_x,
        x_sq
    );
    let r5 = eval_poly!(
        _mm256_set1_ps(PW_TANH_C0_5),
        _mm256_set1_ps(PW_TANH_C1_5),
        _mm256_set1_ps(PW_TANH_C2_5),
        abs_x,
        x_sq
    );
    let r6 = eval_poly!(
        _mm256_set1_ps(PW_TANH_C0_6),
        _mm256_set1_ps(PW_TANH_C1_6),
        _mm256_set1_ps(PW_TANH_C2_6),
        abs_x,
        x_sq
    );

    let mut result = r6;
    result = _mm256_blendv_ps(result, r5, m6);
    result = _mm256_blendv_ps(result, r4, m5);
    result = _mm256_blendv_ps(result, r3, m4);
    result = _mm256_blendv_ps(result, r2, m3);
    result = _mm256_blendv_ps(result, r1, m2);
    result = _mm256_blendv_ps(result, r0, m1);

    result = _mm256_or_ps(result, x_sign);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, result))
}

/// Dual 7-segment piecewise `tanh(x)` — 16 floats via two `__m256` lanes.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let clamp_lo = _mm256_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm256_set1_ps(PADE_TANH_CLAMP);
    let b1 = _mm256_set1_ps(PW_TANH_BOUND_1);
    let b2 = _mm256_set1_ps(PW_TANH_BOUND_2);
    let b3 = _mm256_set1_ps(PW_TANH_BOUND_3);
    let b4 = _mm256_set1_ps(PW_TANH_BOUND_4);
    let b5 = _mm256_set1_ps(PW_TANH_BOUND_5);
    let b6 = _mm256_set1_ps(PW_TANH_BOUND_6);
    let sign_mask = _mm256_set1_ps(-0.0);
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    let x1 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x1));
    let x2 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x2));

    let abs1 = _mm256_andnot_ps(sign_mask, x1);
    let abs2 = _mm256_andnot_ps(sign_mask, x2);
    let sign1 = _mm256_and_ps(sign_mask, x1);
    let sign2 = _mm256_and_ps(sign_mask, x2);
    let sq1 = _mm256_mul_ps(abs1, abs1);
    let sq2 = _mm256_mul_ps(abs2, abs2);

    macro_rules! masks {
        ($abs:expr) => {{
            let m1 = _mm256_cmp_ps($abs, b1, _CMP_LT_OQ);
            let m2 = _mm256_cmp_ps($abs, b2, _CMP_LT_OQ);
            let m3 = _mm256_cmp_ps($abs, b3, _CMP_LT_OQ);
            let m4 = _mm256_cmp_ps($abs, b4, _CMP_LT_OQ);
            let m5 = _mm256_cmp_ps($abs, b5, _CMP_LT_OQ);
            let m6 = _mm256_cmp_ps($abs, b6, _CMP_LT_OQ);
            (m1, m2, m3, m4, m5, m6)
        }};
    }

    let (m11, m12, m13, m14, m15, m16) = masks!(abs1);
    let (m21, m22, m23, m24, m25, m26) = masks!(abs2);

    macro_rules! polys {
        ($abs:expr, $sq:expr) => {{
            let r0 = eval_poly!(
                _mm256_set1_ps(PW_TANH_C0_0),
                _mm256_set1_ps(PW_TANH_C1_0),
                _mm256_set1_ps(PW_TANH_C2_0),
                $abs,
                $sq
            );
            let r1 = eval_poly!(
                _mm256_set1_ps(PW_TANH_C0_1),
                _mm256_set1_ps(PW_TANH_C1_1),
                _mm256_set1_ps(PW_TANH_C2_1),
                $abs,
                $sq
            );
            let r2 = eval_poly!(
                _mm256_set1_ps(PW_TANH_C0_2),
                _mm256_set1_ps(PW_TANH_C1_2),
                _mm256_set1_ps(PW_TANH_C2_2),
                $abs,
                $sq
            );
            let r3 = eval_poly!(
                _mm256_set1_ps(PW_TANH_C0_3),
                _mm256_set1_ps(PW_TANH_C1_3),
                _mm256_set1_ps(PW_TANH_C2_3),
                $abs,
                $sq
            );
            let r4 = eval_poly!(
                _mm256_set1_ps(PW_TANH_C0_4),
                _mm256_set1_ps(PW_TANH_C1_4),
                _mm256_set1_ps(PW_TANH_C2_4),
                $abs,
                $sq
            );
            let r5 = eval_poly!(
                _mm256_set1_ps(PW_TANH_C0_5),
                _mm256_set1_ps(PW_TANH_C1_5),
                _mm256_set1_ps(PW_TANH_C2_5),
                $abs,
                $sq
            );
            let r6 = eval_poly!(
                _mm256_set1_ps(PW_TANH_C0_6),
                _mm256_set1_ps(PW_TANH_C1_6),
                _mm256_set1_ps(PW_TANH_C2_6),
                $abs,
                $sq
            );
            (r0, r1, r2, r3, r4, r5, r6)
        }};
    }

    let (r01, r11, r21, r31, r41, r51, r61) = polys!(abs1, sq1);
    let (r02, r12, r22, r32, r42, r52, r62) = polys!(abs2, sq2);

    macro_rules! blend7 {
        ($r6:expr, $r5:expr, $r4:expr, $r3:expr, $r2:expr, $r1:expr, $r0:expr,
                           $m6:expr, $m5:expr, $m4:expr, $m3:expr, $m2:expr, $m1:expr) => {{
            let mut res = $r6;
            res = _mm256_blendv_ps(res, $r5, $m6);
            res = _mm256_blendv_ps(res, $r4, $m5);
            res = _mm256_blendv_ps(res, $r3, $m4);
            res = _mm256_blendv_ps(res, $r2, $m3);
            res = _mm256_blendv_ps(res, $r1, $m2);
            res = _mm256_blendv_ps(res, $r0, $m1);
            res
        }};
    }

    let res1 = blend7!(
        r61, r51, r41, r31, r21, r11, r01, m16, m15, m14, m13, m12, m11
    );
    let res2 = blend7!(
        r62, r52, r42, r32, r22, r12, r02, m26, m25, m24, m23, m22, m21
    );

    let res1 = _mm256_or_ps(res1, sign1);
    let res2 = _mm256_or_ps(res2, sign2);

    (
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res1)),
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res2)),
    )
}

/// Branchless 7-segment piecewise minimax odd polynomial for `tanh(x)` (AVX-512).
///
/// # Safety
/// The caller must guarantee AVX-512F, AVX-512VL, and AVX-512DQ support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl,avx512dq")]
pub unsafe fn simd_tanh_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-PADE_TANH_CLAMP);
    let clamp_hi = _mm512_set1_ps(PADE_TANH_CLAMP);
    let b1 = _mm512_set1_ps(PW_TANH_BOUND_1);
    let b2 = _mm512_set1_ps(PW_TANH_BOUND_2);
    let b3 = _mm512_set1_ps(PW_TANH_BOUND_3);
    let b4 = _mm512_set1_ps(PW_TANH_BOUND_4);
    let b5 = _mm512_set1_ps(PW_TANH_BOUND_5);
    let b6 = _mm512_set1_ps(PW_TANH_BOUND_6);
    let sign_mask = _mm512_set1_ps(-0.0);
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    let abs_x = _mm512_andnot_ps(sign_mask, x);
    let x_sign = _mm512_and_ps(sign_mask, x);
    let x_sq = _mm512_mul_ps(abs_x, abs_x);

    let m1: __mmask16 = _mm512_cmp_ps_mask(abs_x, b1, _CMP_LT_OQ);
    let m2: __mmask16 = _mm512_cmp_ps_mask(abs_x, b2, _CMP_LT_OQ);
    let m3: __mmask16 = _mm512_cmp_ps_mask(abs_x, b3, _CMP_LT_OQ);
    let m4: __mmask16 = _mm512_cmp_ps_mask(abs_x, b4, _CMP_LT_OQ);
    let m5: __mmask16 = _mm512_cmp_ps_mask(abs_x, b5, _CMP_LT_OQ);
    let m6: __mmask16 = _mm512_cmp_ps_mask(abs_x, b6, _CMP_LT_OQ);

    let r0 = eval_poly_512!(
        _mm512_set1_ps(PW_TANH_C0_0),
        _mm512_set1_ps(PW_TANH_C1_0),
        _mm512_set1_ps(PW_TANH_C2_0),
        abs_x,
        x_sq
    );
    let r1 = eval_poly_512!(
        _mm512_set1_ps(PW_TANH_C0_1),
        _mm512_set1_ps(PW_TANH_C1_1),
        _mm512_set1_ps(PW_TANH_C2_1),
        abs_x,
        x_sq
    );
    let r2 = eval_poly_512!(
        _mm512_set1_ps(PW_TANH_C0_2),
        _mm512_set1_ps(PW_TANH_C1_2),
        _mm512_set1_ps(PW_TANH_C2_2),
        abs_x,
        x_sq
    );
    let r3 = eval_poly_512!(
        _mm512_set1_ps(PW_TANH_C0_3),
        _mm512_set1_ps(PW_TANH_C1_3),
        _mm512_set1_ps(PW_TANH_C2_3),
        abs_x,
        x_sq
    );
    let r4 = eval_poly_512!(
        _mm512_set1_ps(PW_TANH_C0_4),
        _mm512_set1_ps(PW_TANH_C1_4),
        _mm512_set1_ps(PW_TANH_C2_4),
        abs_x,
        x_sq
    );
    let r5 = eval_poly_512!(
        _mm512_set1_ps(PW_TANH_C0_5),
        _mm512_set1_ps(PW_TANH_C1_5),
        _mm512_set1_ps(PW_TANH_C2_5),
        abs_x,
        x_sq
    );
    let r6 = eval_poly_512!(
        _mm512_set1_ps(PW_TANH_C0_6),
        _mm512_set1_ps(PW_TANH_C1_6),
        _mm512_set1_ps(PW_TANH_C2_6),
        abs_x,
        x_sq
    );

    let mut result = r6;
    result = _mm512_mask_blend_ps(m6, result, r5);
    result = _mm512_mask_blend_ps(m5, result, r4);
    result = _mm512_mask_blend_ps(m4, result, r3);
    result = _mm512_mask_blend_ps(m3, result, r2);
    result = _mm512_mask_blend_ps(m2, result, r1);
    result = _mm512_mask_blend_ps(m1, result, r0);

    result = _mm512_or_ps(result, x_sign);
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

/// Scalar version of `tanh` — delegates to `f32::tanh`.
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}
