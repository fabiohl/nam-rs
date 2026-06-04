// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(clippy::excessive_precision)]

//! Experimental branchless 7-segment piecewise minimax odd polynomial for `tanh(x)`.

use crate::math::constants::PADE_TANH_CLAMP;
use core::arch::x86_64::*;

macro_rules! eval_poly {
    ($c0:expr, $c1:expr, $c2:expr, $t:expr, $t_sq:expr) => {{
        let inner = _mm256_fmadd_ps($c2, $t_sq, $c1);
        let poly = _mm256_fmadd_ps(inner, $t_sq, $c0);
        _mm256_mul_ps($t, poly)
    }};
}

#[allow(unused_macros)]
macro_rules! eval_poly_512 {
    ($c0:expr, $c1:expr, $c2:expr, $t:expr, $t_sq:expr) => {{
        let inner = _mm512_fmadd_ps($c2, $t_sq, $c1);
        let poly = _mm512_fmadd_ps(inner, $t_sq, $c0);
        _mm512_mul_ps($t, poly)
    }};
}

/// Segment boundary: |x| = 1.0.
pub const PW_TANH_BOUND_1: f32 = 1.0;
/// Segment boundary: |x| = 1.25.
pub const PW_TANH_BOUND_2: f32 = 1.25;
/// Segment boundary: |x| = 1.5.
pub const PW_TANH_BOUND_3: f32 = 1.5;
/// Segment boundary: |x| = 2.0.
pub const PW_TANH_BOUND_4: f32 = 2.0;
/// Segment boundary: |x| = 2.5.
pub const PW_TANH_BOUND_5: f32 = 2.5;
/// Segment boundary: |x| = 3.0.
pub const PW_TANH_BOUND_6: f32 = 3.0;

/// Coefficient c0 (linear term) for segment 0.
pub const PW_TANH_C0_0: f32 = 0.99999970;
/// Coefficient c1 (cubic in t²) for segment 0.
pub const PW_TANH_C1_0: f32 = -0.32455131;
/// Coefficient c2 (quintic in t⁴) for segment 0.
pub const PW_TANH_C2_0: f32 = 0.08614561;

/// Coefficient c0 (linear term) for segment 1.
pub const PW_TANH_C0_1: f32 = 0.96729100;
/// Coefficient c1 (cubic in t²) for segment 1.
pub const PW_TANH_C1_1: f32 = -0.24295763;
/// Coefficient c2 (quintic in t⁴) for segment 1.
pub const PW_TANH_C2_1: f32 = 0.03726047;

/// Coefficient c0 (linear term) for segment 2.
pub const PW_TANH_C0_2: f32 = 0.93594641;
/// Coefficient c1 (cubic in t²) for segment 2.
pub const PW_TANH_C1_2: f32 = -0.20308490;
/// Coefficient c2 (quintic in t⁴) for segment 2.
pub const PW_TANH_C2_2: f32 = 0.02457593;

/// Coefficient c0 (linear term) for segment 3.
pub const PW_TANH_C0_3: f32 = 0.86742296;
/// Coefficient c1 (cubic in t²) for segment 3.
pub const PW_TANH_C1_3: f32 = -0.14429085;
/// Coefficient c2 (quintic in t⁴) for segment 3.
pub const PW_TANH_C2_3: f32 = 0.01198317;

/// Coefficient c0 (linear term) for segment 4.
pub const PW_TANH_C0_4: f32 = 0.75754636;
/// Coefficient c1 (cubic in t²) for segment 4.
pub const PW_TANH_C1_4: f32 = -0.08811882;
/// Coefficient c2 (quintic in t⁴) for segment 4.
pub const PW_TANH_C2_4: f32 = 0.00480930;

/// Coefficient c0 (linear term) for segment 5.
pub const PW_TANH_C0_5: f32 = 0.65478558;
/// Coefficient c1 (cubic in t²) for segment 5.
pub const PW_TANH_C1_5: f32 = -0.05462651;
/// Coefficient c2 (quintic in t⁴) for segment 5.
pub const PW_TANH_C2_5: f32 = 0.00208080;

/// Coefficient c0 (linear term) for segment 6.
pub const PW_TANH_C0_6: f32 = 0.52481012;
/// Coefficient c1 (cubic in t²) for segment 6.
pub const PW_TANH_C1_6: f32 = -0.02695124;
/// Coefficient c2 (quintic in t⁴) for segment 6.
pub const PW_TANH_C2_6: f32 = 0.00061032;

/// Branchless 7-segment piecewise minimax odd polynomial for `tanh(x)` (AVX2).
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_piecewise_avx2(x: __m256) -> __m256 {
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
