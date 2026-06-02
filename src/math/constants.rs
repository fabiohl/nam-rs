// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(clippy::excessive_precision)]

//! Mathematical constants, Minimax polynomial coefficients, and LUT parameters.
//! Centralized to ensure consistency across scalar, AVX2, and AVX-512 implementations.

// --- Gain LUT Parameters ---

/// Size of the gain LUT table. 4096 points provide sub-0.001 dB precision.
pub const GAIN_LUT_SIZE: usize = 4096;
/// Lower dB limit for the LUT (practical silence floor -96dB).
pub const GAIN_MIN_DB: f32 = -96.0;
/// Upper dB limit for the LUT (+30dB is an extreme boost).
pub const GAIN_MAX_DB: f32 = 30.0;
/// Total gain range in dB supported by the LUT.
pub const GAIN_DB_RANGE: f32 = GAIN_MAX_DB - GAIN_MIN_DB;
/// dB step between each LUT entry.
pub const GAIN_DB_STEP: f32 = GAIN_DB_RANGE / (GAIN_LUT_SIZE as f32 - 1.0);
/// Inverse of the gain step (optimization to avoid divisions in the hot-path).
pub const INV_GAIN_DB_STEP: f32 = 1.0 / GAIN_DB_STEP;

// --- Padé Coefficients for Tanh (Rational [5,4]) ---

/// Numerator constant: x⁴ + 105·x² + 945.
pub const PADE_TANH_NUM_A: f32 = 105.0;
/// Numerator constant.
pub const PADE_TANH_NUM_B: f32 = 945.0;
/// Denominator x⁴ coefficient: 15.
pub const PADE_TANH_DEN_C4: f32 = 15.0;
/// Denominator x² coefficient: 420.
pub const PADE_TANH_DEN_C2: f32 = 420.0;
/// Denominator constant.
pub const PADE_TANH_DEN_A: f32 = 945.0;
/// Clamp limit for the Padé tanh domain (|x| < 4).
pub const PADE_TANH_CLAMP: f32 = 4.0;

// --- Piecewise Minimax Coefficients for Tanh (7 segments, branchless blending) ---
//
// Domain: |x| in [-4, 4], split into 7 symmetric sub-intervals:
//   Seg 0: |x| in [0, 1]       Seg 1: |x| in [1, 1.25]
//   Seg 2: |x| in [1.25, 1.5]  Seg 3: |x| in [1.5, 2]
//   Seg 4: |x| in [2, 2.5]     Seg 5: |x| in [2.5, 3]
//   Seg 6: |x| in [3, 4]
//
// Polynomial form (odd, degree 5 equivalent in x):
//   p(t) = t * (c0 + c1*t² + c2*t⁴)   where t = |x|
//
// Seven narrow segments keep per-segment max absolute error
// < 1.3e-3 (seg 0) and < 2e-4 (all others).
// TODO: Recompute with Sollya's fpminimax for optimal 23-bit mantissa.

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

// Segment 0 [0, 1] — near-linear region.
/// Coefficient c0 (linear term) for segment 0.
pub const PW_TANH_C0_0: f32 = 0.99999970;
/// Coefficient c1 (cubic in t²) for segment 0.
pub const PW_TANH_C1_0: f32 = -0.32455131;
/// Coefficient c2 (quintic in t⁴) for segment 0.
pub const PW_TANH_C2_0: f32 = 0.08614561;

// Segment 1 [1, 1.25]
/// Coefficient c0 (linear term) for segment 1.
pub const PW_TANH_C0_1: f32 = 0.96729100;
/// Coefficient c1 (cubic in t²) for segment 1.
pub const PW_TANH_C1_1: f32 = -0.24295763;
/// Coefficient c2 (quintic in t⁴) for segment 1.
pub const PW_TANH_C2_1: f32 = 0.03726047;

// Segment 2 [1.25, 1.5]
/// Coefficient c0 (linear term) for segment 2.
pub const PW_TANH_C0_2: f32 = 0.93594641;
/// Coefficient c1 (cubic in t²) for segment 2.
pub const PW_TANH_C1_2: f32 = -0.20308490;
/// Coefficient c2 (quintic in t⁴) for segment 2.
pub const PW_TANH_C2_2: f32 = 0.02457593;

// Segment 3 [1.5, 2]
/// Coefficient c0 (linear term) for segment 3.
pub const PW_TANH_C0_3: f32 = 0.86742296;
/// Coefficient c1 (cubic in t²) for segment 3.
pub const PW_TANH_C1_3: f32 = -0.14429085;
/// Coefficient c2 (quintic in t⁴) for segment 3.
pub const PW_TANH_C2_3: f32 = 0.01198317;

// Segment 4 [2, 2.5] — transition to saturation.
/// Coefficient c0 (linear term) for segment 4.
pub const PW_TANH_C0_4: f32 = 0.75754636;
/// Coefficient c1 (cubic in t²) for segment 4.
pub const PW_TANH_C1_4: f32 = -0.08811882;
/// Coefficient c2 (quintic in t⁴) for segment 4.
pub const PW_TANH_C2_4: f32 = 0.00480930;

// Segment 5 [2.5, 3] — near-saturation.
/// Coefficient c0 (linear term) for segment 5.
pub const PW_TANH_C0_5: f32 = 0.65478558;
/// Coefficient c1 (cubic in t²) for segment 5.
pub const PW_TANH_C1_5: f32 = -0.05462651;
/// Coefficient c2 (quintic in t⁴) for segment 5.
pub const PW_TANH_C2_5: f32 = 0.00208080;

// Segment 6 [3, 4] — deep saturation.
/// Coefficient c0 (linear term) for segment 6.
pub const PW_TANH_C0_6: f32 = 0.52481012;
/// Coefficient c1 (cubic in t²) for segment 6.
pub const PW_TANH_C1_6: f32 = -0.02695124;
/// Coefficient c2 (quintic in t⁴) for segment 6.
pub const PW_TANH_C2_6: f32 = 0.00061032;

// --- Minimax Polynomial Coefficients for Direct Sigmoid (Degree 17, 9 odd terms) ---
// Optimized for the interval [-8, 8] via Lawson's weighted minimax algorithm.
// Max absolute error: ~4.09e-4 vs f32::exp native sigmoid.
/// Input clamp for the direct sigmoid polynomial domain (|x| < 8).
pub const SIGMOID_MINIMAX_CLAMP: f32 = 8.0;
/// Minimax coefficient for x^1 term.
pub const SIGMOID_MINIMAX_C0: f32 = 2.4885319190e-01;
/// Minimax coefficient for x^3 term.
pub const SIGMOID_MINIMAX_C1: f32 = -1.9318685012e-02;
/// Minimax coefficient for x^5 term.
pub const SIGMOID_MINIMAX_C2: f32 = 1.4623214305e-03;
/// Minimax coefficient for x^7 term.
pub const SIGMOID_MINIMAX_C3: f32 = -7.9953400187e-05;
/// Minimax coefficient for x^9 term.
pub const SIGMOID_MINIMAX_C4: f32 = 2.9140652422e-06;
/// Minimax coefficient for x^11 term.
pub const SIGMOID_MINIMAX_C5: f32 = -6.8000246432e-08;
/// Minimax coefficient for x^13 term.
pub const SIGMOID_MINIMAX_C6: f32 = 9.6897239158e-10;
/// Minimax coefficient for x^15 term.
pub const SIGMOID_MINIMAX_C7: f32 = -7.6498626314e-12;
/// Minimax coefficient for x^17 term.
pub const SIGMOID_MINIMAX_C8: f32 = 2.5585471676e-14;
