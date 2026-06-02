// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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
