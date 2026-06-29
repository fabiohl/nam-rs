// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the CLAP parameters extension for NAM-rs.

pub mod audio;
pub mod main;

/// Input gain parameter ID.
pub const PARAM_INPUT_GAIN: u32 = 0;
/// Output gain parameter ID.
pub const PARAM_OUTPUT_GAIN: u32 = 1;
/// Noise gate threshold parameter ID.
pub const PARAM_GATE_THRESH: u32 = 2;
/// Plugin bypass parameter ID.
pub const PARAM_BYPASS: u32 = 3;
/// Loaded model name parameter ID (read-only).
pub const PARAM_ACTIVE_MODEL: u32 = 4;
/// Adaptive compute mode parameter ID.
pub const PARAM_ADAPTIVE_COMPUTE: u32 = 5;
/// Manual slim override parameter ID.
pub const PARAM_SLIM_OVERRIDE: u32 = 6;
/// Oversampling factor parameter ID (stepped: Off/2x/4x).
pub const PARAM_OVERSAMPLE: u32 = 7;
/// Activation precision parameter ID (stepped: Standard/HighFidelity).
pub const PARAM_ACTIVATION: u32 = 8;

/// Bypass atomic value: off.
pub const BYPASS_OFF: u32 = 0;
/// Bypass atomic value: on.
pub const BYPASS_ON: u32 = 1;

/// Convert a `u32` atomic bypass value to a `bool`.
#[inline]
pub fn bypass_u32_to_bool(val: u32) -> bool {
    val != 0
}

/// Convert a `bool` bypass to its `u32` atomic representation.
#[inline]
pub fn bypass_bool_to_u32(b: bool) -> u32 {
    if b { BYPASS_ON } else { BYPASS_OFF }
}

/// Convert a `f32` CLAP parameter value to a `bool` bypass.
#[inline]
pub fn bypass_f32_to_bool(val: f32) -> bool {
    val > 0.5
}

/// Activation precision atomic value: Standard.
pub const ACTIVATION_STANDARD: u32 = 0;
/// Activation precision atomic value: HighFidelity.
pub const ACTIVATION_HF: u32 = 1;

/// Convert a `u32` atomic activation value to `ActivationPrecision`.
#[inline]
pub fn activation_u32_to_enum(val: u32) -> crate::math::activations::ActivationPrecision {
    if val == ACTIVATION_HF {
        crate::math::activations::ActivationPrecision::HighFidelity
    } else {
        crate::math::activations::ActivationPrecision::Standard
    }
}

/// Convert an `ActivationPrecision` to its `u32` atomic representation.
#[inline]
pub fn activation_enum_to_u32(mode: crate::math::activations::ActivationPrecision) -> u32 {
    mode as u32
}
