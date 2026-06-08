// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP gain operations, clipping detection, and stereo ramp.
//!
//! Dynamically dispatches to the configured SIMD backend.

mod avx2;
mod avx512;

pub use avx2::*;
pub use avx512::*;

/// Applies constant gain to a mono buffer via SIMD dispatch.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_gain(data: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain(data, gain))
}

/// Applies constant gain in stereo via SIMD dispatch.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain_stereo(left, right, gain))
}

/// Applies gain and detects clipping in mono in a single pass.
/// Returns `true` if any resulting sample has `|x| > 1.0`.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
    crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_mono(data, gain))
}

/// Applies gain and detects clipping in stereo in a single pass.
/// Returns `true` if any resulting sample has `|x| > 1.0`.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_gain_and_detect_clipping_stereo(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_stereo(left, right, gain))
}

/// Applies linear gain ramp in stereo via SIMD dispatch.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    crate::math::common::dispatch_simd!(apply_ramp_stereo(left, right, start, step))
}

/// Applies linear gain ramp to a mono buffer via SIMD dispatch.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
    crate::math::common::dispatch_simd!(apply_ramp(data, start, step))
}

/// Applies the linear gain multiplier to the buffer (safe wrapper).
///
/// Fast-returns if `gain_linear` ~= 1.0 (fast-path bypass).
/// Routes to the SIMD backend via v-table.
pub fn apply_gain_simd(buffer: &mut [f32], gain_linear: f32) {
    if (gain_linear - 1.0).abs() < 1e-6 {
        return;
    }
    unsafe { apply_gain(buffer, gain_linear) };
}

/// Applies a linear gain ramp to the buffer (safe wrapper).
///
/// If the increment is negligible, applies constant gain instead.
/// Routes to the SIMD backend via v-table.
pub fn apply_ramp_simd(buffer: &mut [f32], start: f32, step: f32) {
    if step.abs() < 1e-9 {
        apply_gain_simd(buffer, start);
        return;
    }
    unsafe { apply_ramp(buffer, start, step) };
}

/// Adds a broadcast constant (dither offset) to every element of a mono buffer.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
    crate::math::common::dispatch_simd!(apply_dither_add(data, offset))
}

/// Safe wrapper for dither offset addition via SIMD broadcast + vector add.
pub fn apply_dither_add_simd(buffer: &mut [f32], offset: f32) {
    unsafe { apply_dither_add(buffer, offset) };
}

#[cfg(test)]
#[path = "../gain_test.rs"]
mod gain_test;
