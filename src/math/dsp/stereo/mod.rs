// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! DSP operations for stereo processing and signal metering.

mod convolution_avx2;
mod convolution_avx512;
mod energy;
mod max_diff;
mod peak;

pub use convolution_avx2::{convolve_mono_avx2, convolve_stereo_avx2, convolve_stereo_dual_avx2};
pub use convolution_avx512::{
    convolve_mono_avx512, convolve_stereo_avx512, convolve_stereo_dual_avx512,
};
pub use energy::{
    compute_energy_avx2, compute_energy_avx512, compute_energy_stereo_avx2,
    compute_energy_stereo_avx512,
};
pub use max_diff::{compute_max_diff_avx2, compute_max_diff_avx512};
pub use peak::{compute_peak_abs_stereo_avx2, compute_peak_abs_stereo_avx512};

/// Computes the maximum energy between two audio channels via SIMD dispatch.
///
/// # Safety
/// Uses dynamic dispatch via global v-table.
pub unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_energy_stereo(l, r))
}

/// Computes the maximum absolute difference between two blocks via SIMD dispatch.
///
/// # Safety
/// Uses dynamic dispatch via global v-table.
pub unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_max_diff(a, b))
}

/// Computes the peak absolute value of both stereo channels via SIMD dispatch.
///
/// # Safety
/// Uses dynamic dispatch via global v-table.
pub unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
    crate::math::common::dispatch_simd!(compute_peak_abs_stereo(left, right))
}

/// Stereo convolution (used in the resampler) via SIMD dispatch.
/// Computes the dot product between a filter bank and two input buffers (L/R).
///
/// # Safety
/// `coeffs`, `input_l`, and `input_r` must be valid pointers to at least `taps` elements.
/// `coeffs` must be aligned according to the SIMD register.
pub unsafe fn convolve_stereo(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    crate::math::common::dispatch_simd!(convolve_stereo(coeffs, input_l, input_r, taps))
}

/// Mono convolution (used in the resampler) via SIMD dispatch.
/// Computes the dot product between a filter bank and an input buffer.
///
/// # Safety
/// `coeffs` and `input` must be valid pointers to at least `taps` elements.
/// `coeffs` must be aligned according to the SIMD register.
pub unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    crate::math::common::dispatch_simd!(convolve_mono(coeffs, input, taps))
}
