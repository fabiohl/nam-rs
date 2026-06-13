// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

macro_rules! impl_avx512_dsp {
    () => {
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn convolve_stereo(
            coeffs: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::dsp::stereo::convolve_stereo_avx512(coeffs, input_l, input_r, taps)
            }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn convolve_stereo_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> ((f32, f32), (f32, f32)) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::dsp::stereo::convolve_stereo_dual_avx512(
                    coeffs0, coeffs1, input_l, input_r, taps,
                )
            }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::stereo::convolve_mono_avx512(coeffs, input, taps) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn convolve_mono_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::dsp::stereo::convolve_mono_dual_avx512(coeffs0, coeffs1, input, taps)
            }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::dsp::gain::apply_gain_and_detect_clipping_mono_avx512(data, gain)
            }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_gain_and_detect_clipping_stereo(
            left: &mut [f32],
            right: &mut [f32],
            gain: f32,
        ) -> bool {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::dsp::gain::apply_gain_and_detect_clipping_stereo_avx512(
                    left, right, gain,
                )
            }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::gain::apply_gain_stereo_avx512(left, right, gain) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_gain(data: &mut [f32], gain: f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::gain::apply_gain_avx512(data, gain) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::gain::apply_ramp_avx512(data, start, step) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::gain::apply_ramp_stereo_avx512(left, right, start, step) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::gain::apply_dither_add_avx512(data, offset) }
        }
    };
}
