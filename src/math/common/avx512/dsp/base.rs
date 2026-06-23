// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

macro_rules! impl_avx512_dsp {
    () => {
        #[inline(always)]
        // SAFETY: coeffs is a 64-byte-aligned raw pointer to taps valid f32 elements;
        // input_l and input_r are valid raw pointers to taps valid f32 elements;
        // CPU supports AVX-512F (verified by dispatch). Coeffs use aligned 512-bit loads.
        unsafe fn convolve_stereo(
            coeffs: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: coeffs, input_l, input_r satisfy function invariants (64-byte alignment
            // for coeffs, taps valid elements at each pointer).
            unsafe {
                crate::math::dsp::stereo::convolve_stereo_avx512(coeffs, input_l, input_r, taps)
            }
        }

        #[inline(always)]
        // SAFETY: coeffs0, coeffs1 are 64-byte-aligned pointers to taps valid f32 elements;
        // input_l, input_r are valid pointers to taps valid f32 elements;
        // CPU supports AVX-512F (verified by dispatch).
        unsafe fn convolve_stereo_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> ((f32, f32), (f32, f32)) {
            // SAFETY: all raw pointers satisfy function invariants (aligned coeffs, taps elements).
            unsafe {
                crate::math::dsp::stereo::convolve_stereo_dual_avx512(
                    coeffs0, coeffs1, input_l, input_r, taps,
                )
            }
        }

        #[inline(always)]
        // SAFETY: coeffs is a 64-byte-aligned pointer to taps valid f32 elements;
        // input is a valid pointer to taps valid f32 elements; CPU supports AVX-512F.
        unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
            // SAFETY: coeffs and input satisfy function invariants.
            unsafe { crate::math::dsp::stereo::convolve_mono_avx512(coeffs, input, taps) }
        }

        #[inline(always)]
        // SAFETY: coeffs0, coeffs1 are 64-byte-aligned pointers to taps valid f32 elements;
        // input is a valid pointer to taps valid f32 elements; CPU supports AVX-512F.
        unsafe fn convolve_mono_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: all raw pointers satisfy function invariants.
            unsafe {
                crate::math::dsp::stereo::convolve_mono_dual_avx512(coeffs0, coeffs1, input, taps)
            }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain is a finite f32;
        // CPU supports AVX-512F (verified by dispatch). Kernel uses unaligned loads/stores.
        unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
            // SAFETY: data and gain satisfy function invariants.
            unsafe {
                crate::math::dsp::gain::apply_gain_and_detect_clipping_mono_avx512(data, gain)
            }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // gain is a finite f32; CPU supports AVX-512F.
        unsafe fn apply_gain_and_detect_clipping_stereo(
            left: &mut [f32],
            right: &mut [f32],
            gain: f32,
        ) -> bool {
            // SAFETY: left, right, and gain satisfy function invariants.
            unsafe {
                crate::math::dsp::gain::apply_gain_and_detect_clipping_stereo_avx512(
                    left, right, gain,
                )
            }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // gain is a finite f32; CPU supports AVX-512F.
        unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
            // SAFETY: left, right, and gain satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_stereo_avx512(left, right, gain) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain is a finite f32;
        // CPU supports AVX-512F (verified by dispatch).
        unsafe fn apply_gain(data: &mut [f32], gain: f32) {
            // SAFETY: data and gain satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_avx512(data, gain) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; start and step are finite f32 values;
        // CPU supports AVX-512F.
        unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
            // SAFETY: data, start, and step satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_ramp_avx512(data, start, step) }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // start and step are finite f32 values; CPU supports AVX-512F.
        unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
            // SAFETY: left, right, start, and step satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_ramp_stereo_avx512(left, right, start, step) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; offset is a finite f32;
        // CPU supports AVX-512F.
        unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
            // SAFETY: data and offset satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_dither_add_avx512(data, offset) }
        }

        #[inline(always)]
        // SAFETY: out and pending are valid f32 slices of equal length; t is a finite f32;
        // CPU supports AVX-512F.
        unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32) {
            // SAFETY: out, pending, and t satisfy function invariants.
            unsafe { crate::math::dsp::gain::crossfade_blend_mono_avx512(out, pending, t) }
        }
    };
}
