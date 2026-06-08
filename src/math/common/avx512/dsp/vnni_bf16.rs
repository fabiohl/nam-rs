// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

macro_rules! impl_avx512vnni_bf16_dsp {
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
            unsafe { Avx512Math::convolve_stereo(coeffs, input_l, input_r, taps) }
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
            unsafe { Avx512Math::convolve_stereo_dual(coeffs0, coeffs1, input_l, input_r, taps) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::convolve_mono(coeffs, input, taps) }
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
            unsafe { Avx512Math::convolve_mono_dual(coeffs0, coeffs1, input, taps) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
            Avx512Math::apply_gain_and_detect_clipping_mono(data, gain)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_gain_and_detect_clipping_stereo(
            left: &mut [f32],
            right: &mut [f32],
            gain: f32,
        ) -> bool {
            Avx512Math::apply_gain_and_detect_clipping_stereo(left, right, gain)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
            Avx512Math::apply_gain_stereo(left, right, gain)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_gain(data: &mut [f32], gain: f32) {
            crate::math::dsp::gain::apply_gain_avx512(data, gain)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
            Avx512Math::apply_ramp(data, start, step)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
            Avx512Math::apply_ramp_stereo(left, right, start, step)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
            head1: &[f32],
            head2: &[f32],
            output: &mut [f32],
            scale: f32,
        ) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::wavenet::head::batch_wavenet_head_sum_avx512::<HEAD>(
                    head1, head2, output, scale,
                )
            }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn batch_wavenet_head_sum_dyn(
            head1: &[f32],
            head2: &[f32],
            output: &mut [f32],
            head: usize,
            scale: f32,
        ) {
            Avx512Math::batch_wavenet_head_sum_dyn(head1, head2, output, head, scale)
        }
    };
}
