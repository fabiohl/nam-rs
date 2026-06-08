// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

macro_rules! impl_avx512_reduce {
    () => {
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::common::utility::horizontal_sum_avx512(ptr, N) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::stereo::compute_energy_stereo_avx512(l, r) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_energy(data: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::stereo::compute_energy_avx512(data) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::stereo::compute_max_diff_avx512(a, b) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::stereo::compute_peak_abs_stereo_avx512(left, right) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::dsp::stereo::compute_peak_abs_mono_avx512(data) }
        }
    };
}

macro_rules! impl_avx512vnni_reduce {
    () => {
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
            crate::math::common::utility::horizontal_sum_avx512(ptr, N)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_energy_stereo(l, r) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_energy(data: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_energy(data) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_max_diff(a, b) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_peak_abs_stereo(left, right) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_peak_abs_mono(data) }
        }
    };
}

macro_rules! impl_avx512vnni_bf16_reduce {
    () => {
        const IS_BF16: bool = true;

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
            crate::math::common::utility::horizontal_sum_avx512(ptr, N)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_energy_stereo(l, r) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_energy(data: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_energy(data) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_max_diff(a, b) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_peak_abs_stereo(left, right) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { Avx512Math::compute_peak_abs_mono(data) }
        }
    };
}

pub(super) use impl_avx512_reduce;
pub(super) use impl_avx512vnni_bf16_reduce;
pub(super) use impl_avx512vnni_reduce;
