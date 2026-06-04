// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

macro_rules! impl_avx512_bf16 {
    () => {
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::common::ops::f32_to_bf16_avx512(src, dest) }
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                let v_i = _mm512_castps_si512(v);
                let v_shifted = _mm512_srli_epi32(v_i, 16);
                let packed = _mm512_cvtepi32_epi16(v_shifted);
                _mm256_storeu_si256(ptr as *mut __m256i, packed);
            }
        }
    };
}

macro_rules! impl_avx512vnni_bf16 {
    () => {
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
            Avx512Math::f32_to_bf16(src, dest)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
            Avx512Math::store_bf16(ptr, v)
        }
    };
}

macro_rules! impl_avx512vnni_bf16_bf16 {
    () => {
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
            Avx512Math::f32_to_bf16(src, dest)
        }

        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
            Avx512Math::store_bf16(ptr, v)
        }
    };
}

pub(super) use impl_avx512_bf16;
pub(super) use impl_avx512vnni_bf16;
pub(super) use impl_avx512vnni_bf16_bf16;
