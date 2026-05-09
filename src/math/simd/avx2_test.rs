// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use super::super::{Avx2Math, SimdMath};
    use core::arch::x86_64::*;

    #[test]
    fn test_f32_to_bf16_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let src: Vec<f32> = (0..64).map(|i| i as f32 * 0.123).collect();
        let mut dest_simd = vec![0u16; 64];
        let mut dest_ref = vec![0u16; 64];

        unsafe {
            Avx2Math::f32_to_bf16(&src, &mut dest_simd);
            for i in 0..64 {
                dest_ref[i] = (src[i].to_bits() >> 16) as u16;
            }
        }

        assert_eq!(dest_simd, dest_ref);
    }

    #[test]
    fn test_store_bf16_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let vals = [
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
        ];
        let mut dest = [0u16; 8];
        unsafe {
            let v = _mm256_loadu_ps(vals.as_ptr());
            Avx2Math::store_bf16(dest.as_mut_ptr(), v);
        }
        for i in 0..8 {
            assert_eq!(dest[i], (vals[i].to_bits() >> 16) as u16);
        }
    }

    #[test]
    fn test_apply_gain_and_detect_clipping_stereo_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::math::simd::fallback::FallbackMath;

        let mut left_simd = [0.1, 0.5, 0.9, 1.2, -0.1, -0.5, -0.9, -1.2];
        let mut right_simd = [0.0, 0.2, 0.4, 0.6, -0.0, -0.2, -0.4, -0.6];
        let mut left_ref = left_simd;
        let mut right_ref = right_simd;
        let gain = 1.1;

        unsafe {
            let clipped_simd = Avx2Math::apply_gain_and_detect_clipping_stereo(
                &mut left_simd,
                &mut right_simd,
                gain,
            );
            let clipped_ref = FallbackMath::apply_gain_and_detect_clipping_stereo(
                &mut left_ref,
                &mut right_ref,
                gain,
            );

            assert_eq!(right_simd, right_ref);
        }
    }

    #[test]
    fn test_compute_energy_stereo_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use crate::math::simd::fallback::FallbackMath;

        let left = [0.1, 0.5, 0.9, 1.2, -0.1, -0.5, -0.9, -1.2, 0.5, 0.6, 0.7, 0.8];
        let right = [0.0, 0.2, 0.4, 0.6, -0.0, -0.2, -0.4, -0.6, 0.1, 0.1, 0.1, 0.1];

        unsafe {
            let res_simd = Avx2Math::compute_energy_stereo(&left, &right);
            let res_ref = FallbackMath::compute_energy_stereo(&left, &right);

            assert!((res_simd - res_ref).abs() < 1e-6);
        }
    }
}
