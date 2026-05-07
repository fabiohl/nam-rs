// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use crate::math::simd::{Avx2Math, SimdMath, f32_to_bf16_avx2};
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
            f32_to_bf16_avx2(&src, &mut dest_simd);
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
}
