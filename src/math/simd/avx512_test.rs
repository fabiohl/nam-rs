// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use crate::math::simd::{Avx512Math, SimdMath, f32_to_bf16_avx512};
    use core::arch::x86_64::*;

    #[test]
    fn test_f32_to_bf16_avx512_parity() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let src: Vec<f32> = (0..64).map(|i| i as f32 * 0.123).collect();
        let mut dest_simd = vec![0u16; 64];
        let mut dest_ref = vec![0u16; 64];

        unsafe {
            f32_to_bf16_avx512(&src, &mut dest_simd);
            for i in 0..64 {
                dest_ref[i] = (src[i].to_bits() >> 16) as u16;
            }
        }

        assert_eq!(dest_simd, dest_ref);
    }

    #[test]
    fn test_store_bf16_avx512() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        let vals: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
        let mut dest = [0u16; 16];
        unsafe {
            let v = _mm512_loadu_ps(vals.as_ptr());
            Avx512Math::store_bf16(dest.as_mut_ptr(), v);
        }
        for i in 0..16 {
            assert_eq!(dest[i], (vals[i].to_bits() >> 16) as u16);
        }
    }
}
