// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use crate::math::simd::{Avx512Math, SimdMath};
    use core::arch::x86_64::*;

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

    #[test]
    fn test_gemv_4gate_avx512_parity() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
            return;
        }
        use crate::math::simd::avx512::gemv_4gate_avx512;
        use crate::math::simd::fallback::gemv_4gate_fallback;

        let in_len = 32;
        let out_len = 16;
        let in_frame: Vec<f32> = (0..in_len).map(|i| i as f32 * 0.5).collect();
        let w0: Vec<u16> = (0..in_len * out_len).map(|i| (i as u16) << 4).collect();
        let w1: Vec<u16> = w0.iter().map(|&x| x + 1).collect();
        let w2: Vec<u16> = w0.iter().map(|&x| x + 2).collect();
        let w3: Vec<u16> = w0.iter().map(|&x| x + 3).collect();
        let bias: Vec<f32> = (0..4 * out_len).map(|i| i as f32 * 0.1).collect();
        let mut out_simd = vec![0.0f32; 4 * out_len];
        let mut out_ref = vec![0.0f32; 4 * out_len];

        unsafe {
            gemv_4gate_avx512(&in_frame, &w0, &w1, &w2, &w3, &bias, &mut out_simd, true);
            gemv_4gate_fallback(&in_frame, &w0, &w1, &w2, &w3, &bias, &mut out_ref, true);
        }

        for i in 0..4 * out_len {
            assert!(
                (out_simd[i] - out_ref[i]).abs() < 1e-4,
                "At index {}: SIMD {} != REF {}",
                i,
                out_simd[i],
                out_ref[i]
            );
        }
    }

    #[test]
    fn test_gemv_4gate_bf16_avx512_parity() {
        if !is_x86_feature_detected!("avx512bf16") {
            return;
        }
        use crate::math::simd::avx512::gemv_4gate_bf16_avx512;
        use crate::math::simd::fallback::gemv_4gate_bf16_fallback;

        let in_len = 32;
        let out_len = 16;
        let in_frame: Vec<u16> = (0..in_len).map(|i| ((i as u32 + 1) << 10) as u16).collect();
        let w0: Vec<u16> = (0..in_len * out_len).map(|i| (i as u16) << 4).collect();
        let w1: Vec<u16> = w0.iter().map(|&x| x + 1).collect();
        let w2: Vec<u16> = w0.iter().map(|&x| x + 2).collect();
        let w3: Vec<u16> = w0.iter().map(|&x| x + 3).collect();
        let bias: Vec<f32> = (0..4 * out_len).map(|i| i as f32 * 0.1).collect();
        let mut out_simd = vec![0.0f32; 4 * out_len];
        let mut out_ref = vec![0.0f32; 4 * out_len];

        unsafe {
            gemv_4gate_bf16_avx512(&in_frame, &w0, &w1, &w2, &w3, &bias, &mut out_simd, true);
            gemv_4gate_bf16_fallback(&in_frame, &w0, &w1, &w2, &w3, &bias, &mut out_ref, true);
        }

        for i in 0..4 * out_len {
            assert!(
                (out_simd[i] - out_ref[i]).abs() < 1e-4,
                "At index {}: SIMD {} != REF {}",
                i,
                out_simd[i],
                out_ref[i]
            );
        }
    }
}
