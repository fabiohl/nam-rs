// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use crate::math::common::{Avx512Math, SimdMath};
    use core::arch::x86_64::*;

    /// Garante que o salvamento de dados compactos (bfloat16) usando a largura total do
    /// AVX-512 (512 bits) seja feito sem perdas ou corrupção.
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
