// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::*;

#[test]
fn test_dot_product_avx2_fma() {
    let vec_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let vec_b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let vec_b_u16: Vec<u16> = vec_b
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();

    let result = unsafe { dot_product_avx2(&vec_a, &vec_b_u16) };

    // Expected = (1*2 + 2*2 ... + 8*2) + 9*2
    // 72 * 2 + 18 = 144 + 18 = 90
    let expected: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();

    // FMA math is accurate but compare with epsilon
    assert!(
        (result - expected).abs() < 1e-4,
        "Resultado divergente: esperado {}, obtido {}",
        expected,
        result
    );
}

#[test]
fn test_dot_product_avx512() {
    if crate::math::simd::SimdMathConfig::get().instruction_set
        >= crate::math::simd::SimdInstructionSet::Avx512
    {
        let vec_a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ];
        let vec_b = vec![
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ];
        let vec_b_u16: Vec<u16> = vec_b
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        let result = unsafe { dot_product_avx512(&vec_a, &vec_b_u16) };
        let expected: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (result - expected).abs() < 1e-4,
            "Resultado divergente: esperado {}, obtido {}",
            expected,
            result
        );
    }
}

/// Verifica que `set_daz_ftz` seta corretamente os bits DAZ (6) e FTZ (15) no MXCSR.
#[test]
fn test_set_daz_ftz() {
    unsafe {
        // Ler MXCSR atual e limpar DAZ+FTZ para verificar que a função os seta
        let mut before: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut before);
        let cleared = before & !0x8040;
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &cleared);

        set_daz_ftz();

        let mut after: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut after);
        assert!(
            (after & 0x8040) == 0x8040,
            "set_daz_ftz() não setou DAZ+FTZ: MXCSR=0x{:08X}",
            after
        );

        // Restaurar MXCSR original
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &before);
    }
}

#[test]
fn test_compute_energy_avx2() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let energy = unsafe { compute_energy_avx2(&data) };
    // (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    assert!((energy - 7.5).abs() < 1e-6);

    let data2 = vec![0.0; 16];
    let energy2 = unsafe { compute_energy_avx2(&data2) };
    assert_eq!(energy2, 0.0);
}

#[test]
fn test_compute_max_diff_avx2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.1, 1.9, 3.5, 3.8];
    let max_diff = unsafe { compute_max_diff_avx2(&a, &b) };
    // diffs: [0.1, 0.1, 0.5, 0.2] -> max = 0.5
    assert!((max_diff - 0.5).abs() < 1e-6);

    let a2 = vec![1.0; 8];
    let b2 = vec![1.0; 8];
    let max_diff2 = unsafe { compute_max_diff_avx2(&a2, &b2) };
    assert_eq!(max_diff2, 0.0);
}

#[test]
fn test_horizontal_sum() {
    fn test_n<const N: usize>(data_ptr: *const f32, expected: f32) {
        let res_avx2 = unsafe { Avx2Math::horizontal_sum::<N>(data_ptr) };
        assert!(
            (res_avx2 - expected).abs() < 1e-5,
            "AVX2 N={} failed: got {}, expected {}",
            N,
            res_avx2,
            expected
        );

        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            let res_avx512 = unsafe { Avx512Math::horizontal_sum::<N>(data_ptr) };
            assert!(
                (res_avx512 - expected).abs() < 1e-5,
                "AVX512 N={} failed: got {}, expected {}",
                N,
                res_avx512,
                expected
            );
        }
    }

    let data: Vec<f32> = (0..64).map(|i| i as f32 + 1.0).collect();

    test_n::<1>(data.as_ptr(), data[..1].iter().sum());
    test_n::<4>(data.as_ptr(), data[..4].iter().sum());
    test_n::<6>(data.as_ptr(), data[..6].iter().sum());
    test_n::<8>(data.as_ptr(), data[..8].iter().sum());
    test_n::<12>(data.as_ptr(), data[..12].iter().sum());
    test_n::<16>(data.as_ptr(), data[..16].iter().sum());
    test_n::<32>(data.as_ptr(), data[..32].iter().sum());
}

#[test]
fn test_accumulate_head() {
    fn test_backend<M: SimdMath>() {
        let mut dest = vec![1.0; 32];
        let src = vec![2.0; 32];
        unsafe { M::accumulate_head(&mut dest, &src) };
        for val in dest {
            assert!((val - 3.0).abs() < 1e-6);
        }

        // Test with odd length
        let mut dest2 = vec![1.0; 7];
        let src2 = vec![3.0; 7];
        unsafe { M::accumulate_head(&mut dest2, &src2) };
        for val in dest2 {
            assert!((val - 4.0).abs() < 1e-6);
        }
    }

    test_backend::<ScalarMath>();
    test_backend::<Avx2Math>();
    if std::is_x86_feature_detected!("avx512f") {
        test_backend::<Avx512Math>();
    }
}
