// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::ops::set_daz_ftz;
use super::traits::SimdMath;
use crate::math::common::{Avx2Math, Avx512Math};
use crate::math::dsp::stereo::{compute_energy_avx2, compute_max_diff_avx2};
use crate::math::gemm::dot::{dot_product_avx2, dot_product_avx512};

#[test]
fn test_dot_product_avx2_fma() {
    // Testa a operação de 'Produto Escalar' (multiplicar e somar tudo).
    // É o cálculo matemático mais frequente dentro das redes neurais.
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
    // Versão do teste de produto escalar para processadores ultra-modernos (AVX-512).
    if crate::math::common::SimdMathConfig::get().instruction_set
        >= crate::math::common::InstructionSet::Avx512
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
    // Verifica se o processador está configurado para ignorar números 'fantasmagoricamente'
    // pequenos (denormais). Isso evita lentidão extrema no processamento de áudio.
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
    // Testa o cálculo de energia (volume médio) usando aceleração AVX2.
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
    // Testa o cálculo da maior diferença entre dois sons usando AVX2.
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
    // Testa a soma de todos os números dentro de um 'pacote' SIMD (8 ou 16 números).
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
    // Testa a soma acumulada de dois blocos de som (usada para combinar resultados de camadas).
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

    // test_backend::<ScalarMath>(); // ScalarMath foi removido (Projeto foca em x86-64-v3+)
    test_backend::<Avx2Math>();
    if std::is_x86_feature_detected!("avx512f") {
        test_backend::<Avx512Math>();
    }
}

/// Verifica se a conversão de números de precisão total (f32) para o formato compacto (bfloat16)
/// mantém a paridade matemática rigorosa.
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

/// Garante que o salvamento dos dados compactos (bfloat16) na memória é feito
/// sem perdas ou corrupção de dados.
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
        let v = core::arch::x86_64::_mm256_loadu_ps(vals.as_ptr());
        Avx2Math::store_bf16(dest.as_mut_ptr(), v);
    }
    for i in 0..8 {
        assert_eq!(dest[i], (vals[i].to_bits() >> 16) as u16);
    }
}

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
        let v = core::arch::x86_64::_mm512_loadu_ps(vals.as_ptr());
        Avx512Math::store_bf16(dest.as_mut_ptr(), v);
    }
    for i in 0..16 {
        assert_eq!(dest[i], (vals[i].to_bits() >> 16) as u16);
    }
}

#[test]
fn test_compute_energy_parity() {
    let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let expected = unsafe { crate::math::common::compute_energy_fallback(&data) };

    let res_avx2 = unsafe { Avx2Math::compute_energy(&data) };
    assert!((res_avx2 - expected).abs() < 1e-6);

    if is_x86_feature_detected!("avx512f") {
        let res_avx512 = unsafe { Avx512Math::compute_energy(&data) };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_compute_energy_stereo_parity() {
    let l: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let r: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.01).collect();
    let expected = unsafe { crate::math::common::compute_energy_stereo_fallback(&l, &r) };

    let res_avx2 = unsafe { Avx2Math::compute_energy_stereo(&l, &r) };
    assert!((res_avx2 - expected).abs() < 1e-6);

    if is_x86_feature_detected!("avx512f") {
        let res_avx512 = unsafe { Avx512Math::compute_energy_stereo(&l, &r) };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_compute_max_diff_parity() {
    let a: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..100).map(|i| (i as f32 * 1.1) * 0.01).collect();
    let expected = unsafe { crate::math::common::compute_max_diff_fallback(&a, &b) };

    let res_avx2 = unsafe { Avx2Math::compute_max_diff(&a, &b) };
    assert!((res_avx2 - expected).abs() < 1e-6);

    if is_x86_feature_detected!("avx512f") {
        let res_avx512 = unsafe { Avx512Math::compute_max_diff(&a, &b) };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_convolve_mono_parity() {
    let coeffs = crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.01).collect());
    let input = crate::math::common::AlignedVec::from_vec((0..32).map(|i| (32 - i) as f32 * 0.05).collect());

    let expected = unsafe {
        crate::math::common::convolve_mono_fallback(coeffs.as_ptr(), input.as_ptr(), 32)
    };

    let res_avx2 = unsafe {
        Avx2Math::convolve_mono(coeffs.as_ptr(), input.as_ptr(), 32)
    };
    assert!((res_avx2 - expected).abs() < 1e-6);

    if is_x86_feature_detected!("avx512f") {
        let res_avx512 = unsafe {
            Avx512Math::convolve_mono(coeffs.as_ptr(), input.as_ptr(), 32)
        };
        assert!((res_avx512 - expected).abs() < 1e-6);
    }
}

#[test]
fn test_convolve_stereo_dual_parity() {
    let coeffs0 = crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.01).collect());
    let coeffs1 = crate::math::common::AlignedVec::from_vec((0..32).map(|i| (32 - i) as f32 * 0.007).collect());
    let input_l = crate::math::common::AlignedVec::from_vec((0..32).map(|i| i as f32 * 0.03).collect());
    let input_r = crate::math::common::AlignedVec::from_vec((0..32).map(|i| (32 - i) as f32 * 0.04).collect());

    let expected = unsafe {
        crate::math::common::convolve_stereo_dual_fallback(
            coeffs0.as_ptr(),
            coeffs1.as_ptr(),
            input_l.as_ptr(),
            input_r.as_ptr(),
            32,
        )
    };

    let res_avx2 = unsafe {
        Avx2Math::convolve_stereo_dual(
            coeffs0.as_ptr(),
            coeffs1.as_ptr(),
            input_l.as_ptr(),
            input_r.as_ptr(),
            32,
        )
    };
    assert!((res_avx2.0.0 - expected.0.0).abs() < 1e-5);
    assert!((res_avx2.0.1 - expected.0.1).abs() < 1e-5);
    assert!((res_avx2.1.0 - expected.1.0).abs() < 1e-5);
    assert!((res_avx2.1.1 - expected.1.1).abs() < 1e-5);

    if is_x86_feature_detected!("avx512f") {
        let res_avx512 = unsafe {
            Avx512Math::convolve_stereo_dual(
                coeffs0.as_ptr(),
                coeffs1.as_ptr(),
                input_l.as_ptr(),
                input_r.as_ptr(),
                32,
            )
        };
        assert!((res_avx512.0.0 - expected.0.0).abs() < 1e-5);
        assert!((res_avx512.0.1 - expected.0.1).abs() < 1e-5);
        assert!((res_avx512.1.0 - expected.1.0).abs() < 1e-5);
        assert!((res_avx512.1.1 - expected.1.1).abs() < 1e-5);
    }
}
