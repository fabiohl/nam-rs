// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Testes baseados em propriedades (Property-Based Testing) para as intrínsecas
//! matemáticas críticas (AVX2 / AVX-512).
//!
//! # Objetivo
//! Varre milhões de limites aleatórios na Tangente Hiperbólica e Sigmoide, garantindo:
//! 1. Ausência de falhas computacionais (NaN / Inf).
//! 2. RMSE limite estrito contra `f32::tanh` original para a camada neural.

use nam_rs::math::fastmath;
use nam_rs::math::simd::{dot_product_avx2, dot_product_avx512};
use proptest::prelude::*;

use core::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

prop_compose! {
    /// Gera um array de 8 floats dentro do range espectral típico das camadas ocultas.
    /// Range estendido para [-10.0, 10.0] simulando surtos de ativação transientes.
    fn avx2_input_array()(
        a in -10.0f32..10.0f32,
        b in -10.0f32..10.0f32,
        c in -10.0f32..10.0f32,
        d in -10.0f32..10.0f32,
        e in -10.0f32..10.0f32,
        f in -10.0f32..10.0f32,
        g in -10.0f32..10.0f32,
        h in -10.0f32..10.0f32,
    ) -> [f32; 8] {
        [a, b, c, d, e, f, g, h]
    }

}

prop_compose! {
    /// Gera pares de vetores de tamanho idêntico (1 a 512)
    fn vec_pair_strategy()(len in 1..=512usize)
                          (a in prop::collection::vec(-10.0f32..10.0f32, len..=len),
                           b in prop::collection::vec(-10.0f32..10.0f32, len..=len)) -> (Vec<f32>, Vec<f32>) {
        (a, b)
    }
}

proptest! {
    // Configuração para rodar 10.000 iterações (80.000 floats varridos).
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn prop_simd_tanh_avx2_rmse(input in avx2_input_array()) {
        let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
        let result_vector = unsafe { fastmath::simd_tanh(vector) };

        let mut result = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

        for i in 0..8 {
            let expected = input[i].tanh();
            let actual = result[i];
            let error = (expected - actual).abs();

            // Nenhuma aproximação polinomial pode cuspir NaN.
            assert!(actual.is_finite(), "Saída NaN/Inf gerada pelo polinômio!");

            // O polynomial de grau 5 do Mike Oliphant tem uma incerteza de ~5e-3 nos extremos.
            assert!(
                error <= 5e-3,
                "Falha matemática MS/E. Entrada: {}. Esperado: {}, Obtido: {}, Delta (Erro): {}",
                input[i],
                expected,
                actual,
                error
            );
        }
    }

    #[test]
    fn prop_simd_sigmoid_avx2_rmse(input in avx2_input_array()) {
        let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
        let result_vector = unsafe { fastmath::simd_sigmoid(vector) };

        let mut result = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

        let std_sigmoid = |val: f32| -> f32 { 0.5 * (1.0 + (val * 0.5).tanh()) };

        for i in 0..8 {
            let expected = std_sigmoid(input[i]);
            let actual = result[i];
            let error = (expected - actual).abs();

            assert!(actual.is_finite(), "Sigmoid gerou NaN/Inf!");

            assert!(
                error <= 5e-3,
                "Falha ao validar FastMath Sigmoid em {}. Esperado: {}, Obtido: {}, Delta: {}",
                input[i],
                expected,
                actual,
                error
            );
        }
    }
    #[test]
    fn prop_dot_product_avx2_vs_scalar((vec_a, vec_b) in vec_pair_strategy()) {
        let simd_result = unsafe { dot_product_avx2(&vec_a, &vec_b) };

        // Usamos f64 como "ground truth" porque a acumulação f32 iterativa perde precisão.
        // O SIMD usa FMA que possui precisão interna maior antes de arredondar.
        let scalar_result: f64 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum();
        let l1_norm: f64 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| ((x as f64) * (y as f64)).abs()).sum();

        let error = (simd_result as f64 - scalar_result).abs();
        let threshold = 1e-5 * l1_norm.max(1.0); // Erro escalado pelo L1 norm evita falso positivo por cancelamento

        assert!(
            error <= threshold,
            "Falha matemática no dot_product_avx2! SIMD: {}, Escalar (f64): {}, Erro: {}",
            simd_result,
            scalar_result,
            error
        );
    }

    #[test]
    fn prop_dot_product_avx512_vs_scalar((vec_a, vec_b) in vec_pair_strategy()) {
        if std::is_x86_feature_detected!("avx512f") {
            let simd_result = unsafe { dot_product_avx512(&vec_a, &vec_b) };

            let scalar_result: f64 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum();

            let error = (simd_result as f64 - scalar_result).abs();
            let threshold = (1e-5 * scalar_result.abs()).max(1e-5); // Tolerância para cancelamento catastrófico

            assert!(
                error <= threshold,
                "Falha matemática no dot_product_avx512! SIMD: {}, Escalar (f64): {}, Erro: {}",
                simd_result,
                scalar_result,
                error
            );
        }
    }
}
