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
    /// Estratégia: Gera um array de 8 floats dentro de um intervalo espectral amplo.
    /// O range [-10.0, 10.0] é suficiente para cobrir tanto a zona linear
    /// quanto as zonas de saturação extrema da Tanh e Sigmoid, onde erros de
    /// aproximação polinomial costumam aparecer.
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
    /// Estratégia: Gera pares de vetores de tamanho idêntico para testar o Dot Product.
    /// O tamanho varia de 1 a 512 para garantir que o código lide corretamente
    /// com vetores que não são múltiplos exatos do tamanho do registrador SIMD (trailing elements).
    fn vec_pair_strategy()(len in 1..=512usize)
                          (a in prop::collection::vec(-10.0f32..10.0f32, len..=len),
                           b in prop::collection::vec(-10.0f32..10.0f32, len..=len)) -> (Vec<f32>, Vec<f32>) {
        (a, b)
    }
}

proptest! {
    // Configuração para rodar 10.000 iterações (80.000 floats varridos).
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    /// Valida a precisão da aproximação de Tangente Hiperbólica via AVX2.
    /// Utiliza aproximações Minimax/Padé que priorizam throughput em vez de precisão de 64-bit.
    #[test]
    fn prop_simd_tanh_avx2_rmse(input in avx2_input_array()) {
        // Carregamento não-alinhado seguro para buffers genéricos
        let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
        let result_vector = unsafe { fastmath::simd_tanh(vector) };

        let mut result = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

        for i in 0..8 {
            let expected = input[i].tanh();
            let actual = result[i];
            let error = (expected - actual).abs();

            // Garantia fundamental: nenhuma ativação pode gerar valores inválidos
            assert!(actual.is_finite(), "Saída NaN/Inf gerada pelo polinômio FastMath Tanh!");

            // O threshold de 5e-3 é o limite superior do erro Minimax de grau 5 usado.
            // Para modelagem de áudio (guitarra), esse erro é inaudível (-60dB+ de precisão).
            assert!(
                error <= 5e-3,
                "Falha na precisão Tanh. Entrada: {}. Esperado: {}, Obtido: {}, Delta: {}",
                input[i],
                expected,
                actual,
                error
            );
        }
    }

    /// Valida a precisão da implementação FastMath Sigmoid otimizada para AVX2.
    ///
    /// O teste usa proptest para varrer o domínio espectral e garantir que a
    /// aproximação da função logística não apresente instabilidades ou erros
    /// grosseiros que afetariam o ganho dos gates (onde a sigmoid é comum).
    #[test]
    fn prop_simd_sigmoid_avx2_rmse(input in avx2_input_array()) {
        // Carregamento de 8 floats (256-bit YMM register)
        let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
        // Execução da sigmoid via kernels intrínsecos fastmath
        let result_vector = unsafe { fastmath::simd_sigmoid(vector) };

        let mut result = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

        // Referência escalar de alta fidelidade
        let std_sigmoid = |val: f32| -> f32 { 1.0 / (1.0 + (-val).exp()) };

        for i in 0..8 {
            let expected = std_sigmoid(input[i]);
            let actual = result[i];
            let error = (expected - actual).abs();

            assert!(actual.is_finite(), "Sigmoid gerou NaN/Inf indesejado!");

            // O threshold de 2e-5 reflete a alta precisão da implementação baseada
            // em exp_ps(avx2) ajustada. Erros nesta magnitude são negligenciáveis
            // para controle de dinâmica e modulação de gating.
            assert!(
                error <= 2e-5,
                "Falha ao validar FastMath Sigmoid em {}. Esperado: {}, Obtido: {}, Delta: {}",
                input[i],
                expected,
                actual,
                error
            );
        }
    }
    /// Valida o produto escalar AVX2 com pesos em `f16` (F16C).
    /// Compara contra uma implementação escalar em `f64` (Ground Truth).
    #[test]
    fn prop_dot_product_avx2_vs_scalar((vec_a, vec_b) in vec_pair_strategy()) {
        // Conversão de pesos f32 -> f16 (bits u16)
        let vec_b_u16: Vec<u16> = vec_b.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let simd_result = unsafe { dot_product_avx2(&vec_a, &vec_b_u16) };

        // Ground Truth em f64 para evitar que o erro de acumulação do teste mascare o erro do código
        let scalar_result: f64 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum();

        // Norma L1 para escalar o erro tolerável em vetores grandes
        let l1_norm: f64 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| ((x as f64) * (y as f64)).abs()).sum();

        let error = (simd_result as f64 - scalar_result).abs();

        // 1e-2 de erro é aceitável para a precisão reduzida do f16 (Half precision)
        let threshold = 1e-2 * l1_norm.max(1.0);

        assert!(
            error <= threshold,
            "Produto escalar AVX2 divergiu do escalar! SIMD: {}, Ground Truth (f64): {}, Erro: {}",
            simd_result,
            scalar_result,
            error
        );
    }

    /// Valida o produto escalar AVX-512 (quando disponível no hardware).
    #[test]
    fn prop_dot_product_avx512_vs_scalar((vec_a, vec_b) in vec_pair_strategy()) {
        // Verificação obrigatória em runtime para evitar SIGILL em CPUs antigas
        if std::is_x86_feature_detected!("avx512f") {
            let vec_b_u16: Vec<u16> = vec_b.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
            let simd_result = unsafe { dot_product_avx512(&vec_a, &vec_b_u16) };

            let scalar_result: f64 = vec_a.iter().zip(vec_b.iter()).map(|(&x, &y)| (x as f64) * (y as f64)).sum();

            let error = (simd_result as f64 - scalar_result).abs();
            let threshold = (1e-2 * scalar_result.abs()).max(1e-2);

            assert!(
                error <= threshold,
                "Produto escalar AVX-512 divergiu! SIMD: {}, Ground Truth (f64): {}, Erro: {}",
                simd_result,
                scalar_result,
                error
            );
        }
    }
}
