// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::*;

#[test]
fn test_simd_fastmath_tanh_mse() {
    let input: [f32; 8] = [-4.0, -2.5, -0.75, 0.0, 0.8, 1.2, 3.5, 6.0];
    let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
    let result_vector = unsafe { simd_tanh(vector) };

    let mut result = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

    for i in 0..8 {
        let expected = input[i].tanh();
        let actual = result[i];
        let error = (expected - actual).abs();

        // Tolerância estrita (threshold) ajustada para polinomios < 5 na base original
        assert!(
            error < 1e-4,
            "Falha matemática MS/E. Entrada: {}. Esperado: {}, Obtido: {}, Delta (Erro): {}",
            input[i],
            expected,
            actual,
            error
        );
    }
}

#[test]
fn test_simd_fastmath_sigmoid_mse() {
    let input: [f32; 8] = [-5.0, -1.0, -0.33, 0.0, 0.22, 1.1, 2.8, 4.0];
    let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
    let result_vector = unsafe { simd_sigmoid(vector) };

    let mut result = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

    let std_sigmoid = |val: f32| -> f32 { 1.0 / (1.0 + (-val).exp()) };

    for i in 0..8 {
        let expected = std_sigmoid(input[i]);
        let actual = result[i];
        let error = (expected - actual).abs();

        assert!(
            error < 1e-4,
            "Falha ao validar FastMath Sigmoid em {}. Esperado: {}, Obtido: {}, Delta (Erro): {}",
            input[i],
            expected,
            actual,
            error
        );
    }
}

#[test]
fn test_simd_fastmath_tanh_avx512_mse() {
    if matches!(
        crate::math::simd::SimdMathConfig::get().instruction_set,
        crate::math::simd::InstructionSet::Avx512 | crate::math::simd::InstructionSet::Avx512Vnni
    ) {
        let input: [f32; 16] = [
            -4.0, -2.5, -0.75, 0.0, 0.8, 1.2, 3.5, 6.0, -3.2, -1.1, -0.25, 0.1, 0.5, 2.2, 4.5, 8.0,
        ];
        let vector = unsafe { _mm512_loadu_ps(input.as_ptr()) };
        let result_vector = unsafe { simd_tanh_avx512(vector) };

        let mut result = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(result.as_mut_ptr(), result_vector) };

        for i in 0..16 {
            let expected = input[i].tanh();
            let actual = result[i];
            let error = (expected - actual).abs();

            assert!(
                error < 1e-4,
                "Falha matemática MS/E AVX-512. Entrada: {}. Esperado: {}, Obtido: {}, Delta (Erro): {}",
                input[i],
                expected,
                actual,
                error
            );
        }
    }
}

#[test]
fn test_simd_fastmath_sigmoid_avx512_mse() {
    if matches!(
        crate::math::simd::SimdMathConfig::get().instruction_set,
        crate::math::simd::InstructionSet::Avx512 | crate::math::simd::InstructionSet::Avx512Vnni
    ) {
        let input: [f32; 16] = [
            -5.0, -1.0, -0.33, 0.0, 0.22, 1.1, 2.8, 4.0, -4.2, -2.1, -0.15, 0.5, 0.88, 1.9, 3.2,
            7.0,
        ];
        let vector = unsafe { _mm512_loadu_ps(input.as_ptr()) };
        let result_vector = unsafe { simd_sigmoid_avx512(vector) };

        let mut result = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(result.as_mut_ptr(), result_vector) };

        let std_sigmoid = |val: f32| -> f32 { 1.0 / (1.0 + (-val).exp()) };

        for i in 0..16 {
            let expected = std_sigmoid(input[i]);
            let actual = result[i];
            let error = (expected - actual).abs();

            assert!(
                error < 2e-5,
                "Falha ao validar FastMath Sigmoid AVX-512 em {}. Esperado: {}, Obtido: {}, Delta: {}",
                input[i],
                expected,
                actual,
                error
            );
        }
    }
}

#[test]
fn test_simd_fastmath_tanh_extremes() {
    // Valores que anteriormente causariam NaN devido a overflow em p(x)^2
    let input: [f32; 8] = [
        2000.0,
        5000.0,
        10000.0,
        1e20,
        -2000.0,
        -1e10,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
    let result_vector = unsafe { simd_tanh(vector) };

    let mut result = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

    for i in 0..8 {
        let actual = result[i];
        assert!(
            !actual.is_nan(),
            "Resultado NaN para entrada extrema {}",
            input[i]
        );

        if input[i] > 0.0 {
            assert!(
                actual > 0.99,
                "Falha na saturação positiva para {}: {}",
                input[i],
                actual
            );
            assert!(
                actual <= 1.0001,
                "Saturação excedeu limite superior para {}: {}",
                input[i],
                actual
            );
        } else {
            assert!(
                actual < -0.99,
                "Falha na saturação negativa para {}: {}",
                input[i],
                actual
            );
            assert!(
                actual >= -1.0001,
                "Saturação excedeu limite inferior para {}: {}",
                input[i],
                actual
            );
        }
    }
}
