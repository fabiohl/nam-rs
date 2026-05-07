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

#[test]
fn test_simd_fastmath_relu() {
    let input: [f32; 8] = [-5.0, -1.0, -0.01, 0.0, 0.01, 1.0, 5.0, 10.0];
    let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
    let result_vector = unsafe { simd_relu_avx2(vector) };

    let mut result = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

    for i in 0..8 {
        let expected = if input[i] < 0.0 { 0.0 } else { input[i] };
        assert_eq!(result[i], expected, "Falha em ReLU para {}", input[i]);
    }
}

#[test]
fn test_simd_fastmath_prelu() {
    let input: [f32; 8] = [-5.0, -1.0, -0.01, 0.0, 0.01, 1.0, 5.0, 10.0];
    let alpha_raw: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
    let alpha_vector = unsafe { _mm256_loadu_ps(alpha_raw.as_ptr()) };
    let result_vector = unsafe { simd_prelu_avx2(vector, alpha_vector) };

    let mut result = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

    for i in 0..8 {
        let expected = if input[i] < 0.0 {
            input[i] * alpha_raw[i]
        } else {
            input[i]
        };
        assert!(
            (result[i] - expected).abs() < 1e-6,
            "Falha em PReLU para {}",
            input[i]
        );
    }
}

#[test]
fn test_simd_fastmath_softsign() {
    let input: [f32; 8] = [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0];
    let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
    let result_vector = unsafe { simd_softsign_avx2(vector) };

    let mut result = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

    for i in 0..8 {
        let expected = input[i] / (1.0 + input[i].abs());
        let error = (result[i] - expected).abs();
        assert!(
            error < 1e-6,
            "Falha em Softsign para {}. Esperado: {}, Obtido: {}",
            input[i],
            expected,
            result[i]
        );
    }
}

#[test]
fn test_simd_fastmath_silu() {
    let input: [f32; 8] = [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0];
    let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
    let result_vector = unsafe { simd_silu_avx2(vector) };

    let mut result = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

    let silu = |x: f32| x / (1.0 + (-x).exp());

    for i in 0..8 {
        let expected = silu(input[i]);
        let error = (result[i] - expected).abs();
        assert!(
            error < 1e-5,
            "Falha em SiLU para {}. Esperado: {}, Obtido: {}",
            input[i],
            expected,
            result[i]
        );
    }
}

#[test]
fn test_fastmath_slices() {
    let data = [
        -5.0, -1.0, 0.0, 1.0, 5.0, -2.0, 2.0, 0.5, -0.5, 3.0, -3.0, 4.0, -4.0, 10.0, -10.0, 0.1,
    ];
    let mut data_relu = data;
    let mut data_softsign = data;
    let mut data_silu = data;
    let mut data_prelu = data;

    unsafe {
        relu_slice_avx2(&mut data_relu);
        softsign_slice_avx2(&mut data_softsign);
        silu_slice_avx2(&mut data_silu);
        prelu_slice_avx2(&mut data_prelu, &[0.1]);
    }

    for i in 0..data.len() {
        let x = data[i];
        assert_eq!(data_relu[i], if x < 0.0 { 0.0 } else { x });
        assert!((data_softsign[i] - x / (1.0 + x.abs())).abs() < 1e-6);
        assert!((data_silu[i] - x / (1.0 + (-x).exp())).abs() < 1e-5);
        assert!((data_prelu[i] - if x < 0.0 { x * 0.1 } else { x }).abs() < 1e-6);
    }
}

#[test]
fn test_prelu_slice_periodic() {
    let mut data = [-1.0; 16];
    let slopes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    unsafe {
        prelu_slice_avx2(&mut data, &slopes);
    }
    for i in 0..16 {
        assert!((data[i] - (-slopes[i % 8])).abs() < 1e-6);
    }
}
#[test]
fn test_simd_tanh_sigmoid_dual_parity_avx2() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }
    unsafe {
        let xt = _mm256_set_ps(0.1, -0.5, 1.2, -2.3, 3.4, -4.5, 5.6, -6.7);
        let xs = _mm256_set_ps(-0.1, 0.5, -1.2, 2.3, -3.4, 4.5, -5.6, 6.7);

        let (t_fused, s_fused) = simd_tanh_sigmoid_dual_avx2(xt, xs);
        let t_ref = simd_tanh_avx2(xt);
        let s_ref = simd_sigmoid_avx2(xs);

        let mut res_t_fused = [0.0f32; 8];
        let mut res_s_fused = [0.0f32; 8];
        let mut res_t_ref = [0.0f32; 8];
        let mut res_s_ref = [0.0f32; 8];

        _mm256_storeu_ps(res_t_fused.as_mut_ptr(), t_fused);
        _mm256_storeu_ps(res_s_fused.as_mut_ptr(), s_fused);
        _mm256_storeu_ps(res_t_ref.as_mut_ptr(), t_ref);
        _mm256_storeu_ps(res_s_ref.as_mut_ptr(), s_ref);

        for i in 0..8 {
            assert!((res_t_fused[i] - res_t_ref[i]).abs() < 1e-6);
            assert!((res_s_fused[i] - res_s_ref[i]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_simd_tanh_sigmoid_dual_parity_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }
    unsafe {
        let xt = _mm512_set_ps(
            0.1, -0.5, 1.2, -2.3, 3.4, -4.5, 5.6, -6.7, 7.8, -8.9, 9.0, -10.1, 11.2, -12.3, 13.4,
            -14.5,
        );
        let xs = _mm512_set_ps(
            -0.1, 0.5, -1.2, 2.3, -3.4, 4.5, -5.6, 6.7, -7.8, 8.9, -9.0, 10.1, -11.2, 12.3, -13.4,
            14.5,
        );

        let (t_fused, s_fused) = simd_tanh_sigmoid_dual_avx512(xt, xs);
        let t_ref = simd_tanh_avx512(xt);
        let s_ref = simd_sigmoid_avx512(xs);

        let mut res_t_fused = [0.0f32; 16];
        let mut res_s_fused = [0.0f32; 16];
        let mut res_t_ref = [0.0f32; 16];
        let mut res_s_ref = [0.0f32; 16];

        _mm512_storeu_ps(res_t_fused.as_mut_ptr(), t_fused);
        _mm512_storeu_ps(res_s_fused.as_mut_ptr(), s_fused);
        _mm512_storeu_ps(res_t_ref.as_mut_ptr(), t_ref);
        _mm512_storeu_ps(res_s_ref.as_mut_ptr(), s_ref);

        for i in 0..16 {
            assert!((res_t_fused[i] - res_t_ref[i]).abs() < 1e-6);
            assert!((res_s_fused[i] - res_s_ref[i]).abs() < 1e-6);
        }
    }
}
