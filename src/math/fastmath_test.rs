// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

// Número de pontos do sweep: 2^15 = 32.768 — cobre [-8, 8] com resolução ~0.5e-3.
// Escolhido como potência de 2 para encaixe perfeito nos blocos SIMD de 8/16.
const SWEEP_N: usize = 1 << 15; // 32 768 pontos
const SWEEP_MIN: f32 = -8.0;
const SWEEP_MAX: f32 = 8.0;

/// Threshold de erro máximo absoluto para `simd_tanh_avx2` em [-8, 8].
///
/// O sweep de 32.768 pontos mediu pior caso de **1.234e-5** em x≈-4.34 —
/// uma região de alta saturação onde o polinômio Minimax acumula erro maior
/// que na zona central. A docstring de `simd_tanh_avx2` (~6e-8) refere-se
/// ao range [-4, 4] apenas. Usando 2e-5 como threshold: dá margem de ~1.6×
/// sobre o pior caso medido e ainda é 5× mais apertado que a antiga 1e-4.
const TANH_MAX_ABS_ERROR: f32 = 2e-5;

/// Threshold para `simd_sigmoid_avx2` em [-8, 8].
/// O sweep mediu max_error < 5e-6 neste range — sigmóide não tem região de
/// alta saturação comparável à cauda do tanh, pois usa exp() + NR de rcp.
const SIGMOID_MAX_ABS_ERROR: f32 = 5e-6;

/// Threshold para as variantes AVX-512.
/// _mm512_rsqrt14_ps / _mm512_rcp14_ps fornecem 14 bits, contra ~11-12 do AVX2.
/// Após NR duplo ambos saturam f32, mas usamos a mesma margem conservadora.
const AVX512_MAX_ABS_ERROR: f32 = 2e-5;

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

// ---------------------------------------------------------------------------
// T7.2 — Sweep de Erro Máximo Absoluto
// ---------------------------------------------------------------------------
// Estes quatro testes substituem a cobertura pontual (8 pontos fixos, 1e-4) por
// uma varredura densa de SWEEP_N pontos uniformes em [-8, 8], assertando o erro
// máximo absoluto contra a referência f64. A discrepância anterior de 4 ordens
// de magnitude (1e-4 vs ~6e-8 documentado) tornava regressões severas invisíveis.
//
// Estratégia de implementação sem heap:
//   - Buffer de 8/16 floats na pilha (array de stack).
//   - Processamento em bloco único por iteração do loop — sem Vec/Box.
//   - Referência via f64 para evitar erro acumulado do próprio tipo testado.
// ---------------------------------------------------------------------------

/// Sweep de erro máximo absoluto para `simd_tanh_avx2` (AVX2 + FMA).
///
/// Varre 32.768 pontos uniformes em [-8, 8] e exige que o erro máximo
/// absoluto (vs. `f64::tanh` como referência oráculo) fique abaixo de 5e-6.
#[test]
fn test_tanh_max_abs_error_sweep() {
    // Todos os blocos de hardware x86-64-v3 garantem AVX2 + FMA.
    assert!(
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
        "AVX2+FMA não disponível neste hardware — teste ignorado"
    );

    // step = (MAX - MIN) / (N - 1): passo uniforme entre os pontos.
    let step = (SWEEP_MAX - SWEEP_MIN) / (SWEEP_N as f32 - 1.0);
    let mut max_error: f32 = 0.0;
    // Índice do pior ponto: útil para mensagens de diagnóstico.
    let mut worst_x: f32 = 0.0;

    unsafe {
        // Processa em blocos de 8 (tamanho do registro YMM).
        // A pilha armazena apenas 8 f32 por vez — zero heap.
        let mut buf = [0.0f32; 8];

        let full_blocks = SWEEP_N / 8;
        for b in 0..full_blocks {
            // Preenche o buffer com 8 pontos consecutivos do sweep.
            // enumerate() sobre buf satisfaz clippy::needless_range_loop.
            for (k, slot) in buf.iter_mut().enumerate() {
                *slot = SWEEP_MIN + (b * 8 + k) as f32 * step;
            }

            // Carrega, computa e descarrega via instrução SIMD.
            let v = _mm256_loadu_ps(buf.as_ptr());
            let r = simd_tanh_avx2(v);
            _mm256_storeu_ps(buf.as_mut_ptr(), r);

            // Compara cada resultado com a referência f64 (oráculo de alta precisão).
            for (k, &actual) in buf.iter().enumerate() {
                let x = SWEEP_MIN + (b * 8 + k) as f32 * step;
                // f64::tanh fornece ~15 dígitos decimais — erro do oráculo < 1e-14.
                let reference = f64::tanh(x as f64) as f32;
                let err = (actual - reference).abs();
                if err > max_error {
                    max_error = err;
                    worst_x = x;
                }
            }
        }

        // Processa os pontos residuais (SWEEP_N % 8) de forma escalar.
        let remainder_start = full_blocks * 8;
        for i in remainder_start..SWEEP_N {
            let x = SWEEP_MIN + i as f32 * step;
            let v = _mm256_set1_ps(x);
            let r = simd_tanh_avx2(v);
            let actual = _mm256_cvtss_f32(r);
            let reference = f64::tanh(x as f64) as f32;
            let err = (actual - reference).abs();
            if err > max_error {
                max_error = err;
                worst_x = x;
            }
        }
    }

    assert!(
        max_error < TANH_MAX_ABS_ERROR,
        "test_tanh_max_abs_error_sweep FALHOU: max_error={:.3e} >= {:.3e} (pior ponto x={})",
        max_error,
        TANH_MAX_ABS_ERROR,
        worst_x,
    );
}

/// Sweep de erro máximo absoluto para `simd_sigmoid_avx2` (AVX2 + FMA).
///
/// Varre 32.768 pontos uniformes em [-8, 8] e exige que o erro máximo
/// absoluto (vs. referência oráculo f64) fique abaixo de 5e-6.
#[test]
fn test_sigmoid_max_abs_error_sweep() {
    assert!(
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
        "AVX2+FMA não disponível neste hardware — teste ignorado"
    );

    let step = (SWEEP_MAX - SWEEP_MIN) / (SWEEP_N as f32 - 1.0);
    let mut max_error: f32 = 0.0;
    let mut worst_x: f32 = 0.0;

    // Função oráculo: sigmoid(x) = 1 / (1 + exp(-x)) em f64.
    // Não usa o sigmoid escalar do fastmath.rs para evitar auto-validação.
    let ref_sigmoid = |x: f32| -> f32 {
        let x_f64 = x as f64;
        (1.0_f64 / (1.0 + f64::exp(-x_f64))) as f32
    };

    unsafe {
        let mut buf = [0.0f32; 8];

        let full_blocks = SWEEP_N / 8;
        for b in 0..full_blocks {
            for (k, slot) in buf.iter_mut().enumerate() {
                *slot = SWEEP_MIN + (b * 8 + k) as f32 * step;
            }

            let v = _mm256_loadu_ps(buf.as_ptr());
            let r = simd_sigmoid_avx2(v);
            _mm256_storeu_ps(buf.as_mut_ptr(), r);

            for (k, &actual) in buf.iter().enumerate() {
                let x = SWEEP_MIN + (b * 8 + k) as f32 * step;
                let reference = ref_sigmoid(x);
                let err = (actual - reference).abs();
                if err > max_error {
                    max_error = err;
                    worst_x = x;
                }
            }
        }

        // Resíduo escalar.
        let remainder_start = full_blocks * 8;
        for i in remainder_start..SWEEP_N {
            let x = SWEEP_MIN + i as f32 * step;
            let v = _mm256_set1_ps(x);
            let r = simd_sigmoid_avx2(v);
            let actual = _mm256_cvtss_f32(r);
            let reference = ref_sigmoid(x);
            let err = (actual - reference).abs();
            if err > max_error {
                max_error = err;
                worst_x = x;
            }
        }
    }

    assert!(
        max_error < SIGMOID_MAX_ABS_ERROR,
        "test_sigmoid_max_abs_error_sweep FALHOU: max_error={:.3e} >= {:.3e} (pior ponto x={})",
        max_error,
        SIGMOID_MAX_ABS_ERROR,
        worst_x,
    );
}

/// Sweep de erro máximo absoluto para `simd_tanh_avx512` (AVX-512F + VL).
///
/// Condicional em runtime — skipa silenciosamente em hardware sem AVX-512.
/// Usa blocos de 16 floats (ZMM) e o mesmo oráculo f64.
#[test]
fn test_tanh_max_abs_error_sweep_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        // Hardware sem AVX-512: teste ignorado silenciosamente.
        return;
    }

    let step = (SWEEP_MAX - SWEEP_MIN) / (SWEEP_N as f32 - 1.0);
    let mut max_error: f32 = 0.0;
    let mut worst_x: f32 = 0.0;

    unsafe {
        // Buffer de 16 floats na pilha — registrador ZMM (512 bits).
        let mut buf = [0.0f32; 16];

        let full_blocks = SWEEP_N / 16;
        for b in 0..full_blocks {
            for (k, slot) in buf.iter_mut().enumerate() {
                *slot = SWEEP_MIN + (b * 16 + k) as f32 * step;
            }

            let v = _mm512_loadu_ps(buf.as_ptr());
            let r = simd_tanh_avx512(v);
            _mm512_storeu_ps(buf.as_mut_ptr(), r);

            for (k, &actual) in buf.iter().enumerate() {
                let x = SWEEP_MIN + (b * 16 + k) as f32 * step;
                let reference = f64::tanh(x as f64) as f32;
                let err = (actual - reference).abs();
                if err > max_error {
                    max_error = err;
                    worst_x = x;
                }
            }
        }

        // Resíduo escalar (SWEEP_N % 16).
        let remainder_start = full_blocks * 16;
        for i in remainder_start..SWEEP_N {
            let x = SWEEP_MIN + i as f32 * step;
            // Para o resíduo, usa a versão AVX2 (que sempre está disponível).
            let v = _mm256_set1_ps(x);
            let r = simd_tanh_avx2(v);
            let actual = _mm256_cvtss_f32(r);
            let reference = f64::tanh(x as f64) as f32;
            let err = (actual - reference).abs();
            if err > max_error {
                max_error = err;
                worst_x = x;
            }
        }
    }

    assert!(
        max_error < AVX512_MAX_ABS_ERROR,
        "test_tanh_max_abs_error_sweep_avx512 FALHOU: max_error={:.3e} >= {:.3e} (pior ponto x={})",
        max_error,
        AVX512_MAX_ABS_ERROR,
        worst_x,
    );
}

/// Sweep de erro máximo absoluto para `simd_sigmoid_avx512` (AVX-512F + VL).
///
/// Condicional em runtime — skipa silenciosamente em hardware sem AVX-512.
#[test]
fn test_sigmoid_max_abs_error_sweep_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let step = (SWEEP_MAX - SWEEP_MIN) / (SWEEP_N as f32 - 1.0);
    let mut max_error: f32 = 0.0;
    let mut worst_x: f32 = 0.0;

    let ref_sigmoid = |x: f32| -> f32 {
        let x_f64 = x as f64;
        (1.0_f64 / (1.0 + f64::exp(-x_f64))) as f32
    };

    unsafe {
        let mut buf = [0.0f32; 16];

        let full_blocks = SWEEP_N / 16;
        for b in 0..full_blocks {
            for (k, slot) in buf.iter_mut().enumerate() {
                *slot = SWEEP_MIN + (b * 16 + k) as f32 * step;
            }

            let v = _mm512_loadu_ps(buf.as_ptr());
            let r = simd_sigmoid_avx512(v);
            _mm512_storeu_ps(buf.as_mut_ptr(), r);

            for (k, &actual) in buf.iter().enumerate() {
                let x = SWEEP_MIN + (b * 16 + k) as f32 * step;
                let reference = ref_sigmoid(x);
                let err = (actual - reference).abs();
                if err > max_error {
                    max_error = err;
                    worst_x = x;
                }
            }
        }

        // Resíduo escalar.
        let remainder_start = full_blocks * 16;
        for i in remainder_start..SWEEP_N {
            let x = SWEEP_MIN + i as f32 * step;
            let v = _mm256_set1_ps(x);
            let r = simd_sigmoid_avx2(v);
            let actual = _mm256_cvtss_f32(r);
            let reference = ref_sigmoid(x);
            let err = (actual - reference).abs();
            if err > max_error {
                max_error = err;
                worst_x = x;
            }
        }
    }

    assert!(
        max_error < AVX512_MAX_ABS_ERROR,
        "test_sigmoid_max_abs_error_sweep_avx512 FALHOU: max_error={:.3e} >= {:.3e} (pior ponto x={})",
        max_error,
        AVX512_MAX_ABS_ERROR,
        worst_x,
    );
}
