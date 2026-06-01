// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Testa a aplicação básica de ganho.
/// Verifica bypass (1.0), amplificação (2.0) e comportamento com zeros.
#[test]
fn test_apply_gain_simd() {
    let mut buffer = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let original = buffer;

    // Neutro: Ganho 1.0 não deve alterar o buffer.
    apply_gain_simd(&mut buffer, 1.0);
    assert_eq!(buffer, original);

    // Ganho = 2.0: Multiplica todas as amostras por 2.
    apply_gain_simd(&mut buffer, 2.0);
    assert_eq!(
        buffer,
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    );

    // Ganho em Zeros: Multiplicar zero por qualquer coisa deve resultar em zero.
    let mut zeros = [0.0; 25];
    apply_gain_simd(&mut zeros, 5.5);
    for &z in &zeros {
        assert_eq!(z, 0.0);
    }
}

/// Valida o cálculo combinado de gain staging (user + model metadata).
/// Simula a lógica exata de `update_gain_multipliers` de pw_host.rs.
/// Garante que a conversão dB -> Linear e a aplicação estão corretas.
#[test]
fn test_combined_gain_staging() {
    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    let user_input_db: f32 = 6.0;
    let model_input_adj_db: f32 = -3.0;

    // Na nova arquitetura, a Main Thread envia multiplicadores lineares.
    let user_input_mult = lut.db_to_linear(user_input_db);
    let model_input_adj_mult = lut.db_to_linear(model_input_adj_db);

    // A thread RT apenas os multiplica (zero powf/LUT no hot-path).
    let gain_linear = user_input_mult * model_input_adj_mult;

    let mut buffer = [1.0f32; 16];
    apply_gain_simd(&mut buffer, gain_linear);

    let expected = lut.db_to_linear(3.0);
    for &sample in &buffer {
        assert!(
            (sample - expected).abs() < 1e-5,
            "Esperado ~{expected:.5}, obteve {sample:.5}"
        );
    }
}

/// Verifica estabilidade com ganho extremo negativo (-60dB ≈ 0.001)
/// e positivo (+24dB ≈ 15.85) sem underflow/overflow em Float32.
#[test]
fn test_extreme_gain_values() {
    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    // -60 dB → gain ≈ 0.001
    let gain_neg60 = lut.db_to_linear(-60.0);
    assert!(gain_neg60 > 0.0 && gain_neg60.is_finite());

    let mut buffer = [1.0f32; 20];
    apply_gain_simd(&mut buffer, gain_neg60);
    for &s in &buffer {
        assert!(
            s.is_finite() && s > 0.0 && s < 0.01,
            "Underflow em -60dB: {s}"
        );
    }

    // +24 dB → gain ≈ 15.85
    let gain_pos24 = lut.db_to_linear(24.0);
    assert!(gain_pos24 > 10.0 && gain_pos24.is_finite());

    let mut buffer2 = [0.5f32; 20];
    apply_gain_simd(&mut buffer2, gain_pos24);
    for &s in &buffer2 {
        assert!(
            s.is_finite() && s > 5.0 && s < 10.0,
            "Overflow em +24dB: {s}"
        );
    }
}

/// Verifica que `apply_gain_simd` com gain=1.0 é true-bypass bitwise (sem alterar nenhum bit).
#[test]
fn test_gain_true_bypass() {
    // Cenário 1: Buffer alinhado (múltiplo de 8)
    let original_aligned: [f32; 16] = [
        0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6,
    ];
    let mut buf_aligned = original_aligned;
    apply_gain_simd(&mut buf_aligned, 1.0);
    assert_eq!(
        buf_aligned, original_aligned,
        "Gain 1.0 deve preservar buffer alinhado bitwise (true-bypass)"
    );

    // Cenário 2: Buffer não-alinhado (tail escalar)
    let original_tail: [f32; 13] = [
        0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10, 0.11, -0.12, 0.13,
    ];
    let mut buf_tail = original_tail;
    apply_gain_simd(&mut buf_tail, 1.0);
    assert_eq!(
        buf_tail, original_tail,
        "Gain 1.0 deve preservar buffer não-alinhado bitwise (true-bypass tail)"
    );

    // Cenário 3: Gain muito próximo de 1.0 (dentro do epsilon de bypass)
    let mut buf_eps = original_aligned;
    apply_gain_simd(&mut buf_eps, 1.0 + 1e-7);
    assert_eq!(
        buf_eps, original_aligned,
        "Gain ≈1.0 (dentro de 1e-6) deve acionar fast-path bypass"
    );
}

/// Roundtrip +6dB → -6dB deve preservar o sinal original (MSE < 1e-10).
#[test]
fn test_gain_roundtrip_6db() {
    let original: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect();

    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    let gain_up = lut.db_to_linear(6.0); // +6 dB
    let gain_down = lut.db_to_linear(-6.0); // -6 dB

    let mut buffer = original.clone();
    apply_gain_simd(&mut buffer, gain_up);
    apply_gain_simd(&mut buffer, gain_down);

    let mse: f64 = original
        .iter()
        .zip(buffer.iter())
        .map(|(a, b)| {
            let d = (*a as f64) - (*b as f64);
            d * d
        })
        .sum::<f64>()
        / (original.len() as f64);

    assert!(
        mse < 1e-10,
        "Roundtrip +6dB/-6dB MSE={mse:.2e} excede 1e-10"
    );
}

/// Valida a rampa linear SIMD contra uma implementação escalar de referência.
#[test]
fn test_apply_ramp_simd() {
    let len = 37; // Tamanho não múltiplo de 8 para testar tail
    let mut buffer_simd = vec![1.0f32; len];
    let mut buffer_scalar = vec![1.0f32; len];

    let start = 0.5f32;
    let step = 0.01f32;

    // Referência escalar
    let mut m = start;
    for s in buffer_scalar.iter_mut() {
        *s *= m;
        m += step;
    }

    // Implementação SIMD
    apply_ramp_simd(&mut buffer_simd, start, step);

    // Verifica MSE entre as duas implementações
    for i in 0..len {
        assert!(
            (buffer_simd[i] - buffer_scalar[i]).abs() < 1e-6,
            "Divergência na rampa no índice {i}: SIMD={} Scalar={}",
            buffer_simd[i],
            buffer_scalar[i]
        );
    }
}
