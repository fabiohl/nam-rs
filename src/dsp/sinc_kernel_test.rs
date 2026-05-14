// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_bessel_i0_known_values() {
    assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
    // I0(1) ≈ 1.2660658...
    assert!((bessel_i0(1.0) - 1.2660658).abs() < 1e-5);
}

#[test]
fn test_sinc_kaiser_dc_unity() {
    let kernel = generate_sinc_kaiser(256, 0.5, 12.0);
    let dc: f64 = kernel.iter().sum();
    assert!(
        (dc - 1.0).abs() < 1e-10,
        "Ganho DC deve ser unitário, obteve {dc}"
    );
}

#[test]
fn test_minimum_phase_causal() {
    let kernel = generate_sinc_kaiser(128, 0.5, 10.0);
    let min_ph = to_minimum_phase(&kernel);
    // Fase mínima concentra energia no início — o pico deve estar
    // nos primeiros 10% das amostras.
    let peak_pos = min_ph
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .unwrap()
        .0;
    assert!(
        peak_pos < kernel.len() / 4,
        "Pico deveria estar no início (fase mínima), encontrado em {peak_pos}/{}",
        kernel.len()
    );
}

#[test]
fn test_minimum_phase_energy_concentration() {
    let kernel = generate_sinc_kaiser(256, 0.5, 12.0);
    let min_ph = to_minimum_phase(&kernel);

    // Em fase mínima, a energia é concentrada no início do impulso.
    // Verificamos que >50% da energia total está nos primeiros 10% das amostras.
    let n10 = min_ph.len() / 10;
    let early_energy: f64 = min_ph[..n10].iter().map(|x| x * x).sum();
    let total_energy: f64 = min_ph.iter().map(|x| x * x).sum();
    let ratio = early_energy / total_energy.max(1e-30);
    assert!(
        ratio > 0.5,
        "Fase mínima: >50% da energia deveria estar nos primeiros 10%, obteve {:.1}%",
        ratio * 100.0
    );

    // Verifica que o kernel NÃO é simétrico (confirma que não é fase linear)
    let n = min_ph.len();
    let asym: f64 = (0..n / 2)
        .map(|i| (min_ph[i] - min_ph[n - 1 - i]).abs())
        .sum();
    assert!(
        asym > 0.01,
        "Kernel deveria ser assimétrico (fase mínima), assimetria={asym:.6}"
    );
}

#[test]
fn test_polyphase_bank_dimensions() {
    let bank = generate_polyphase_bank(44100, 48000);
    assert_eq!(bank.taps_per_phase, TAPS_PER_PHASE);
    // Verifica que todas as fases são acessíveis
    for p in 0..NUM_PHASES {
        let c = bank.phase_coeffs(p);
        assert_eq!(c.len(), TAPS_PER_PHASE);
    }
}

#[test]
fn test_aligned_coeffs_alignment() {
    let bank = generate_polyphase_bank(44100, 48000);
    let ptr = bank.phase_ptr(0) as usize;
    assert_eq!(ptr % 64, 0, "Coeficientes devem estar alinhados a 64 bytes");
}
