// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::*;

#[test]
fn test_gain_lut_initialization() {
    let lut = GainLUT::new();
    // Teste de borda: ganho mínimo
    assert!((lut.db_to_linear(GAIN_MIN_DB) - 10.0f32.powf(GAIN_MIN_DB / 20.0)).abs() < 1e-6);
    // Teste de borda: ganho máximo
    assert!((lut.db_to_linear(GAIN_MAX_DB) - 10.0f32.powf(GAIN_MAX_DB / 20.0)).abs() < 1e-6);
}

#[test]
fn test_gain_lut_interpolation() {
    let lut = GainLUT::new();
    let mid_db = (GAIN_MIN_DB + GAIN_MAX_DB) / 2.0;
    let linear = lut.db_to_linear(mid_db);
    let expected = 10.0f32.powf(mid_db / 20.0);
    // Interpolação linear em curva exponencial introduz pequeno erro,
    // mas deve ser bem próximo com step de 0.1dB.
    assert!((linear - expected).abs() < 1e-3);
}

#[test]
fn test_gain_lut_clamping() {
    let lut = GainLUT::new();
    let linear_min = lut.db_to_linear(GAIN_MIN_DB - 10.0);
    let linear_max = lut.db_to_linear(GAIN_MAX_DB + 10.0);
    assert_eq!(linear_min, lut.db_to_linear(GAIN_MIN_DB));
    assert_eq!(linear_max, lut.db_to_linear(GAIN_MAX_DB));
}
