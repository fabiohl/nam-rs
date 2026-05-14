// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tabela de Look-Up (LUT) para conversão ultra-rápida de dB para ganho linear.
//!
//! Projetada para ser instanciada via `OnceLock` e acessada em threads RT.
//! Migrada de `math::fastmath` como parte da Tarefa 3.5 (Épico 3).

use crate::math::constants::*;
use std::sync::OnceLock;

/// Tabela de Look-Up para conversão ultra-rápida de dB para ganho linear.
/// Projetada para ser instanciada via `OnceLock` e acessada em threads RT.
pub struct GainLUT {
    table: [f32; GAIN_LUT_SIZE],
}

impl GainLUT {
    /// Inicializa a LUT pré-calculando os valores de ganho linear.
    pub fn new() -> Self {
        let mut table = [0.0f32; GAIN_LUT_SIZE];
        for (i, item) in table.iter_mut().enumerate() {
            let db =
                (i as f32) * (GAIN_MAX_DB - GAIN_MIN_DB) / (GAIN_LUT_SIZE as f32) + GAIN_MIN_DB;
            *item = 10.0f32.powf(db / 20.0);
        }
        Self { table }
    }

    /// Converte dB para ganho linear usando a LUT com interpolação linear.
    #[inline(always)]
    pub fn db_to_linear(&self, db: f32) -> f32 {
        let db_clamped = db.clamp(GAIN_MIN_DB, GAIN_MAX_DB - 0.001);
        let idx_f =
            (db_clamped - GAIN_MIN_DB) * (GAIN_LUT_SIZE as f32) / (GAIN_MAX_DB - GAIN_MIN_DB);
        let idx = idx_f as usize;
        let frac = idx_f - (idx as f32);

        let v0 = self.table[idx];
        let v1 = self.table[(idx + 1).min(GAIN_LUT_SIZE - 1)];
        v0 + frac * (v1 - v0)
    }
}

impl Default for GainLUT {
    fn default() -> Self {
        Self::new()
    }
}

/// Acesso global e seguro à LUT de ganho.
pub static GAIN_LUT: OnceLock<GainLUT> = OnceLock::new();

/// Retorna a referência global para a LUT de ganho, inicializando-a se necessário.
pub fn get_gain_lut() -> &'static GainLUT {
    GAIN_LUT.get_or_init(GainLUT::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_lut_initialization() {
        let lut = GainLUT::new();
        // Teste de borda: ganho mínimo
        assert!((lut.db_to_linear(GAIN_MIN_DB) - 10.0f32.powf(GAIN_MIN_DB / 20.0)).abs() < 1e-6);
        // Teste de borda: ganho máximo dentro do range de interpolação seguro
        let max_db = GAIN_MAX_DB - GAIN_DB_STEP;
        let max_val = lut.db_to_linear(max_db);
        let expected_max = 10.0f32.powf(max_db / 20.0);
        assert!(
            (max_val - expected_max).abs() < 1e-3,
            "GAIN_MAX_DB={GAIN_MAX_DB}, step={GAIN_DB_STEP}: lut={max_val}, expected={expected_max}"
        );
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
}
