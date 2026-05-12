// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de FastMath (Minimax & Pade) para otimização de operações matemáticas.
//! Atualmente contém a LUT de ganho. As funções de ativação foram migradas para `math::activations`.

pub use crate::math::activations::sigmoid::simd_sigmoid_avx2 as simd_sigmoid;
pub use crate::math::activations::tanh::simd_tanh_avx2 as simd_tanh;
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
        let v1 = self.table[idx + 1];
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
