// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model Dispatcher — converte `NamModelData` (parsers JSON/NAMB) em `Box<DynamicModel>`.
//!
//! Toda a lógica de construção e alocação ocorre exclusivamente na thread CLI,
//! garantindo que o `Box<DynamicModel>` resultante esteja pronto para injeção na
//! thread DSP via SPSC sem nenhuma alocação no caminho RT.
//!
//! Os pesos são consumidos sequencialmente por um `WeightCursor` cursor-forward,
//! com verificação de exaustão ao final para detectar modelos inconsistentes.

use crate::loader::nam_json::NamModelData;
use crate::models::DynamicModel;
use anyhow::bail;

// =============================================================================
// WeightCursor — Leitura sequencial determinística dos pesos planificados
// =============================================================================

/// Cursor de leitura forward-only sobre o vetor de pesos planificados.
///
/// Garante:
/// - Nenhum peso é lido fora dos limites (`read_slice` / `read_f32`)
/// - Todos os pesos foram consumidos no final (`verify_exhausted`)
pub(crate) struct WeightCursor<'a> {
    /// Referência ao slice completo de pesos do modelo.
    data: &'a [f32],
    /// Posição corrente do cursor (avança a cada leitura).
    pos: usize,
    /// Layout dos pesos informado no cabeçalho binário.
    pub layout: crate::loader::nam_json::WeightsLayout,
}

impl<'a> WeightCursor<'a> {
    /// Cria um novo cursor sobre a fatia de pesos com layout especificado.
    #[cold]
    pub fn new(data: &'a [f32], layout: crate::loader::nam_json::WeightsLayout) -> Self {
        Self {
            data,
            pos: 0,
            layout,
        }
    }

    /// Verifica se os pesos estão em formato entrelaçado (WaveNet v2).
    pub fn is_interleaved4(&self) -> bool {
        self.layout == crate::loader::nam_json::WeightsLayout::Interleaved4WaveNet
    }

    /// Verifica se os pesos estão em formato Gate-Major (LSTM v2).
    pub fn is_gate_major_lstm(&self) -> bool {
        self.layout == crate::loader::nam_json::WeightsLayout::GateMajorLstm
    }

    /// Lê uma fatia contígua de `len` pesos, avançando o cursor.
    fn read_slice(&mut self, len: usize) -> anyhow::Result<&'a [f32]> {
        if self.pos + len > self.data.len() {
            bail!(
                "Insufficient weights: required {} starting from position {}, available {}",
                len,
                self.pos,
                self.data.len()
            );
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    /// Lê um único escalar `f32`, avançando o cursor.
    fn read_f32(&mut self) -> anyhow::Result<f32> {
        let s = self.read_slice(1)?;
        Ok(s[0])
    }

    /// Verifica que todos os pesos foram consumidos. Falha se restarem pesos.
    fn verify_exhausted(&self) -> anyhow::Result<()> {
        if self.pos != self.data.len() {
            bail!(
                "Model with inconsistent weights: consumed {}, total {}",
                self.pos,
                self.data.len()
            );
        }
        Ok(())
    }
}

// =============================================================================
// Ponto de Entrada Público
// =============================================================================

/// Constrói um `Box<DynamicModel>` a partir dos dados brutos parseados.
///
/// Bifurca por arquitetura (`"WaveNet"` / `"LSTM"`) e delega para os
/// construtores especializados com const generics.
pub fn build_model(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    match data.architecture.as_str() {
        "WaveNet" => wavenet::build_wavenet(data),
        "LSTM" => lstm::build_lstm(data),
        other => bail!("Unsupported architecture: '{}'", other),
    }
}

/// Módulo de construção de modelos LSTM
pub mod lstm;
/// Módulo de construção de modelos WaveNet
pub mod wavenet;

pub use lstm::build_lstm_dynamic;
pub use wavenet::build_wavenet_dynamic;
