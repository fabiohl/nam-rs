// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo WaveNet — Arquitetura Neural Causal Dilatada para emulação de amplificadores e pedais.
//!
//! Este módulo fornece motores de inferência WaveNet otimizados para execução DSP em tempo real:
//!
//! - **Caminho Estático** (`model`): usa Const Generics para dimensões fixas (ex: 16 canais),
//!   eliminando verificações de bounds e maximizando throughput.
//! - **Caminho Dinâmico** (`model_dyn`): fallback para topologias não-cobertas por Const Generics,
//!   com alocação única no construtor e zero-alocação no hot-path.
//!
//! ## Sub-módulos
//!
//! | Módulo        | Descrição                                                            |
//! | ------------- | -------------------------------------------------------------------- |
//! | `common`      | Constantes e tipos fundamentais (`WaveNetLayerState`, `WavenetProcessContext`) |
//! | `conv1d`      | Convolução 1D causal estática (`Conv1d`) + trait `ConvInput`        |
//! | `conv1d_dyn`  | Convolução 1D causal dinâmica (`Conv1dDyn`)                         |
//! | `dense`       | Camada densa 1x1 estática (`DenseLayer`)                            |
//! | `model`       | Modelo estático completo (`WaveNetModel`, `WaveNetLayerArray`)      |
//! | `model_dyn`   | Modelo dinâmico com dimensões runtime (`WaveNetDynModel`)           |

pub mod common;
pub mod conv1d;
pub mod conv1d_dyn;
/// Camada densa 1x1 estática (`DenseLayer<IN, OUT>`).
pub mod dense;
/// Modelo estático (`WaveNetModel`, `WaveNetLayerArray`, `WaveNetLayer`).
pub mod model;
pub mod model_dyn;

use super::NamModel;

// =============================================================================
// NamModel para WaveNet (Const Generics)
// =============================================================================

impl<const CH: usize, const K: usize, const HEAD: usize> NamModel
    for model::WaveNetModel<CH, K, HEAD>
{
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Delega ao método inherent WaveNetModel::process (métodos inherent têm prioridade)
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        // WaveNet prewarm é one-shot: preenche o campo receptivo via copy_buffer.
        // O C++ executa `model->Prewarm()` sem parâmetro (diferente do LSTM).
        self.prewarm();
    }
}

// =============================================================================
// NamModel para WaveNet Dinâmico
// =============================================================================

impl NamModel for model_dyn::WaveNetDynModel {
    /// Delega o processamento para a implementação interna do modelo WaveNet dinâmico.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// O "aquecimento" da WaveNet é simplificado pois ela não possui memória infinita
    /// como a LSTM, apenas um buffer de delay (campo receptivo).
    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }
}

// =============================================================================
// Re-exports públicos
// =============================================================================

pub use common::{
    LAYER_ARRAY_BUFFER_PADDING, WAVENET_MAX_NUM_FRAMES, WaveNetLayerState, WavenetProcessContext,
};
pub use conv1d::Conv1d;
pub use conv1d_dyn::Conv1dDyn;
pub use dense::DenseLayer;
pub use model::{WaveNetLayer, WaveNetLayerArray, WaveNetModel};
pub use model_dyn::{DenseLayerDyn, WaveNetDynModel, WaveNetLayerArrayDyn, WaveNetLayerDyn};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
