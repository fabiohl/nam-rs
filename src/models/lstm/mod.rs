// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Módulo LSTM — Modelos Recorrentes para Inferência NAM.
//!
//! Contém as implementações estáticas (Const Generics) e dinâmica (fallback)
//! dos modelos LSTM, suas camadas, type aliases por perfil de performance
//! e a integração com a trait `NamModel` para despacho pelo host DSP.
//!
//! ## Arquitetura
//!
//! ```text
//! lstm/
//! ├── mod.rs        # Re-exports, type aliases, NamModel impls, LstmLike trait
//! ├── layer.rs      # LstmLayer + macros SIMD de processamento por amostra
//! ├── model1.rs     # LstmModel1 + macro define_lstm1_process!
//! ├── model2.rs     # LstmModel2 + macro define_lstm2_process_pipelined!
//! ├── model_dyn.rs  # LstmDynLayer + LstmDynModel (fallback dinâmico)
//! └── tests.rs      # Testes unitários e de paridade SIMD vs escalar
//! ```

use super::NamModel;

pub mod layer;
pub mod model1;
pub mod model2;
pub mod model_dyn;

// =============================================================================
// Re-exports — Structs públicas
// =============================================================================

pub use layer::LstmLayer;
pub use model_dyn::{LstmDynLayer, LstmDynModel};
pub use model1::LstmModel1;
pub use model2::LstmModel2;

// =============================================================================
// Type Aliases — Perfis LSTM NAM Comuns
// =============================================================================

/// LSTM 1 camada × 8 unidades ocultas (Nano/Feather).
pub type Lstm1x8 = LstmModel1<8, 9, 32>;
/// LSTM 1 camada × 12 unidades ocultas (Lite).
pub type Lstm1x12 = LstmModel1<12, 13, 48>;
/// LSTM 1 camada × 16 unidades ocultas (Standard).
pub type Lstm1x16 = LstmModel1<16, 17, 64>;
/// LSTM 1 camada × 24 unidades ocultas (Heavy Standard).
pub type Lstm1x24 = LstmModel1<24, 25, 96>;
/// LSTM 1 camada × 40 unidades ocultas (Tone Matching).
pub type Lstm1x40 = LstmModel1<40, 41, 160>;

/// LSTM 2 camadas × 8 unidades ocultas.
pub type Lstm2x8 = LstmModel2<8, 9, 16, 32>;
/// LSTM 2 camadas × 12 unidades ocultas.
pub type Lstm2x12 = LstmModel2<12, 13, 24, 48>;
/// LSTM 2 camadas × 16 unidades ocultas.
pub type Lstm2x16 = LstmModel2<16, 17, 32, 64>;
/// LSTM 2 camadas × 24 unidades ocultas.
pub type Lstm2x24 = LstmModel2<24, 25, 48, 96>;

// =============================================================================
// NamModel para LSTM — 1 Camada
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H_H4: usize> NamModel
    for LstmModel1<H, H1_IH, H_H4>
{
    /// Executa o processamento de áudio da LSTM.
    /// Note que `self.process` chama o método inerente da struct, que já possui
    /// a lógica de despacho SIMD (AVX2/512) otimizada.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Safety: A verificação de compatibilidade de hardware é feita no início da aplicação.
        self.process(input, output);
    }

    /// O prewarm na LSTM é vital. Como é um modelo recorrente, o estado interno (memória)
    /// precisa de um tempo processando silêncio para "estabilizar" antes do áudio real.
    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        lstm_prewarm_common(self, num_samples);
    }

    /// Reset leve para LSTM: apenas zera os estados internos sem reprocessar
    /// o prewarm completo com silêncio.
    fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        self.reset_states();
    }
}

// =============================================================================
// NamModel para LSTM — 2 Camadas
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> NamModel
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Processamento idêntico ao modelo de 1 camada, mas operando sobre a cadeia de 2 camadas.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// Prewarm para o modelo empilhado. Ambas as camadas são estabilizadas sequencialmente.
    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        lstm_prewarm_common(self, num_samples);
    }

    /// Reset leve para LSTM de 2 camadas: apenas zera os estados internos sem
    /// reprocessar o prewarm completo com silêncio.
    fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        self.reset_states();
    }
}

// =============================================================================
// NamModel para LSTM Dinâmico
// =============================================================================

impl NamModel for LstmDynModel {
    /// Implementação para modelos onde o tamanho do hidden state é definido em tempo de execução.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// O prewarm dinâmico já encapsula internamente a lógica de loop de silêncio.
    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        self.prewarm(num_samples);
    }

    /// Reset leve para LSTM dinâmico: apenas zera os estados internos sem
    /// reprocessar o prewarm completo com silêncio.
    fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        self.reset_states();
    }
}

// =============================================================================
// Helpers Internos — Redução de Boilerplate
// =============================================================================

/// Trait interno para unificar modelos que possuem estado LSTM resetável.
trait LstmLike: NamModel {
    fn reset_input_slots(&mut self);
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmLike
    for LstmModel1<H, H1_IH, H_H4>
{
    fn reset_input_slots(&mut self) {
        self.layer.reset_input_slot();
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> LstmLike
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn reset_input_slots(&mut self) {
        self.layer1.reset_input_slot();
        self.layer2.reset_input_slot();
    }
}

/// Implementação genérica de aquecimento (prewarm) para modelos baseados em LSTM.
/// Zera apenas os slots de entrada, preservando os estados oculto e de célula
/// carregados do arquivo NAM (`_xh` e `_c`), e processa silêncio para estabilização.
fn lstm_prewarm_common(model: &mut impl LstmLike, num_samples: usize) {
    // 1. Zera apenas o slot de entrada de cada camada, preservando _xh e _c do arquivo.
    model.reset_input_slots();

    // 2. Processa amostras de valor zero.
    const CHUNK: usize = 512;
    let zero_in = [0.0f32; CHUNK];
    let mut zero_out = [0.0f32; CHUNK];
    let mut rem = num_samples;

    while rem > 0 {
        let n = rem.min(CHUNK);
        model.process(&zero_in[..n], &mut zero_out[..n]);
        rem -= n;
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod lstm_tests;
