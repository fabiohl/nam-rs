// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de Motores Cerebrais (Arquiteturas Inferenciais Neurais) para o NAM-rs.
//!
//! Este módulo contém os cérebros acústicos do programa: redes neurais que aprenderam como,
//! por exemplo, um amplificador ou pedal verdadeiro distorce e colore o som de uma guitarra.
//! Aqui temos implementações super rápidas ("Zero-Allocation") baseadas no
//! comportamento exato treinado pelo Neural Amp Modeler (NAM) original.

pub mod lstm;
pub mod lstm_dyn;
pub mod wavenet;
pub mod wavenet_dyn;

/// A interface (o conector padrão) para qualquer modelo neural (amplificadores, pedais, etc.).
///
/// Pense nisso como um cabo de áudio padrão: não importa a marca do pedal interno (WaveNet ou LSTM),
/// o sistema de áudio (host principal) sabe enviar som para ele e receber de volta através deste contrato/trait.
pub trait NamModel: Send + Sync {
    /// Invocado pelo DSP RT-Thread para processar blocos de amostragem acústica (Float32).
    /// O áudio deverá ser processado *in-place* ou lido de input para output dependendo da geometria vetorial.
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// "Aquece" as válvulas virtuais do motor neural (`prewarm`).
    /// Injeta um curto zumbido ou silêncio (`num_samples`) por dentro do amplificador antes de expor
    /// uma guitarra de verdade a ele. Isso acalma o "campo magnético virtual", garantindo que não
    /// hajam estalos (pops) estridentes quando você liga ou troca de modelo de súbito.
    fn prewarm(&mut self, num_samples: usize);
}

/// Wrapper para preservar a vtable do trait object. Transita como thin pointer seguro.
pub struct DynamicModel(pub Box<dyn NamModel>);

// =============================================================================
// NamModel para WaveNet
// =============================================================================

impl<const CH: usize, const K: usize, const HEAD: usize> NamModel
    for wavenet::WaveNetModel<CH, K, HEAD>
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

impl NamModel for wavenet_dyn::WaveNetDynModel {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }
}

// =============================================================================
// NamModel para LSTM — 1 Camada
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H_H4: usize> NamModel
    for lstm::LstmModel1<H, H1_IH, H_H4>
{
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Safety: AVX2+FMA verificado no startup (main.rs).
        // O método inherent unsafe tem prioridade sobre o trait — sem recursão.
        self.process(input, output);
    }

    fn prewarm(&mut self, num_samples: usize) {
        // Zera os estados internos antes do prewarm (como esperado pelo C++)
        self.reset_states();

        // LSTM requer processamento de `num_samples` amostras zero para estabilizar estados.
        // Ref C++: Prewarm(2048, 64) — processa 2048 amostras com bloco de 64.
        const CHUNK: usize = 512;
        let zero_in = [0.0f32; CHUNK];
        let mut zero_out = [0.0f32; CHUNK];
        let mut rem = num_samples;
        while rem > 0 {
            let n = rem.min(CHUNK);
            // Safety: AVX2+FMA verificado no startup (main.rs).
            self.process(&zero_in[..n], &mut zero_out[..n]);
            rem -= n;
        }
    }
}

// =============================================================================
// NamModel para LSTM — 2 Camadas
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> NamModel
    for lstm::LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Safety: AVX2+FMA verificado no startup (main.rs).
        self.process(input, output);
    }

    fn prewarm(&mut self, num_samples: usize) {
        // Zera os estados internos antes do prewarm
        self.reset_states();

        const CHUNK: usize = 512;
        let zero_in = [0.0f32; CHUNK];
        let mut zero_out = [0.0f32; CHUNK];
        let mut rem = num_samples;
        while rem > 0 {
            let n = rem.min(CHUNK);
            // Safety: AVX2+FMA verificado no startup (main.rs).
            self.process(&zero_in[..n], &mut zero_out[..n]);
            rem -= n;
        }
    }
}

// =============================================================================
// NamModel para LSTM Dinâmico
// =============================================================================

impl NamModel for lstm_dyn::LstmDynModel {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, num_samples: usize) {
        self.prewarm(num_samples);
    }
}

// =============================================================================
// Type Aliases — Perfis LSTM NAM Comuns
// =============================================================================

/// LSTM 1 camada × 8 unidades ocultas (Nano/Feather).
pub type Lstm1x8 = lstm::LstmModel1<8, 9, 32>;
/// LSTM 1 camada × 12 unidades ocultas (Lite).
pub type Lstm1x12 = lstm::LstmModel1<12, 13, 48>;
/// LSTM 1 camada × 16 unidades ocultas (Standard).
pub type Lstm1x16 = lstm::LstmModel1<16, 17, 64>;
/// LSTM 1 camada × 24 unidades ocultas (Heavy Standard).
pub type Lstm1x24 = lstm::LstmModel1<24, 25, 96>;

/// LSTM 2 camadas × 8 unidades ocultas.
pub type Lstm2x8 = lstm::LstmModel2<8, 9, 16, 32>;
/// LSTM 2 camadas × 12 unidades ocultas.
pub type Lstm2x12 = lstm::LstmModel2<12, 13, 24, 48>;
/// LSTM 2 camadas × 16 unidades ocultas.
pub type Lstm2x16 = lstm::LstmModel2<16, 17, 32, 64>;
