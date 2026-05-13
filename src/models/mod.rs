// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de Motores Cerebrais (Arquiteturas Inferenciais Neurais) para o NAM-rs.
//!
//! Este módulo contém os cérebros acústicos do programa: redes neurais que aprenderam como,
//! por exemplo, um amplificador ou pedal verdadeiro distorce e colora o som de uma guitarra.

pub mod a2;
pub mod lstm;
pub mod wavenet;

// =============================================================================
// Trait NamModel — Contrato Público
// =============================================================================

/// A interface (o conector padrão) para qualquer modelo neural (amplificadores, pedais, etc.).
pub trait NamModel: Send + Sync {
    /// Invocado pelo DSP RT-Thread para processar blocos de amostragem acústica (Float32).
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// "Aquece" as válvulas virtuais do motor neural (`prewarm`).
    fn prewarm(&mut self, num_samples: usize);
}

/// Wrapper enum para as variantes de modelo treinadas.
/// Permite despacho estático das chamadas DSP para a variante concreta evitando overhead de vtable.
pub enum DynamicModel {
    /// WaveNet Standard (16 canais, kernel 3, dilation 8).
    WavenetStandard(Box<wavenet::WaveNetModel<16, 3, 8>>),
    /// WaveNet Lite (12 canais, kernel 3, dilation 6).
    WavenetLite(Box<wavenet::WaveNetModel<12, 3, 6>>),
    /// WaveNet Feather (8 canais, kernel 3, dilation 4).
    WavenetFeather(Box<wavenet::WaveNetModel<8, 3, 4>>),
    /// WaveNet Nano (4 canais, kernel 3, dilation 2).
    WavenetNano(Box<wavenet::WaveNetModel<4, 3, 2>>),
    /// WaveNet Dinâmico (usado como fallback para arquiteturas não-padrão).
    WavenetDyn(Box<wavenet::WaveNetDynModel>),
    /// WaveNet A2 (Placeholder para arquitetura nova).
    WavenetA2(Box<a2::WavenetA2Placeholder>),
    /// LSTM 1 Camada × 8 unidades ocultas.
    Lstm1x8(Box<lstm::Lstm1x8>),
    /// LSTM 1 Camada × 12 unidades ocultas.
    Lstm1x12(Box<lstm::Lstm1x12>),
    /// LSTM 1 Camada × 16 unidades ocultas.
    Lstm1x16(Box<lstm::Lstm1x16>),
    /// LSTM 1 Camada × 24 unidades ocultas.
    Lstm1x24(Box<lstm::Lstm1x24>),
    /// LSTM 2 Camadas × 8 unidades ocultas.
    Lstm2x8(Box<lstm::Lstm2x8>),
    /// LSTM 2 Camadas × 12 unidades ocultas.
    Lstm2x12(Box<lstm::Lstm2x12>),
    /// LSTM 2 Camadas × 16 unidades ocultas.
    Lstm2x16(Box<lstm::Lstm2x16>),
    /// LSTM Dinâmico (usado como fallback).
    LstmDyn(Box<lstm::LstmDynModel>),
}

impl NamModel for DynamicModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        match self {
            Self::WavenetStandard(m) => m.process(input, output),
            Self::WavenetLite(m) => m.process(input, output),
            Self::WavenetFeather(m) => m.process(input, output),
            Self::WavenetNano(m) => m.process(input, output),
            Self::WavenetDyn(m) => m.process(input, output),
            Self::WavenetA2(m) => m.process(input, output),
            Self::Lstm1x8(m) => m.process(input, output),
            Self::Lstm1x12(m) => m.process(input, output),
            Self::Lstm1x16(m) => m.process(input, output),
            Self::Lstm1x24(m) => m.process(input, output),
            Self::Lstm2x8(m) => m.process(input, output),
            Self::Lstm2x12(m) => m.process(input, output),
            Self::Lstm2x16(m) => m.process(input, output),
            Self::LstmDyn(m) => m.process(input, output),
        }
    }

    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        match self {
            Self::WavenetStandard(m) => m.prewarm(),
            Self::WavenetLite(m) => m.prewarm(),
            Self::WavenetFeather(m) => m.prewarm(),
            Self::WavenetNano(m) => m.prewarm(),
            Self::WavenetDyn(m) => m.prewarm(),
            Self::WavenetA2(m) => m.prewarm(num_samples),
            Self::Lstm1x8(m) => m.prewarm(num_samples),
            Self::Lstm1x12(m) => m.prewarm(num_samples),
            Self::Lstm1x16(m) => m.prewarm(num_samples),
            Self::Lstm1x24(m) => m.prewarm(num_samples),
            Self::Lstm2x8(m) => m.prewarm(num_samples),
            Self::Lstm2x12(m) => m.prewarm(num_samples),
            Self::Lstm2x16(m) => m.prewarm(num_samples),
            Self::LstmDyn(m) => m.prewarm(num_samples),
        }
    }
}
