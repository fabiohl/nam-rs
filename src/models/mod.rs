// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Módulo de Motores Cerebrais (Arquiteturas Inferenciais Neurais) para o NAM-rs.
//!
//! Este módulo contém os cérebros acústicos do programa: redes neurais que aprenderam como,
//! por exemplo, um amplificador ou pedal verdadeiro distorce e colora o som de uma guitarra.

pub mod a2;
pub mod lstm;
pub mod wavenet;

use crate::common::spsc::RtStatusFlags;
use std::sync::Arc;

// =============================================================================
// Trait NamModel — Contrato Público
// =============================================================================

/// A interface (o conector padrão) para qualquer modelo neural (amplificadores, pedais, etc.).
pub trait NamModel: Send + Sync {
    /// Invocado pelo DSP RT-Thread para processar blocos de amostragem acústica (Float32).
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// "Aquece" as válvulas virtuais do motor neural (`prewarm`).
    fn prewarm(&mut self, num_samples: usize);

    /// Reseta o estado interno do modelo com nova taxa de amostragem e tamanho máximo de buffer.
    ///
    /// A implementação default chama `prewarm(max_buffer_size)`, o que é adequado para
    /// arquiteturas como WaveNet que precisam preencher o campo receptivo com silêncio.
    /// Arquiteturas com estado recorrente (LSTM) podem sobrescrever para um reset
    /// mais leve (apenas zerar os estados internos sem reprocessar prewarm completo).
    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) {
        self.prewarm(max_buffer_size);
    }
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
    /// LSTM 1 Camada × 40 unidades ocultas.
    Lstm1x40(Box<lstm::Lstm1x40>),
    /// LSTM 2 Camadas × 24 unidades ocultas.
    Lstm2x24(Box<lstm::Lstm2x24>),
    /// LSTM Dinâmico (usado como fallback).
    LstmDyn(Box<lstm::LstmDynModel>),
}

impl DynamicModel {
    /// Injeta `RtStatusFlags` na variante `WavenetA2` para que o placeholder
    /// possa sinalizar seu estado ao UI via flags atômicas.
    pub fn inject_rt_status(&mut self, rt_status: Arc<RtStatusFlags>) {
        if let Self::WavenetA2(m) = self {
            m.inject_rt_status(rt_status);
        }
    }
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
            Self::Lstm1x40(m) => m.process(input, output),
            Self::Lstm2x24(m) => m.process(input, output),
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
            Self::Lstm1x40(m) => m.prewarm(num_samples),
            Self::Lstm2x24(m) => m.prewarm(num_samples),
            Self::LstmDyn(m) => m.prewarm(num_samples),
        }
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) {
        match self {
            Self::WavenetStandard(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetLite(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetFeather(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetNano(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetDyn(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x8(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x12(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x16(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x24(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x8(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x12(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x16(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x40(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x24(m) => m.reset(sample_rate, max_buffer_size),
            Self::LstmDyn(m) => m.reset(sample_rate, max_buffer_size),
        }
    }
}
