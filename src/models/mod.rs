// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de arquiteturas inferenciais neurais para o NAM-rs.
//!
//! Este módulo contém implementações "Zero-Allocation" usando Const Generics
//! baseadas no modelo "Neural Amp Modeler" original.

pub mod lstm;
pub mod wavenet;

/// Trait base para todos os modelos neurais inferenciais da aplicação.
///
/// Garante interface unificada para despacho dentro do loop lock-free RT de DSP.
pub trait NamModel: Send + Sync {
    /// Invocado pelo DSP RT-Thread para processar blocos de amostragem acústica (Float32).
    /// O áudio deverá ser processado *in-place* ou lido de input para output dependendo da geometria vetorial.
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// Inicializa a arquitetura matemática injetando carga zero (`num_samples`)
    /// para estabilizar buffers (ex: conv1d e receptive field da WaveNet ou state da LSTM)
    /// anulando vazamentos espúrios e oscilações do momento pré-transiente real.
    fn prewarm(&mut self, num_samples: usize);
}

/// Wrapper para preservar a vtable do trait object. Transita como thin pointer seguro.
pub struct DynamicModel(pub Box<dyn NamModel>);
