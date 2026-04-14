// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Traits genéricos para processamento digital de sinais.
//!
//! Este módulo introduz as fundações recomendadas pelos relatórios e premissas
//! de auditoria DSP em tempo real. Ele provê blocos construtores que induzem
//! "static dispatch" (monomorfização) em vias quentes, e garante tipagem forte para
//! amostras, abstrações por quadros de canal (*frames*), e topologia iterável em blocos
//! vetoriais (*signal/block processing*).
//!
//! Para uso geral nas arquiteturas de inferência, resamplers e conversores deste motor,
//! impedindo acoplamento exclusivo em fatias brutas `&[f32]`.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

/// Menor unidade escalar e indivisível capturada pela amostragem num respectivo canal.
/// Tipicamente `f32` no NAM-rs por ser a resolução nativa do *Pipeline Neural*,
/// mas assegura polimorfismo para `f64` ou inteiros em pipelines extendidos.
pub trait Sample:
    Copy
    + Clone
    + Send
    + Sync
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// O elemento representativo de zero estrito na corrente DC.
    const EQUILIBRIUM: Self;

    /// Transição do tipo matemático primário do núcleo (f32) limitando o custo em cast dinâmico.
    fn from_f32(v: f32) -> Self;

    /// Injeção do Sample genérico para um float-32 nativo, compatível nativamente
    /// com a malha PipeWire ou vetores intrínsecos AVX do motor.
    fn to_f32(self) -> f32;
}

impl Sample for f32 {
    const EQUILIBRIUM: Self = 0.0;

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

impl Sample for f64 {
    const EQUILIBRIUM: Self = 0.0;

    #[inline(always)]
    fn from_f32(v: f32) -> Self {
        v as f64
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

/// Contêiner de encapsulamento espacial de amostras em um instante `T`.
/// Encapsula perfeitamente o número estrito de canais `CHANNELS` (Mono, Estéreo, etc).
pub trait Frame: Copy + Clone + Send + Sync {
    /// Unidade amostral basal contida nesta estrutura iterável.
    type Sample: Sample;

    /// O número de divisões espaciais (Canais representados).
    const CHANNELS: usize;

    /// Exposição imutável do núcleo para manipulação matricial iterativa (Zero-cost abstrato).
    fn as_slice(&self) -> &[Self::Sample];

    /// Invasão em nível *in-place* por canal (Mutação temporal instantânea).
    fn as_mut_slice(&mut self) -> &mut [Self::Sample];
}

// Extende nativamente toda matriz fixa (Mono `[T; 1]`, Estéreo `[T; 2]`) como um Frame matemático.
impl<S: Sample, const N: usize> Frame for [S; N] {
    type Sample = S;
    const CHANNELS: usize = N;

    #[inline(always)]
    fn as_slice(&self) -> &[Self::Sample] {
        self
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [Self::Sample] {
        self
    }
}

/// Representa a topologia operante de *Block Processing* vetorizada em RT Audio threads.
/// Modificadores de fluxo assinam este contrato para indicar transformações absolutas de onda.
pub trait Signal {
    /// O esqueleto dimensional das amostras contidas no fluxo de inferência ou conversão.
    type Frame: Frame;

    /// Operação vetorial padrão (Obrigatória em arquiteturas sem *Branch Misprediction*).
    /// Abastece a matriz de *output* integralmente com representações do *input*.
    /// Permite monomorfização quando empacotado puramente.
    fn process_block(&mut self, input: &[Self::Frame], output: &mut [Self::Frame]);

    /// Manipulação baseada na sobreposição no mesmo buffer (*in-place*).
    /// Deve ser assinado deliberadamente para expurgar cópias auxiliares
    /// em componentes compatíveis (Ex: processadores puramente estáticos ou Multi-Gain Filters).
    fn process_block_in_place(&mut self, buffer: &mut [Self::Frame]);
}
