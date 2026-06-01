// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Implementações AVX-512 da trait `SimdMath`.
//!
//! Contém `Avx512Math`, `Avx512VnniMath` e `Avx512VnniBf16Math`.
//! `Avx512VnniMath` tem implementações reais (BF16 dot product nativo via `_mm512_dpbf16_ps`).
//! Os métodos delegam para funções-kernel em `math::gemm`, `math::wavenet`,
//! `math::lstm`, `math::dsp`, `math::common::ops` e `math::common::utility`.
//!
//! # Submódulos
//! - `gemv`: Kernels GEMV/GEMM, dot products e 4-gate LSTM.
//! - `activations`: Funções de ativação (tanh/sigmoid), acúmulo e fusões LSTM.
//! - `bf16`: Conversões FP32↔BF16 e wrappers `store_bf16`.
//! - `reduce`: Soma horizontal, energia e max-diff.
//! - `dsp`: Convolução, ganho, rampa e head-sum do WaveNet.

mod activations;
mod bf16;
mod dsp;
mod gemv;
mod reduce;

use crate::math::common::scalar_ref::*;
use crate::math::common::traits::SimdMath;
use core::arch::x86_64::*;

/// Implementação SIMD via AVX-512.
/// Esta struct agrupa todas as funções matemáticas otimizadas para processadores que suportam AVX-512.
pub struct Avx512Math;

/// Implementação estática para AVX-512 com suporte a VNNI (Instruções Neurais de Vetor).
/// É ainda mais rápido para calcular redes neurais em processadores Intel modernos (ex: Cascade Lake e mais novos),
/// pois possui instruções especializadas para "moer" números de redes neurais com mais eficiência.
pub struct Avx512VnniMath;

/// Implementação estática para AVX-512 com suporte a VNNI e BF16 (Brain Float 16).
/// Esta é a "Ferrari" do processamento de áudio, disponível em CPUs Intel muito recentes (ex: Sapphire Rapids).
/// O formato BF16 permite que o chip processe o dobro de números com quase a mesma precisão do f32 original.
pub struct Avx512VnniBf16Math;

// ── Avx512Math ──

impl SimdMath for Avx512Math {
    type V = __m512;

    gemv::impl_avx512_gemv!();
    activations::impl_avx512_activations!();
    bf16::impl_avx512_bf16!();
    reduce::impl_avx512_reduce!();
    dsp::impl_avx512_dsp!();
}

// ── Avx512VnniMath ──

impl SimdMath for Avx512VnniMath {
    type V = __m512;

    gemv::impl_avx512vnni_gemv!();
    activations::impl_avx512vnni_activations!();
    bf16::impl_avx512vnni_bf16!();
    reduce::impl_avx512vnni_reduce!();
    dsp::impl_avx512vnni_dsp!();
}

// ── Avx512VnniBf16Math ──

impl SimdMath for Avx512VnniBf16Math {
    type V = __m512;

    gemv::impl_avx512vnni_bf16_gemv!();
    activations::impl_avx512vnni_bf16_activations!();
    bf16::impl_avx512vnni_bf16_bf16!();
    reduce::impl_avx512vnni_bf16_reduce!();
    dsp::impl_avx512vnni_bf16_dsp!();
}
