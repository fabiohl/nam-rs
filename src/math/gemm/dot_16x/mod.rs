// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de Dot Product 16x (16 canais de saída simultâneos) — AVX-512.
//!
//! Processa 16 output channels por chamada utilizando registradores `__m512`.
//! Disponível apenas em CPUs com suporte a AVX-512F.

pub mod dot_f32_avx512;
pub mod scalar;

pub use dot_f32_avx512::*;
pub use scalar::*;

#[cfg(test)]
#[path = "dot_16x_test.rs"]
mod dot_16x_test;
