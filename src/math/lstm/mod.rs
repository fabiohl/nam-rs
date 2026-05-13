// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels SIMD para redes LSTM (Long Short-Term Memory).
//!
//! Este módulo foca na eficiência de atualização de estado das células LSTM,
//! utilizando fusão de operações para reduzir a latência de inferência.
//!
//! # Destaques
//! - **Fused Gate Processing**: O cálculo das 4 portas (input, forget, cell, output) é
//!   feito em uma única passagem sobre os registradores SIMD.
//! - **State Persistence**: Gerenciamento otimizado do `cell_state` e `hidden_state`.
//! - **Parallelism**: Processamento simultâneo de 8 (AVX2) ou 16 (AVX-512) células por vez.
//!
//! Extraídos de `activations/fused.rs` e `simd/avx2.rs`/`simd/avx512.rs`
//! durante a Tarefa 3.3.

pub mod gates;

pub use gates::*;
