// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels SIMD para redes LSTM (Long Short-Term Memory).
//!
//! Extraídos de `activations/fused.rs` e `simd/avx2.rs`/`simd/avx512.rs`
//! durante a Tarefa 3.3.

pub mod gates;

pub use gates::*;
