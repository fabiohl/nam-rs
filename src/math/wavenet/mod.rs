// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels SIMD para arquitetura WaveNet (Head Sum, Accumulate, Gated Activation).
//!
//! Extraídos de `simd/avx2.rs`, `simd/avx512.rs` e `common/scalar_ref.rs`
//! durante a Tarefa 3.4.

pub mod accumulate;
pub mod head;

pub use accumulate::*;
pub use head::*;
