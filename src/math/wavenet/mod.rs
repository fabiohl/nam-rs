// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels SIMD para arquitetura WaveNet (Head Sum, Accumulate, Gated Activation).
//!
//! Este módulo contém kernels altamente especializados para a cascata de camadas
//! WaveNet, focando em minimizar acessos à memória e maximizar a densidade aritmética.
//!
//! # Operações Críticas
//! - **Head Accumulate**: Soma dos vetores de skip-connection para a saída final.
//! - **Gated Activation**: Fusão de `tanh` e `sigmoid` em um único kernel SIMD.
//! - **Dialated Conv Fetches**: Otimização de busca de dados em linhas de atraso (delay lines).
//!
//! Extraídos de `simd/avx2.rs`, `simd/avx512.rs` e `common/scalar_ref.rs`
//! durante a Tarefa 3.4.

pub mod accumulate;
pub mod head;

pub use accumulate::*;
pub use head::*;
