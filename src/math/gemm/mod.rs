// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels de álgebra linear SIMD (GEMM, GEMV, Dot Product).
//!
//! Este módulo é o motor de alta vazão do NAM-rs, responsável pela multiplicação
//! massiva de pesos por estados dos modelos neurais.
//!
//! # Estratégias de Performance
//! - **ILP (Instruction Level Parallelism)**: Múltiplos acumuladores para saturar as portas FMA.
//! - **F16C Compression**: Pesos armazenados em meia-precisão para dobrar o throughput de cache.
//! - **Interleaved Layout**: Pesos organizados para maximizar o reúso de dados nos registros.
//! - **Tiling**: Processamento em blocos para otimizar a localidade de dados.
//!
//! Extraídos de `simd/avx2.rs` e `simd/avx512.rs` durante a Tarefa 3.2.
//! Contém implementações AVX2 e AVX-512 lado a lado, organizadas por operação.

pub mod dot;
pub mod dot_4x;
pub mod gemm_batch;
pub mod gemv;
pub mod gemv_4gate;
pub mod gemv_bf16;

pub use dot::*;
pub use dot_4x::*;
pub use gemm_batch::*;
pub use gemv::*;
pub use gemv_4gate::*;
#[allow(unused_imports)]
pub use gemv_bf16::*;
