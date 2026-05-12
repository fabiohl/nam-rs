// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Kernels de álgebra linear SIMD (GEMM, GEMV, Dot Product).
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
