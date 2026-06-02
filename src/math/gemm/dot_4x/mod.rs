// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de Dot Product 4x (ILP interleaved, dual frame, batch) — AVX2 e AVX-512.
//!
//! Organized into submodules by ISA and variant:
//! - `avx2` / `avx2_dual`: kernels AVX2 com F16C.
//! - `avx512` / `avx512_dual`: kernels AVX-512 com permutexvar.
//! - `scalar`: reference scalar implementations.
//! - `avx512_bf16`: placeholder para VNNI BF16 (`vdpbf16ps`).

pub mod avx2;
pub mod avx2_dual;
pub mod avx512;
pub mod avx512_bf16;
pub mod avx512_dual;
pub mod scalar;

pub use avx2::*;
pub use avx2_dual::*;
pub use avx512::*;
pub use avx512_bf16::*;
pub use avx512_dual::*;
pub use scalar::*;

#[cfg(test)]
#[path = "dot_4x_test.rs"]
mod dot_4x_test;
