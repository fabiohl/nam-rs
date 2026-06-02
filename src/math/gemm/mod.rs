// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SIMD linear algebra kernels (GEMM, GEMV, Dot Product).
//!
//! This module is the high-throughput engine of NAM-rs, responsible for the
//! massive multiplication of weights by neural network states.
//!
//! # Performance Strategies
//! - **ILP (Instruction Level Parallelism)**: Multiple accumulators to saturate the FMA ports.
//! - **F16C Compression**: Weights stored in half-precision to double cache throughput.
//! - **Interleaved Layout**: Weights organized to maximize data reuse in registers.
//! - **Tiling**: Block processing to optimize data locality.
//!
//! Extracted from `simd/avx2.rs` and `simd/avx512.rs` during Task 3.2.
//! Contains AVX2 and AVX-512 implementations side by side, organized by operation.

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
