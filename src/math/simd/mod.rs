// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Ponte de re-exportação para compatibilidade legada.
//! [T13] Infraestrutura movida para `math::common`.

pub mod avx2;
pub mod avx512;

// Re-exports de infraestrutura (agora em common)
pub use crate::math::common::aligned;
pub use crate::math::common::dispatch;
pub use crate::math::common::ops;
pub use crate::math::common::scalar_ref;
pub use crate::math::common::traits;
pub use crate::math::common::utility;

// Re-exports de kernels GEMM (Tarefa 3.2)
pub use crate::math::gemm::*;

// Re-exports de kernels Wavenet (Tarefa 3.4)
pub use crate::math::wavenet::*;

pub use aligned::AlignedVec;
pub use avx2::*;
pub use avx512::*;
pub use dispatch::{InstructionSet, SIMD_MATH, SimdMathConfig};
pub use ops::*;
pub use scalar_ref::*;
pub use traits::SimdMath;

// Re-export do macro dispatch_simd de common
pub use crate::math::common::dispatch_simd;

// DSP Re-exports (movidos para dsp/stereo)
/// Ponte de compatibilidade para operações DSP movidas para math::dsp.
pub mod ops_dsp {
    pub use crate::math::dsp::stereo::*;
}
pub use ops_dsp::*;

#[cfg(test)]
#[path = "simd_test_bridge.rs"]
mod simd_test_bridge;
