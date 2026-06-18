// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Tanh (Hyperbolic Tangent) activation kernels.
//!
//! - **Production path:** Padé [5,4] rational approximant with hardware division
//!   (`simd_tanh_avx2`, `simd_tanh_dual_avx2`, `simd_tanh_avx512`).
//! - **Reference path:** Padé NR2 variants retained for benchmarking and
//!   documentation (`simd_tanh_pade_nr2_avx2`, `simd_tanh_pade_nr2_avx512`).

/// Polynomial exp-based tanh/sigmoid kernels (degree-6 Taylor, ≤ 1e-6 error).
pub mod high_fidelity;
pub mod production;
/// Experimental / reference Padé NR2 variants retained for benchmarking.
pub mod reference;

pub use high_fidelity::*;
pub use production::*;
pub use reference::*;
