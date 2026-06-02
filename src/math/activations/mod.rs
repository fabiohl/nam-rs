// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Central activations module with automatic SIMD dispatch.
//!
//! Provides ultra-fast and branchless implementations of `tanh` and `sigmoid`.
//! - `tanh(x)` uses piecewise minimax odd polynomials (degree 5) with branchless
//!   blending for |x| < 4, saturated to [-1, 1] (7 symmetric sub-intervals).
//! - `sigmoid(x)` uses a direct degree-17 minimax polynomial (9 odd terms) for |x| < 8,
//!   saturated to [0, 1].  Max error ~4.09e-4 vs `f32::exp` reference.
//!
//! Reference for tanh: VDT library (CERN), Mineiro & Vorlicek (2016).
//! Sigmoid coefficients: Lawson-weighted minimax on [-8, 8].
//!
//! # Features
//! - **FastMath**: Approximations that prioritize speed while keeping error < 5e-3 (FP16-compatible).
//! - **Branchless**: Zero branches — only SIMD masks (max/min/blend).
//! - **Dispatch**: Automatic kernel selection (AVX2, AVX-512) via CPUID.
//! - **Zero-Alloc**: Operations strictly safe for use in real-time threads.
//! - **AMX/AVX10.2-ready**: Branchless approach compatible with future extensions.

pub mod fused;
pub mod prelu;
pub mod relu;
pub mod sigmoid;
pub mod silu;
pub mod softsign;
pub mod tanh;

#[cfg(test)]
mod tests;

pub use fused::*;
pub use prelu::*;
pub use relu::*;
pub use sigmoid::*;
pub use silu::*;
pub use softsign::*;
pub use tanh::*;

use crate::math::common::{InstructionSet, SIMD_MATH};

/// Applies Tanh activation to a slice of f32 with automatic dispatch to the best SIMD implementation.
#[inline(always)]
pub fn tanh_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { tanh::tanh_slice_avx512(data) };
        }
        _ => {
            unsafe { tanh::tanh_slice_avx2(data) };
        }
    }
}

/// Applies Sigmoid activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn sigmoid_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { sigmoid::sigmoid_slice_avx512(data) };
        }
        _ => {
            unsafe { sigmoid::sigmoid_slice_avx2(data) };
        }
    }
}

/// Applies ReLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn relu_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { relu::relu_slice_avx512(data) };
        }
        _ => {
            unsafe { relu::relu_slice_avx2(data) };
        }
    }
}

/// Applies PReLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn prelu_slice(data: &mut [f32], slopes: &[f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { prelu::prelu_slice_avx512(data, slopes) };
        }
        _ => {
            unsafe { prelu::prelu_slice_avx2(data, slopes) };
        }
    }
}

/// Applies Softsign activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn softsign_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { softsign::softsign_slice_avx512(data) };
        }
        _ => {
            unsafe { softsign::softsign_slice_avx2(data) };
        }
    }
}

/// Applies SiLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn silu_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { silu::silu_slice_avx512(data) };
        }
        _ => {
            unsafe { silu::silu_slice_avx2(data) };
        }
    }
}
