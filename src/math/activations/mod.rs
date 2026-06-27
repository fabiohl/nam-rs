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

pub mod fast_tanh;
pub mod fused;
pub mod hard_swish;
pub mod hard_tanh;
pub mod kernel_macro;
pub mod leaky_hard_tanh;
pub mod prelu;
pub mod relu;
pub mod sigmoid;
pub mod silu;
pub mod softsign;
pub mod tanh;

#[cfg(test)]
mod activations_test;

pub use fast_tanh::*;
pub use fused::*;
pub use hard_swish::*;
pub use hard_tanh::*;
pub use leaky_hard_tanh::*;
pub use prelu::*;
pub use relu::*;
pub use sigmoid::*;
pub use silu::*;
pub use softsign::*;
pub use tanh::*;

/// Activation precision mode for tanh/sigmoid approximation.
///
/// Controls whether the production Padé/minimax kernels or the
/// high-fidelity polynomial exp-based kernels are used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
#[allow(clippy::module_name_repetitions)]
pub enum ActivationPrecision {
    /// Padé [5,4] tanh (~2.32e-3) + minimax degree-17 sigmoid (~4.09e-4).
    /// Current production default — fastest path.
    Standard = 0,
    /// Polynomial exp-based tanh/sigmoid (~2.4e-7 / ~2.1e-7).
    /// Higher cost, lower error, lower aliasing (offline/mixdown).
    HighFidelity = 1,
}

/// Global activation precision mode (default: `Standard`).
///
/// # Thread Safety
/// This is a relaxed atomic — the mode is set once during initialisation
/// (or on a hot-swap rebuild) and never changed mid-`process`.  The
/// branch predictor will specialise to whichever path is stable.
static ACTIVATION_MODE: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(ActivationPrecision::Standard as usize);

/// Set the global activation precision mode.
///
/// Must be called **outside** the real-time audio thread (during model build
/// or hot-swap), before the first call to `tanh_slice`/`sigmoid_slice`.
#[inline]
pub fn set_activation_precision(mode: ActivationPrecision) {
    ACTIVATION_MODE.store(mode as usize, core::sync::atomic::Ordering::Relaxed);
}

/// Returns the current global activation precision mode.
#[inline]
pub fn activation_precision() -> ActivationPrecision {
    match ACTIVATION_MODE.load(core::sync::atomic::Ordering::Relaxed) {
        1 => ActivationPrecision::HighFidelity,
        _ => ActivationPrecision::Standard,
    }
}

/// Applies Tanh activation to a slice of f32 with automatic dispatch to the best SIMD implementation.
#[inline(always)]
pub fn tanh_slice(data: &mut [f32]) {
    if activation_precision() == ActivationPrecision::HighFidelity {
        crate::math::common::dispatch_simd!(tanh_slice_hf(data));
    } else {
        crate::math::common::dispatch_simd!(tanh_slice(data));
    }
}

/// Applies Sigmoid activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn sigmoid_slice(data: &mut [f32]) {
    if activation_precision() == ActivationPrecision::HighFidelity {
        crate::math::common::dispatch_simd!(sigmoid_slice_hf(data));
    } else {
        crate::math::common::dispatch_simd!(sigmoid_slice(data));
    }
}

/// Applies ReLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn relu_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(relu_slice(data));
}

/// Applies PReLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn prelu_slice(data: &mut [f32], slopes: &[f32]) {
    crate::math::common::dispatch_simd!(prelu_slice(data, slopes));
}

/// Applies Softsign activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn softsign_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(softsign_slice(data));
}

/// Applies SiLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn silu_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(silu_slice(data));
}

/// Applies HardTanh activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn hard_tanh_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(hard_tanh_slice(data));
}

/// Applies HardSwish activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn hard_swish_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(hard_swish_slice(data));
}

/// Applies FastTanh activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn fast_tanh_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(fast_tanh_slice(data));
}

/// Applies LeakyHardTanh activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn leaky_hard_tanh_slice(
    data: &mut [f32],
    min_val: f32,
    max_val: f32,
    min_slope: f32,
    max_slope: f32,
) {
    crate::math::common::dispatch_simd!(leaky_hard_tanh_slice(
        data, min_val, max_val, min_slope, max_slope
    ));
}
