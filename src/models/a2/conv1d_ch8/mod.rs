// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! CH=8 T=8 frame-tiled tap-major convolution with broadcast-FMA (T2.2).
//!
//! When `out_ch == 8`, the generic 4-wide interleaved scheme processes frames
//! one at a time. This module implements a **block-level** kernel that processes
//! 8 consecutive frames per tile, amortizing weight loads across all 8 frames
//! via SIMD broadcast-FMA instructions. Weights are stored in **col-major-per-tap**
//! layout (`w[k * 64 + in * 8 + out]`) so that the 8 output-channel weights for
//! a single (tap, input_channel) pair are contiguous — one `_mm256_loadu_ps` loads
//! them all.
//!
//! For each tile of T=8 frames, the inner loop is:
//!
//! ```text
//! a[f][o] += Wcol[o] * h[f]   (o vectorized, h[f] scalar broadcast)
//! ```
//!
//! On x86-64-v3 this emits `vfmadd231ps` (broadcast-FMA).
//!
//! ## Source of truth
//! - `a2_fast.cpp:617-681` (strategy `Channels >= 8`, T=4 tap-major).
//!   Elevated to T=8 to saturate FMA ports.

use crate::math::common::AlignedVec;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

/// Maximum frames per kernel invocation.
/// Guaranteed by `process()` internal chunking (T2.1).
const MAX_KERNEL_FRAMES: usize = WAVENET_MAX_NUM_FRAMES;

// =============================================================================
// A2Conv1dCh8 — CH=8 convolution with col-major-per-tap f32 weights
// =============================================================================

/// CH=8 dilated causal Conv1D weights in col-major-per-tap layout.
///
/// Layout: `w[k * 64 + in_ch * 8 + out_ch]`
/// - `k`: kernel tap index (0..K-1)
/// - `in_ch`: input channel (0..7)
/// - `out_ch`: output channel (0..7)
///
/// For a given `(k, in_ch)`, the 8 output weights are contiguous → one AVX2 load.
#[derive(Clone)]
#[repr(align(64))]
pub struct A2Conv1dCh8 {
    /// Col-major-per-tap f32 weights: `kernel_size * 64` elements.
    pub weights: AlignedVec<f32>,
    /// Bias vector [8], f32.
    pub bias: AlignedVec<f32>,
    /// Temporal dilation factor.
    pub dilation: usize,
    /// Kernel size (6 or 15 for A2).
    pub kernel: usize,
}

impl A2Conv1dCh8 {
    /// Builds a CH=8 conv1d from the weight data read in NAM JSON order.
    ///
    /// `raw` is in NAM JSON row-major order: `[out_ch][in_ch][kernel]`.
    /// This constructor permutes to col-major-per-tap: `[kernel][in_ch][out_ch]`.
    pub fn new(
        raw_weights: &[f32],
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        dilation: usize,
        bias: AlignedVec<f32>,
    ) -> Self {
        debug_assert_eq!(out_ch, 8);
        debug_assert_eq!(in_ch, 8);
        debug_assert!(kernel == 6 || kernel == 15);
        debug_assert_eq!(raw_weights.len(), out_ch * in_ch * kernel);
        debug_assert_eq!(bias.len(), out_ch);

        let mut weights = AlignedVec::new(kernel * 64, 0.0f32);
        for out in 0..out_ch {
            for inp in 0..in_ch {
                for k in 0..kernel {
                    let src = out * in_ch * kernel + inp * kernel + k;
                    let dst = k * 64 + inp * 8 + out;
                    weights[dst] = raw_weights[src];
                }
            }
        }

        Self {
            weights,
            bias,
            dilation,
            kernel,
        }
    }
}

mod scalar;
mod simd;
pub use scalar::*;
pub use simd::*;

#[cfg(test)]
#[path = "../conv1d_ch8_test.rs"]
mod tests;
