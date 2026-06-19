// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Activation functions module for the NAM A2 architecture.
//!
//! This module provides implementations of various non-linear activation functions
//! used in Neural Amp Modeler models, ensuring parity with the
//! original C++ implementation.
//!
//! ## Escopo atual (fast-path A2-Full/Lite)
//!
//! Apenas `LeakyReLU { negative_slope: 0.01 }` é utilizada pelo fast-path
//! (`a2_fast.cpp`), aplicada de forma homogênea em todas as 23 camadas.
//!
//! ## Reservado p/ motor A2 geral (futuro)
//!
//! Todas as demais variantes de `ActivationType` são preservadas para suporte
//! futuro ao motor A2 completo (ativações heterogêneas por camada, FiLM, gating).

use core::arch::x86_64::*;

/// Activation types supported by NAM A2.
///
/// Apenas `LeakyReLU` é exercitada pelo fast-path A2-Full/Lite.
/// Demais variantes: reservadas p/ motor A2 geral (futuro).
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    /// Hyperbolic Tangent (standard).
    /// NOTE: reservado p/ motor A2 geral (futuro).
    Tanh,
    /// Hyperbolic Tangent with hard saturation within [-1, 1].
    /// NOTE: reservado p/ motor A2 geral (futuro).
    HardTanh,
    /// Fast rational approximation for Tanh.
    /// NOTE: reservado p/ motor A2 geral (futuro).
    FastTanh,
    /// Rectified Linear Unit: max(0, x).
    /// NOTE: reservado p/ motor A2 geral (futuro).
    ReLU,
    /// Leaky ReLU with fixed negative slope (fast-path A2-Full/Lite: slope = 0.01).
    LeakyReLU {
        /// Slope applied when x < 0.
        negative_slope: f32,
    },
    /// Parametric ReLU with per-channel or global negative slope.
    /// NOTE: reservado p/ motor A2 geral (futuro).
    PReLU {
        /// Vector of slopes for negative values.
        negative_slopes: Vec<f32>,
    },
    /// Logistic Sigmoid function: 1 / (1 + exp(-x)).
    /// NOTE: reservado p/ motor A2 geral (futuro).
    Sigmoid,
    /// Sigmoid Linear Unit: x * sigmoid(x).
    /// NOTE: reservado p/ motor A2 geral (futuro).
    SiLU,
    /// Efficient version of the Swish function.
    /// NOTE: reservado p/ motor A2 geral (futuro).
    HardSwish,
    /// HardTanh with configurable slopes for saturation regions.
    /// NOTE: reservado p/ motor A2 geral (futuro).
    LeakyHardTanh {
        /// Minimum value for linear saturation.
        min_val: f32,
        /// Maximum value for linear saturation.
        max_val: f32,
        /// Slope applied below `min_val`.
        min_slope: f32,
        /// Slope applied above `max_val`.
        max_slope: f32,
    },
    /// Softsign: x / (1 + |x|).
    /// NOTE: reservado p/ motor A2 geral (futuro).
    Softsign,
}

/// Trait for applying activation functions on audio buffers.
pub trait ActivationFn {
    /// Applies the activation function in-place on the provided buffer.
    fn apply(&self, data: &mut [f32]);
}

impl ActivationFn for ActivationType {
    fn apply(&self, data: &mut [f32]) {
        match self {
            // Tanh (Hyperbolic Tangent): Smooth S-shaped curve that compresses
            // any input value to the range [-1.0, 1.0].
            Self::Tanh => {
                crate::math::activations::tanh_slice(data);
            }
            // HardTanh (Rigid Hyperbolic Tangent): Abrupt saturation (hard clipping/culling).
            // Limits values strictly to a minimum of -1.0 and a maximum of 1.0.
            Self::HardTanh => {
                if is_x86_feature_detected!("avx2") {
                    unsafe { hard_tanh_slice_avx2(data) }
                } else {
                    for x in data.iter_mut() {
                        *x = x.clamp(-1.0, 1.0);
                    }
                }
            }
            // FastTanh: A fast mathematical approximation of the processor's native Tanh function.
            // Uses a rational polynomial to avoid computing slow exponentials.
            Self::FastTanh => {
                for x in data.iter_mut() {
                    *x = fast_tanh(*x);
                }
            }
            // ReLU (Rectified Linear Unit): Zeros all negative values, letting
            // positive values pass through without any change.
            Self::ReLU => {
                crate::math::activations::relu_slice(data);
            }
            // LeakyReLU: Similar to ReLU, but instead of completely zeroing negative values,
            // applies a small multiplier (fixed negative slope) retaining a fraction of the signal.
            Self::LeakyReLU { negative_slope } => {
                let slopes = [*negative_slope];
                crate::math::activations::prelu_slice(data, &slopes);
            }
            // PReLU (Parametric ReLU): Allows adjustable negative slopes learned by the model.
            // The multiplicative factors can vary per channel or element.
            Self::PReLU { negative_slopes } => {
                if negative_slopes.is_empty() {
                    return;
                }
                crate::math::activations::prelu_slice(data, negative_slopes);
            }
            // Sigmoid: Logistic function that smoothly maps the input between 0.0 and 1.0,
            // widely used to compute gating factors (gates) for switching signals on/off.
            Self::Sigmoid => {
                crate::math::activations::sigmoid_slice(data);
            }
            // SiLU (Sigmoid Linear Unit): Multiplies the input value by its own sigmoid
            // activation. Also known as Swish.
            Self::SiLU => {
                crate::math::activations::silu_slice(data);
            }
            // HardSwish: A linear approximation of the Swish/SiLU function designed to be
            // computed efficiently without calculating complex exponential functions.
            Self::HardSwish => {
                for x in data.iter_mut() {
                    let t = *x + 3.0;
                    let clamped = t.clamp(0.0, 6.0);
                    *x *= clamped * (1.0 / 6.0);
                }
            }
            // LeakyHardTanh: A hybrid version that acts like HardTanh, but in the saturation
            // zones (outside min_val/max_val) the signal continues to grow with attenuated gain.
            Self::LeakyHardTanh {
                min_val,
                max_val,
                min_slope,
                max_slope,
            } => {
                for x in data.iter_mut() {
                    if *x < *min_val {
                        *x = (*x - *min_val) * *min_slope + *min_val;
                    } else if *x > *max_val {
                        *x = (*x - *max_val) * *max_slope + *max_val;
                    }
                }
            }
            // Softsign: Smooth symmetric curve similar to Tanh, given by the formula x / (1 + |x|),
            // being smoother at the edges and cheaper to compute on the processor.
            Self::Softsign => {
                crate::math::activations::softsign_slice(data);
            }
        }
    }
}

/// AVX2-accelerated HardTanh: `clamp(x, -1.0, 1.0)` over a slice.
///
/// Processes 16 elements per iteration (2× `__m256`), then 8, then scalar remainder.
///
/// # Safety
/// Requires AVX2 support.
#[target_feature(enable = "avx2")]
pub unsafe fn hard_tanh_slice_avx2(data: &mut [f32]) {
    let neg_one = _mm256_set1_ps(-1.0_f32);
    let pos_one = _mm256_set1_ps(1.0_f32);
    let mut i = 0;
    let len = data.len();
    while i + 16 <= len {
        unsafe {
            let x1 = _mm256_loadu_ps(data.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
            _mm256_storeu_ps(
                data.as_mut_ptr().add(i),
                _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x1)),
            );
            _mm256_storeu_ps(
                data.as_mut_ptr().add(i + 8),
                _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x2)),
            );
        }
        i += 16;
    }
    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(data.as_ptr().add(i));
            _mm256_storeu_ps(
                data.as_mut_ptr().add(i),
                _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x)),
            );
        }
        i += 8;
    }
    for x in data.iter_mut().skip(i) {
        *x = x.clamp(-1.0, 1.0);
    }
}

/// Fast rational approximation for the tanh function.
/// Parity with `NAM/activations.h` L111-122.
#[inline(always)]
#[allow(clippy::excessive_precision)]
fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;

    (x * (2.45550750702956
        + 2.45550750702956 * ax
        + (0.893229853513558 + 0.821226666969744 * ax) * x2))
        / (2.44506634652299 + (2.44506634652299 + x2) * (x + 0.814642734961073 * x * ax).abs())
}

#[cfg(test)]
#[path = "activations_test.rs"]
mod activations_test;
