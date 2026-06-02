// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Activation functions module for the NAM A2 architecture.
//!
//! This module provides implementations of various non-linear activation functions
//! used in Neural Amp Modeler models, ensuring parity with the
//! original C++ implementation.
//!
//! IMPORTANT: A2 architecture support is in "placeholder" stage
//! pending stabilization of the reference implementation.

/// Activation types supported by NAM A2.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    /// Hyperbolic Tangent (standard).
    Tanh,
    /// Hyperbolic Tangent with hard saturation within [-1, 1].
    HardTanh,
    /// Fast rational approximation for Tanh.
    FastTanh,
    /// Rectified Linear Unit: max(0, x).
    ReLU,
    /// Leaky ReLU with fixed negative slope.
    LeakyReLU {
        /// Slope applied when x < 0.
        negative_slope: f32,
    },
    /// Parametric ReLU with per-channel or global negative slope.
    PReLU {
        /// Vector of slopes for negative values.
        negative_slopes: Vec<f32>,
    },
    /// Logistic Sigmoid function: 1 / (1 + exp(-x)).
    Sigmoid,
    /// Sigmoid Linear Unit: x * sigmoid(x).
    SiLU,
    /// Efficient version of the Swish function.
    HardSwish,
    /// HardTanh with configurable slopes for saturation regions.
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
                for x in data.iter_mut() {
                    *x = x.clamp(-1.0, 1.0);
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
mod tests {
    use super::*;

    #[test]
    fn test_activation_tanh() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::Tanh.apply(&mut data);
        let expected = [
            -5.0f32.tanh(),
            -1.0f32.tanh(),
            0.0f32.tanh(),
            1.0f32.tanh(),
            5.0f32.tanh(),
        ];
        for (i, &v) in data.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
        }
    }

    #[test]
    fn test_activation_hard_tanh() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::HardTanh.apply(&mut data);
        let expected = [-1.0, -1.0, 0.0, 1.0, 1.0];
        assert_eq!(data, expected);
    }

    #[test]
    fn test_activation_fast_tanh() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::FastTanh.apply(&mut data);
        // Values should be within the tanh output domain: (-1, 1).
        for &v in data.iter() {
            assert!((-1.0..=1.0).contains(&v));
        }
        // Parity with tanh at 0: fast_tanh(0) = 0 exactly.
        assert!((data[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_activation_relu() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::ReLU.apply(&mut data);
        let expected = [0.0, 0.0, 0.0, 1.0, 5.0];
        assert_eq!(data, expected);
    }

    #[test]
    fn test_activation_leaky_relu() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::LeakyReLU {
            negative_slope: 0.1,
        }
        .apply(&mut data);
        let expected = [-0.5, -0.1, 0.0, 1.0, 5.0];
        for (i, &v) in data.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
        }
    }

    #[test]
    fn test_activation_prelu() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::PReLU {
            negative_slopes: vec![0.1, 0.2],
        }
        .apply(&mut data);
        // data[0] (-5.0) uses slope[0]=0.1 -> -0.5
        // data[1] (-1.0) uses slope[1]=0.2 -> -0.2
        // data[2] (0.0) -> 0.0
        // data[3] (1.0) -> 1.0 (positive, passthrough)
        // data[4] (5.0) -> 5.0 (positive, passthrough)
        let expected = [-0.5, -0.2, 0.0, 1.0, 5.0];
        for (i, &v) in data.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
        }
    }

    #[test]
    fn test_activation_sigmoid() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::Sigmoid.apply(&mut data);
        fn sig(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }
        let expected = [sig(-5.0), sig(-1.0), sig(0.0), sig(1.0), sig(5.0)];
        for (i, &v) in data.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
        }
    }

    #[test]
    fn test_activation_silu() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::SiLU.apply(&mut data);
        fn silu(x: f32) -> f32 {
            x / (1.0 + (-x).exp())
        }
        let expected = [silu(-5.0), silu(-1.0), silu(0.0), silu(1.0), silu(5.0)];
        for (i, &v) in data.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
        }
    }

    #[test]
    fn test_activation_hard_swish() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::HardSwish.apply(&mut data);
        fn hswish(x: f32) -> f32 {
            x * (x + 3.0).clamp(0.0, 6.0) / 6.0
        }
        let expected = [
            hswish(-5.0),
            hswish(-1.0),
            hswish(0.0),
            hswish(1.0),
            hswish(5.0),
        ];
        for (i, &v) in data.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
        }
    }

    #[test]
    fn test_activation_leaky_hardtanh() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::LeakyHardTanh {
            min_val: -1.0,
            max_val: 1.0,
            min_slope: 0.1,
            max_slope: 0.2,
        }
        .apply(&mut data);
        // -5.0: < -1.0 -> (-5 - -1)*0.1 + -1 = -4*0.1 - 1 = -1.4
        // -1.0: in range -> -1.0
        // 0.0:  in range -> 0.0
        // 1.0:  in range -> 1.0
        // 5.0:  > 1.0   -> (5 - 1)*0.2 + 1 = 4*0.2 + 1 = 1.8
        let expected = [-1.4, -1.0, 0.0, 1.0, 1.8];
        for (i, &v) in data.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
        }
    }

    #[test]
    fn test_activation_softsign() {
        let mut data = [-5.0, -1.0, 0.0, 1.0, 5.0];
        ActivationType::Softsign.apply(&mut data);
        fn ss(x: f32) -> f32 {
            x / (1.0 + x.abs())
        }
        let expected = [ss(-5.0), ss(-1.0), ss(0.0), ss(1.0), ss(5.0)];
        for (i, &v) in data.iter().enumerate() {
            assert!((v - expected[i]).abs() < 1e-6, "At index {}", i);
        }
    }

    #[test]
    fn test_prelu_empty_slopes() {
        let mut data = [-1.0, 1.0];
        ActivationType::PReLU {
            negative_slopes: vec![],
        }
        .apply(&mut data);
        // With empty slopes, should operate as identity (early return).
        assert_eq!(data, [-1.0, 1.0]);
    }

    #[test]
    fn test_prelu_cycle() {
        let mut data = [-1.0, -1.0, -1.0, -1.0];
        ActivationType::PReLU {
            negative_slopes: vec![0.1, 0.5],
        }
        .apply(&mut data);
        // Slopes ciclam: [0.1, 0.5, 0.1, 0.5]
        assert_eq!(data, [-0.1, -0.5, -0.1, -0.5]);
    }
}
