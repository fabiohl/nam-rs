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

/// Activation types supported by NAM A2.
///
/// Apenas `LeakyReLU` é exercitada pelo fast-path A2-Full/Lite.
/// Demais variantes: reservadas p/ motor A2 geral (futuro).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type")]
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
                crate::math::activations::hard_tanh_slice(data);
            }
            // FastTanh: A fast mathematical approximation of the processor's native Tanh function.
            // Uses a rational polynomial to avoid computing slow exponentials.
            Self::FastTanh => {
                crate::math::activations::fast_tanh_slice(data);
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
                crate::math::activations::hard_swish_slice(data);
            }
            // LeakyHardTanh: A hybrid version that acts like HardTanh, but in the saturation
            // zones (outside min_val/max_val) the signal continues to grow with attenuated gain.
            Self::LeakyHardTanh {
                min_val,
                max_val,
                min_slope,
                max_slope,
            } => {
                crate::math::activations::leaky_hard_tanh_slice(
                    data, *min_val, *max_val, *min_slope, *max_slope,
                );
            }
            // Softsign: Smooth symmetric curve similar to Tanh, given by the formula x / (1 + |x|),
            // being smoother at the edges and cheaper to compute on the processor.
            Self::Softsign => {
                crate::math::activations::softsign_slice(data);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::models::a2::activations::*;

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
            assert!((v - expected[i]).abs() < 5e-3, "At index {}", i);
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
        for &v in data.iter() {
            assert!((-1.0..=1.0).contains(&v));
        }
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
            assert!((v - expected[i]).abs() < 5e-4, "At index {}", i);
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
            assert!((v - expected[i]).abs() < 5e-3, "At index {}", i);
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
        assert_eq!(data, [-1.0, 1.0]);
    }

    #[test]
    fn test_prelu_cycle() {
        let mut data = [-1.0, -1.0, -1.0, -1.0];
        ActivationType::PReLU {
            negative_slopes: vec![0.1, 0.5],
        }
        .apply(&mut data);
        assert_eq!(data, [-0.1, -0.5, -0.1, -0.5]);
    }

    #[test]
    fn test_hard_tanh_avx2_parity() {

        #[target_feature(enable = "avx2")]
        unsafe fn scalar_ref(data: &mut [f32]) {
            for x in data.iter_mut() {
                *x = x.clamp(-1.0, 1.0);
            }
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe { crate::math::activations::hard_tanh_slice_avx2(&mut simd_data) };
        unsafe { scalar_ref(&mut ref_data) };

        for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "AVX2 parity mismatch at index {}: simd={}, ref={}",
                i,
                s,
                r
            );
        }
    }

    #[test]
    fn test_hard_tanh_avx2_large_slice() {

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            *x = x.clamp(-1.0, 1.0);
        }

        unsafe { crate::math::activations::hard_tanh_slice_avx2(&mut data) };

        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "AVX2 mismatch at index {}: got={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_hard_swish_avx2_parity() {

        #[target_feature(enable = "avx2")]
        unsafe fn scalar_ref(data: &mut [f32]) {
            for x in data.iter_mut() {
                let t = *x + 3.0;
                *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
            }
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe { crate::math::activations::hard_swish_slice_avx2(&mut simd_data) };
        unsafe { scalar_ref(&mut ref_data) };

        for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "AVX2 parity mismatch at index {}: simd={}, ref={}",
                i,
                s,
                r
            );
        }
    }

    #[test]
    fn test_hard_swish_avx2_large_slice() {

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            let t = *x + 3.0;
            *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
        }

        unsafe { crate::math::activations::hard_swish_slice_avx2(&mut data) };

        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "AVX2 mismatch at index {}: got={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_leaky_hard_tanh_avx2_parity() {

        let min_val = -1.0_f32;
        let max_val = 1.0_f32;
        let min_slope = 0.15_f32;
        let max_slope = 0.25_f32;

        #[target_feature(enable = "avx2")]
        unsafe fn scalar_ref(
            data: &mut [f32],
            min_val: f32,
            max_val: f32,
            min_slope: f32,
            max_slope: f32,
        ) {
            for x in data.iter_mut() {
                if *x < min_val {
                    *x = (*x - min_val) * min_slope + min_val;
                } else if *x > max_val {
                    *x = (*x - max_val) * max_slope + max_val;
                }
            }
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe {
            crate::math::activations::leaky_hard_tanh_slice_avx2(
                &mut simd_data,
                min_val,
                max_val,
                min_slope,
                max_slope,
            )
        };
        unsafe { scalar_ref(&mut ref_data, min_val, max_val, min_slope, max_slope) };

        for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "AVX2 parity mismatch at index {}: simd={}, ref={}",
                i,
                s,
                r
            );
        }
    }

    #[test]
    fn test_leaky_hard_tanh_avx2_large_slice() {

        let min_val = -1.5_f32;
        let max_val = 1.5_f32;
        let min_slope = 0.1_f32;
        let max_slope = 0.3_f32;

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            if *x < min_val {
                *x = (*x - min_val) * min_slope + min_val;
            } else if *x > max_val {
                *x = (*x - max_val) * max_slope + max_val;
            }
        }

        unsafe {
            crate::math::activations::leaky_hard_tanh_slice_avx2(
                &mut data, min_val, max_val, min_slope, max_slope,
            )
        };

        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "AVX2 mismatch at index {}: got={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_fast_tanh_avx2_parity() {

        #[target_feature(enable = "avx2")]
        #[allow(clippy::excessive_precision)]
        unsafe fn scalar_ref(data: &mut [f32]) {
            for x in data.iter_mut() {
                let xv = *x;
                let ax = xv.abs();
                let x2 = xv * xv;
                *x = (xv
                    * (2.45550750702956
                        + 2.45550750702956 * ax
                        + (0.893229853513558 + 0.821226666969744 * ax) * x2))
                    / (2.44506634652299
                        + (2.44506634652299 + x2) * (xv + 0.814642734961073 * xv * ax).abs());
            }
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe { crate::math::activations::fast_tanh_slice_avx2(&mut simd_data) };
        unsafe { scalar_ref(&mut ref_data) };

        for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-5,
                "AVX2 parity mismatch at index {}: simd={}, ref={}",
                i,
                s,
                r
            );
        }
    }

    #[test]
    fn test_fast_tanh_avx2_large_slice() {

        #[allow(clippy::excessive_precision)]
        fn fast_tanh_scalar(x: f32) -> f32 {
            let ax = x.abs();
            let x2 = x * x;
            (x * (2.45550750702956
                + 2.45550750702956 * ax
                + (0.893229853513558 + 0.821226666969744 * ax) * x2))
                / (2.44506634652299
                    + (2.44506634652299 + x2) * (x + 0.814642734961073 * x * ax).abs())
        }

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            *x = fast_tanh_scalar(*x);
        }

        unsafe { crate::math::activations::fast_tanh_slice_avx2(&mut data) };

        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "AVX2 mismatch at index {}: got={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    // --- AVX-512 parity tests ---

    #[test]
    fn test_hard_tanh_avx512_parity() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe { crate::math::activations::hard_tanh_slice_avx512(&mut simd_data) };
        for x in ref_data.iter_mut() {
            *x = x.clamp(-1.0, 1.0);
        }

        for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "AVX-512 parity mismatch at index {}: simd={}, ref={}",
                i,
                s,
                r
            );
        }
    }

    #[test]
    fn test_hard_tanh_avx512_large_slice() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            *x = x.clamp(-1.0, 1.0);
        }

        unsafe { crate::math::activations::hard_tanh_slice_avx512(&mut data) };

        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "AVX-512 mismatch at index {}: got={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_hard_swish_avx512_parity() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe { crate::math::activations::hard_swish_slice_avx512(&mut simd_data) };
        for x in ref_data.iter_mut() {
            let t = *x + 3.0;
            *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
        }

        for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "AVX-512 parity mismatch at index {}: simd={}, ref={}",
                i,
                s,
                r
            );
        }
    }

    #[test]
    fn test_hard_swish_avx512_large_slice() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            let t = *x + 3.0;
            *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
        }

        unsafe { crate::math::activations::hard_swish_slice_avx512(&mut data) };

        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "AVX-512 mismatch at index {}: got={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_leaky_hard_tanh_avx512_parity() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let min_val = -1.0_f32;
        let max_val = 1.0_f32;
        let min_slope = 0.15_f32;
        let max_slope = 0.25_f32;

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe {
            crate::math::activations::leaky_hard_tanh_slice_avx512(
                &mut simd_data,
                min_val,
                max_val,
                min_slope,
                max_slope,
            )
        };
        for x in ref_data.iter_mut() {
            if *x < min_val {
                *x = (*x - min_val) * min_slope + min_val;
            } else if *x > max_val {
                *x = (*x - max_val) * max_slope + max_val;
            }
        }

        for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-6,
                "AVX-512 parity mismatch at index {}: simd={}, ref={}",
                i,
                s,
                r
            );
        }
    }

    #[test]
    fn test_leaky_hard_tanh_avx512_large_slice() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        let min_val = -1.5_f32;
        let max_val = 1.5_f32;
        let min_slope = 0.1_f32;
        let max_slope = 0.3_f32;

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            if *x < min_val {
                *x = (*x - min_val) * min_slope + min_val;
            } else if *x > max_val {
                *x = (*x - max_val) * max_slope + max_val;
            }
        }

        unsafe {
            crate::math::activations::leaky_hard_tanh_slice_avx512(
                &mut data, min_val, max_val, min_slope, max_slope,
            )
        };

        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "AVX-512 mismatch at index {}: got={}, expected={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_fast_tanh_avx512_parity() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        #[allow(clippy::excessive_precision)]
        fn fast_tanh_scalar(x: f32) -> f32 {
            let ax = x.abs();
            let x2 = x * x;
            (x * (2.45550750702956
                + 2.45550750702956 * ax
                + (0.893229853513558 + 0.821226666969744 * ax) * x2))
                / (2.44506634652299
                    + (2.44506634652299 + x2) * (x + 0.814642734961073 * x * ax).abs())
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe { crate::math::activations::fast_tanh_slice_avx512(&mut simd_data) };
        for x in ref_data.iter_mut() {
            *x = fast_tanh_scalar(*x);
        }

        for (i, (&s, &r)) in simd_data.iter().zip(ref_data.iter()).enumerate() {
            assert!(
                (s - r).abs() < 1e-5,
                "AVX-512 parity mismatch at index {}: simd={}, ref={}",
                i,
                s,
                r
            );
        }
    }

    #[test]
    fn test_fast_tanh_avx512_large_slice() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }

        #[allow(clippy::excessive_precision)]
        fn fast_tanh_scalar(x: f32) -> f32 {
            let ax = x.abs();
            let x2 = x * x;
            (x * (2.45550750702956
                + 2.45550750702956 * ax
                + (0.893229853513558 + 0.821226666969744 * ax) * x2))
                / (2.44506634652299
                    + (2.44506634652299 + x2) * (x + 0.814642734961073 * x * ax).abs())
        }

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            *x = fast_tanh_scalar(*x);
        }

        unsafe { crate::math::activations::fast_tanh_slice_avx512(&mut data) };

        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "AVX-512 mismatch at index {}: got={}, expected={}",
                i,
                a,
                b
            );
        }
    }
}
