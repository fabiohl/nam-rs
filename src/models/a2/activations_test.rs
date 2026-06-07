// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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
