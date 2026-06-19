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

    #[test]
    fn test_hard_tanh_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        #[target_feature(enable = "avx2")]
        unsafe fn scalar_ref(data: &mut [f32]) {
            for x in data.iter_mut() {
                *x = x.clamp(-1.0, 1.0);
            }
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe { super::super::hard_tanh_slice_avx2(&mut simd_data) };
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
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            *x = x.clamp(-1.0, 1.0);
        }

        unsafe { super::super::hard_tanh_slice_avx2(&mut data) };

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
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        #[target_feature(enable = "avx2")]
        unsafe fn scalar_ref(data: &mut [f32]) {
            for x in data.iter_mut() {
                let t = *x + 3.0;
                *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
            }
        }

        let mut simd_data: Vec<f32> = (-20..21).map(|i| i as f32 * 0.43).collect();
        let mut ref_data = simd_data.clone();

        unsafe { super::super::hard_swish_slice_avx2(&mut simd_data) };
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
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let mut data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let mut expected = data.clone();
        for x in expected.iter_mut() {
            let t = *x + 3.0;
            *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
        }

        unsafe { super::super::hard_swish_slice_avx2(&mut data) };

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
        if !is_x86_feature_detected!("avx2") {
            return;
        }

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
            super::super::leaky_hard_tanh_slice_avx2(
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
        if !is_x86_feature_detected!("avx2") {
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
            super::super::leaky_hard_tanh_slice_avx2(
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
        if !is_x86_feature_detected!("avx2") {
            return;
        }

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

        unsafe { super::super::fast_tanh_slice_avx2(&mut simd_data) };
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
        if !is_x86_feature_detected!("avx2") {
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

        unsafe { super::super::fast_tanh_slice_avx2(&mut data) };

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
}
