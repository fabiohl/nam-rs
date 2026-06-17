// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for high-fidelity tanh/sigmoid SIMD kernels (T-HF1.1, T-HF1.3).
//!
//! Validates max absolute error ≤ 1e-6 vs `f32::tanh`/`f32::exp` references
//! on [-20, 20] dense sweep, edge cases, and saturation at extremes.
//!
//! AVX-512 tests are gated on `is_x86_feature_detected!("avx512f")`.

use super::*;

const DENSE_SWEEP_POINTS: usize = 4001; // 0.01 step on [-20, 20]

// ══════════════════════════════════════════════════════════════════════════════
// Tanh sweep — AVX2
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_hifi_avx2_sweep() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(8) {
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y = simd_tanh_hifi_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "tanh_hifi({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    let remainder = sweep.chunks_exact(8).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 8];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        unsafe {
            let x = _mm256_loadu_ps(batch.as_ptr());
            let y = simd_tanh_hifi_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "tanh_hifi({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    eprintln!("[T-HF1.1] tanh_hifi AVX2 sweep max error: {max_error:.4e} (limit 1e-6)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Sigmoid sweep — AVX2
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sigmoid_hifi_avx2_sweep() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(8) {
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y = simd_sigmoid_hifi_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = 1.0 / (1.0 + (-input).exp());
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "sigmoid_hifi({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    let remainder = sweep.chunks_exact(8).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 8];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        unsafe {
            let x = _mm256_loadu_ps(batch.as_ptr());
            let y = simd_sigmoid_hifi_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = 1.0 / (1.0 + (-input).exp());
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "sigmoid_hifi({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    eprintln!("[T-HF1.1] sigmoid_hifi AVX2 sweep max error: {max_error:.4e} (limit 1e-6)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Edge cases
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_hifi_edge_cases() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let test_vals: [f32; 9] = [-100.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 100.0, f32::NAN];

    unsafe {
        for &x in &test_vals {
            let vx = _mm256_set1_ps(x);
            let vy = simd_tanh_hifi_avx2(vx);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), vy);
            let y = result[0];

            if x.is_nan() {
                assert!(y.is_nan(), "tanh_hifi(NaN) should be NaN, got {y}");
            } else {
                assert!(
                    (-1.0..=1.0).contains(&y),
                    "tanh_hifi({x}) = {y} out of [-1, 1]"
                );
            }
        }
    }
}

#[test]
fn test_sigmoid_hifi_edge_cases() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let test_vals: [f32; 9] = [-100.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 100.0, f32::NAN];

    unsafe {
        for &x in &test_vals {
            let vx = _mm256_set1_ps(x);
            let vy = simd_sigmoid_hifi_avx2(vx);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), vy);
            let y = result[0];

            if x.is_nan() {
                assert!(y.is_nan(), "sigmoid_hifi(NaN) should be NaN, got {y}");
            } else {
                assert!(
                    (0.0..=1.0).contains(&y),
                    "sigmoid_hifi({x}) = {y} out of [0, 1]"
                );
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Saturation
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_hifi_saturation() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    unsafe {
        let mut result = [0.0_f32; 8];

        // Large positive → 1
        let vx = _mm256_set1_ps(1000.0f32);
        let vy = simd_tanh_hifi_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0] - 1.0).abs() < 1e-6, "tanh(+∞) should be 1");

        // Large negative → -1
        let vx = _mm256_set1_ps(-1000.0f32);
        let vy = simd_tanh_hifi_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0] + 1.0).abs() < 1e-6, "tanh(-∞) should be -1");

        // Zero → 0
        let vx = _mm256_set1_ps(0.0f32);
        let vy = simd_tanh_hifi_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert_eq!(result[0], 0.0, "tanh(0) should be 0");
    }
}

#[test]
fn test_sigmoid_hifi_saturation() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    unsafe {
        let mut result = [0.0_f32; 8];

        // Large positive → 1
        let vx = _mm256_set1_ps(1000.0f32);
        let vy = simd_sigmoid_hifi_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0] - 1.0).abs() < 1e-6, "σ(+∞) should be 1");

        // Large negative → 0
        let vx = _mm256_set1_ps(-1000.0f32);
        let vy = simd_sigmoid_hifi_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0]).abs() < 1e-6, "σ(-∞) should be 0");

        // Zero → 0.5
        let vx = _mm256_set1_ps(0.0f32);
        let vy = simd_sigmoid_hifi_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0] - 0.5).abs() < 1e-6, "σ(0) should be 0.5");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Dual gate
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_sigmoid_dual_hifi_avx2() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let test_vals: [f32; 7] = [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0];

    unsafe {
        for &x1_val in &test_vals {
            for &x2_val in &test_vals {
                let x1 = _mm256_set1_ps(x1_val);
                let x2 = _mm256_set1_ps(x2_val);
                let (t, s) = simd_tanh_sigmoid_dual_hifi_avx2(x1, x2);

                let mut t_arr = [0.0_f32; 8];
                let mut s_arr = [0.0_f32; 8];
                _mm256_storeu_ps(t_arr.as_mut_ptr(), t);
                _mm256_storeu_ps(s_arr.as_mut_ptr(), s);

                let expected_tanh = x1_val.tanh();
                let expected_sig = 1.0 / (1.0 + (-x2_val).exp());

                assert!(
                    (t_arr[0] - expected_tanh).abs() <= 1e-6,
                    "dual tanh({x1_val}) = {}, expected {expected_tanh}",
                    t_arr[0],
                );
                assert!(
                    (s_arr[0] - expected_sig).abs() <= 1e-6,
                    "dual sigmoid({x2_val}) = {}, expected {expected_sig}",
                    s_arr[0],
                );
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AVX-512 sweeps (T-HF1.3) — gated on hardware availability
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_hifi_avx512_sweep() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(16) {
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y = simd_tanh_hifi_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "tanh_hifi_avx512({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    let remainder = sweep.chunks_exact(16).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 16];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        unsafe {
            let x = _mm512_loadu_ps(batch.as_ptr());
            let y = simd_tanh_hifi_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "tanh_hifi_avx512({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    eprintln!("[T-HF1.3] tanh_hifi AVX-512 sweep max error: {max_error:.4e} (limit 1e-6)");
}

#[test]
fn test_sigmoid_hifi_avx512_sweep() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(16) {
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y = simd_sigmoid_hifi_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = 1.0 / (1.0 + (-input).exp());
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "sigmoid_hifi_avx512({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    let remainder = sweep.chunks_exact(16).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 16];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        unsafe {
            let x = _mm512_loadu_ps(batch.as_ptr());
            let y = simd_sigmoid_hifi_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = 1.0 / (1.0 + (-input).exp());
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "sigmoid_hifi_avx512({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    eprintln!("[T-HF1.3] sigmoid_hifi AVX-512 sweep max error: {max_error:.4e} (limit 1e-6)");
}

#[test]
fn test_tanh_sigmoid_dual_hifi_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let test_vals: [f32; 7] = [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0];

    unsafe {
        for &x1_val in &test_vals {
            for &x2_val in &test_vals {
                let x1 = _mm512_set1_ps(x1_val);
                let x2 = _mm512_set1_ps(x2_val);
                let (t, s) = simd_tanh_sigmoid_dual_hifi_avx512(x1, x2);

                let mut t_arr = [0.0_f32; 16];
                let mut s_arr = [0.0_f32; 16];
                _mm512_storeu_ps(t_arr.as_mut_ptr(), t);
                _mm512_storeu_ps(s_arr.as_mut_ptr(), s);

                let expected_tanh = x1_val.tanh();
                let expected_sig = 1.0 / (1.0 + (-x2_val).exp());

                assert!(
                    (t_arr[0] - expected_tanh).abs() <= 1e-6,
                    "dual_avx512 tanh({x1_val}) = {}, expected {expected_tanh}",
                    t_arr[0],
                );
                assert!(
                    (s_arr[0] - expected_sig).abs() <= 1e-6,
                    "dual_avx512 sigmoid({x2_val}) = {}, expected {expected_sig}",
                    s_arr[0],
                );
            }
        }
    }
}
