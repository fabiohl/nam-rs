// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::scalar_ref;

/// Generates test data: weights `[[u16; 4]]` and state `[f32]`.
fn make_data(len: usize) -> (Vec<[u16; 4]>, Vec<f32>) {
    let weights: Vec<[u16; 4]> = (0..len)
        .map(|i| {
            let v = (i as f32 * 0.1).sin() * 0.5 + 0.5;
            let bits = half::f16::from_f32(v).to_bits();
            [
                bits,
                bits.wrapping_mul(3),
                bits.wrapping_add(0x1000),
                bits.wrapping_mul(7),
            ]
        })
        .collect();
    let state: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
    (weights, state)
}

#[test]
fn test_dot_4x_interleaved_avx512_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_data(len);
        let expected = unsafe { scalar_ref::dot_product_4x_interleaved_fallback(&weights, &state) };
        let result = unsafe { dot_product_4x_interleaved_avx512(&weights, &state) };
        for j in 0..4 {
            assert!(
                (result[j] - expected[j]).abs() < 1e-3,
                "len={} channel={}: simd={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_4x_interleaved_dual_frame_avx512_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state_f0) = make_data(len);
        let state_f1: Vec<f32> = state_f0.iter().map(|&s| s * 0.8 + 0.1).collect();

        let expected = unsafe {
            scalar_ref::dot_product_4x_interleaved_dual_frame_fallback(
                &weights, &state_f0, &state_f1,
            )
        };
        let result =
            unsafe { dot_product_4x_interleaved_dual_frame_avx512(&weights, &state_f0, &state_f1) };
        for j in 0..4 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 1e-3,
                "len={} frame0 channel={}: simd={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 1e-3,
                "len={} frame1 channel={}: simd={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

#[test]
fn test_dot_4x_interleaved_avx512_stress() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state) = make_data(len);
        let expected = unsafe { scalar_ref::dot_product_4x_interleaved_fallback(&weights, &state) };
        let result = unsafe { dot_product_4x_interleaved_avx512(&weights, &state) };
        for j in 0..4 {
            assert!(
                (result[j] - expected[j]).abs() < 1e-3,
                "stress len={} channel={}: simd={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_4x_interleaved_dual_frame_avx512_stress() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state_f0) = make_data(len);
        let state_f1: Vec<f32> = state_f0.iter().map(|&s| s * 0.8 + 0.1).collect();

        let expected = unsafe {
            scalar_ref::dot_product_4x_interleaved_dual_frame_fallback(
                &weights, &state_f0, &state_f1,
            )
        };
        let result =
            unsafe { dot_product_4x_interleaved_dual_frame_avx512(&weights, &state_f0, &state_f1) };
        for j in 0..4 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 1e-3,
                "stress len={} f0 ch={}: simd={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 1e-3,
                "stress len={} f1 ch={}: simd={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

#[test]
fn test_dot_4x_interleaved_avx512_vs_avx2() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [0, 1, 3, 8, 16, 64, 128, 256, 512, 1024];
    for &len in &sizes {
        let (weights, state) = make_data(len);
        let avx2_result = unsafe { dot_product_4x_interleaved_avx2(&weights, &state) };
        let avx512_result = unsafe { dot_product_4x_interleaved_avx512(&weights, &state) };
        for j in 0..4 {
            assert!(
                (avx512_result[j] - avx2_result[j]).abs() < 1e-3,
                "len={} channel={}: avx512={}, avx2={}",
                len,
                j,
                avx512_result[j],
                avx2_result[j]
            );
        }
    }
}

#[test]
fn test_dot_4x_interleaved_dual_frame_avx512_vs_avx2() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [0, 1, 3, 8, 16, 64, 128, 256, 512, 1024];
    for &len in &sizes {
        let (weights, state_f0) = make_data(len);
        let state_f1: Vec<f32> = state_f0.iter().map(|&s| s * 0.8 + 0.1).collect();

        let avx2_result =
            unsafe { dot_product_4x_interleaved_dual_frame_avx2(&weights, &state_f0, &state_f1) };
        let avx512_result =
            unsafe { dot_product_4x_interleaved_dual_frame_avx512(&weights, &state_f0, &state_f1) };
        for j in 0..4 {
            assert!(
                (avx512_result.0[j] - avx2_result.0[j]).abs() < 1e-3,
                "len={} f0 ch={}: avx512={}, avx2={}",
                len,
                j,
                avx512_result.0[j],
                avx2_result.0[j]
            );
            assert!(
                (avx512_result.1[j] - avx2_result.1[j]).abs() < 1e-3,
                "len={} f1 ch={}: avx512={}, avx2={}",
                len,
                j,
                avx512_result.1[j],
                avx2_result.1[j]
            );
        }
    }
}
