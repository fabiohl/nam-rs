// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::scalar_ref;

fn make_f32_16x_data(len: usize) -> (Vec<[f32; 16]>, Vec<f32>) {
    let weights: Vec<[f32; 16]> = (0..len)
        .map(|i| {
            let base = (i as f32 * 0.1).sin() * 0.5 + 0.5;
            [
                base,
                base * 1.1,
                base * 1.2,
                base * 1.3,
                base * 0.9,
                base * 0.8,
                base * 1.4,
                base * 1.5,
                base * 1.05,
                base * 0.95,
                base * 1.15,
                base * 0.85,
                base * 1.25,
                base * 0.75,
                base * 1.35,
                base * 0.65,
            ]
        })
        .collect();
    let state: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
    (weights, state)
}

#[test]
fn test_dot_16x_f32_avx512_vs_scalar() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_16x_data(len);
        let expected = unsafe { scalar_ref::dot_product_16x_f32_scalar(&weights, &state) };
        let result = unsafe { dot_product_16x_f32_avx512(&weights, &state) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 5e-4,
                "len={} channel={}: avx512={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_avx512_stress() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state) = make_f32_16x_data(len);
        let expected = unsafe { scalar_ref::dot_product_16x_f32_scalar(&weights, &state) };
        let result = unsafe { dot_product_16x_f32_avx512(&weights, &state) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 2e-3,
                "stress len={} channel={}: avx512={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_avx512_vs_4x_decompose() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_16x_data(len);
        let result_16x = unsafe { dot_product_16x_f32_avx512(&weights, &state) };

        let w0: Vec<[f32; 4]> = weights.iter().map(|w| [w[0], w[1], w[2], w[3]]).collect();
        let w1: Vec<[f32; 4]> = weights.iter().map(|w| [w[4], w[5], w[6], w[7]]).collect();
        let w2: Vec<[f32; 4]> = weights.iter().map(|w| [w[8], w[9], w[10], w[11]]).collect();
        let w3: Vec<[f32; 4]> = weights
            .iter()
            .map(|w| [w[12], w[13], w[14], w[15]])
            .collect();
        let r0 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(&w0, &state) };
        let r1 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(&w1, &state) };
        let r2 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(&w2, &state) };
        let r3 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(&w3, &state) };
        assert!(
            (result_16x[0] - r0[0]).abs() < 5e-5,
            "len={}: 16x[0]={}, 4x[0]={}",
            len,
            result_16x[0],
            r0[0]
        );
        assert!(
            (result_16x[4] - r1[0]).abs() < 5e-5,
            "len={}: 16x[4]={}, 4x[4]={}",
            len,
            result_16x[4],
            r1[0]
        );
        assert!(
            (result_16x[8] - r2[0]).abs() < 5e-5,
            "len={}: 16x[8]={}, 4x[8]={}",
            len,
            result_16x[8],
            r2[0]
        );
        assert!(
            (result_16x[12] - r3[0]).abs() < 5e-5,
            "len={}: 16x[12]={}, 4x[12]={}",
            len,
            result_16x[12],
            r3[0]
        );
    }
}
