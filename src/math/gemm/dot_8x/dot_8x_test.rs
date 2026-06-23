// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::scalar_ref;

fn make_f32_8x_data(len: usize) -> (Vec<[f32; 8]>, Vec<f32>) {
    let weights: Vec<[f32; 8]> = (0..len)
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
            ]
        })
        .collect();
    let state: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
    (weights, state)
}

#[test]
fn test_dot_8x_f32_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_8x_data(len);
        let expected = unsafe { scalar_ref::dot_product_8x_f32_scalar(&weights, &state) };
        let result = unsafe { dot_product_8x_f32_avx2(&weights, &state) };
        for j in 0..8 {
            assert!(
                (result[j] - expected[j]).abs() < 5e-4,
                "len={} channel={}: avx2={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_8x_f32_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state) = make_f32_8x_data(len);
        let expected = unsafe { scalar_ref::dot_product_8x_f32_scalar(&weights, &state) };
        let result = unsafe { dot_product_8x_f32_avx2(&weights, &state) };
        for j in 0..8 {
            assert!(
                (result[j] - expected[j]).abs() < 2e-3,
                "stress len={} channel={}: avx2={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_8x_f32_avx2_decompose_vs_4x() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_8x_data(len);
        let result_8x = unsafe { dot_product_8x_f32_avx2(&weights, &state) };

        let w0: Vec<[f32; 4]> = weights.iter().map(|w| [w[0], w[1], w[2], w[3]]).collect();
        let w1: Vec<[f32; 4]> = weights.iter().map(|w| [w[4], w[5], w[6], w[7]]).collect();
        let r0 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx2(&w0, &state) };
        let r1 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx2(&w1, &state) };
        assert!(
            (result_8x[0] - r0[0]).abs() < 5e-4,
            "len={}: 8x={}, 4x_0={}",
            len,
            result_8x[0],
            r0[0]
        );
        assert!(
            (result_8x[4] - r1[0]).abs() < 5e-4,
            "len={}: 8x={}, 4x_1={}",
            len,
            result_8x[4],
            r1[0]
        );
    }
}
