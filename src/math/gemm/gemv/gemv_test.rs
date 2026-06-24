// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::scalar_ref;

fn make_gemv_data(in_len: usize, out_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_frames: Vec<f32> = (0..in_len).map(|i| (i as f32 * 0.07).sin()).collect();
    let weights: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.25)
        .collect();
    let bias: Vec<f32> = (0..out_len)
        .map(|i| (i as f32 * 0.13).sin() * 0.1)
        .collect();
    (in_frames, weights, bias)
}

const GEMV_OUT_LENS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const GEMV_IN_LENS: &[usize] = &[1, 2, 3, 4, 5, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128];

// ── AVX2 with_bias ──────────────────────────────────────────────────────────

#[test]
fn test_gemv_with_bias_f32_avx2_vs_fallback() {
    let num_frames = 1;
    for &out_len in GEMV_OUT_LENS {
        for &in_len in GEMV_IN_LENS {
            let (in_frames, weights, bias) = make_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len * num_frames];
            let mut out_scalar = vec![0.0f32; out_len * num_frames];

            unsafe {
                gemv_with_bias_f32_avx2(&in_frames, &weights, &bias, &mut out_simd, num_frames);
                scalar_ref::gemv_with_bias_f32_fallback(
                    &in_frames,
                    &weights,
                    &bias,
                    &mut out_scalar,
                    num_frames,
                );
            }
            for c in 0..out_len {
                assert!(
                    (out_simd[c] - out_scalar[c]).abs() < 5e-4,
                    "in_len={} out_len={} ch={}: avx2={}, scalar={}",
                    in_len,
                    out_len,
                    c,
                    out_simd[c],
                    out_scalar[c]
                );
            }
        }
    }
}

#[test]
fn test_gemv_with_bias_f32_avx2_batch_vs_fallback() {
    for &num_frames in &[1, 2, 3, 4, 8] {
        let out_len = 8;
        let in_len = 64;
        let (in_frames, weights, bias) = make_gemv_data(in_len * num_frames, out_len);
        let mut out_simd = vec![0.0f32; out_len * num_frames];
        let mut out_scalar = vec![0.0f32; out_len * num_frames];

        unsafe {
            gemv_with_bias_f32_avx2(&in_frames, &weights, &bias, &mut out_simd, num_frames);
            scalar_ref::gemv_with_bias_f32_fallback(
                &in_frames,
                &weights,
                &bias,
                &mut out_scalar,
                num_frames,
            );
        }
        for f in 0..num_frames {
            for c in 0..out_len {
                assert!(
                    (out_simd[f * out_len + c] - out_scalar[f * out_len + c]).abs() < 5e-4,
                    "batch frames={} f={} ch={}: avx2={}, scalar={}",
                    num_frames,
                    f,
                    c,
                    out_simd[f * out_len + c],
                    out_scalar[f * out_len + c]
                );
            }
        }
    }
}

// ── AVX2 no_bias ────────────────────────────────────────────────────────────

#[test]
fn test_gemv_no_bias_f32_avx2_vs_fallback() {
    let num_frames = 1;
    for &out_len in GEMV_OUT_LENS {
        for &in_len in GEMV_IN_LENS {
            let (in_frames, weights, _) = make_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len * num_frames];
            let mut out_scalar = vec![0.0f32; out_len * num_frames];

            unsafe {
                gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_simd, num_frames);
                scalar_ref::gemv_no_bias_f32_fallback(
                    &in_frames,
                    &weights,
                    &mut out_scalar,
                    num_frames,
                );
            }
            for c in 0..out_len {
                assert!(
                    (out_simd[c] - out_scalar[c]).abs() < 5e-4,
                    "in_len={} out_len={} ch={}: avx2={}, scalar={}",
                    in_len,
                    out_len,
                    c,
                    out_simd[c],
                    out_scalar[c]
                );
            }
        }
    }
}

#[test]
fn test_gemv_no_bias_f32_avx2_batch_vs_fallback() {
    for &num_frames in &[1, 2, 3, 4, 8] {
        let out_len = 8;
        let in_len = 64;
        let (in_frames, weights, _) = make_gemv_data(in_len * num_frames, out_len);
        let mut out_simd = vec![0.0f32; out_len * num_frames];
        let mut out_scalar = vec![0.0f32; out_len * num_frames];

        unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_simd, num_frames);
            scalar_ref::gemv_no_bias_f32_fallback(
                &in_frames,
                &weights,
                &mut out_scalar,
                num_frames,
            );
        }
        for f in 0..num_frames {
            for c in 0..out_len {
                assert!(
                    (out_simd[f * out_len + c] - out_scalar[f * out_len + c]).abs() < 5e-4,
                    "batch frames={} f={} ch={}: avx2={}, scalar={}",
                    num_frames,
                    f,
                    c,
                    out_simd[f * out_len + c],
                    out_scalar[f * out_len + c]
                );
            }
        }
    }
}

// ── AVX-512 with_bias ───────────────────────────────────────────────────────

#[test]
fn test_gemv_with_bias_f32_avx512_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let num_frames = 1;
    for &out_len in GEMV_OUT_LENS {
        for &in_len in GEMV_IN_LENS {
            let (in_frames, weights, bias) = make_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len * num_frames];
            let mut out_scalar = vec![0.0f32; out_len * num_frames];

            unsafe {
                gemv_with_bias_f32_avx512(&in_frames, &weights, &bias, &mut out_simd, num_frames);
                scalar_ref::gemv_with_bias_f32_fallback(
                    &in_frames,
                    &weights,
                    &bias,
                    &mut out_scalar,
                    num_frames,
                );
            }
            for c in 0..out_len {
                assert!(
                    (out_simd[c] - out_scalar[c]).abs() < 5e-4,
                    "in_len={} out_len={} ch={}: avx512={}, scalar={}",
                    in_len,
                    out_len,
                    c,
                    out_simd[c],
                    out_scalar[c]
                );
            }
        }
    }
}

#[test]
fn test_gemv_with_bias_f32_avx512_batch_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    for &num_frames in &[1, 2, 3, 4, 8] {
        let out_len = 8;
        let in_len = 64;
        let (in_frames, weights, bias) = make_gemv_data(in_len * num_frames, out_len);
        let mut out_simd = vec![0.0f32; out_len * num_frames];
        let mut out_scalar = vec![0.0f32; out_len * num_frames];

        unsafe {
            gemv_with_bias_f32_avx512(&in_frames, &weights, &bias, &mut out_simd, num_frames);
            scalar_ref::gemv_with_bias_f32_fallback(
                &in_frames,
                &weights,
                &bias,
                &mut out_scalar,
                num_frames,
            );
        }
        for f in 0..num_frames {
            for c in 0..out_len {
                assert!(
                    (out_simd[f * out_len + c] - out_scalar[f * out_len + c]).abs() < 5e-4,
                    "batch frames={} f={} ch={}: avx512={}, scalar={}",
                    num_frames,
                    f,
                    c,
                    out_simd[f * out_len + c],
                    out_scalar[f * out_len + c]
                );
            }
        }
    }
}

// ── AVX-512 no_bias ─────────────────────────────────────────────────────────

#[test]
fn test_gemv_no_bias_f32_avx512_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let num_frames = 1;
    for &out_len in GEMV_OUT_LENS {
        for &in_len in GEMV_IN_LENS {
            let (in_frames, weights, _) = make_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len * num_frames];
            let mut out_scalar = vec![0.0f32; out_len * num_frames];

            unsafe {
                gemv_no_bias_f32_avx512(&in_frames, &weights, &mut out_simd, num_frames);
                scalar_ref::gemv_no_bias_f32_fallback(
                    &in_frames,
                    &weights,
                    &mut out_scalar,
                    num_frames,
                );
            }
            for c in 0..out_len {
                assert!(
                    (out_simd[c] - out_scalar[c]).abs() < 5e-4,
                    "in_len={} out_len={} ch={}: avx512={}, scalar={}",
                    in_len,
                    out_len,
                    c,
                    out_simd[c],
                    out_scalar[c]
                );
            }
        }
    }
}

#[test]
fn test_gemv_no_bias_f32_avx512_batch_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    for &num_frames in &[1, 2, 3, 4, 8] {
        let out_len = 8;
        let in_len = 64;
        let (in_frames, weights, _) = make_gemv_data(in_len * num_frames, out_len);
        let mut out_simd = vec![0.0f32; out_len * num_frames];
        let mut out_scalar = vec![0.0f32; out_len * num_frames];

        unsafe {
            gemv_no_bias_f32_avx512(&in_frames, &weights, &mut out_simd, num_frames);
            scalar_ref::gemv_no_bias_f32_fallback(
                &in_frames,
                &weights,
                &mut out_scalar,
                num_frames,
            );
        }
        for f in 0..num_frames {
            for c in 0..out_len {
                assert!(
                    (out_simd[f * out_len + c] - out_scalar[f * out_len + c]).abs() < 5e-4,
                    "batch frames={} f={} ch={}: avx512={}, scalar={}",
                    num_frames,
                    f,
                    c,
                    out_simd[f * out_len + c],
                    out_scalar[f * out_len + c]
                );
            }
        }
    }
}
