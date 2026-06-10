// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::AlignedVec;
use crate::models::a2::A2_DILATIONS;
use crate::models::a2::conv1d_fallback::a2_conv1d_single_frame_fallback;
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;

fn make_ch3_test_weights(kernel: usize, seed: u32) -> (AlignedVec<u16>, AlignedVec<f32>) {
    let in_ch = 3usize;
    let out_ch = 3usize;
    let num_blocks = 1; // out_ch.div_ceil(4) = 1 for CH=3
    let total_w = num_blocks * 4 * in_ch * kernel;
    let mut weights = AlignedVec::new(total_w, 0u16);

    let mut state = seed;
    for i in 0..total_w {
        // Lane 3 stays 0 (CH=3 only has 3 output channels).
        if i % 4 == 3 {
            continue;
        }
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
        weights[i] = half::f16::from_f32(v).to_bits();
    }

    let mut bias = AlignedVec::new(out_ch, 0.0f32);
    for i in 0..out_ch {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        bias[i] = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
    }

    (weights, bias)
}

fn make_ch3_conv_dyn(
    weights: AlignedVec<u16>,
    bias: AlignedVec<f32>,
    do_bias: bool,
    dilation: usize,
    kernel: usize,
) -> Conv1dDyn {
    let in_ch = 3usize;
    let out_ch = 3usize;
    Conv1dDyn {
        weights,
        bias,
        do_bias,
        dilation,
        in_ch,
        out_ch,
        num_blocks: 1,
        kernel,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    }
}

/// K=6 parity: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k6_parity() {
    let kernel = 6;
    let dilation = A2_DILATIONS[5]; // 101

    let (weights, bias) = make_ch3_test_weights(kernel, 42);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 99u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=6 channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// K=15 parity: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k15_parity() {
    let kernel = 15;
    let dilation = A2_DILATIONS[15]; // 13

    let (weights, bias) = make_ch3_test_weights(kernel, 123);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 77u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=15 channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// K=6 with mixin: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k6_with_mixin() {
    let kernel = 6;
    let dilation = A2_DILATIONS[3]; // 17

    let (weights, bias) = make_ch3_test_weights(kernel, 555);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 31u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let mut state = 88u32;
    let mixin: Vec<f32> = (0..3)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect();

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, Some(&mixin));
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        Some(&mixin),
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=6+mixin channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// K=15 with mixin: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k15_with_mixin() {
    let kernel = 15;
    let dilation = A2_DILATIONS[14]; // 1

    let (weights, bias) = make_ch3_test_weights(kernel, 777);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 13u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let mut state = 99u32;
    let mixin: Vec<f32> = (0..3)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect();

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, Some(&mixin));
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        Some(&mixin),
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=15+mixin channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// All K=6 dilations: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k6_all_dilations() {
    let kernel = 6;

    let dilation_set: Vec<usize> = A2_DILATIONS
        .iter()
        .filter(|&&d| {
            let idx = A2_DILATIONS.iter().position(|&x| x == d).unwrap();
            crate::models::a2::A2_KERNEL_SIZES[idx] == 6
        })
        .copied()
        .collect();

    let (weights, bias) = make_ch3_test_weights(kernel, 111);

    let buf_frames = 4096;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 17u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    for &dilation in &dilation_set {
        let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

        let frame_idx = 3500;

        let mut ch3_out = vec![0.0f32; 3];
        let mut scalar_out = vec![0.0f32; 3];

        unsafe {
            conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
        }

        a2_conv1d_single_frame_fallback(
            &weights,
            &bias,
            true,
            dilation,
            3,
            3,
            kernel,
            &layer_buffer,
            frame_idx,
            None,
            &mut scalar_out,
        );

        for c in 0..3 {
            let diff = (ch3_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "dilation={} channel {}: ch3_unrolled={}, scalar={}, diff={}",
                dilation,
                c,
                ch3_out[c],
                scalar_out[c],
                diff
            );
        }
    }
}

/// No bias: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k6_no_bias() {
    let kernel = 6;
    let dilation = A2_DILATIONS[7]; // 1

    let (weights, bias) = make_ch3_test_weights(kernel, 42);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), false, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 55u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        false,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "no_bias channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// Consistency: independent runs produce identical results.
#[test]
fn test_ch3_unrolled_k6_deterministic() {
    let kernel = 6;
    let dilation = A2_DILATIONS[5];

    let (weights, bias) = make_ch3_test_weights(kernel, 123);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 44u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut out1 = vec![0.0f32; 3];
    let mut out2 = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut out1, frame_idx, None);
        conv.process_single_ch3_unrolled(&layer_buffer, &mut out2, frame_idx, None);
    }

    for c in 0..3 {
        assert_eq!(
            out1[c], out2[c],
            "deterministic failure: run1[{}]={} != run2[{}]={}",
            c, out1[c], c, out2[c]
        );
    }
}

/// Unrolled K=6 vs generic SIMD path: results must match within tolerance.
#[test]
fn test_ch3_unrolled_k6_vs_generic() {
    let kernel = 6;
    let dilation = A2_DILATIONS[5];

    let (weights, bias) = make_ch3_test_weights(kernel, 42);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 99u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut generic_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
        conv.process_single_frame_generic::<crate::math::common::Avx2Math, f32>(
            &layer_buffer,
            &mut generic_out,
            frame_idx,
            None,
        );
    }

    for c in 0..3 {
        let diff = (ch3_out[c] - generic_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=6 ch3 vs generic channel {}: ch3_unrolled={}, generic={}, diff={}",
            c,
            ch3_out[c],
            generic_out[c],
            diff
        );
    }
}

/// Unrolled K=15 vs generic SIMD path: results must match within tolerance.
#[test]
fn test_ch3_unrolled_k15_vs_generic() {
    let kernel = 15;
    let dilation = A2_DILATIONS[15];

    let (weights, bias) = make_ch3_test_weights(kernel, 123);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 77u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut generic_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
        conv.process_single_frame_generic::<crate::math::common::Avx2Math, f32>(
            &layer_buffer,
            &mut generic_out,
            frame_idx,
            None,
        );
    }

    for c in 0..3 {
        let diff = (ch3_out[c] - generic_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=15 ch3 vs generic channel {}: ch3_unrolled={}, generic={}, diff={}",
            c,
            ch3_out[c],
            generic_out[c],
            diff
        );
    }
}
