// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::*;
use half::f16;

#[test]
fn test_fallback_dot_product() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b_f32 = [0.5, 1.5, 2.5, 3.5];
    let b_u16: Vec<u16> = b_f32.iter().map(|&x| f16::from_f32(x).to_bits()).collect();

    let result = unsafe { FallbackMath::dot_product(&a, &b_u16) };
    let expected: f32 = a.iter().zip(b_f32.iter()).map(|(x, y)| x * y).sum();

    assert!((result - expected).abs() < 1e-5);
}

#[test]
fn test_fallback_fused_add_gemv() {
    let in_frame = vec![1.0, 2.0];
    let out_len = 4;
    // Weights (out_len x in_len) interleaved conceptually as (in_len x out_len)
    // weights = [w00, w01, w02, w03, w10, w11, w12, w13]
    let w_f32 = [
        0.1, 0.2, 0.3, 0.4, // Weights for in[0]
        0.5, 0.6, 0.7, 0.8, // Weights for in[1]
    ];
    let w_u16: Vec<u16> = w_f32.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let bias = vec![1.0, 1.0, 1.0, 1.0];
    let mut out_frame = vec![10.0, 10.0, 10.0, 10.0];

    unsafe {
        FallbackMath::fused_add_gemv(&in_frame, &w_u16, &bias, &mut out_frame, true);
    }

    // Expected:
    // out[0] = 10.0 + 1.0 + (1.0 * 0.1 + 2.0 * 0.5) = 11.0 + 1.1 = 12.1
    // out[1] = 10.0 + 1.0 + (1.0 * 0.2 + 2.0 * 0.6) = 11.0 + 1.4 = 12.4
    // out[2] = 10.0 + 1.0 + (1.0 * 0.3 + 2.0 * 0.7) = 11.0 + 1.7 = 12.7
    // out[3] = 10.0 + 1.0 + (1.0 * 0.4 + 2.0 * 0.8) = 11.0 + 2.0 = 13.0

    let expected = [12.1, 12.4, 12.7, 13.0];
    for i in 0..out_len {
        assert!((out_frame[i] - expected[i]).abs() < 1e-3);
    }
}

#[test]
fn test_fallback_gemv_overwrite_4gate() {
    let in_frame = vec![1.0, 2.0];
    let hidden_size = 2;
    // 4 gates x hidden_size = 8 outputs total
    // Re-organize weights for the layout: (in_len * out_total)
    // Actually FallbackMath::gemv_overwrite_4gate uses gemv_overwrite_fallback internally.
    // gemv_overwrite_fallback(in, w, bias, out, do_bias)
    // where w is (in_len * out_len)

    let in_len = 2;
    let out_total = 8;
    let mut w_full = vec![0.0f32; in_len * out_total];
    for (i, w) in w_full.iter_mut().enumerate() {
        *w = i as f32 * 0.1;
    }
    let w_u16: Vec<u16> = w_full.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let bias = vec![0.5; out_total];
    let mut out_gates = vec![0.0; out_total];

    unsafe {
        FallbackMath::gemv_overwrite_4gate(
            &in_frame,
            &w_u16,
            &bias,
            &mut out_gates,
            hidden_size,
            true,
        );
    }

    assert!((out_gates[0] - 0.9).abs() < 1e-3);
}

#[test]
fn test_fallback_lstm_gates() {
    let hidden_size = 2;
    let mut gates = [
        0.0, 0.0, // Input gate (i)
        10.0, 10.0, // Forget gate (f) -> will be ~1.0
        0.0, 0.0, // Cell gate (g)
        0.0, 0.0, // Output gate (o)
    ];
    let mut cell_state = vec![1.0, 1.0];
    let mut hidden_state = vec![0.0, 0.0];

    unsafe {
        FallbackMath::fused_lstm_gates_dyn(
            &mut gates,
            &mut cell_state,
            &mut hidden_state,
            hidden_size,
        );
    }

    // i = sigmoid(0) = 0.5
    // f = sigmoid(10) \approx 1.0
    // g = tanh(0) = 0.0
    // o = sigmoid(0) = 0.5
    // c_new = f * c_old + i * g = 1.0 * 1.0 + 0.5 * 0.0 = 1.0
    // h_new = o * tanh(c_new) = 0.5 * tanh(1.0) \approx 0.5 * 0.76159 = 0.380795

    assert!((cell_state[0] - 1.0).abs() < 1e-3);
    let expected_h = 0.5 * 1.0f32.tanh();
    assert!((hidden_state[0] - expected_h).abs() < 1e-3);
}

#[test]
fn test_fallback_activations() {
    let mut data = [-1.0, 0.0, 1.0];
    let mut data_copy = data;

    unsafe {
        FallbackMath::tanh_slice(&mut data);
        FallbackMath::sigmoid_slice(&mut data_copy);
    }

    assert!((data[0] - (-1.0f32).tanh()).abs() < 1e-6);
    assert!((data[1] - 0.0f32.tanh()).abs() < 1e-6);
    assert!((data[2] - 1.0f32.tanh()).abs() < 1e-6);

    assert!((data_copy[0] - (1.0 / (1.0 + 1.0f32.exp()))).abs() < 1e-6);
    assert!((data_copy[1] - 0.5).abs() < 1e-6);
    assert!((data_copy[2] - (1.0 / (1.0 + (-1.0f32).exp()))).abs() < 1e-6);
}

#[test]
fn test_fallback_apply_gain_and_detect_clipping_stereo() {
    let mut left = [0.5, 0.8, -1.2, 0.1];
    let mut right = [0.1, -0.9, 0.4, 0.2];
    let gain = 2.0;

    // Expected values after gain:
    // left:  [1.0, 1.6, -2.4, 0.2] -> clipping at index 1 and 2
    // right: [0.2, -1.8, 0.8, 0.4] -> clipping at index 1
    let clipped =
        unsafe { FallbackMath::apply_gain_and_detect_clipping_stereo(&mut left, &mut right, gain) };

    assert!(clipped);
    assert_eq!(left[0], 1.0);
    assert_eq!(left[1], 1.6);
    assert_eq!(right[1], -1.8);

    // Test without clipping
    let mut left2 = [0.1, 0.2];
    let mut right2 = [0.3, 0.4];
    let gain2 = 0.5;
    let clipped2 = unsafe {
        FallbackMath::apply_gain_and_detect_clipping_stereo(&mut left2, &mut right2, gain2)
    };
    assert!(!clipped2);
    assert_eq!(left2[0], 0.05);
}

#[test]
fn test_fallback_compute_energy_stereo() {
    let l = [1.0, 2.0, 3.0, 4.0];
    let r = [0.5, 1.5, 2.5, 3.5];

    // energy_l = (1^2 + 2^2 + 3^2 + 4^2) / 4 = (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    // energy_r = (0.5^2 + 1.5^2 + 2.5^2 + 3.5^2) / 4 = (0.25 + 2.25 + 6.25 + 12.25) / 4 = 21 / 4 = 5.25
    // max(7.5, 5.25) = 7.5

    let result = unsafe { FallbackMath::compute_energy_stereo(&l, &r) };
    assert!((result - 7.5).abs() < 1e-6);

    let l2 = [0.1, 0.2];
    let r2 = [1.0, 2.0]; // energy_r2 = (1 + 4)/2 = 2.5
    let result2 = unsafe { FallbackMath::compute_energy_stereo(&l2, &r2) };
    assert!((result2 - 2.5).abs() < 1e-6);
}
