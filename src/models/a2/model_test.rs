// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_wavenet_a2_receptive_field_ch3() {
    let model = WaveNetA2::<3>::new();
    // Reference: computed from A2_KERNEL_SIZES and A2_DILATIONS arrays.
    let expected = {
        let mut sum = 0usize;
        for i in 0..A2_NUM_LAYERS {
            sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
        }
        sum + (A2_HEAD_KERNEL_SIZE - 1)
    };
    assert_eq!(model.receptive_field_size, expected);
    assert_eq!(model.receptive_field(), expected);
    assert_eq!(model.channels(), 3);
}

#[test]
fn test_wavenet_a2_receptive_field_ch8() {
    let model = WaveNetA2::<8>::new();
    let expected = {
        let mut sum = 0usize;
        for i in 0..A2_NUM_LAYERS {
            sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
        }
        sum + (A2_HEAD_KERNEL_SIZE - 1)
    };
    assert_eq!(model.receptive_field_size, expected);
    assert_eq!(model.channels(), 8);
}

#[test]
fn test_wavenet_a2_process_stub_output_silence() {
    let mut model = WaveNetA2::<3>::new();
    let input = vec![0.5f32; 64];
    let mut output = vec![1.0f32; 64];
    model.process(&input, &mut output);
    for v in &output {
        assert!(v.abs() < 1e-9, "expected silence, got {}", v);
    }
}

#[test]
fn test_wavenet_a2_process_empty_input() {
    let mut model = WaveNetA2::<3>::new();
    let input: [f32; 0] = [];
    let mut output: [f32; 0] = [];
    model.process(&input, &mut output);
    // Empty input should be a no-op.
}

#[test]
fn test_wavenet_a2_prewarm_fills_buffers() {
    let mut model = WaveNetA2::<3>::new();
    for buf in &mut model.layer_buffers {
        let len = buf.size();
        buf[..len].fill(0.5);
    }
    model.head_accum.fill(0.5);
    model.layer_in.fill(0.5);
    model.prewarm();
    for buf in &model.layer_buffers {
        let len = buf.size();
        for &v in buf[..len].iter() {
            assert!(v.abs() < 1e-9, "layer_buffer not zeroed");
        }
    }
    for v in model.head_accum.iter() {
        assert!(v.abs() < 1e-9, "head_accum not zeroed");
    }
    for v in model.layer_in.iter() {
        assert!(v.abs() < 1e-9, "layer_in not zeroed");
    }
    assert_eq!(model.head_write_pos, model.receptive_field_size);
}

#[test]
fn test_wavenet_a2_reset_reallocates_and_prewarms() {
    let mut model = WaveNetA2::<3>::new();
    let orig_rings: Vec<usize> = model.layer_ring_sizes.clone();
    model.reset(48000, 128);
    assert!(model.max_buffer_size == 128);
    for (i, &size) in model.layer_ring_sizes.iter().enumerate() {
        assert!(size >= orig_rings[i], "layer ring {} shrank", i);
    }
    for buf in &model.layer_buffers {
        let len = buf.size();
        for &v in buf[..len].iter() {
            assert!(v.abs() < 1e-9, "reset layer_buffer not zeroed");
        }
    }
}

#[test]
fn test_wavenet_a2_set_max_buffer_size_noop_on_smaller() {
    let mut model = WaveNetA2::<3>::new();
    let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
    model.set_max_buffer_size(32);
    assert_eq!(model.layer_ring_sizes, orig_sizes);
    assert_eq!(model.max_buffer_size, WAVENET_MAX_NUM_FRAMES);
}

#[test]
fn test_wavenet_a2_set_max_buffer_size_grows() {
    let mut model = WaveNetA2::<8>::new();
    let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
    model.set_max_buffer_size(256);
    assert!(model.max_buffer_size == 256);
    assert_eq!(model.layer_ring_sizes.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_buffer_starts.len(), A2_NUM_LAYERS);
    // At least one ring should have grown.
    let any_grew = orig_sizes
        .iter()
        .zip(model.layer_ring_sizes.iter())
        .any(|(a, b)| b > a);
    assert!(
        any_grew,
        "at least one ring should grow with larger max_buffer_size"
    );
}

#[test]
fn test_wavenet_a2_default_creates_valid_model() {
    let model = WaveNetA2::<3>::default();
    assert_eq!(model.channels(), 3);
    assert!(model.receptive_field_size > 0);
    assert!(!model.head_accum.is_empty());
    assert!(!model.layer_buffers.is_empty());
    assert_eq!(model.rechannel_w.len(), 3);
    assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_ring_sizes.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_buffer_starts.len(), A2_NUM_LAYERS);
    assert_eq!(model.layer_in.len(), 3 * model.max_buffer_size);
}

#[test]
fn test_wavenet_a2_const_receptive_field_matches_runtime() {
    let rf_const = a2_receptive_field();
    let model3 = WaveNetA2::<3>::new();
    let model8 = WaveNetA2::<8>::new();
    assert_eq!(model3.receptive_field_size, rf_const);
    assert_eq!(model8.receptive_field_size, rf_const);
}

// ── set_weights tests (T1.6) ───────────────────────────────────────

#[allow(dead_code)]
fn expected_weight_count(ch: usize) -> usize {
    let mut count = ch; // rechannel_w
    for &k in &A2_KERNEL_SIZES {
        count += ch * ch * k; // conv_w
        count += ch; // conv_b
        count += ch; // mixin_w
        count += ch * ch; // l1x1_w
        count += ch; // l1x1_b
    }
    count += A2_HEAD_KERNEL_SIZE * ch; // head_w
    count += 1; // head_b
    count += 1; // head_scale
    count
}

#[allow(dead_code)]
fn make_test_weights(n: usize, seed: u32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        v.push(((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25);
    }
    v
}

#[test]
fn test_set_weights_exact_count_ch3() {
    let mut model = WaveNetA2::<3>::new();
    let count = expected_weight_count(3);
    assert_eq!(count, 1871); // sanity-check known count
    let weights = make_test_weights(count, 42);
    assert!(model.set_weights(&weights).is_ok());
    assert!(model.has_weights());
    assert_eq!(model.layers.len(), A2_NUM_LAYERS);
    assert!(model.head_conv.is_some());
}

#[test]
fn test_set_weights_exact_count_ch8() {
    let mut model = WaveNetA2::<8>::new();
    let count = expected_weight_count(8);
    assert_eq!(count, 12146); // sanity-check known count
    let weights = make_test_weights(count, 77);
    assert!(model.set_weights(&weights).is_ok());
    assert!(model.has_weights());
    assert_eq!(model.layers.len(), A2_NUM_LAYERS);
    assert!(model.head_conv.is_some());
}

#[test]
fn test_set_weights_too_few_ch3() {
    let mut model = WaveNetA2::<3>::new();
    let count = expected_weight_count(3);
    let weights = make_test_weights(count - 10, 42);
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "expected error with too few weights");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("stream exhausted"),
        "error should mention exhaustion, got: {err_msg}"
    );
    assert!(!model.has_weights());
}

#[test]
fn test_set_weights_too_many_ch3() {
    let mut model = WaveNetA2::<3>::new();
    let count = expected_weight_count(3);
    let weights = make_test_weights(count + 5, 42);
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "expected error with too many weights");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("unconsumed"),
        "error should mention unconsumed, got: {err_msg}"
    );
}

#[test]
fn test_set_weights_too_few_ch8() {
    let mut model = WaveNetA2::<8>::new();
    let count = expected_weight_count(8);
    let weights = make_test_weights(count - 1, 99);
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "expected error with too few weights");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("stream exhausted"),
        "error should mention exhaustion"
    );
}

#[test]
fn test_set_weights_too_many_ch8() {
    let mut model = WaveNetA2::<8>::new();
    let count = expected_weight_count(8);
    let weights = make_test_weights(count + 1, 88);
    let err = model.set_weights(&weights);
    assert!(err.is_err(), "expected error with too many weights");
    let err_msg = err.unwrap_err();
    assert!(
        err_msg.contains("unconsumed"),
        "error should mention unconsumed"
    );
}

#[test]
fn test_set_weights_has_weights_flag_ch3() {
    let mut model = WaveNetA2::<3>::new();
    assert!(!model.has_weights());
    let count = expected_weight_count(3);
    let weights = make_test_weights(count, 123);
    model.set_weights(&weights).unwrap();
    assert!(model.has_weights());
}

/// Smoke: load random weights, prewarm, process 1 frame — output should be non-zero
/// (random weights almost certainly produce non-zero output).
#[test]
fn test_set_weights_process_smoke_ch3() {
    let mut model = WaveNetA2::<3>::new();
    let count = expected_weight_count(3);
    let weights = make_test_weights(count, 42);
    model.set_weights(&weights).unwrap();
    model.prewarm();

    let input = vec![0.5f32; 16];
    let mut output = vec![0.0f32; 16];
    model.process(&input, &mut output);

    // With random weights, output should be non-zero (statistical certainty).
    let any_nonzero = output.iter().any(|&v| v.abs() > 1e-30);
    assert!(
        any_nonzero,
        "process should produce non-zero output after weight loading"
    );
}

#[test]
fn test_set_weights_process_smoke_ch8() {
    let mut model = WaveNetA2::<8>::new();
    let count = expected_weight_count(8);
    let weights = make_test_weights(count, 77);
    model.set_weights(&weights).unwrap();
    model.prewarm();

    let input = vec![0.5f32; 16];
    let mut output = vec![0.0f32; 16];
    model.process(&input, &mut output);

    let any_nonzero = output.iter().any(|&v| v.abs() > 1e-30);
    assert!(
        any_nonzero,
        "process should produce non-zero output after weight loading"
    );
}
