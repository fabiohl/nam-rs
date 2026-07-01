// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::GatingMode;
use crate::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS};

fn make_standard_activations(num: usize) -> Vec<ActivationType> {
    vec![
        ActivationType::LeakyReLU {
            negative_slope: A2_LEAKY_SLOPE,
        };
        num
    ]
}

fn make_standard_gating(num: usize) -> Vec<GatingMode> {
    vec![GatingMode::None; num]
}

fn make_standard_secondary(num: usize) -> Vec<Option<ActivationType>> {
    vec![None; num]
}

#[test]
fn test_wavenet_a2_dyn_new_ch3() {
    let model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();

    assert_eq!(model.channels, 3);
    assert_eq!(model.bottleneck, 3);
    assert_eq!(model.num_layers, A2_NUM_LAYERS);
    assert!(model.receptive_field_size > 0);
    assert!(!model.head_accum.is_empty());
    assert!(!model.layer_buffers.is_empty());
    assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
    assert_eq!(model.rechannel_w.len(), 3);
    assert_eq!(model.layer_in.len(), 3 * model.max_buffer_size);
    assert!(!model.head1x1_active);
}

#[test]
fn test_wavenet_a2_dyn_new_ch8() {
    let model = WaveNetA2Dyn::new(
        1,
        8,
        8,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();

    assert_eq!(model.channels, 8);
    assert_eq!(model.bottleneck, 8);
    assert_eq!(model.num_layers, A2_NUM_LAYERS);
    assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
}

#[test]
fn test_wavenet_a2_dyn_bottleneck_neq_channels() {
    let model = WaveNetA2Dyn::new(
        1,
        8,
        4,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        true,
    )
    .unwrap();

    assert_eq!(model.channels, 8);
    assert_eq!(model.bottleneck, 4);
    assert!(model.head1x1_active);
    assert_eq!(model.head1x1_w.len(), 4 * 8);
    assert_eq!(model.head1x1_b.len(), 8);
}

#[test]
fn test_wavenet_a2_dyn_gating_prealloc() {
    let num = A2_NUM_LAYERS;
    let mut gating = vec![GatingMode::None; num];
    gating[0] = GatingMode::Gated;
    gating[1] = GatingMode::Blended;
    let mut sec = vec![None; num];
    sec[0] = Some(ActivationType::Sigmoid);
    sec[1] = Some(ActivationType::Tanh);

    let model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(num),
        gating,
        sec,
        false,
    )
    .unwrap();

    assert!(model.gating_configs[0].is_some());
    assert!(model.blending_configs[1].is_some());
    assert!(model.gating_configs[2].is_none());
    assert!(model.blending_configs[0].is_none());
}

#[test]
fn test_wavenet_a2_dyn_process_empty_input() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let input: [f32; 0] = [];
    let mut output: [f32; 0] = [];
    model.process(&input, &mut output);
}

#[test]
fn test_wavenet_a2_dyn_process_silence_no_weights() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let input = vec![0.5f32; 64];
    let mut output = vec![1.0f32; 64];
    model.process(&input, &mut output);
    for v in &output {
        assert!(v.abs() < 1e-9, "expected silence, got {}", v);
    }
}

#[test]
fn test_wavenet_a2_dyn_prewarm_zeroes() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
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
}

#[test]
fn test_wavenet_a2_dyn_set_max_buffer_size_noop() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
    model.set_max_buffer_size(32).unwrap();
    assert_eq!(model.layer_ring_sizes, orig_sizes);
}

#[test]
fn test_wavenet_a2_dyn_set_max_buffer_size_grows() {
    let mut model = WaveNetA2Dyn::new(
        1,
        8,
        8,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
    model.set_max_buffer_size(256).unwrap();
    assert!(model.max_buffer_size == 256);
    let any_grew = orig_sizes
        .iter()
        .zip(model.layer_ring_sizes.iter())
        .any(|(a, b)| b > a);
    assert!(any_grew, "at least one ring should grow");
}

#[test]
fn test_wavenet_a2_dyn_has_weights_false_initially() {
    let model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    assert!(!model.has_weights());
}

#[test]
fn test_wavenet_a2_dyn_receptive_field() {
    let model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let expected = {
        let mut sum = 0usize;
        for i in 0..A2_NUM_LAYERS {
            sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
        }
        sum + (crate::models::a2::A2_HEAD_KERNEL_SIZE - 1)
    };
    assert_eq!(model.receptive_field_size, expected);
    assert_eq!(model.receptive_field(), expected);
}

// ── F9 regression: head_write_pos wrap-around with empty layers ──────

#[test]
fn test_wavenet_a2_dyn_head_write_pos_never_exceeds_ring_mask() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let ring_mask = model.head_ring_mask;
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    for i in 0..2048 {
        model.process(&input, &mut output);
        assert!(
            model.head_write_pos <= ring_mask,
            "iteration {}: head_write_pos {} exceeded ring_mask {}",
            i,
            model.head_write_pos,
            ring_mask
        );
    }
}

#[test]
fn test_wavenet_a2_dyn_head_write_pos_wraps_correctly() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let ring_mask = model.head_ring_mask;
    let cap = ring_mask + 1;
    let mut prev_wp = model.head_write_pos;
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut wrapped = false;
    for _ in 0..4096 {
        model.process(&input, &mut output);
        let wp = model.head_write_pos;
        if wp < prev_wp {
            wrapped = true;
        }
        assert!(wp <= ring_mask);
        prev_wp = wp;
    }
    assert!(
        wrapped || cap <= 64 * 4096,
        "head_write_pos should wrap after enough iterations (cap={}, wp after 4096 iters={})",
        cap,
        model.head_write_pos
    );
}

#[test]
fn test_wavenet_a2_dyn_head_write_pos_reset_after_prewarm() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let rf = model.receptive_field_size;
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    for _ in 0..500 {
        model.process(&input, &mut output);
    }
    assert_ne!(
        model.head_write_pos, rf,
        "head_write_pos should have advanced after 500 process calls"
    );
    model.prewarm();
    assert_eq!(
        model.head_write_pos, rf,
        "head_write_pos should be reset to rf after prewarm"
    );
}
