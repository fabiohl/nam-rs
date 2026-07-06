// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::activations::ActivationType;

fn build_single_block_model() -> ConvNetModel {
    let mut block =
        ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block");

    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);

    let bn_scale = vec![1.0f32];
    let bn_offset = vec![0.0f32];
    block.set_bn_params(&bn_scale, &bn_offset).unwrap();

    ConvNetModel {
        blocks: vec![block],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
    }
}

#[test]
fn test_single_block_process() {
    let mut model = build_single_block_model();

    let input = [0.5f32];
    let mut output = [0.0f32];

    model.process(&input, &mut output);

    assert!((output[0] - 0.5f32.tanh()).abs() < 1e-4);
}

#[test]
fn test_head_scale() {
    let mut model = build_single_block_model();
    model.head_scale = 2.0;

    let input = [0.5f32];
    let mut output = [0.0f32];

    model.process(&input, &mut output);

    let expected = 2.0 * 0.5f32.tanh();
    assert!((output[0] - expected).abs() < 1e-4);
}

#[test]
fn test_empty_model_outputs_silence() {
    let mut model = ConvNetModel {
        blocks: vec![],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
    };

    let input = [0.5f32, -0.3];
    let mut output = [1.0f32; 2];
    model.process(&input, &mut output);
    assert_eq!(output, [0.0, 0.0]);
}

#[test]
fn test_empty_input_noop() {
    let mut model = build_single_block_model();

    let input: [f32; 0] = [];
    let mut output: [f32; 0] = [];

    model.process(&input, &mut output);
}

#[test]
fn test_prewarm_no_panic() {
    let mut model = build_single_block_model();
    model.prewarm();

    let input = [0.0f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);
    assert!(output[0].is_finite());
}

#[test]
fn test_two_block_chain() {
    let mut block0 =
        ConvNetBlock::new(1, 2, 1, 1, false, ActivationType::ReLU, 0).expect("block 0");
    let weights0 = vec![1.0f32, 2.0, 0.0, 0.0];
    block0.set_conv_weights(&weights0);
    block0
        .set_bn_params(&[1.0f32, 1.0], &[0.0f32, 0.0])
        .unwrap();

    let mut block1 =
        ConvNetBlock::new(2, 1, 1, 1, false, ActivationType::Tanh, 1).expect("block 1");
    let weights1 = vec![0.5f32, 0.0, 0.0, 0.0, 0.5f32, 0.0, 0.0, 0.0];
    block1.set_conv_weights(&weights1);
    block1.set_bn_params(&[1.0f32], &[0.0f32]).unwrap();

    let model = ConvNetModel {
        blocks: vec![block0, block1],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(2 * WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
    };

    let mut model = model;
    let input = [2.0f32];
    let mut output = [0.0f32];

    model.process(&input, &mut output);

    let _b0_c0: f32 = 2.0 * 1.0;
    let _b0_c1: f32 = 2.0 * 2.0;
    let b1_out: f32 = 4.0 * 0.5 + 2.0 * 0.5;
    let expected = b1_out.tanh();
    assert!(
        (output[0] - expected).abs() < 5e-4,
        "output[0]={}, expected={}, diff={}",
        output[0],
        expected,
        (output[0] - expected).abs()
    );
}

#[test]
fn test_post_stack_head_integration() {
    use crate::loader::nam_json::model::HeadConfig;
    use crate::models::wavenet::PostStackHead;

    let mut block = ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::ReLU, 0).expect("block");
    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);
    block.set_bn_params(&[1.0f32], &[0.0f32]).unwrap();

    let head_config = HeadConfig {
        channels: Some(1),
        bias: Some(false),
        out_channels: Some(1),
        activation: Some("Tanh".to_string()),
        kernel_size: Some(1),
    };
    let head = PostStackHead::from_config(&head_config, 1).expect("head");

    let model = ConvNetModel {
        blocks: vec![block],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: Some(head),
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
    };

    let mut model = model;
    model
        .post_stack_head
        .as_mut()
        .unwrap()
        .set_weights(&[1.0, 0.0, 0.0, 0.0]);

    let input = [0.5f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);
    assert!((output[0] - 0.5f32.tanh()).abs() < 1e-4);
}

#[test]
fn test_prewarm_with_head() {
    use crate::loader::nam_json::model::HeadConfig;

    let mut block = ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::ReLU, 0).expect("block");
    block.set_conv_weights(&[1.0, 0.0, 0.0, 0.0]);
    block.set_bn_params(&[1.0f32], &[0.0f32]).unwrap();

    let head_config = HeadConfig {
        channels: Some(1),
        bias: Some(false),
        out_channels: Some(1),
        activation: None,
        kernel_size: Some(1),
    };
    let head = PostStackHead::from_config(&head_config, 1).expect("head");

    let mut model = ConvNetModel {
        blocks: vec![block],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: Some(head),
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
    };

    model
        .post_stack_head
        .as_mut()
        .unwrap()
        .set_weights(&[1.0, 0.0, 0.0, 0.0]);
    model.prewarm();

    let input = [0.0f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);
    assert!(output[0].is_finite());
}

#[test]
fn test_struct_alignment() {
    assert_eq!(std::mem::align_of::<ConvNetModel>(), 64);
}
