// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Interface Tests.
//!
//! Validates that the Rust A2 constants mirror `a2_fast.h` exactly,
//! and that `is_a2_shape` accepts valid A2 topologies while rejecting
//! non-A2 shapes.
//!
//! Real A2 inference is validated by golden tests (see Sprint 1.4).

use nam_rs::loader::nam_json::{
    NamConfig, NamLayerConfig, NamModelData, WeightsLayout, get_wavenet_topology, is_a2_shape,
};
use nam_rs::models::a2::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS,
    A2_VALID_CHANNELS,
};

// =============================================================================
// 1. Constants cross-check with a2_fast.h (raw C++ source embedded as string)
// =============================================================================

#[test]
fn test_a2_constants_match_cpp_reference() {
    // Raw C++ constants from github.com/NeuralAmpModelerCore/NAM/wavenet/a2_fast.h
    assert_eq!(A2_NUM_LAYERS, 23, "kNumLayers mismatch");

    assert_eq!(A2_HEAD_KERNEL_SIZE, 16, "kHeadKernelSize mismatch");

    assert!(
        (A2_LEAKY_SLOPE - 0.01f32).abs() < f32::EPSILON,
        "kLeakySlope mismatch: got {}, expected 0.01",
        A2_LEAKY_SLOPE
    );

    // Cross-check kernel sizes against raw a2_fast.h values
    let cpp_kernel_sizes: [usize; 23] = [
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
    ];
    assert_eq!(A2_KERNEL_SIZES, cpp_kernel_sizes, "kKernelSizes mismatch");

    // Cross-check dilations against raw a2_fast.h values
    let cpp_dilations: [usize; 23] = [
        1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
    ];
    assert_eq!(A2_DILATIONS, cpp_dilations, "kDilations mismatch");

    // Cross-check valid channels
    let cpp_valid_channels: [u8; 2] = [3, 8];
    assert_eq!(
        A2_VALID_CHANNELS, cpp_valid_channels,
        "valid channels mismatch"
    );
}

// =============================================================================
// 2. is_a2_shape accepts valid A2 topologies, rejects others
// =============================================================================

/// Helper: creates NamModelData for testing is_a2_shape.
fn make_a2_data(channels: u8, dilations: Vec<usize>) -> NamModelData {
    NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                head_size: None,
                channels: Some(channels as usize),
                kernel_size: None,
                dilations: Some(dilations),
                activation: Some("LeakyReLU".to_string()),
                gated: None,
                head_bias: None,
            }],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

#[test]
fn test_is_a2_shape_accepts_channels_3() {
    let data = make_a2_data(3, A2_DILATIONS.to_vec());
    let result = is_a2_shape(&data);
    assert!(
        result.is_some(),
        "is_a2_shape should accept channels=3 with correct dilations"
    );
    assert_eq!(result.unwrap(), 3, "should return channels=3");
}

#[test]
fn test_is_a2_shape_accepts_channels_8() {
    let data = make_a2_data(8, A2_DILATIONS.to_vec());
    let result = is_a2_shape(&data);
    assert!(
        result.is_some(),
        "is_a2_shape should accept channels=8 with correct dilations"
    );
    assert_eq!(result.unwrap(), 8, "should return channels=8");
}

#[test]
fn test_is_a2_shape_rejects_channels_4() {
    let data = make_a2_data(4, A2_DILATIONS.to_vec());
    assert!(
        is_a2_shape(&data).is_none(),
        "channels=4 should not match A2 shape"
    );
}

#[test]
fn test_is_a2_shape_rejects_channels_16() {
    let data = make_a2_data(16, A2_DILATIONS.to_vec());
    assert!(
        is_a2_shape(&data).is_none(),
        "channels=16 should not match A2 shape"
    );
}

#[test]
fn test_is_a2_shape_rejects_wrong_dilations() {
    let wrong_dils: Vec<usize> = vec![
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4,
    ];
    let data = make_a2_data(8, wrong_dils);
    assert!(
        is_a2_shape(&data).is_none(),
        "dilations not matching A2_DILATIONS should be rejected"
    );
}

#[test]
fn test_is_a2_shape_rejects_multiple_layers() {
    let data = NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![
                NamLayerConfig {
                    input_size: Some(1),
                    condition_size: Some(1),
                    head_size: None,
                    channels: Some(8),
                    kernel_size: None,
                    dilations: Some(A2_DILATIONS.to_vec()),
                    activation: Some("LeakyReLU".to_string()),
                    gated: None,
                    head_bias: None,
                },
                NamLayerConfig {
                    input_size: Some(1),
                    condition_size: Some(1),
                    head_size: None,
                    channels: Some(8),
                    kernel_size: None,
                    dilations: Some(vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
                    activation: Some("Tanh".to_string()),
                    gated: None,
                    head_bias: None,
                },
            ],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    };
    assert!(
        is_a2_shape(&data).is_none(),
        "models with more than one layer should not match A2 shape"
    );
}

#[test]
fn test_is_a2_shape_rejects_non_wavenet_architecture() {
    let data = NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "LSTM".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                head_size: None,
                channels: Some(8),
                kernel_size: None,
                dilations: Some(A2_DILATIONS.to_vec()),
                activation: Some("LeakyReLU".to_string()),
                gated: None,
                head_bias: None,
            }],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    };
    assert!(
        is_a2_shape(&data).is_none(),
        "LSTM architecture should not match A2 shape"
    );
}

#[test]
fn test_is_a2_shape_rejects_wrong_dilations_length() {
    let short_dils: Vec<usize> = vec![1, 2, 3];
    let data = make_a2_data(8, short_dils);
    assert!(
        is_a2_shape(&data).is_none(),
        "dilations with wrong length should be rejected"
    );
}

#[test]
fn test_a2_shape_does_not_match_standard_wavenet() {
    // Standard WaveNet topology (16 channels, non-A2 dilations) should not match
    let data = NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![
                NamLayerConfig {
                    channels: Some(16),
                    dilations: Some(vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
                    activation: Some("Tanh".to_string()),
                    input_size: None,
                    condition_size: None,
                    head_size: None,
                    kernel_size: None,
                    gated: Some(false),
                    head_bias: Some(false),
                },
                NamLayerConfig {
                    channels: Some(16),
                    dilations: Some(vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
                    activation: Some("Tanh".to_string()),
                    input_size: None,
                    condition_size: None,
                    head_size: None,
                    kernel_size: None,
                    gated: Some(false),
                    head_bias: Some(true),
                },
            ],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    };

    assert!(
        is_a2_shape(&data).is_none(),
        "Standard WaveNet topology should not match A2 shape"
    );
    assert!(
        get_wavenet_topology(&data).is_some(),
        "Standard WaveNet should still be detected by get_wavenet_topology"
    );
}
