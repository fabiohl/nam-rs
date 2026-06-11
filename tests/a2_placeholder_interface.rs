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
            receptive_field: None,
            bias: None,
            submodels: None,
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
            receptive_field: None,
            bias: None,
            submodels: None,
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
            receptive_field: None,
            bias: None,
            submodels: None,
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
            receptive_field: None,
            bias: None,
            submodels: None,
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

// =============================================================================
// 3. Dispatch table: A1 with version >= 0.6.0 must NOT be misrouted to A2
// =============================================================================

/// Table-driven test ensuring `is_wavenet_a2()` uses shape as the primary
/// detector. An A1 WaveNet model with a high version string must NOT be
/// classified as A2 — version is telemetry only.
#[test]
fn test_dispatch_table_a1_high_version_not_misrouted() {
    use nam_rs::models::a2::A2_DILATIONS;

    // A1 Standard dilations
    let a1_std_dils: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    // A1 Lite layer-0 dilations
    let a1_lite_dils_0: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64];
    // A1 Lite layer-1 dilations
    let a1_lite_dils_1: Vec<usize> = vec![128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    /// Each case: description, NamModelData factory, expected is_a2_shape, expected is_wavenet_a2
    struct Case {
        desc: &'static str,
        data: NamModelData,
        expect_a2_shape: Option<u8>,
        expect_wavenet_a2: bool,
    }

    let cases = vec![
        // ── A1 topologies with high version — must NOT be A2 ──
        Case {
            desc: "A1 Standard v0.5.4 — baseline A1",
            data: NamModelData {
                version: Some("0.5.4".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![
                        NamLayerConfig {
                            channels: Some(16),
                            dilations: Some(a1_std_dils.clone()),
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
                            dilations: Some(a1_std_dils.clone()),
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
                    head_scale: None,
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        Case {
            desc: "A1 Standard v0.7.0 — high version, NOT A2",
            data: NamModelData {
                version: Some("0.7.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![
                        NamLayerConfig {
                            channels: Some(16),
                            dilations: Some(a1_std_dils.clone()),
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
                            dilations: Some(a1_std_dils.clone()),
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
                    head_scale: None,
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        Case {
            desc: "A1 Lite v0.8.0 — high version, NOT A2",
            data: NamModelData {
                version: Some("0.8.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![
                        NamLayerConfig {
                            channels: Some(12),
                            dilations: Some(a1_lite_dils_0.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(false),
                        },
                        NamLayerConfig {
                            channels: Some(12),
                            dilations: Some(a1_lite_dils_1.clone()),
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
                    head_scale: None,
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        Case {
            desc: "A1 Feather v0.9.0 — high version, NOT A2",
            data: NamModelData {
                version: Some("0.9.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![
                        NamLayerConfig {
                            channels: Some(8),
                            dilations: Some(a1_lite_dils_0.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(false),
                        },
                        NamLayerConfig {
                            channels: Some(8),
                            dilations: Some(a1_lite_dils_1.clone()),
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
                    head_scale: None,
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        // ── Real A2 topologies — must be A2 ──
        Case {
            desc: "A2 real CH=3 (Lite shape)",
            data: NamModelData {
                version: Some("0.6.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(3),
                        dilations: Some(A2_DILATIONS.to_vec()),
                        activation: Some("LeakyReLU".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: Some(3),
            expect_wavenet_a2: true,
        },
        Case {
            desc: "A2 real CH=8 (Full shape)",
            data: NamModelData {
                version: Some("0.6.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(8),
                        dilations: Some(A2_DILATIONS.to_vec()),
                        activation: Some("LeakyReLU".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: Some(8),
            expect_wavenet_a2: true,
        },
        // ── Ambiguous shapes — version is telemetry only, shape is primary ──
        Case {
            desc: "Ambiguous: CH=3, wrong dils, LeakyReLU — A2 via activation",
            data: NamModelData {
                version: Some("0.6.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(3),
                        dilations: Some(vec![1, 2, 4, 8]),
                        activation: Some("LeakyReLU".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: true, // non-Tanh activation
        },
        Case {
            desc: "Ambiguous: CH=8, wrong dils, Tanh, high version — NOT A2",
            data: NamModelData {
                version: Some("0.7.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(8),
                        dilations: Some(vec![1, 2, 4, 8, 16, 32, 64]),
                        activation: Some("Tanh".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false, // version alone is NOT sufficient
        },
        Case {
            desc: "Ambiguous: CH=4 (A1 Nano channels), single layer, high version, Tanh — NOT A2",
            data: NamModelData {
                version: Some("0.6.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(4),
                        dilations: Some(vec![1, 2, 4, 8, 16, 32, 64]),
                        activation: Some("Tanh".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
    ];

    for case in &cases {
        let shape_result = is_a2_shape(&case.data);
        let wavenet_a2_result = case.data.is_wavenet_a2();
        assert_eq!(
            shape_result, case.expect_a2_shape,
            "[{}] is_a2_shape mismatch: expected {:?}, got {:?}",
            case.desc, case.expect_a2_shape, shape_result,
        );
        assert_eq!(
            wavenet_a2_result, case.expect_wavenet_a2,
            "[{}] is_wavenet_a2 mismatch: expected {}, got {}",
            case.desc, case.expect_wavenet_a2, wavenet_a2_result,
        );
    }
}
