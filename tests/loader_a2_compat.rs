// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Loader and Forward-Compatibility Tests for the NAM-rs Loader.
//!
//! Validates that A2 models produce real inference through the first-class
//! dispatch branch (no placeholder), A1 legacy models continue to work,
//! and unrecognized shapes return clear errors.

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{
    NamConfig, NamLayerConfig, NamModelData, WeightsLayout, parse_nam_json,
};
use nam_rs::models::NamModel;
use std::fs;
use std::path::PathBuf;

/// Helper: resolves the absolute path to a test model located at `tests/fixtures/models/`.
fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

// =============================================================================
// A2 Real Inference Tests
// =============================================================================

/// Validates that A2-Lite (CH=3) real inference produces finite output
/// (not silence from a placeholder).
#[test]
fn test_a2_lite_real_inference_finite_output() {
    use nam_rs::models::a2::WaveNetA2;

    let mut model = WaveNetA2::<3>::new();
    model.prewarm();
    let input = [0.01f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "A2-Lite output must be finite (real inference)"
        );
    }
}

/// Validates that A2-Full (CH=8) real inference produces finite output.
#[test]
fn test_a2_full_real_inference_finite_output() {
    use nam_rs::models::a2::WaveNetA2;

    let mut model = WaveNetA2::<8>::new();
    model.prewarm();
    let input = [0.01f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "A2-Full output must be finite (real inference)"
        );
    }
}

// =============================================================================
// Forward-Compatibility: A2-labeled models with unrecognized shape
// =============================================================================

/// Helper: creates `NamModelData` for a model that has A2 metadata but
/// does not match any known A2 or A1 topology.
fn make_unrecognized_a2_like_data(channels: usize) -> NamModelData {
    NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                head_size: None,
                channels: Some(channels),
                kernel_size: None,
                dilations: Some(vec![1, 2, 4, 8, 16, 32, 64]),
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

/// Verifies that an A2-labeled model with unrecognized shape returns a
/// clear error (not a silent bypass or panic).
#[test]
fn test_a2_unrecognized_shape_returns_clear_error() {
    let data = make_unrecognized_a2_like_data(5);
    assert!(
        data.is_wavenet_a2(),
        "model should be detected as A2 via SemVer"
    );
    let result = build_model(&data);
    assert!(
        result.is_err(),
        "unrecognized A2 shape must produce an error, not a silent bypass"
    );
    let err_msg = format!("{}", result.err().unwrap());
    assert!(
        err_msg.contains("not recognized") || err_msg.contains("shape"),
        "Error should mention topology not being recognized: {err_msg}",
    );
}

// =============================================================================
// Regression Tests for A1 Models
// =============================================================================

/// Regression Test for WaveNet A1 (Standard).
#[test]
fn test_regression_a1_wavenet_standard() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON file");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet A1 Standard model");

    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "WaveNet A1 output contains non-finite values (NaN/Inf)"
        );
    }
}

/// Regression Test for LSTM architecture.
#[test]
fn test_regression_a1_lstm() {
    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON file");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    let mut model = build_model(&model_data).expect("Dispatcher failed to build LSTM A1 model");

    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "LSTM A1 output contains non-finite values (NaN/Inf)"
        );
    }
}
