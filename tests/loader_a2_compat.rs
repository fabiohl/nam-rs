// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Forward-Compatibility Tests for the NAM-rs Loader.
//!
//! Ensures that models with A2 architecture or future fields do not panic
//! and gracefully fall back to placeholders, while A1 models
//! continue to work without regressions.

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::{DynamicModel, NamModel};
use std::fs;
use std::path::PathBuf;

/// Helper: resolves the absolute path to a test model located at `tests/fixtures/models/`.
/// Uses `CARGO_MANIFEST_DIR` to ensure the test works regardless of the execution directory.
fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

/// Forward-Compatibility Test for WaveNet A2.
///
/// Verifies that the inference engine gracefully handles v0.6+ (A2) models.
/// Currently, NAM-rs does not implement all A2 features (such as FiLM or dynamic Gate),
/// so it should load the model without panic and fall back to a placeholder
/// that outputs silence, informing the host that the model is incompatible but safe.
#[test]
fn test_forward_compatibility_wavenet_a2() {
    let path = model_path("mock_a2.nam");

    if !path.exists() {
        // Critical failure if the test fixture is missing
        panic!(
            "Fixture mock_a2.nam not found at {path:?}. Check if the fixtures submodule was downloaded."
        );
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read mock_a2.nam");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    // Validate that the model metadata was correctly identified as A2 architecture
    assert!(
        model_data.is_wavenet_a2(),
        "mock_a2.nam must be detected as A2 architecture (v0.6+)"
    );

    // The dispatcher should accept the model and return the placeholder variant
    let mut model = build_model(&model_data).expect(
        "The dispatcher should have fallen back to the A2 placeholder instead of failing",
    );

    // Explicitly verify that the returned variant is the Placeholder
    match *model {
        DynamicModel::WavenetA2(_) => {
            println!("Fallback to WavenetA2Placeholder confirmed successfully.");
        }
        _ => panic!(
            "Architecture error: The loader should have returned DynamicModel::WavenetA2 for this file"
        ),
    }

    // RT safety validation: the placeholder must not process audio, only silence the buffer.
    let input = [1.0f32; 64];
    let mut output = [1.0f32; 64];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert_eq!(
            s, 0.0,
            "A2 placeholder must guarantee absolute silence to prevent unwanted noise. Failure at index {i}"
        );
    }
}

/// Regression Test for WaveNet A1 (Standard).
/// Ensures legacy models continue to load and process audio normally.
#[test]
fn test_regression_a1_wavenet_standard() {
    let path = model_path("BossWN-standard.nam");

    // Skip if the real model is not present (large files are usually not in git)
    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON file");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet A1 Standard model");

    // Fill delay buffers
    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    // Validate that the output is numerical and finite (no NaNs or Infs from instability)
    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "WaveNet A1 output contains non-finite values (NaN/Inf)"
        );
    }
}

/// Regression Test for LSTM architecture.
/// Ensures the legacy recurrent engine (v0.5.x) maintains its functional integrity.
#[test]
fn test_regression_a1_lstm() {
    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON file");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM A1 model");

    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    // LSTMs are more prone to numerical instability; this test ensures functional parity
    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "LSTM A1 output contains non-finite values (NaN/Inf)"
        );
    }
}
