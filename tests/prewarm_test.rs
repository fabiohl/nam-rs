// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration tests for optional prewarm control via `LoadOptions`.
//!
//! Validates that:
//! 1. `prewarm: Some(false)` skips the initial prewarm during loading.
//! 2. `reset()` with `prewarm_on_reset == false` does not execute prewarm.
//! 3. `set_prewarm_on_reset(false)` propagates through ContainerModel submodels.

use nam_rs::common::diagnostics::SystemSnapshot;
use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::loader::{LoadOptions, load_and_build_model};
use nam_rs::models::{NamModel, StaticModel};

mod common;
use common::io_helpers::{model_path, process_in_blocks};

const BLOCK_SIZE: usize = 64;

// =============================================================================
// Helpers
// =============================================================================

fn sys() -> SystemSnapshot {
    SystemSnapshot::capture()
}

fn load_with_opts(
    path: &std::path::Path,
    prewarm: Option<bool>,
) -> nam_rs::loader::LoadedModelPair {
    load_and_build_model(path, &sys(), false, LoadOptions { prewarm })
        .expect("Failed to load model for prewarm test")
}

// =============================================================================
// Test 1: prewarm: Some(false) skips initial prewarm
// =============================================================================

/// Load a model with `prewarm: Some(false)` and verify prewarm was skipped.
///
/// Verifies that `prewarm_on_reset()` returns `false` after loading with
/// `prewarm: Some(false)`, and returns `true` with the default `LoadOptions`.
#[test]
fn test_load_with_prewarm_skip() {
    let path = model_path("linear_test.nam");

    let pair_skip = load_with_opts(&path, Some(false));
    let model_skip = pair_skip.model_l.as_ref().unwrap();
    assert!(
        !model_skip.prewarm_on_reset(),
        "prewarm_on_reset should be false when loaded with prewarm: Some(false)"
    );

    let pair_default = load_with_opts(&path, None);
    let model_default = pair_default.model_l.as_ref().unwrap();
    assert!(
        model_default.prewarm_on_reset(),
        "prewarm_on_reset should be true when loaded with default LoadOptions"
    );
}

/// Verify that loading with `prewarm: Some(false)` produces a model that
/// processes audio correctly (no panics, finite output).
#[test]
fn test_skip_prewarm_output_is_valid() {
    let path = model_path("linear_test.nam");

    let mut pair_skip = load_with_opts(&path, Some(false));
    let model = pair_skip.model_l.as_mut().unwrap();

    let input = vec![0.5f32; BLOCK_SIZE * 4];
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(model, &input, &mut output, BLOCK_SIZE);

    for &s in &output {
        assert!(s.is_finite(), "Output sample should be finite");
    }
}

/// Ensure deterministic output comparison between prewarm-skip and default
/// after both models are manually reset with prewarm_on_reset enabled.
/// After identical reset, outputs must match.
#[test]
fn test_skip_vs_default_deterministic_after_reset() {
    let path = model_path("linear_test.nam");

    let mut pair_skip = load_with_opts(&path, Some(false));
    let mut pair_default = load_with_opts(&path, None);

    let model_skip = pair_skip.model_l.as_mut().unwrap();
    let model_default = pair_default.model_l.as_mut().unwrap();

    model_skip
        .reset(48000, BLOCK_SIZE)
        .expect("reset with prewarm_on_reset=false should succeed");
    model_default
        .reset(48000, BLOCK_SIZE)
        .expect("reset with prewarm_on_reset=true should succeed");

    let input = {
        let mut v = vec![0.0f32; BLOCK_SIZE * 4];
        for (i, s) in v.iter_mut().enumerate() {
            *s = (i as f32 * 0.1).sin();
        }
        v
    };
    let mut out_skip = vec![0.0f32; input.len()];
    let mut out_default = vec![0.0f32; input.len()];

    process_in_blocks(model_skip, &input, &mut out_skip, BLOCK_SIZE);
    process_in_blocks(model_default, &input, &mut out_default, BLOCK_SIZE);

    for (&a, &b) in out_skip.iter().zip(out_default.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Outputs should match after identical reset"
        );
    }
}

// =============================================================================
// Test 2: reset() without prewarm_on_reset skips prewarm computations
// =============================================================================

/// Build model, set `prewarm_on_reset(false)`, call `reset()` and verify
/// no panic occurs and output is still valid.
#[test]
fn test_reset_without_prewarm_no_panic() {
    let path = model_path("linear_test.nam");
    let json = std::fs::read_to_string(&path).expect("Failed to read linear_test.nam");
    let data = parse_nam_json(&json).expect("Failed to parse linear_test.nam");
    let mut model = build_model(&data).expect("Failed to build linear model");

    model.set_prewarm_on_reset(false);
    assert!(!model.prewarm_on_reset());

    model
        .reset(48000, BLOCK_SIZE)
        .expect("reset() with prewarm_on_reset=false should succeed");

    let input = vec![0.5f32; BLOCK_SIZE * 4];
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, BLOCK_SIZE);

    for &s in &output {
        assert!(
            s.is_finite(),
            "Output should be finite after reset without prewarm"
        );
    }
}

/// Verify that reset() with prewarm_on_reset=false does NOT clear state,
/// while reset() with prewarm_on_reset=true DOES clear it.
/// Process audio first (fill internal state), then compare reset outcomes.
#[test]
fn test_reset_outcome_differs_prewarm_vs_noprewarm() {
    let path = model_path("linear_test.nam");
    let json = std::fs::read_to_string(&path).expect("Failed to read linear_test.nam");
    let data = parse_nam_json(&json).expect("Failed to parse linear_test.nam");

    let input = {
        let mut v = vec![0.0f32; BLOCK_SIZE * 4];
        for (i, s) in v.iter_mut().enumerate() {
            *s = (i as f32 * 0.25).sin();
        }
        v
    };

    // Build and process some audio through the model to fill internal state
    let mut model_a = build_model(&data).expect("Failed to build");
    let mut dummy_out = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_a, &input, &mut dummy_out, BLOCK_SIZE);
    // Reset with prewarm → clears state
    model_a.set_prewarm_on_reset(true);
    model_a.reset(48000, BLOCK_SIZE).unwrap();
    let mut out_a = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_a, &input, &mut out_a, BLOCK_SIZE);

    // Build separate model, process audio, reset without prewarm → keeps state
    let mut model_b = build_model(&data).expect("Failed to build");
    process_in_blocks(&mut model_b, &input, &mut dummy_out, BLOCK_SIZE);
    model_b.set_prewarm_on_reset(false);
    model_b.reset(48000, BLOCK_SIZE).unwrap();
    let mut out_b = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_b, &input, &mut out_b, BLOCK_SIZE);

    let mut any_differ = false;
    for (&a, &b) in out_a.iter().zip(out_b.iter()) {
        if (a - b).abs() > 1e-6 {
            any_differ = true;
        }
        assert!(a.is_finite());
        assert!(b.is_finite());
    }
    assert!(
        any_differ,
        "Outputs should differ: prewarm clears state, no-prewarm retains previous state"
    );
}

// =============================================================================
// Test 3: ContainerModel flag propagation to nested submodels
// =============================================================================

/// Create a ContainerModel, call `set_prewarm_on_reset(false)`, and verify
/// all submodels also have their prewarm_on_reset set to false.
#[test]
fn test_container_prewarm_propagation() {
    let full_nam_path = model_path("wavenet_a2_full.nam");
    let lite_nam_path = model_path("wavenet_a2_lite.nam");
    if !full_nam_path.exists() || !lite_nam_path.exists() {
        eprintln!("SKIP: A2 model files not found. Container prewarm propagation test impossible.");
        return;
    }

    let full_json = std::fs::read_to_string(&full_nam_path).expect("Failed to read A2-Full");
    let full_data = parse_nam_json(&full_json).expect("Failed to parse A2-Full");
    let full_model = build_model(&full_data).expect("Failed to build A2-Full");

    let lite_json = std::fs::read_to_string(&lite_nam_path).expect("Failed to read A2-Lite");
    let lite_data = parse_nam_json(&lite_json).expect("Failed to parse A2-Lite");
    let lite_model = build_model(&lite_data).expect("Failed to build A2-Lite");

    let sample_rate = full_data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let mut container = nam_rs::models::container::ContainerModel::new(
        vec![(0.5, lite_model), (1.0, full_model)],
        sample_rate,
    )
    .expect("Failed to create ContainerModel");

    assert!(container.prewarm_on_reset(), "Default should be true");

    container.set_prewarm_on_reset(false);
    assert!(
        !container.prewarm_on_reset(),
        "Container should be false after set"
    );

    for (idx, (_threshold, submodel)) in container.submodels().iter().enumerate() {
        assert!(
            !submodel.prewarm_on_reset(),
            "Submodel[{}] prewarm_on_reset should be false after container propagation",
            idx
        );
    }
}

/// Verify that loading a slimmable_container.nam with `prewarm: Some(false)`
/// propagates the flag through both ContainerModel and its submodels.
#[test]
fn test_container_load_skip_propagation() {
    let path = model_path("slimmable_container.nam");
    if !path.exists() {
        eprintln!("SKIP: slimmable_container.nam not found.");
        return;
    }

    let pair = load_with_opts(&path, Some(false));
    let model = pair.model_l.as_ref().unwrap();

    assert!(
        !model.prewarm_on_reset(),
        "Container loaded with prewarm: Some(false) should have prewarm_on_reset=false"
    );

    if let StaticModel::Container(container) = model.as_ref() {
        for (idx, (_threshold, submodel)) in container.submodels().iter().enumerate() {
            assert!(
                !submodel.prewarm_on_reset(),
                "Container submodel[{}] should have prewarm_on_reset=false",
                idx
            );
        }
    }
}
