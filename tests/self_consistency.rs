// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Rust self-consistency (determinism) tests.
//!
//! # Objective
//! Validate that the Rust inference engine produces bitwise-identical results
//! across independent runs with the same model weights and inputs.
//!
//! Each test loads the same `.nam` model twice, builds two independent
//! `StaticModel` instances, runs prewarm and processes the same 440 Hz sine
//! test signal, and asserts that the MSE between the two output buffers is
//! exactly 0.0 (no floating-point divergence).
//!
//! These tests do not depend on C++ golden vectors.

use std::fs;

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::NamModel;

mod common;
use common::*;

/// Test 5: WaveNet self-consistency — absolute determinism.
///
/// Loads `BossWN-standard.nam` twice, builds two identical `StaticModel`s,
/// runs prewarm and processes the same 440 Hz sine signal (512 samples).
/// The MSE between the two outputs must be exactly 0.0 (bitwise identical).
///
/// This test does not depend on C++ golden vectors and validates that the Rust engine
/// is deterministic across independent runs with the same weights and inputs.
#[test]
fn test_auto_consistency_wavenet() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        eprintln!(
            "SKIP: BossWN-standard.nam not found at {path:?}. Skipping WaveNet self-consistency."
        );
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let mut model_a =
        build_model(&model_data).expect("Dispatcher failed (model_a) for self-consistency");
    let mut model_b =
        build_model(&model_data).expect("Dispatcher failed (model_b) for self-consistency");

    model_a.prewarm(2048);
    model_b.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_a = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_b = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(&mut model_a, &input, &mut out_a, GOLDEN_BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut out_b, GOLDEN_BLOCK_SIZE);

    let mse = compute_mse(&out_a, &out_b);
    let mae = compute_max_abs_error(&out_a, &out_b);

    println!("[WaveNet Self-Consistency] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "Rust WaveNet engine non-deterministic! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

/// Test 5b: WaveNet A2-Full self-consistency — absolute determinism.
///
/// Loads `wavenet_a2_full.nam` twice, builds two identical `StaticModel`s,
/// runs prewarm and processes the same 440 Hz sine signal (2048 samples).
/// The MSE between the two outputs must be exactly 0.0 (bitwise identical).
#[test]
fn test_auto_consistency_wavenet_a2_full() {
    let path = model_path("wavenet_a2_full.nam");

    if !path.exists() {
        eprintln!(
            "SKIP: wavenet_a2_full.nam not found at {path:?}. Skipping A2-Full self-consistency."
        );
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let mut model_a =
        build_model(&model_data).expect("Dispatcher failed (model_a) for A2-Full self-consistency");
    let mut model_b =
        build_model(&model_data).expect("Dispatcher failed (model_b) for A2-Full self-consistency");

    model_a.prewarm(2048);
    model_b.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_a = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_b = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(&mut model_a, &input, &mut out_a, GOLDEN_BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut out_b, GOLDEN_BLOCK_SIZE);

    let mse = compute_mse(&out_a, &out_b);
    let mae = compute_max_abs_error(&out_a, &out_b);

    println!("[WaveNet A2-Full Self-Consistency] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "Rust WaveNet A2-Full engine non-deterministic! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

/// Test 5c: WaveNet A2-Lite self-consistency — absolute determinism.
///
/// Loads `wavenet_a2_lite.nam` twice, builds two identical `StaticModel`s,
/// runs prewarm and processes the same 440 Hz sine signal (2048 samples).
/// The MSE between the two outputs must be exactly 0.0 (bitwise identical).
#[test]
fn test_auto_consistency_wavenet_a2_lite() {
    let path = model_path("wavenet_a2_lite.nam");

    if !path.exists() {
        eprintln!(
            "SKIP: wavenet_a2_lite.nam not found at {path:?}. Skipping A2-Lite self-consistency."
        );
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let mut model_a =
        build_model(&model_data).expect("Dispatcher failed (model_a) for A2-Lite self-consistency");
    let mut model_b =
        build_model(&model_data).expect("Dispatcher failed (model_b) for A2-Lite self-consistency");

    model_a.prewarm(2048);
    model_b.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_a = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_b = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(&mut model_a, &input, &mut out_a, GOLDEN_BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut out_b, GOLDEN_BLOCK_SIZE);

    let mse = compute_mse(&out_a, &out_b);
    let mae = compute_max_abs_error(&out_a, &out_b);

    println!("[WaveNet A2-Lite Self-Consistency] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "Rust WaveNet A2-Lite engine non-deterministic! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

/// Test 6: LSTM self-consistency — absolute determinism.
///
/// Loads `BossLSTM-1x16.nam` twice, builds two identical `StaticModel`s,
/// runs prewarm and processes the same 440 Hz sine signal (512 samples).
/// The MSE between the two outputs must be exactly 0.0 (bitwise identical).
#[test]
fn test_auto_consistency_lstm() {
    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        eprintln!("SKIP: BossLSTM-1x16.nam not found at {path:?}. Skipping LSTM self-consistency.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read LSTM model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let mut model_a =
        build_model(&model_data).expect("Dispatcher failed (model_a) for LSTM self-consistency");
    let mut model_b =
        build_model(&model_data).expect("Dispatcher failed (model_b) for LSTM self-consistency");

    model_a.prewarm(2048);
    model_b.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_a = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_b = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(&mut model_a, &input, &mut out_a, GOLDEN_BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut out_b, GOLDEN_BLOCK_SIZE);

    let mse = compute_mse(&out_a, &out_b);
    let mae = compute_max_abs_error(&out_a, &out_b);

    println!("[LSTM Self-Consistency] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "Rust LSTM engine non-deterministic! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

/// Test 16: LSTM 2x8 self-consistency — absolute determinism.
#[test]
fn test_auto_consistency_lstm_2x8() {
    let path = model_path("BossLSTM-2x8.nam");

    if !path.exists() {
        eprintln!(
            "SKIP: BossLSTM-2x8.nam not found at {path:?}. Skipping LSTM 2x8 self-consistency."
        );
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read LSTM 2x8 model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let mut model_a = build_model(&model_data)
        .expect("Dispatcher failed (model_a) for LSTM 2x8 self-consistency");
    let mut model_b = build_model(&model_data)
        .expect("Dispatcher failed (model_b) for LSTM 2x8 self-consistency");

    model_a.prewarm(2048);
    model_b.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_a = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_b = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(&mut model_a, &input, &mut out_a, GOLDEN_BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut out_b, GOLDEN_BLOCK_SIZE);

    let mse = compute_mse(&out_a, &out_b);
    let mae = compute_max_abs_error(&out_a, &out_b);

    println!("[LSTM 2x8 Self-Consistency] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "Rust LSTM 2x8 engine non-deterministic! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}
