// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use nam_rs::loader::dispatcher::{build_model, build_wavenet_dynamic};
use nam_rs::loader::nam_json::{NamConfig, NamModelData};
use nam_rs::models::NamModel;

/// Generates a sine wave signal to be used as test input.
/// The predictability of a sine wave helps diagnose numerical drift.
fn generate_sine(freq: f32, sr: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect()
}

/// Computes the Mean Squared Error (MSE) between two signals.
/// Used to validate numerical parity between different implementations
/// of the same algorithm (e.g. Scalar vs SIMD or Static vs Dynamic).
fn calculate_mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Audio buffers must have the same size for comparison"
    );
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    sum / a.len() as f32
}

/// Tests whether the Dynamic WaveNet implementation (used as fallback) produces
/// numerically equivalent results to the Static (optimized) implementation.
/// The error tolerance (1e-10) ensures that floating-point accumulation
/// differences are minimal.
#[test]
fn test_wavenet_dynamic_parity() {
    // Define a reduced topology (3 dilations per block) to keep the test fast.
    // The goal is to test the dispatch logic and convolution loop, not raw throughput.
    let dils_short = vec![1, 2, 4];

    // Layer 1: Rechannel from 1 to 8 channels
    let layer_s1 = nam_rs::loader::nam_json::NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        channels: Some(8),
        dilations: Some(dils_short.clone()),
        kernel_size: Some(3),
        head_size: Some(4),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(false),
    };

    // Layer 2: Process from 8 to 4 channels
    let layer_s2 = nam_rs::loader::nam_json::NamLayerConfig {
        input_size: Some(8),
        condition_size: Some(1),
        channels: Some(4),
        dilations: Some(dils_short),
        kernel_size: Some(3),
        head_size: Some(1),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(true),
    };

    // Weight calculation must be exact to avoid dispatcher failures.
    // Weight breakdown (bias + kernel + linear):
    // Array 1: rechannel (8) + layers (3 * (8*3*8 + 8 + 8 + 64 + 8) = 840) + head (32) = 880
    // Array 2: rechannel (32) + layers (3 * (4*3*4 + 4 + 4 + 16 + 4) = 228) + head (5) = 265
    // Scale: 1 -> Total = 1146
    let total_weights = 1146;

    let data = NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![layer_s1, layer_s2],
            head: None,
            head_scale: None,
            num_layers: None,
            hidden_size: None,
        },
        weights: vec![0.01; total_weights],
        weights_layout: nam_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    };

    // Instantiate both inference engines
    let mut model_static = build_model(&data).expect("Failed to build static model");
    let mut model_dyn = build_wavenet_dynamic(&data).expect("Failed to build dynamic model");

    // Prewarm is necessary to stabilize internal states (delays)
    model_static.prewarm(1024);
    model_dyn.prewarm(1024);

    let input = generate_sine(440.0, 48000.0, 128);
    let mut out_static = vec![0.0f32; 128];
    let mut out_dyn = vec![0.0f32; 128];

    // Parallel execution of the models
    model_static.process(&input, &mut out_static);
    model_dyn.process(&input, &mut out_dyn);

    // Parity validation
    let mse = calculate_mse(&out_static, &out_dyn);
    println!("MSE Static vs Dynamic: {}", mse);
    assert!(
        mse < 1e-10,
        "WaveNet numerical parity failed: MSE={} (Dynamic implementation diverged from Static)",
        mse
    );
}
