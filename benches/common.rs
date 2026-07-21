// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared benchmark utilities.
//!
//! Provides deterministic signal generators, synthetic model-data builders,
//! and model-loader helpers used across the bench suite.
//!
//! This module is compiled into multiple bench binaries; individual
//! functions may appear unused in some binaries during phased migration.
#![allow(dead_code)]
#![allow(unused_imports)]

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{NamConfig, NamLayerConfig, NamModelData, parse_nam_json};
use nam_rs::models::NamModel;
use nam_rs::models::lstm::lstm_weight_count;
use std::fs;
use std::path::PathBuf;

pub fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    const F0: f64 = 440.0;
    const SR: f64 = 48_000.0;
    let omega = 2.0 * std::f64::consts::PI * F0 / SR;
    (0..num_samples)
        .map(|i| ((i as f64 * omega).sin()) as f32)
        .collect()
}

pub fn make_lstm_data(num_layers: usize, hidden_size: usize) -> NamModelData {
    let total_weights = lstm_weight_count(num_layers, hidden_size);
    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "LSTM".to_string(),
        config: NamConfig {
            layers: vec![],
            head: None,
            head_scale: None,
            num_layers: Some(num_layers),
            hidden_size: Some(hidden_size),
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights: vec![0.01; total_weights],
        weights_layout: nam_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

pub fn make_wavenet_dyn_data() -> NamModelData {
    let channels = 5usize;
    let kernel_size = 3usize;
    let condition_size = 3usize;
    let head_1 = 5usize;
    let head_2 = 1usize;
    let dilations = [vec![1, 2, 4, 8, 16], vec![1, 2, 4, 8, 16]];
    let num_layers_per_array = 5usize;

    let array1_rechannel = channels;
    let array2_rechannel = channels * channels;
    let per_layer = channels * kernel_size * channels
        + channels
        + condition_size * channels
        + channels * channels
        + channels;
    let array1_head = channels * head_1;
    let array2_head = channels * head_2 + head_2;
    let total_weights = array1_rechannel
        + num_layers_per_array * per_layer
        + array1_head
        + array2_rechannel
        + num_layers_per_array * per_layer
        + array2_head
        + 1;

    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![
                NamLayerConfig {
                    input_size: Some(1),
                    condition_size: Some(condition_size),
                    head_size: Some(head_1),
                    channels: Some(channels),
                    kernel_size: Some(kernel_size),
                    dilations: Some(dilations[0].clone()),
                    activation: Some("Tanh".to_string()),
                    gated: Some(false),
                    head_bias: Some(false),
                    ..Default::default()
                },
                NamLayerConfig {
                    input_size: Some(channels),
                    condition_size: Some(condition_size),
                    head_size: Some(head_2),
                    channels: Some(channels),
                    kernel_size: Some(kernel_size),
                    dilations: Some(dilations[1].clone()),
                    activation: Some("Tanh".to_string()),
                    gated: Some(false),
                    head_bias: Some(true),
                    ..Default::default()
                },
            ],
            head: None,
            head_scale: Some(0.02),
            ..Default::default()
        },
        weights: vec![0.01; total_weights],
        weights_layout: nam_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

pub fn make_wavenet_a2_dyn_data() -> NamModelData {
    use nam_rs::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES};

    let channels = 4usize;
    let bottleneck = 4usize;
    let head_k = nam_rs::models::a2::params::A2_HEAD_KERNEL_SIZE;

    let mut total_weights = channels;
    for &ksize in A2_KERNEL_SIZES.iter() {
        total_weights += channels * bottleneck * ksize;
        total_weights += bottleneck;
        total_weights += bottleneck;
        total_weights += bottleneck * channels;
        total_weights += channels;
    }
    total_weights += head_k * channels;
    total_weights += 1;
    total_weights += 1;

    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                channels: Some(channels),
                bottleneck: Some(bottleneck),
                kernel_sizes: Some(A2_KERNEL_SIZES.to_vec()),
                dilations: Some(A2_DILATIONS.to_vec()),
                activation: Some("LeakyReLU".to_string()),
                gated: Some(true),
                head_bias: Some(true),
                ..Default::default()
            }],
            head: None,
            head_scale: Some(0.02),
            ..Default::default()
        },
        weights: vec![0.01; total_weights],
        weights_layout: nam_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

pub fn model_path(filename: &str) -> PathBuf {
    let mut base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut nondist = base.clone();
    nondist.push("tests/fixtures/models-nondist");
    nondist.push(filename);
    if nondist.exists() {
        nondist
    } else {
        base.push("tests/fixtures/models");
        base.push(filename);
        base
    }
}

pub fn load_and_prewarm(filename: &str) -> Option<nam_rs::models::StaticModel> {
    let path = model_path(filename);
    if !path.exists() {
        return None;
    }
    let json_data = fs::read_to_string(&path).ok()?;
    let model_data = parse_nam_json(&json_data).ok()?;
    let mut model = build_model(&model_data).ok()?;
    model.prewarm(2048);
    Some(*model)
}

pub fn load_model_data(filename: &str) -> Option<NamModelData> {
    let path = model_path(filename);
    let json_data = fs::read_to_string(&path).ok()?;
    parse_nam_json(&json_data).ok()
}

pub fn synth_ir(len: usize, freq: f32, decay: f32) -> Vec<f32> {
    const SR: f32 = 48000.0;
    (0..len)
        .map(|n| {
            let t = n as f32 / SR;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}
