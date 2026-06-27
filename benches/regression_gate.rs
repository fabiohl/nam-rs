// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance regression gate — Criterion benches with adequate statistical power
//! for CI regression detection against persisted baselines.
//!
//! ## Running
//!
//! ```sh
//! # Save a baseline (first run, or after verified optimization):
//! taskset -c 0 cargo bench --bench regression_gate -- --save-baseline ci-baseline
//!
//! # CI run: compare against saved baseline, fail on regression:
//! taskset -c 0 cargo bench --bench regression_gate -- --baseline ci-baseline
//! ```
//!
//! ## Bench layout
//!
//! Each bench function loads the real model file, warms up, and measures
//! `process()` for a single 64-sample block. The Criterion framework
//! handles statistical comparison against the saved baseline.
//!
//! ## Model coverage
//!
//! WaveNet Std/Feather/Nano/Lite, A2-Full/Lite, LSTM 1x16/2x8, Linear, ConvNet.

use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::NamModel;
use std::fs;
use std::path::PathBuf;

fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

fn model_path(filename: &str) -> PathBuf {
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

fn load_and_prewarm(filename: &str) -> Option<nam_rs::models::StaticModel> {
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

macro_rules! regression_bench {
    ($c:expr, $label:expr, $file:expr) => {
        if let Some(mut model) = load_and_prewarm($file) {
            let input = generate_sine_440hz(64);
            let mut output = vec![0.0f32; 64];
            $c.bench_function($label, |b| {
                b.iter(|| model.process(&input, &mut output));
            });
        }
    };
}

fn bench_wavenet_standard(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Std_CH16", "BossWN-standard.nam");
}

fn bench_wavenet_feather(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Feather_CH8", "BossWN-feather.nam");
}

fn bench_wavenet_lite(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Lite_CH12", "BossWN-lite.nam");
}

fn bench_wavenet_nano(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Nano_CH4", "BossWN-nano.nam");
}

fn bench_a2_full(c: &mut Criterion) {
    regression_bench!(c, "RT_A2_Full_CH8", "wavenet_a2_full.nam");
}

fn bench_a2_lite(c: &mut Criterion) {
    regression_bench!(c, "RT_A2_Lite_CH3", "wavenet_a2_lite.nam");
}

fn bench_lstm_1x16(c: &mut Criterion) {
    regression_bench!(c, "RT_LSTM_1x16", "BossLSTM-1x16.nam");
}

fn bench_lstm_2x8(c: &mut Criterion) {
    regression_bench!(c, "RT_LSTM_2x8", "BossLSTM-2x8.nam");
}

fn bench_linear(c: &mut Criterion) {
    regression_bench!(c, "RT_Linear", "linear_test.nam");
}

fn bench_convnet(c: &mut Criterion) {
    regression_bench!(c, "RT_ConvNet", "convnet_test.nam");
}

criterion_group!(
    name = regression_gates;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1))
        .noise_threshold(0.02);
    targets =
        bench_wavenet_standard,
        bench_wavenet_feather,
        bench_wavenet_lite,
        bench_wavenet_nano,
        bench_a2_full,
        bench_a2_lite,
        bench_lstm_1x16,
        bench_lstm_2x8,
        bench_linear,
        bench_convnet,
);

criterion_main!(regression_gates);
