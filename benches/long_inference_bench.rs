// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Long-running (soak) inference benchmarks for the NAM-rs engine.
//!
//! These benchmarks use extended measurement times (35 s) with large buffer
//! sizes to validate CPU thermal stability and detect performance jitter /
//! throttling over time.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --features standalone,long_bench --bench long_inference_bench
//! ```
//!
//! Without the `long_bench` feature this binary compiles to a no-op so that
//! `cargo bench` (default pass) does not re-run the long soak benchmarks.

#[cfg(feature = "long_bench")]
#[path = "common.rs"]
mod common;

#[cfg(feature = "long_bench")]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "long_bench")]
use nam_rs::loader::dispatcher::build_model;
#[cfg(feature = "long_bench")]
use nam_rs::loader::nam_json::parse_nam_json;
#[cfg(feature = "long_bench")]
use nam_rs::models::NamModel;

#[cfg(feature = "long_bench")]
fn bench_wavenet_long_run(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed");
    model.prewarm(4096);
    let size = 4096;
    let input = common::generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_WaveNet");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_WaveNet_Standard_CH16_4096samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_lstm_long_run(c: &mut Criterion) {
    let data = common::make_lstm_data(2, 16);
    let mut model = build_model(&data).expect("Dispatcher failed");
    model.prewarm(4096);
    let size = 4096;
    let input = common::generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_LSTM");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_LSTM_2x16_4096samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_resampler_long_run(c: &mut Criterion) {
    use nam_rs::dsp::resampler::NamResampler;
    let size = 4096;
    let mut rs = NamResampler::new(44_100, 48_000, size).unwrap();
    let in_l = vec![0.0f32; size];
    let in_r = vec![0.0f32; size];
    let mut out_l = vec![0.0f32; size * 2];
    let mut out_r = vec![0.0f32; size * 2];
    let mut group = c.benchmark_group("Long_Run_Resampler");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_Resampler_44100_to_48000_4096samp", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_a2_full_long_run(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/wavenet_a2_full.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed");
    model.prewarm(4096);
    let size = 4096;
    let input = common::generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_A2Full");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_A2Full_CH8_4096samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_a2_lite_long_run(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/wavenet_a2_lite.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed");
    model.prewarm(4096);
    let size = 4096;
    let input = common::generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_A2Lite");
    group.measurement_time(std::time::Duration::from_secs(35));
    group.sample_size(100);
    group.bench_function("Long_A2Lite_CH3_4096samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
fn bench_cabsim_long_run(c: &mut Criterion) {
    use nam_rs::dsp::cabsim::conv::ConvEngine;
    let ir = common::synth_ir(16384, 440.0, 10.0);
    let mut engine = ConvEngine::new(&ir, 64).expect("bench ConvEngine allocation failed");
    let mut input = vec![0.0f32; 4096];
    let mut output = vec![0.0f32; 4096];

    for i in 0..engine.num_partitions().max(1) {
        let mut buf_in = vec![0.0f32; 64];
        let mut buf_out = vec![0.0f32; 64];
        for (j, v) in buf_in.iter_mut().enumerate() {
            *v = ((i * 64 + j) as f32 * 0.01).sin();
        }
        engine.process(&buf_in, &mut buf_out);
        output[..64].copy_from_slice(&buf_out);
    }

    let mut group = c.benchmark_group("Cabsim_LongRun");
    group.sample_size(100);
    group.measurement_time(std::time::Duration::from_secs(35));
    group.bench_function("4096samp_block", |b| {
        b.iter(|| {
            for (j, v) in input.iter_mut().enumerate() {
                *v = (j as f32 * 0.01).sin();
            }
            for chunk in 0..(4096 / 64) {
                let start = chunk * 64;
                engine.process(
                    std::hint::black_box(&input[start..start + 64]),
                    std::hint::black_box(&mut output[start..start + 64]),
                );
            }
            std::hint::black_box(&output);
        });
    });
    group.finish();
}

#[cfg(feature = "long_bench")]
criterion_group!(
    name = long_benches;
    config = Criterion::default();
    targets = bench_wavenet_long_run, bench_lstm_long_run, bench_resampler_long_run, bench_a2_full_long_run, bench_a2_lite_long_run, bench_cabsim_long_run
);

#[cfg(feature = "long_bench")]
criterion_main!(long_benches);

#[cfg(not(feature = "long_bench"))]
fn main() {}
