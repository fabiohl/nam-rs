// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Benchmarks formais de latência de inferência para o motor NAM-rs.
//!
//! Utiliza `criterion` para medir a performance de:
//! - WaveNet Standard (CH=16, K=3, HEAD=8) — 64 amostras
//! - LSTM 2×16 — 64 amostras
//! - FastMath `tanh_slice` — 256 amostras
//! - FastMath `sigmoid_slice` — 256 amostras
//!
//! Execute com: `cargo bench --bench inference_bench`

use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{NamConfig, NamModelData, parse_nam_json};

/// Gera sinal senoidal determinístico de 440 Hz a 48 kHz.
fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

/// Gera um `NamModelData` LSTM sintético com pesos zerados (0.01).
fn make_lstm_data(num_layers: usize, hidden_size: usize, total_weights: usize) -> NamModelData {
    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "LSTM".to_string(),
        config: NamConfig {
            layers: vec![],
            head: None,
            head_scale: None,
            num_layers: Some(num_layers),
            hidden_size: Some(hidden_size),
        },
        weights: vec![0.01; total_weights],
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Benchmark: WaveNet Standard (real model) — 64 amostras.
fn bench_wavenet_standard_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("github.com/mikeoliphant/NeuralAudio/Utils/Models/BossWN-standard.nam");

    if !path.exists() {
        eprintln!(
            "SKIP bench: BossWN-standard.nam não encontrado em {path:?}. Ignorando benchmark WaveNet."
        );
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model = build_model(&model_data).expect("Dispatcher falhou para benchmark");
    model.0.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("wavenet_standard_64samp", |b| {
        b.iter(|| {
            model.0.process(&input, &mut output);
        });
    });
}

/// Benchmark: LSTM 2×16 (sintético) — 64 amostras.
fn bench_lstm_2x16_process(c: &mut Criterion) {
    // LSTM 2×16: 3345 pesos
    let data = make_lstm_data(2, 16, 3345);
    let mut model = build_model(&data).expect("Dispatcher falhou para LSTM benchmark");
    model.0.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("lstm_2x16_64samp", |b| {
        b.iter(|| {
            model.0.process(&input, &mut output);
        });
    });
}

/// Benchmark: FastMath tanh_slice — 256 amostras.
fn bench_tanh_slice_256(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

        c.bench_function("tanh_slice_256", |b| {
            let mut buf = base.clone();
            b.iter(|| {
                buf.copy_from_slice(&base);
                unsafe { nam_rs::math::fastmath::tanh_slice_avx2(&mut buf) };
            });
        });
    }
}

/// Benchmark: FastMath sigmoid_slice — 256 amostras.
fn bench_sigmoid_slice_256(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

        c.bench_function("sigmoid_slice_256", |b| {
            let mut buf = base.clone();
            b.iter(|| {
                buf.copy_from_slice(&base);
                unsafe { nam_rs::math::fastmath::sigmoid_slice_avx2(&mut buf) };
            });
        });
    }
}

criterion_group!(
    benches,
    bench_wavenet_standard_process,
    bench_lstm_2x16_process,
    bench_tanh_slice_256,
    bench_sigmoid_slice_256,
);
criterion_main!(benches);
