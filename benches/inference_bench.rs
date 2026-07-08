// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! End-to-end neural network inference latency benchmarks for the NAM-rs engine.
//!
//! Measures the processing time of 1 DSP block (64 samples at 48 kHz = 1.33 ms
//! deadline) for WaveNet, LSTM, A2, ConvNet, and Linear neural networks.
//!
//! ## Available benchmarks
//!
//! | ID                                      | Description                             | Practical context                        |
//! | --------------------------------------- | --------------------------------------- | ---------------------------------------- |
//! | `WaveNet_Standard_CH16_64samp_48kHz`    | Complete WaveNet Standard inference     | ~284 KB model, 10+10 dilated layers      |
//! | `LSTM_2x16_64samp_48kHz`                | LSTM 2 layers × 16 hidden inference     | Heaviest supported recurrent network     |
//! | `A2Full_CH8_64samp_48kHz`               | A2-Full (CH=8) inference                | AVX2 col-major T=4 broadcast-FMA         |
//! | `A2Lite_CH3_64samp_48kHz`               | A2-Lite (CH=3) inference                | u16 interleaved GEMV, CPU-efficient      |
//! | `WaveNet_Dynamic_CH5_64samp_48kHz`      | WaveNet Dynamic free-geom inference     | Fallback for non-cataloged WaveNet geom  |
//! | `LSTM_Dynamic_1x7_64samp_48kHz`         | LSTM Dynamic 1×7 inference              | Fallback for non-cataloged LSTM geom     |
//! | `ConvNet_Model_64samp_48kHz`            | ConvNet end-to-end model inference      | Full pipeline: 2 blocks CH=8→4 + head    |
//! | `A2Dyn_Gated_64samp_48kHz`              | A2 Dynamic CH=4 gated inference         | Fallback for non-cataloged A2 geom       |
//!
//! ## Interpreting the results
//!
//! - The real-time deadline at 48 kHz with a 64-sample buffer is **1.33 ms**.
//! - If any inference benchmark exceeds this deadline, the engine will cause
//!   xruns (buffer underruns) in production with this buffer size.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench inference_bench
//! ```

mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::NamModel;
use nam_rs::models::StaticModel;
use nam_rs::models::container::ContainerModel;
use nam_rs::models::slimmable::SlimmableModel;

use common::{
    generate_sine_440hz, make_lstm_data, make_wavenet_a2_dyn_data, make_wavenet_dyn_data,
};

/// Measures the processing time of a real WaveNet model ("Standard").
/// This benchmark is the most representative for common guitar use,
/// testing the effectiveness of dilated convolutions and optimized SIMD kernels.
fn bench_wavenet_standard_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");

    if !path.exists() {
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let mut model = build_model(&model_data).expect("Dispatcher failed for benchmark");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("WaveNet_Standard_CH16_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of a 2x16 LSTM recurrent network.
/// LSTMs are known for their high sequential computational load,
/// making them the ideal stress test to verify feedback loop latency.
fn bench_lstm_2x16_process(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_2x16_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Compares performance between Scalar (baseline) and SIMD implementations for LSTM 1x8.
/// This benchmark validates the performance gain achieved with "T3: Fused Gates",
/// where the 4 LSTM gates are computed simultaneously in AVX2 registers.
fn bench_lstm_1x8_comparison(c: &mut Criterion) {
    let data = make_lstm_data(1, 8);
    let mut model_simd = build_model(&data).expect("Dispatcher failed for LSTM 1x8 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher failed for LSTM 1x8 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_1x8_Comparison");

    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    #[cfg(test)]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::StaticModel::Lstm1x8(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm1x8"),
    });
    group.finish();
}

/// Comparative Benchmark (T15): LSTM 2x16 Scalar vs SIMD (Fused Gates T3).
fn bench_lstm_2x16_comparison(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    let mut model_simd = build_model(&data).expect("Dispatcher failed for LSTM 2x16 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher failed for LSTM 2x16 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_2x16_Comparison");
    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    #[cfg(test)]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::StaticModel::Lstm2x16(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm2x16"),
    });
    group.finish();
}

/// Benchmarks WaveNet Standard (P10) inference at small RT buffer sizes (1, 16, 64
/// samples) covering the full RT range: 1 (per-sample minimum), 16 (small plugin
/// buffer), 64 (common CLAP/JACK buffer).
///
/// Record throughput (elem/s) and latency (ns) for each size.
fn bench_wavenet_p10_small_block_sizes(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Dispatcher failed for P10 bench");
    let mut model = build_model(&model_data).expect("Dispatcher failed for P10 bench");
    model.prewarm(2048);

    for &size in &[1usize, 16, 64] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("WaveNet_P10_{}", size), |b| {
            b.iter(|| model.process(&input, &mut output));
        });
    }
}

/// Evaluates how WaveNet scales with different DSP buffer sizes.
/// Larger buffers allow better cache utilization and prefetching,
/// but increase the total latency perceived by the musician.
fn bench_wavenet_standard_block_sizes(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed");
    model.prewarm(2048);

    for &size in &[32, 128, 256, 512] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("WaveNet_Standard_CH16_{}samp_48kHz", size), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Evaluates LSTM scaling with different buffer sizes.
/// Unlike WaveNet, the LSTM is purely sequential (sample by sample),
/// so per-sample overhead tends to be more constant regardless of block size.
fn bench_lstm_2x16_block_sizes(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    let mut model = build_model(&data).expect("Dispatcher failed");
    model.prewarm(2048);
    for &size in &[32, 128, 256, 512] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("LSTM_2x16_{}samp_48kHz", size), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

// ── A2-Full (CH=8) inference benchmarks ──

/// Measures the processing time of an A2-Full (CH=8) WaveNet model.
/// A2-Full is the Criterion variant with 8 channels,
/// using AVX2 T=4 broadcast-FMA convolution for maximum throughput.
fn bench_a2_full_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/wavenet_a2_full.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for A2-Full benchmark");
    model.prewarm(2048);
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Full_CH8_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures A2-Full (CH=8) scaling across different DSP buffer sizes.
/// A2 benefit from SIMD col-major layout which keeps 8 weights
/// contiguous per (tap, input) pair, maximizing L1 cache locality.
fn bench_a2_full_block_sizes(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/wavenet_a2_full.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed");
    model.prewarm(2048);

    for &size in &[32, 128, 256, 512] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("A2Full_CH8_{}samp_48kHz", size), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures A2-Full (CH=8) prewarm cost.
/// Prewarm fills the causal history buffers — essential to measure
/// so model switching latency is known for live performance.
fn bench_prewarm_a2_full(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/wavenet_a2_full.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    c.bench_function("Prewarm_A2Full_CH8_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&model_data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

// ── A2-Lite (CH=3) inference benchmarks ──

/// Measures the processing time of an A2-Lite (CH=3) WaveNet model.
/// A2-Lite is the CPU-efficient variant with 3 channels, designed
/// for reduced computational load while preserving timbre character.
fn bench_a2_lite_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/wavenet_a2_lite.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for A2-Lite benchmark");
    model.prewarm(2048);
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Lite_CH3_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures A2-Lite (CH=3) scaling across different DSP buffer sizes.
/// With fewer channels and the u16 interleaved kernel path, A2-Lite
/// targets minimal CPU usage for low-power / high-polyphony scenarios.
fn bench_a2_lite_block_sizes(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/wavenet_a2_lite.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed");
    model.prewarm(2048);

    for &size in &[32, 128, 256, 512] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(&format!("A2Lite_CH3_{}samp_48kHz", size), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures A2-Lite (CH=3) prewarm cost.
fn bench_prewarm_a2_lite(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/wavenet_a2_lite.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    c.bench_function("Prewarm_A2Lite_CH3_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&model_data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

/// Measures the time spent in the `prewarm` function.
/// Although prewarm runs outside the audio thread, it must be fast enough
/// so that model switching during a live performance is imperceptible.
fn bench_prewarm_wavenet_standard(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    c.bench_function("Prewarm_WaveNet_Standard_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&model_data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

fn bench_prewarm_lstm_2x16(c: &mut Criterion) {
    let data = make_lstm_data(2, 16);
    c.bench_function("Prewarm_LSTM_2x16_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

fn bench_lstm_1x40_process(c: &mut Criterion) {
    let data = make_lstm_data(1, 40);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_1x40_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

fn bench_lstm_2x24_process(c: &mut Criterion) {
    let data = make_lstm_data(2, 24);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_2x24_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

fn bench_lstm_1x40_comparison(c: &mut Criterion) {
    let data = make_lstm_data(1, 40);
    let mut model_simd = build_model(&data).expect("Dispatcher failed for LSTM 1x40 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher failed for LSTM 1x40 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_1x40_Comparison");
    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    #[cfg(test)]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::StaticModel::Lstm1x40(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm1x40"),
    });
    group.finish();
}

fn bench_lstm_2x24_comparison(c: &mut Criterion) {
    let data = make_lstm_data(2, 24);
    let mut model_simd = build_model(&data).expect("Dispatcher failed for LSTM 2x24 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher failed for LSTM 2x24 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_2x24_Comparison");
    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    #[cfg(test)]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::StaticModel::Lstm2x24(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm2x24"),
    });
    group.finish();
}

/// Benchmarks the LinearModel dot product kernel (AVX2/AVX-512 SIMD vs scalar).
/// With RF 256, the scalar path performs 16k FMAs per 64-sample block;
/// the SIMD path reduces this by 4-8×.
fn bench_linear_model_dot_product(c: &mut Criterion) {
    use nam_rs::models::linear::LinearModel;

    let rf = 256;
    let weights: Vec<f32> = (0..rf).map(|i| (i as f32 * 0.01).sin()).collect();
    let bias = 0.1;
    let mut model = LinearModel::new(
        weights,
        bias,
        nam_rs::loader::nam_json::LinearImplementation::default(),
    )
    .unwrap();
    model.prewarm(0);

    let input = (0..64)
        .map(|i| (i as f32 * 0.05).sin())
        .collect::<Vec<f32>>();
    let mut output = vec![0.0f32; 64];

    c.bench_function("LinearModel_RF256_64samp_SIMD", |b| {
        b.iter(|| unsafe {
            model.process(
                std::hint::black_box(&input),
                std::hint::black_box(&mut output),
            );
        });
    });
}

/// Measures the processing cost of a ContainerModel crossfade block
/// (dual inference + SIMD blend via FMA), the worst-case per-block cost
/// during slimmable submodel switching.
fn bench_container_crossfade_64samp(c: &mut Criterion) {
    let full_path = {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/models/wavenet_a2_full.nam");
        p
    };
    let lite_path = {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/models/wavenet_a2_lite.nam");
        p
    };
    if !full_path.exists() || !lite_path.exists() {
        return;
    }

    let full_json = std::fs::read_to_string(&full_path).expect("Failed to read A2-Full");
    let full_data = parse_nam_json(&full_json).expect("Failed in JSON parser");
    let full_model = build_model(&full_data).expect("Dispatcher failed for A2-Full");

    let lite_json = std::fs::read_to_string(&lite_path).expect("Failed to read A2-Lite");
    let lite_data = parse_nam_json(&lite_json).expect("Failed in JSON parser");
    let lite_model = build_model(&lite_data).expect("Dispatcher failed for A2-Lite");

    let sr = full_data.sample_rate.map(|s| s as u32).unwrap_or(48000);
    let mut container = ContainerModel::new(vec![(0.5, lite_model), (1.0, full_model)], sr)
        .expect("Failed to create ContainerModel benchmark");

    container.set_slimmable_size(0.25);
    assert!(container.is_crossfading());

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut model = nam_rs::models::StaticModel::Container(Box::new(container));
    c.bench_function("Container_Crossfade_64samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of a free-geometry WaveNet Dynamic model (CH=5, K=3, COND=3).
///
/// CH=5 and condition_size=3 force the dispatcher to route to `WaveNetModelDyn`
/// instead of a const-generic SKU. This benchmark covers the dynamic hot-path
/// that is exercised when a user loads a model outside the catalog.
fn bench_wavenet_dynamic_process(c: &mut Criterion) {
    let data = make_wavenet_dyn_data();
    let mut model = build_model(&data).expect("Dispatcher failed for WaveNet Dynamic benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("WaveNet_Dynamic_CH5_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of an LSTM Dynamic model (1 layer × 7 hidden).
///
/// H=7 is not in the const-generic dispatch table ({3,8,12,16,24,40}),
/// forcing routing to `LstmModelDyn`. This covers the dynamic LSTM hot-path.
fn bench_lstm_dynamic_process(c: &mut Criterion) {
    let data = make_lstm_data(1, 7);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM Dynamic benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_Dynamic_1x7_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of a WaveNet A2 Dynamic model (CH=4, gated).
///
/// CH=4 is not in the A2 const-generic dispatch table ({3, 8}),
/// forcing routing to `WaveNetA2Dyn`. This covers the dynamic A2 hot-path
/// with gating active on the first layer, exercising the full dynamic engine.
fn bench_wavenet_a2_dyn_gated_process(c: &mut Criterion) {
    let data = make_wavenet_a2_dyn_data();
    let mut model = build_model(&data).expect("Dispatcher failed for WaveNet A2 Dynamic benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Dyn_Gated_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Comparative benchmark: WaveNet Standard (CH=16, cataloged) vs WaveNet Dynamic
/// (CH=5, fallback) side by side. Measures the dispatch overhead and performance
/// delta between the const-generic fast path and the dynamic free-geometry fallback.
fn bench_wavenet_comparison(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");

    let mut group = c.benchmark_group("WaveNet_Comparison");
    group.sample_size(50);

    if path.exists() {
        let json_data = std::fs::read_to_string(&path).expect("Failed to read WaveNet model");
        let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
        let mut model = build_model(&model_data).expect("Dispatcher failed");
        model.prewarm(2048);

        let input = generate_sine_440hz(64);
        let mut output = vec![0.0f32; 64];

        group.bench_function("Standard_CH16_cataloged", |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }

    let dyn_data = make_wavenet_dyn_data();
    let mut dyn_model =
        build_model(&dyn_data).expect("Dispatcher failed for WaveNet Dynamic benchmark");
    dyn_model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    group.bench_function("Dynamic_CH5_fallback", |b| {
        b.iter(|| {
            dyn_model.process(&input, &mut output);
        });
    });

    group.finish();
}

/// Comparative benchmark: A2-Full (CH=8), A2-Lite (CH=3), and A2-Dyn (CH=4, gated)
/// side by side. Validates the performance spread across the static const-generic
/// fast paths and the dynamic free-geometry fallback of the A2 architecture.
fn bench_a2_comparison(c: &mut Criterion) {
    let full_path = {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/models/wavenet_a2_full.nam");
        p
    };
    let lite_path = {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("tests/fixtures/models/wavenet_a2_lite.nam");
        p
    };

    let mut group = c.benchmark_group("A2_Comparison");
    group.sample_size(50);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    if full_path.exists() {
        let json = std::fs::read_to_string(&full_path).expect("Failed to read A2-Full");
        let data = parse_nam_json(&json).expect("Failed in JSON parser");
        let mut model = build_model(&data).expect("Dispatcher failed for A2-Full");
        model.prewarm(2048);

        group.bench_function("Full_CH8", |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }

    if lite_path.exists() {
        let json = std::fs::read_to_string(&lite_path).expect("Failed to read A2-Lite");
        let data = parse_nam_json(&json).expect("Failed in JSON parser");
        let mut model = build_model(&data).expect("Dispatcher failed for A2-Lite");
        model.prewarm(2048);

        group.bench_function("Lite_CH3", |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }

    let dyn_data = make_wavenet_a2_dyn_data();
    let mut dyn_model = build_model(&dyn_data).expect("Dispatcher failed for A2 Dynamic benchmark");
    dyn_model.prewarm(2048);

    group.bench_function("Dynamic_CH4_gated", |b| {
        b.iter(|| {
            dyn_model.process(&input, &mut output);
        });
    });

    group.finish();
}

/// Measures inference latency for any present non-distributable models.
fn bench_nondist_models(c: &mut Criterion) {
    let mut nondist_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    nondist_path.push("tests/fixtures/models-nondist");

    if !nondist_path.exists() {
        return;
    }

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&nondist_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext == "nam" || ext == "json")
            {
                models.push(path);
            }
        }
    }

    for model_path in models {
        let filename = model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let json_data = match std::fs::read_to_string(&model_path) {
            Ok(data) => data,
            Err(_) => continue,
        };
        let model_data = match parse_nam_json(&json_data) {
            Ok(data) => data,
            Err(_) => continue,
        };
        let mut model = match build_model(&model_data) {
            Ok(m) => m,
            Err(_) => continue,
        };
        model.prewarm(2048);

        let input = generate_sine_440hz(64);
        let mut output = vec![0.0f32; 64];

        c.bench_function(&format!("NonDist_Model_{}_64samp", filename), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures the end-to-end inference cost of a full ConvNet model (2 blocks, CH=8→4, K=3).
///
/// Unlike the ConvNetBlock-level benches (now in separate benches), this loads the
/// `convnet_test.nam` fixture, exercises the full model pipeline (multi-block chaining
/// + head_scale), and profiles the dispatcher build_model path.
fn bench_convnet_model_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/convnet_test.nam");

    if !path.exists() {
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Failed to read ConvNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for ConvNet benchmark");
    model.prewarm(2048);

    let num_out = match model.as_ref() {
        StaticModel::ConvNet(c) => c.out_channels(),
        _ => 1,
    };
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64 * num_out];

    c.bench_function("ConvNet_Model_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

criterion_group!(
    name = benches;
    config = criterion::Criterion::default().sample_size(50);
    targets = bench_wavenet_standard_process,
    bench_wavenet_p10_small_block_sizes,
    bench_wavenet_standard_block_sizes,
    bench_lstm_2x16_process,
    bench_lstm_2x16_block_sizes,
    bench_lstm_1x8_comparison,
    bench_lstm_2x16_comparison,
    bench_lstm_1x40_process,
    bench_lstm_2x24_process,
    bench_lstm_1x40_comparison,
    bench_lstm_2x24_comparison,
    bench_a2_full_process,
    bench_a2_full_block_sizes,
    bench_a2_lite_process,
    bench_a2_lite_block_sizes,
    bench_prewarm_wavenet_standard,
    bench_prewarm_lstm_2x16,
    bench_prewarm_a2_full,
    bench_prewarm_a2_lite,
    bench_linear_model_dot_product,
    bench_container_crossfade_64samp,
    bench_wavenet_dynamic_process,
    bench_lstm_dynamic_process,
    bench_wavenet_a2_dyn_gated_process,
    bench_wavenet_comparison,
    bench_a2_comparison,
    bench_nondist_models,
    bench_convnet_model_process
);

criterion_main!(benches);
