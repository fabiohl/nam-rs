// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Formal inference latency benchmarks for the NAM-rs engine.
//!
//! Measures the processing time of 1 DSP block (64 samples at 48 kHz = 1.33 ms
//! deadline) for WaveNet and LSTM neural networks, plus FastMath kernels
//! that comprise the SIMD activation functions.
//!
//! ## Available benchmarks
//!
//! | ID                                      | Description                             | Practical context                        |
//! | --------------------------------------- | --------------------------------------- | ---------------------------------------- |
//! | `WaveNet_Standard_CH16_64samp_48kHz`    | Complete WaveNet Standard inference     | ~284 KB model, 10+10 dilated layers      |
//! | `LSTM_2x16_64samp_48kHz`                | LSTM 2 layers × 16 hidden inference     | Heaviest supported recurrent network     |
//! | `A2Full_CH8_64samp_48kHz`               | A2-Full (CH=8) inference                | AVX2 col-major T=4 broadcast-FMA         |
//! | `A2Lite_CH3_64samp_48kHz`               | A2-Lite (CH=3) inference                | u16 interleaved GEMV, CPU-efficient      |
//! | `FastMath_tanh_AVX2_256elem`            | Padé×rsqrt tanh activation over 256 f32 | Kernel called N×layers/block in WaveNet  |
//! | `FastMath_sigmoid_AVX2_256elem`         | Sigmoid activation derived from tanh    | Kernel called N×gates/block in LSTM      |
//! | `WaveNet_Dynamic_CH5_64samp_48kHz`  | WaveNet Dynamic free-geom inference  | Fallback for non-cataloged WaveNet geometries |
//! | `LSTM_Dynamic_1x7_64samp_48kHz`      | LSTM Dynamic 1×7 inference           | Fallback for non-cataloged LSTM geometries     |
//! | `ConvNet_Model_64samp_48kHz`         | ConvNet end-to-end model inference   | Full pipeline: 2 blocks CH=8→4 + head_scale    |
//! | `A2Dyn_Gated_64samp_48kHz`           | A2 Dynamic CH=4 gated inference      | Fallback for non-cataloged A2 geometries       |
//!
//! ## Interpreting the results
//!
//! - The real-time deadline at 48 kHz with a 64-sample buffer is **1.33 ms**.
//! - If any inference benchmark exceeds this deadline, the engine will cause
//!   xruns (buffer underruns) in production with this buffer size.
//! - FastMath kernels are sub-components called hundreds of times per block;
//!   their total time contributes to the full inference latency.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench inference_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{NamConfig, NamLayerConfig, NamModelData, parse_nam_json};
use nam_rs::math::common::AlignedVec;
use nam_rs::models::NamModel;
use nam_rs::models::StaticModel;
use nam_rs::models::a2::activations::ActivationType;
use nam_rs::models::container::ContainerModel;
use nam_rs::models::convnet::ConvNetBlock;
use nam_rs::models::lstm::lstm_weight_count;
use nam_rs::models::slimmable::SlimmableModel;
use nam_rs::models::wavenet::dense::DenseLayer;

/// Generates a deterministic 440 Hz sinusoidal signal at a 48 kHz sample rate.
/// Used as a stable input to ensure processing load is consistent across
/// benchmark iterations, avoiding fluctuations from random signals.
fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

/// Creates a synthetic `NamModelData` struct configured as an LSTM network.
/// Allows testing different topologies (layers and hidden size) without depending
/// on external files, easing raw performance validation of the inference engine.
fn make_lstm_data(num_layers: usize, hidden_size: usize) -> NamModelData {
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
        // Weights initialized with a small value (0.01) to avoid premature saturation/infs
        // during repeated benchmark runs.
        weights: vec![0.01; total_weights],
        weights_layout: nam_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Creates a synthetic `NamModelData` for WaveNet Dynamic with free geometry.
///
/// CH=5, K=3, COND=3 — forces routing to `WaveNetModelDyn` because
/// CH ∉ {4,8,12,16} and `condition_size > 1` disqualifies catalog matching.
fn make_wavenet_dyn_data() -> NamModelData {
    let channels = 5usize;
    let kernel_size = 3usize;
    let condition_size = 3usize;
    let head_1 = 5usize;
    let head_2 = 1usize;
    let dilations = [vec![1, 2, 4, 8, 16], vec![1, 2, 4, 8, 16]];
    let num_layers_per_array = 5usize;

    let array1_rechannel = channels;
    let array2_rechannel = channels * channels;
    let per_layer = channels * kernel_size * channels + channels              // conv1d + bias
        + condition_size * channels                                // input_mixin
        + channels * channels + channels; // one_by_one + bias
    let array1_head = channels * head_1; // no bias
    let array2_head = channels * head_2 + head_2; // with bias
    let total_weights = array1_rechannel
        + num_layers_per_array * per_layer
        + array1_head
        + array2_rechannel
        + num_layers_per_array * per_layer
        + array2_head
        + 1; // head_scale

    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![
                nam_rs::loader::nam_json::NamLayerConfig {
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
                nam_rs::loader::nam_json::NamLayerConfig {
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

/// Measures the processing time of a real WaveNet model ("Standard").
/// This benchmark is the most representative for common guitar use,
/// testing the effectiveness of dilated convolutions and optimized SIMD kernels.
fn bench_wavenet_standard_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");

    // Silently ignores if the fixture model is not present
    if !path.exists() {
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    // The dispatcher picks the fastest implementation (static vs dynamic)
    let mut model = build_model(&model_data).expect("Dispatcher failed for benchmark");

    // Prewarm is CRITICAL to populate state buffers and prevent initial resource
    // allocation inside the model from skewing the benchmark average.
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("WaveNet_Standard_CH16_64samp_48kHz", |b| {
        b.iter(|| {
            // Runs inference on a full block (64 samples)
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
            // LSTM processing tends to be heavier than WaveNet for small blocks
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

    // Optimized path (SIMD / Auto-vectorized)
    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    // Explicit scalar path to measure theoretical speedup
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

    // Tests buffers from 32 to 512 samples
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

/// Creates a synthetic `NamModelData` for WaveNet A2 Dynamic with free geometry.
///
/// CH=4 forces routing to `WaveNetA2Dyn` because CH ∉ {3,8}
/// disqualifies the const-generic fast-path (A2-Full/Lite).
fn make_wavenet_a2_dyn_data() -> NamModelData {
    use nam_rs::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES};

    let channels = 4usize;
    let bottleneck = 4usize;
    let head_k = nam_rs::models::a2::params::A2_HEAD_KERNEL_SIZE;

    let mut total_weights = channels; // rechannel_w
    for &ksize in A2_KERNEL_SIZES.iter() {
        total_weights += channels * bottleneck * ksize; // conv_w
        total_weights += bottleneck; // conv_b
        total_weights += bottleneck; // mixin_w
        total_weights += bottleneck * channels; // l1x1_w
        total_weights += channels; // l1x1_b
    }
    total_weights += head_k * channels; // head_w
    total_weights += 1; // head_b
    total_weights += 1; // head_scale

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

// ── A2 long-run soak benches → moved to benches/long_inference_bench.rs (T2.2.3) ──

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

// --- Long Benchmarks (Soak Testing) → moved to benches/long_inference_bench.rs (T2.2.3) ---

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

/// Measures the processing time of `process_block` (head_rechannel FP32)
/// for three representative shapes, verifying the SIMD vectorization speedup.
///
/// Tested shapes:
/// - `DenseLayer<16,8>`: array1 (head_rechannel of WaveNet Standard)
/// - `DenseLayer<8,1>`:  array2 dominant case (final head 8→1)
/// - `DenseLayer<16,1>`: LSTM head single-output
fn bench_head_rechannel_fp32(c: &mut Criterion) {
    let num_frames: usize = 64;
    let mut group = c.benchmark_group("head_rechannel_fp32");

    let avx512_supported =
        std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl");

    // ── DenseLayer<16,8> ──
    {
        let in_size: usize = 16;
        let out_size: usize = 8;
        let weights: AlignedVec<f32> =
            AlignedVec::new(in_size * out_size, 0.01).expect("bench allocation failed");
        let bias: AlignedVec<f32> =
            AlignedVec::new(out_size, 0.0).expect("bench allocation failed");
        let layer = DenseLayer::<16, 8> {
            weights,
            bias,
            do_bias: true,
        };
        let input = vec![0.01f32; num_frames * in_size];
        let mut output = vec![0.0f32; num_frames * out_size];

        group.bench_function("DenseLayer_16x8_64f_AVX2", |b| {
            b.iter(|| unsafe {
                layer.process_block::<nam_rs::math::common::Avx2Math>(
                    &input,
                    &mut output,
                    num_frames,
                )
            });
        });

        if avx512_supported {
            group.bench_function("DenseLayer_16x8_64f_AVX512", |b| {
                b.iter(|| unsafe {
                    layer.process_block::<nam_rs::math::common::Avx512Math>(
                        &input,
                        &mut output,
                        num_frames,
                    )
                });
            });
        }

        group.bench_function("DenseLayer_16x8_64f_Scalar", |b| {
            b.iter(|| {
                nam_rs::math::common::scalar_ref::gemv_with_bias_f32_fallback(
                    &input,
                    &layer.weights,
                    &layer.bias,
                    &mut output,
                    num_frames,
                )
            });
        });
    }

    // ── DenseLayer<8,1> (dominant case) ──
    {
        let in_size: usize = 8;
        let out_size: usize = 1;
        let weights: AlignedVec<f32> =
            AlignedVec::new(in_size * out_size, 0.01).expect("bench allocation failed");
        let bias: AlignedVec<f32> =
            AlignedVec::new(out_size, 0.0).expect("bench allocation failed");
        let layer = DenseLayer::<8, 1> {
            weights,
            bias,
            do_bias: true,
        };
        let input = vec![0.01f32; num_frames * in_size];
        let mut output = vec![0.0f32; num_frames * out_size];

        group.bench_function("DenseLayer_8x1_64f_AVX2", |b| {
            b.iter(|| unsafe {
                layer.process_block::<nam_rs::math::common::Avx2Math>(
                    &input,
                    &mut output,
                    num_frames,
                )
            });
        });

        if avx512_supported {
            group.bench_function("DenseLayer_8x1_64f_AVX512", |b| {
                b.iter(|| unsafe {
                    layer.process_block::<nam_rs::math::common::Avx512Math>(
                        &input,
                        &mut output,
                        num_frames,
                    )
                });
            });
        }

        group.bench_function("DenseLayer_8x1_64f_Scalar", |b| {
            b.iter(|| {
                nam_rs::math::common::scalar_ref::gemv_with_bias_f32_fallback(
                    &input,
                    &layer.weights,
                    &layer.bias,
                    &mut output,
                    num_frames,
                )
            });
        });
    }

    // ── DenseLayer<16,1> (LSTM head) ──
    {
        let in_size: usize = 16;
        let out_size: usize = 1;
        let weights: AlignedVec<f32> =
            AlignedVec::new(in_size * out_size, 0.01).expect("bench allocation failed");
        let bias: AlignedVec<f32> =
            AlignedVec::new(out_size, 0.0).expect("bench allocation failed");
        let layer = DenseLayer::<16, 1> {
            weights,
            bias,
            do_bias: true,
        };
        let input = vec![0.01f32; num_frames * in_size];
        let mut output = vec![0.0f32; num_frames * out_size];

        group.bench_function("DenseLayer_16x1_64f_AVX2", |b| {
            b.iter(|| unsafe {
                layer.process_block::<nam_rs::math::common::Avx2Math>(
                    &input,
                    &mut output,
                    num_frames,
                )
            });
        });

        if avx512_supported {
            group.bench_function("DenseLayer_16x1_64f_AVX512", |b| {
                b.iter(|| unsafe {
                    layer.process_block::<nam_rs::math::common::Avx512Math>(
                        &input,
                        &mut output,
                        num_frames,
                    )
                });
            });
        }

        group.bench_function("DenseLayer_16x1_64f_Scalar", |b| {
            b.iter(|| {
                nam_rs::math::common::scalar_ref::gemv_with_bias_f32_fallback(
                    &input,
                    &layer.weights,
                    &layer.bias,
                    &mut output,
                    num_frames,
                )
            });
        });
    }

    group.finish();
}
// CLAP benchmarks → moved to benches/clap_bench.rs// ── Gate FSM Benchmarks ──

// ── LinearModel Dot Product Benchmarks ──

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

// ── ConvNet Multi-Channel Convolution Benchmarks (T3.2.4) ──

/// Creates a synthetic `ConvNetBlock` with identity-like weights for benchmarking
/// the Conv1D + BatchNorm + Activation pipeline at a specific (in_ch, out_ch) pair.
fn make_convnet_block(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    do_bias: bool,
) -> ConvNetBlock {
    let mut block = ConvNetBlock::new(
        in_ch,
        out_ch,
        kernel,
        dilation,
        do_bias,
        ActivationType::Tanh,
        0,
    )
    .expect("create convnet block for bench");

    let num_blocks = out_ch.div_ceil(4);
    let padded_total = num_blocks * kernel * in_ch * 4;

    let mut weights = vec![0.0f32; padded_total];
    for b in 0..num_blocks {
        for k in 0..kernel {
            for c_in in 0..in_ch {
                for c_out in 0..4usize {
                    let dst_c = b * 4 + c_out;
                    if dst_c < out_ch {
                        let idx = c_out + c_in * 4 + k * in_ch * 4 + b * kernel * in_ch * 4;
                        weights[idx] = if dst_c == c_in.min(out_ch - 1) {
                            1.0
                        } else {
                            0.01
                        };
                    }
                }
            }
        }
    }
    block.set_conv_weights(&weights);

    if do_bias {
        let bias = vec![0.0f32; out_ch];
        block.set_conv_bias(&bias);
    }

    let bn_scale = vec![1.0f32; out_ch];
    let bn_offset = vec![0.0f32; out_ch];
    block.set_bn_params(&bn_scale, &bn_offset).unwrap();

    block
}

/// Generates an interleaved multi-channel signal for benchmarking.
/// Layout: `[f0_c0, f0_c1, ..., f0_c{n-1}, f1_c0, ..., f{t-1}_c{n-1}]`
fn generate_multichannel_sine(num_frames: usize, num_channels: usize) -> Vec<f32> {
    (0..num_frames)
        .flat_map(|f| {
            let base = (2.0 * std::f32::consts::PI * 440.0 * (f as f32) / 48000.0).sin();
            (0..num_channels).map(move |c| base * (1.0 + 0.2 * c as f32))
        })
        .collect()
}

/// Benchmarks ConvNetBlock convolution throughput across channel dimensions.
///
/// Covers all 9 combinations of `in_ch ∈ {1, 2, 4}` × `out_ch ∈ {1, 2, 4}`
/// with kernel=3, dilation=1, bias=true — measuring Conv1D + BatchNorm + Tanh
/// over 64 audio frames at 48 kHz.
fn bench_convnet_multichannel(c: &mut Criterion) {
    const NUM_FRAMES: usize = 64;
    const KERNEL: usize = 3;
    const DILATION: usize = 1;
    const DO_BIAS: bool = true;

    let combos: [(usize, usize); 9] = [
        (1, 1),
        (1, 2),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 4),
        (4, 1),
        (4, 2),
        (4, 4),
    ];

    let mut group = c.benchmark_group("ConvNet_MultiChannel_64samp");
    group.sample_size(50);

    for &(in_ch, out_ch) in &combos {
        let mut block = make_convnet_block(in_ch, out_ch, KERNEL, DILATION, DO_BIAS);
        let input = generate_multichannel_sine(NUM_FRAMES, in_ch);
        let mut output = vec![0.0f32; NUM_FRAMES * out_ch];

        unsafe {
            block.process_block(&input, &mut output, NUM_FRAMES);
        }

        group.bench_function(format!("ConvNetBlock_in{}_out{}", in_ch, out_ch), |b| {
            b.iter(|| unsafe {
                block.process_block(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                    NUM_FRAMES,
                );
            });
        });
    }

    group.finish();
}

/// Benchmarks ConvNetBlock with larger kernel sizes (5 and 7) to stress
/// the tap-pointer prefetch logic in Conv1dDyn convolution.
fn bench_convnet_large_kernels(c: &mut Criterion) {
    const NUM_FRAMES: usize = 64;
    const DO_BIAS: bool = true;

    let mut group = c.benchmark_group("ConvNet_LargeKernel_64samp");
    group.sample_size(50);

    for &((in_ch, out_ch, kernel), label) in &[
        ((1, 1, 5), "ConvNetBlock_in1_out1_k5"),
        ((2, 2, 5), "ConvNetBlock_in2_out2_k5"),
        ((4, 4, 5), "ConvNetBlock_in4_out4_k5"),
        ((1, 1, 7), "ConvNetBlock_in1_out1_k7"),
        ((2, 2, 7), "ConvNetBlock_in2_out2_k7"),
        ((4, 4, 7), "ConvNetBlock_in4_out4_k7"),
    ] {
        let mut block = make_convnet_block(in_ch, out_ch, kernel, 1, DO_BIAS);
        let input = generate_multichannel_sine(NUM_FRAMES, in_ch);
        let mut output = vec![0.0f32; NUM_FRAMES * out_ch];

        unsafe {
            block.process_block(&input, &mut output, NUM_FRAMES);
        }

        group.bench_function(label, |b| {
            b.iter(|| unsafe {
                block.process_block(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                    NUM_FRAMES,
                );
            });
        });
    }

    group.finish();
}

/// Benchmarks ConvNetBlock with dilated convolutions (dilation ∈ {1, 2, 4}),
/// which exercise the full tap-pointer offset computation in Conv1dDyn.
fn bench_convnet_dilated(c: &mut Criterion) {
    const NUM_FRAMES: usize = 64;
    const DO_BIAS: bool = true;
    const KERNEL: usize = 3;

    let mut group = c.benchmark_group("ConvNet_Dilated_64samp");
    group.sample_size(50);

    for &(dilation, label) in &[
        (1, "ConvNetBlock_d1_in4_out4"),
        (2, "ConvNetBlock_d2_in4_out4"),
        (4, "ConvNetBlock_d4_in4_out4"),
    ] {
        let (in_ch, out_ch) = (4, 4);
        let mut block = make_convnet_block(in_ch, out_ch, KERNEL, dilation, DO_BIAS);
        let input = generate_multichannel_sine(NUM_FRAMES, in_ch);
        let mut output = vec![0.0f32; NUM_FRAMES * out_ch];

        unsafe {
            block.process_block(&input, &mut output, NUM_FRAMES);
        }

        group.bench_function(label, |b| {
            b.iter(|| unsafe {
                block.process_block(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                    NUM_FRAMES,
                );
            });
        });
    }

    group.finish();
}

// PGO: name uses _64samp suffix for build-release.sh profiling filter
/// Measures the end-to-end inference cost of a full ConvNet model (2 blocks, CH=8→4, K=3).
///
/// Unlike the ConvNetBlock-level benches, this loads the `convnet_test.nam` fixture,
/// exercises the full model pipeline (multi-block chaining + head_scale), and profiles
/// the dispatcher build_model path.
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

// cabsim benchmarks → moved to benches/cabsim_bench.rs
// cabsim_long_run → moved to benches/long_inference_bench.rs (T2.2.3)

// Main benchmark group definition (inference latency and DSP kernels)
criterion_group!(
    name = benches;
    // sample_size(50) is a balance between statistical accuracy and total runtime.
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
    bench_a2_head_ch8,
    bench_a2_head_ch3,
    bench_head_rechannel_fp32,
    bench_linear_model_dot_product,
    bench_container_crossfade_64samp,
    bench_wavenet_dynamic_process,
    bench_lstm_dynamic_process,
    bench_wavenet_a2_dyn_gated_process,
    bench_wavenet_comparison,
    bench_a2_comparison,
    bench_nondist_models,
    bench_convnet_multichannel,
    bench_convnet_large_kernels,
    bench_convnet_dilated,
    bench_convnet_model_process
);

/// Bench: A2 Head Conv CH=8 — scalar vs AVX2+FMA (16 frames).
///
/// Tests the isolated A2 Head Conv kernel with 8 channels over 16
/// frames, comparing the scalar reference implementation against
/// the AVX2+FMA SIMD kernel with T=4 frame-tiling.
fn bench_a2_head_ch8(c: &mut Criterion) {
    const NUM_FRAMES: usize = 16;
    const RING_SIZE: usize = 256;
    const RING_MASK: usize = RING_SIZE - 1;
    const CH: usize = 8;
    const K: usize = 16;
    let write_pos: usize = 200;

    let mut state: u32 = 42;
    let mut w = AlignedVec::new(K * CH, 0.0f32).expect("bench allocation failed");
    for val in &mut *w {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
    }
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let head_b = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let head_scale = ((state as f32) / (u32::MAX as f32)) * 0.5 + 0.75;

    let mut history = vec![0.0f32; CH * RING_SIZE];
    state = 99;
    for val in &mut history {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }

    let mut output = vec![0.0f32; NUM_FRAMES];
    let avx2_available =
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma");

    let mut group = c.benchmark_group("A2HeadConv_CH8");
    group.bench_function("a2_head_ch8_scalar", |b| {
        b.iter(|| {
            nam_rs::models::a2::a2_head_block_scalar_ref(
                std::hint::black_box(&w),
                std::hint::black_box(head_b),
                std::hint::black_box(head_scale),
                std::hint::black_box(CH),
                std::hint::black_box(&history),
                std::hint::black_box(write_pos),
                std::hint::black_box(RING_MASK),
                std::hint::black_box(NUM_FRAMES),
                std::hint::black_box(&mut output),
            );
        });
    });

    if avx2_available {
        group.bench_function("a2_head_ch8_avx2", |b| {
            b.iter(|| unsafe {
                nam_rs::models::a2::head_process_ch8_avx2(
                    std::hint::black_box(&w),
                    std::hint::black_box(head_b),
                    std::hint::black_box(head_scale),
                    std::hint::black_box(&history),
                    std::hint::black_box(write_pos),
                    std::hint::black_box(RING_MASK),
                    std::hint::black_box(NUM_FRAMES),
                    std::hint::black_box(&mut output),
                );
            });
        });
    }
    group.finish();
}

/// Bench: A2 Head Conv CH=3 — scalar vs SSE+FMA (16 frames).
///
/// Tests the isolated A2 Head Conv kernel with 3 channels over 16
/// frames, comparing the scalar reference implementation against
/// the SSE+FMA SIMD kernel.
fn bench_a2_head_ch3(c: &mut Criterion) {
    const NUM_FRAMES: usize = 16;
    const RING_SIZE: usize = 256;
    const RING_MASK: usize = RING_SIZE - 1;
    const CH: usize = 3;
    const K: usize = 16;
    let write_pos: usize = 200;

    let mut state: u32 = 42;
    let mut w = AlignedVec::new(K * CH, 0.0f32).expect("bench allocation failed");
    for val in &mut *w {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
    }
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let head_b = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let head_scale = ((state as f32) / (u32::MAX as f32)) * 0.5 + 0.75;

    let mut history = vec![0.0f32; CH * RING_SIZE];
    state = 99;
    for val in &mut history {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }

    let mut output = vec![0.0f32; NUM_FRAMES];
    let sse_available = std::arch::is_x86_feature_detected!("fma");

    let mut group = c.benchmark_group("A2HeadConv_CH3");
    group.bench_function("a2_head_ch3_scalar", |b| {
        b.iter(|| {
            nam_rs::models::a2::a2_head_block_scalar_ref(
                std::hint::black_box(&w),
                std::hint::black_box(head_b),
                std::hint::black_box(head_scale),
                std::hint::black_box(CH),
                std::hint::black_box(&history),
                std::hint::black_box(write_pos),
                std::hint::black_box(RING_MASK),
                std::hint::black_box(NUM_FRAMES),
                std::hint::black_box(&mut output),
            );
        });
    });

    if sse_available {
        group.bench_function("a2_head_ch3_sse", |b| {
            b.iter(|| unsafe {
                nam_rs::models::a2::head_process_ch3_sse(
                    std::hint::black_box(&w),
                    std::hint::black_box(head_b),
                    std::hint::black_box(head_scale),
                    std::hint::black_box(&history),
                    std::hint::black_box(write_pos),
                    std::hint::black_box(RING_MASK),
                    std::hint::black_box(NUM_FRAMES),
                    std::hint::black_box(&mut output),
                );
            });
        });
    }
    group.finish();
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

// PGO: group name uses _64samp suffix to match build-release.sh profiling filter
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

// PGO: group name uses _64samp suffix to match build-release.sh profiling filter
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

// PGO: name uses _64samp suffix for build-release.sh profiling filter
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

        // Run with standard 64 sample block size at 48kHz
        let input = generate_sine_440hz(64);
        let mut output = vec![0.0f32; 64];

        c.bench_function(&format!("NonDist_Model_{}_64samp", filename), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

criterion_main!(benches);
