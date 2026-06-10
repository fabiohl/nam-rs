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
//! | `FastMath_tanh_AVX2_256elem`            | Padé×rsqrt tanh activation over 256 f32 | Kernel called N×layers/block in WaveNet  |
//! | `FastMath_sigmoid_AVX2_256elem`         | Sigmoid activation derived from tanh    | Kernel called N×gates/block in LSTM      |
//! | `WaveNet_Dynamic_Standard_64samp_48kHz` | WaveNet Dynamic inference (fallback)    | Measures overhead of path without const generics |
//! | `LSTM_Dynamic_1x16_64samp_48kHz`        | LSTM Dynamic 1×16 inference (fallback)  | Measures overhead of path without const generics |
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
use nam_rs::loader::nam_json::{NamConfig, NamModelData, parse_nam_json};
use nam_rs::math::common::AlignedVec;
use nam_rs::models::NamModel;
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
        // Weights initialized with a small value (0.01) to avoid premature saturation/infs
        // during repeated benchmark runs.
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
    let data = make_lstm_data(2, 16, 3345);
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
    let data = make_lstm_data(1, 8, 345);
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
    #[cfg(any(test, feature = "long_bench"))]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::DynamicModel::Lstm1x8(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm1x8"),
    });
    group.finish();
}

/// Comparative Benchmark (T15): LSTM 2x16 Scalar vs SIMD (Fused Gates T3).
fn bench_lstm_2x16_comparison(c: &mut Criterion) {
    let data = make_lstm_data(2, 16, 3345);
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

    #[cfg(any(test, feature = "long_bench"))]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::DynamicModel::Lstm2x16(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm2x16"),
    });
    group.finish();
}

/// Measures the performance of the AVX2-optimized `tanh` activation kernel.
/// Uses piecewise minimax odd polynomials (degree 5) with branchless blending
/// for maximum throughput at the expense of sub-sample precision.
fn bench_tanh_slice_256(c: &mut Criterion) {
    // Input range covering the linear and saturation regions of tanh
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

    c.bench_function("FastMath_tanh_AVX2_256elem", |b| {
        let mut buf = base.clone();
        b.iter(|| {
            // Copy original data to ensure the kernel always processes the same
            // values, simulating the actual load of a neural layer.
            buf.copy_from_slice(&base);
            unsafe { nam_rs::math::activations::tanh_slice_avx2(&mut buf) };
        });
    });
}

/// E8.T04: Padé [5,4] tanh with double Newton-Raphson on reciprocal (AVX2).
/// Evaluates rcp_ps + 2×NR iteration throughput vs piecewise minimax.
fn bench_tanh_pade_nr2_256(c: &mut Criterion) {
    use std::arch::x86_64::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PadeNR2_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for chunk in buf.chunks_exact_mut(8) {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    let y = nam_rs::math::activations::simd_tanh_pade_nr2_avx2(x);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), y);
                }
            }
        });
    });
}

/// E8.T04: Padé [5,4] tanh with hardware division oracle (AVX2).
/// IEEE 754 full-precision reference — maximum fidelity, minimum throughput.
fn bench_tanh_pade_div_256(c: &mut Criterion) {
    use std::arch::x86_64::*;
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
    c.bench_function("FastMath_tanh_PadeDiv_AVX2_256elem", |b| {
        b.iter(|| {
            let mut buf = base.clone();
            unsafe {
                for chunk in buf.chunks_exact_mut(8) {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    let y = nam_rs::math::activations::simd_tanh_avx2(x);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), y);
                }
            }
        });
    });
}

/// Measures the performance of the AVX2-optimized `sigmoid` activation kernel.
/// Essential for LSTM models, this kernel converts the approximate tanh into a
/// logistic function (0 to 1) to control memory gates.
fn bench_sigmoid_slice_256(c: &mut Criterion) {
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

    c.bench_function("FastMath_sigmoid_AVX2_256elem", |b| {
        let mut buf = base.clone();
        b.iter(|| {
            buf.copy_from_slice(&base);
            unsafe { nam_rs::math::activations::sigmoid_slice_avx2(&mut buf) };
        });
    });
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
    let data = make_lstm_data(2, 16, 3345);
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

/// Measures the throughput of the AVX2 dot product with f16 (Half Precision) weights.
/// This technique reduces memory bandwidth usage by 50% and improves
/// L1 cache locality, crucial for WaveNet dense layers.
fn bench_dot_product_avx2_256(c: &mut Criterion) {
    let vec_a = AlignedVec::from_vec((0..256).map(|i| (i as f32) * 0.1).collect());
    let vec_b = AlignedVec::from_vec(
        (0..256)
            .map(|i| half::f16::from_f32((i as f32) * -0.1).to_bits())
            .collect(),
    );

    c.bench_function("DotProduct_AVX2_256elem", |b| {
        b.iter(|| unsafe {
            // black_box prevents the compiler from optimizing away the entire loop,
            // ensuring the math computation is actually executed.
            nam_rs::math::gemm::dot::dot_product_avx2(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

/// Dot product version for small vectors (64 elements).
/// Represents the typical intermediate layer size in lightweight models.
fn bench_dot_product_avx2_64(c: &mut Criterion) {
    let vec_a = AlignedVec::from_vec((0..64).map(|i| (i as f32) * 0.1).collect());
    let vec_b = AlignedVec::from_vec(
        (0..64)
            .map(|i| half::f16::from_f32((i as f32) * -0.1).to_bits())
            .collect(),
    );
    c.bench_function("DotProduct_AVX2_64elem", |b| {
        b.iter(|| unsafe {
            nam_rs::math::gemm::dot::dot_product_avx2(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

/// Measures the resampler cost when converting from 44.1 kHz to 48 kHz.
/// The resampler is one of the most sensitive components, as it involves FIR filtering.
/// `process_input` and `process_output` are measured separately to identify
/// bottlenecks in input (buffering) vs output (interpolation).
fn bench_resampler_44100_to_48000_256samp(c: &mut Criterion) {
    use nam_rs::dsp::resampler::NamResampler;
    let size = 256;
    let mut rs = NamResampler::new(44_100, 48_000, size).unwrap();
    let in_l = vec![0.0f32; size];
    let in_r = vec![0.0f32; size];
    let mut out_l = vec![0.0f32; size * 2];
    let mut out_r = vec![0.0f32; size * 2];
    let mut group = c.benchmark_group("Resampler_44100_to_48000_256samp");
    group.bench_function("process_input", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_input_mono", |b| {
        b.iter(|| {
            rs.process_input_mono(&in_l, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_output", |b| {
        b.iter(|| {
            rs.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_output_mono", |b| {
        b.iter(|| {
            rs.process_output_mono(&in_l, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

/// Measures 96 kHz to 48 kHz conversion (downsampling).
/// Generally lighter than upsampling, but still requires anti-aliasing filtering.
fn bench_resampler_96000_to_48000_256samp(c: &mut Criterion) {
    use nam_rs::dsp::resampler::NamResampler;
    let size = 256;
    let mut rs = NamResampler::new(96_000, 48_000, size).unwrap();
    let in_l = vec![0.0f32; size];
    let in_r = vec![0.0f32; size];
    let mut out_l = vec![0.0f32; size * 2];
    let mut out_r = vec![0.0f32; size * 2];
    let mut group = c.benchmark_group("Resampler_96000_to_48000_256samp");
    group.bench_function("process_input", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_input_mono", |b| {
        b.iter(|| {
            rs.process_input_mono(&in_l, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_output", |b| {
        b.iter(|| {
            rs.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_output_mono", |b| {
        b.iter(|| {
            rs.process_output_mono(&in_l, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

/// Measures the performance of the latency histogram `record` function.
/// Simulates 64 calls (equivalent to processing 1 second of audio at 48 kHz
/// with a 64-sample buffer, or about 750 callbacks). The benchmark validates that
/// `fetch_add` is significantly faster than `fetch_update` (CAS-loop).
fn bench_record(c: &mut Criterion) {
    use nam_rs::dsp::telemetry::LatencyHistogram;
    let hist = LatencyHistogram::new();
    let durations: Vec<u64> = (0..64).map(|i| (i * 100) as u64).collect();

    c.bench_function("bench_record_64calls", |b| {
        b.iter(|| {
            for &d in &durations {
                hist.record(d);
            }
        });
    });
}

/// Measures the resampler overhead when sample rates are equal.
/// Serves to validate that the "bypass" path is efficient.
fn bench_resampler_48000_bypass(c: &mut Criterion) {
    use nam_rs::dsp::resampler::NamResampler;
    let size = 256;
    let mut rs = NamResampler::new(48_000, 48_000, size).unwrap();
    let in_l = vec![0.0f32; size];
    let in_r = vec![0.0f32; size];
    let mut out_l = vec![0.0f32; size];
    let mut out_r = vec![0.0f32; size];
    c.bench_function("Resampler_48000_bypass_256samp", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
}

/// Benchmarks for processors that support AVX-512 (e.g. AMD Zen 4, Intel Ice Lake+).
/// AVX-512 allows processing 16 floats simultaneously (512 bits), theoretically
/// doubling throughput compared to AVX2.
fn bench_tanh_avx512_256elem(c: &mut Criterion) {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
        c.bench_function("FastMath_tanh_AVX512_256elem", |b| {
            let mut buf = base.clone();
            b.iter(|| {
                buf.copy_from_slice(&base);
                unsafe { nam_rs::math::activations::tanh_slice_avx512(&mut buf) };
            });
        });
    }
}

fn bench_sigmoid_avx512_256elem(c: &mut Criterion) {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
        c.bench_function("FastMath_sigmoid_AVX512_256elem", |b| {
            let mut buf = base.clone();
            b.iter(|| {
                buf.copy_from_slice(&base);
                unsafe { nam_rs::math::activations::sigmoid_slice_avx512(&mut buf) };
            });
        });
    }
}

/// E8.T04: Padé [5,4] tanh with double Newton-Raphson (AVX-512).
fn bench_tanh_pade_nr2_avx512_256elem(c: &mut Criterion) {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
        use std::arch::x86_64::*;
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
        c.bench_function("FastMath_tanh_PadeNR2_AVX512_256elem", |b| {
            b.iter(|| {
                let mut buf = base.clone();
                unsafe {
                    for chunk in buf.chunks_exact_mut(16) {
                        let x = _mm512_loadu_ps(chunk.as_ptr());
                        let y = nam_rs::math::activations::simd_tanh_pade_nr2_avx512(x);
                        _mm512_storeu_ps(chunk.as_mut_ptr(), y);
                    }
                }
            });
        });
    }
}

/// E8.T04: Padé [5,4] tanh with hardware division oracle (AVX-512).
fn bench_tanh_pade_div_avx512_256elem(c: &mut Criterion) {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
        use std::arch::x86_64::*;
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
        c.bench_function("FastMath_tanh_PadeDiv_AVX512_256elem", |b| {
            b.iter(|| {
                let mut buf = base.clone();
                unsafe {
                    for chunk in buf.chunks_exact_mut(16) {
                        let x = _mm512_loadu_ps(chunk.as_ptr());
                        let y = nam_rs::math::activations::simd_tanh_avx512(x);
                        _mm512_storeu_ps(chunk.as_mut_ptr(), y);
                    }
                }
            });
        });
    }
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
    let data = make_lstm_data(2, 16, 3345);
    c.bench_function("Prewarm_LSTM_2x16_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&data).expect("Dispatcher failed"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

// --- Long Benchmarks (Soak Testing) ---
// These benchmarks only run if the "long_bench" feature is enabled.
// They are intended to validate CPU thermal stability and detect performance
// variations over time (jitters, throttling).

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
    let input = generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_WaveNet");
    // Extended run (35 seconds) to ensure statistical convergence and avoid timeout warnings
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
    let data = make_lstm_data(2, 16, 3345);
    let mut model = build_model(&data).expect("Dispatcher failed");
    model.prewarm(4096);
    let size = 4096;
    let input = generate_sine_440hz(size);
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
            // Validate that the resampler maintains stability and does not accumulate
            // phase errors or variable latency over long periods.
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

fn bench_lstm_1x40_process(c: &mut Criterion) {
    let data = make_lstm_data(1, 40, 6841);
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
    let data = make_lstm_data(2, 24, 7321);
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
    let data = make_lstm_data(1, 40, 6841);
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

    #[cfg(any(test, feature = "long_bench"))]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::DynamicModel::Lstm1x40(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm1x40"),
    });
    group.finish();
}

fn bench_lstm_2x24_comparison(c: &mut Criterion) {
    let data = make_lstm_data(2, 24, 7321);
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

    #[cfg(any(test, feature = "long_bench"))]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::DynamicModel::Lstm2x24(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Model is not Lstm2x24"),
    });
    group.finish();
}

/// Measures the processing time of `process_block_f32_native` (head_rechannel FP32)
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
        let weights: AlignedVec<f32> = AlignedVec::new(in_size * out_size, 0.01);
        let bias: AlignedVec<f32> = AlignedVec::new(out_size, 0.0);
        let layer = DenseLayer::<16, 8> {
            weights: AlignedVec::new(0, 0u16),
            bias,
            do_bias: true,
            f32_weights: Some(weights),
        };
        let input = vec![0.01f32; num_frames * in_size];
        let mut output = vec![0.0f32; num_frames * out_size];

        group.bench_function("DenseLayer_16x8_64f_AVX2", |b| {
            b.iter(|| unsafe {
                layer.process_block_f32_native::<nam_rs::math::common::Avx2Math>(
                    &input,
                    &mut output,
                    num_frames,
                )
            });
        });

        if avx512_supported {
            group.bench_function("DenseLayer_16x8_64f_AVX512", |b| {
                b.iter(|| unsafe {
                    layer.process_block_f32_native::<nam_rs::math::common::Avx512Math>(
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
                    layer.f32_weights.as_ref().unwrap(),
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
        let weights: AlignedVec<f32> = AlignedVec::new(in_size * out_size, 0.01);
        let bias: AlignedVec<f32> = AlignedVec::new(out_size, 0.0);
        let layer = DenseLayer::<8, 1> {
            weights: AlignedVec::new(0, 0u16),
            bias,
            do_bias: true,
            f32_weights: Some(weights),
        };
        let input = vec![0.01f32; num_frames * in_size];
        let mut output = vec![0.0f32; num_frames * out_size];

        group.bench_function("DenseLayer_8x1_64f_AVX2", |b| {
            b.iter(|| unsafe {
                layer.process_block_f32_native::<nam_rs::math::common::Avx2Math>(
                    &input,
                    &mut output,
                    num_frames,
                )
            });
        });

        if avx512_supported {
            group.bench_function("DenseLayer_8x1_64f_AVX512", |b| {
                b.iter(|| unsafe {
                    layer.process_block_f32_native::<nam_rs::math::common::Avx512Math>(
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
                    layer.f32_weights.as_ref().unwrap(),
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
        let weights: AlignedVec<f32> = AlignedVec::new(in_size * out_size, 0.01);
        let bias: AlignedVec<f32> = AlignedVec::new(out_size, 0.0);
        let layer = DenseLayer::<16, 1> {
            weights: AlignedVec::new(0, 0u16),
            bias,
            do_bias: true,
            f32_weights: Some(weights),
        };
        let input = vec![0.01f32; num_frames * in_size];
        let mut output = vec![0.0f32; num_frames * out_size];

        group.bench_function("DenseLayer_16x1_64f_AVX2", |b| {
            b.iter(|| unsafe {
                layer.process_block_f32_native::<nam_rs::math::common::Avx2Math>(
                    &input,
                    &mut output,
                    num_frames,
                )
            });
        });

        if avx512_supported {
            group.bench_function("DenseLayer_16x1_64f_AVX512", |b| {
                b.iter(|| unsafe {
                    layer.process_block_f32_native::<nam_rs::math::common::Avx512Math>(
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
                    layer.f32_weights.as_ref().unwrap(),
                    &layer.bias,
                    &mut output,
                    num_frames,
                )
            });
        });
    }

    group.finish();
}

#[cfg(feature = "clap-plugin")]
struct BenchHostShared;
#[cfg(feature = "clap-plugin")]
impl clack_host::prelude::SharedHandler<'_> for BenchHostShared {
    fn request_restart(&self) {}
    fn request_process(&self) {}
    fn request_callback(&self) {}
}

#[cfg(feature = "clap-plugin")]
struct BenchHost;
#[cfg(feature = "clap-plugin")]
impl clack_host::prelude::HostHandlers for BenchHost {
    type Shared<'a> = BenchHostShared;
    type MainThread<'a> = ();
    type AudioProcessor<'a> = ();
}

#[cfg(feature = "clap-plugin")]
fn bench_clap_process_block_64samp(c: &mut criterion::Criterion) {
    use clack_common::events::Pckn;
    use clack_common::events::event_types::ParamValueEvent;
    use clack_common::utils::{ClapId, Cookie};
    use clack_host::prelude::*;
    use nam_rs::clap::extensions::params::{PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN};
    use nam_rs::clap::plugin::NamClapPlugin;

    let entry =
        PluginEntry::load_from_clack::<clack_plugin::entry::SinglePluginEntry<NamClapPlugin>>(
            c"/bench",
        )
        .expect("Failed to load PluginEntry");

    let host_info = HostInfo::new(
        "NAM-rs Bench Host",
        "Fabio Lima",
        "https://github.com/fabiohl/nam-rs",
        "0.1.0",
    )
    .unwrap();

    let mut plugin_instance = PluginInstance::<BenchHost>::new(
        |_| BenchHostShared,
        |_| (),
        &entry,
        c"br.eti.fabiolima.nam-rs",
        &host_info,
    )
    .expect("Failed to create plugin instance");

    let audio_config = PluginAudioConfiguration {
        sample_rate: 48000.0,
        min_frames_count: 64,
        max_frames_count: 64,
    };

    let stopped_processor = plugin_instance
        .activate(|_, _| (), audio_config)
        .expect("Failed to activate plugin");

    let mut started_processor = stopped_processor
        .start_processing()
        .expect("Failed to start processing");

    // Prepare non-silent audio buffers (sine wave at 440Hz)
    // Non-silent so the gate stays open and the DSP/gain/peaks code is fully exercised
    let sine = generate_sine_440hz(64);
    let mut input_audio_buffers = [sine.clone(), sine.clone()];
    let mut output_audio_buffers = [[0.0f32; 64], [0.0f32; 64]];

    let mut input_ports = AudioPorts::with_capacity(2, 1);
    let mut output_ports = AudioPorts::with_capacity(2, 1);

    let mut output_events_buffer = EventBuffer::with_capacity(10);

    let mut group = c.benchmark_group("CLAP_process_block_64samp");

    // 1. Fast Path (SIMD) - stable parameters, no changes
    group.bench_function("SIMD_FastPath", |b| {
        b.iter(|| {
            let input_events = InputEvents::empty();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            // Reconstruct the port views to keep it realistic
            let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_audio_buffers.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(
                    output_audio_buffers
                        .iter_mut()
                        .map(|buf| buf.as_mut_slice()),
                ),
            }]);

            let status = started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("Process failed");

            std::hint::black_box(status);
        });
    });

    // 2. Slow Path (Scalar) - forced parameter changes on every block
    let mut input_events_buffer = EventBuffer::new();
    let event_in = ParamValueEvent::new(
        0,
        ClapId::new(PARAM_INPUT_GAIN),
        Pckn::match_all(),
        0.0,
        Cookie::empty(),
    );
    let event_out = ParamValueEvent::new(
        0,
        ClapId::new(PARAM_OUTPUT_GAIN),
        Pckn::match_all(),
        0.0,
        Cookie::empty(),
    );
    input_events_buffer.push(&event_in);
    input_events_buffer.push(&event_out);
    let input_events = InputEvents::from_buffer(&input_events_buffer);

    group.bench_function("Scalar_SlowPath", |b| {
        b.iter(|| {
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            // Reconstruct the port views to keep it realistic
            let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_audio_buffers.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(
                    output_audio_buffers
                        .iter_mut()
                        .map(|buf| buf.as_mut_slice()),
                ),
            }]);

            let status = started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("Process failed");

            std::hint::black_box(status);
        });
    });

    group.finish();

    let stopped_processor = started_processor.stop_processing();
    plugin_instance.deactivate(stopped_processor);
}

#[cfg(not(feature = "clap-plugin"))]
fn bench_clap_process_block_64samp(_c: &mut criterion::Criterion) {}

// Main benchmark group definition (inference latency and DSP kernels)
criterion_group!(
    name = benches;
    // sample_size(50) is a balance between statistical accuracy and total runtime.
    config = criterion::Criterion::default().sample_size(50);
    targets = bench_wavenet_standard_process,
    bench_wavenet_standard_block_sizes,
    bench_lstm_2x16_process,
    bench_lstm_2x16_block_sizes,
    bench_lstm_1x8_comparison,
    bench_lstm_2x16_comparison,
    bench_lstm_1x40_process,
    bench_lstm_2x24_process,
    bench_lstm_1x40_comparison,
    bench_lstm_2x24_comparison,
    bench_tanh_slice_256,
    bench_tanh_pade_nr2_256,
    bench_tanh_pade_div_256,
    bench_sigmoid_slice_256,
    bench_dot_product_avx2_256,
    bench_dot_product_avx2_64,
    bench_resampler_44100_to_48000_256samp,
    bench_resampler_96000_to_48000_256samp,
    bench_resampler_48000_bypass,
    bench_record,
    bench_tanh_avx512_256elem,
    bench_tanh_pade_nr2_avx512_256elem,
    bench_tanh_pade_div_avx512_256elem,
    bench_sigmoid_avx512_256elem,
    bench_prewarm_wavenet_standard,
    bench_prewarm_lstm_2x16,
    bench_head_rechannel_fp32,
    bench_clap_process_block_64samp
);

// Long-running benchmark group definition (Soak Tests)
#[cfg(feature = "long_bench")]
criterion_group!(
    name = long_benches;
    config = Criterion::default();
    targets = bench_wavenet_long_run, bench_lstm_long_run, bench_resampler_long_run
);

// Conditional entry point depending on stress feature activation
#[cfg(not(feature = "long_bench"))]
criterion_main!(benches);

#[cfg(feature = "long_bench")]
criterion_main!(benches, long_benches);
