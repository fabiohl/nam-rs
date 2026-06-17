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
use nam_rs::math::common::half::f32_to_f16_bits;
use nam_rs::models::NamModel;
use nam_rs::models::container::ContainerModel;
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

/// Compares WaveNet Standard inference at small RT buffer sizes (1, 16, 64 samples)
/// across lo-fi (default, u16 weights + Padé tanh) and hi-fi (f32 weights + high-acc
/// tanh) modes. Run in both modes to populate T-HF4.1 of TODO-sprints.md:
///
/// ```
/// cargo bench --bench inference_bench -- "WaveNet_P10_Comparison"
/// cargo bench --bench inference_bench --features high-fidelity -- "WaveNet_P10_Comparison"
/// ```
///
/// Record throughput (elem/s) and latency (ns) for each size and mode.
fn bench_wavenet_p10_lofi_vs_hifi(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Dispatcher failed for P10 bench");
    let mut model = build_model(&model_data).expect("Dispatcher failed for P10 bench");
    model.prewarm(2048);

    // Sizes chosen to cover the full RT range: 1 (per-sample minimum), 16 (small plugin
    // buffer), 64 (common CLAP/JACK buffer). Suffix "LF" / "HF" is determined at
    // compile time by the `high-fidelity` feature flag.
    #[cfg(not(feature = "high-fidelity"))]
    let mode_tag = "LF";
    #[cfg(feature = "high-fidelity")]
    let mode_tag = "HF";

    for &size in &[1usize, 16, 64] {
        let input = generate_sine_440hz(size);
        let mut output = vec![0.0f32; size];
        c.bench_function(
            &format!("WaveNet_P10_Comparison_{}_{}", mode_tag, size),
            |b| {
                b.iter(|| model.process(&input, &mut output));
            },
        );
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

/// Measures the throughput of the AVX2 dot product with f16 (Half Precision) weights.
/// This technique reduces memory bandwidth usage by 50% and improves
/// L1 cache locality, crucial for WaveNet dense layers.
fn bench_dot_product_avx2_256(c: &mut Criterion) {
    let vec_a = AlignedVec::from_vec((0..256).map(|i| (i as f32) * 0.1).collect());
    let vec_b = AlignedVec::from_vec(
        (0..256)
            .map(|i| f32_to_f16_bits((i as f32) * -0.1))
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
            .map(|i| f32_to_f16_bits((i as f32) * -0.1))
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

// ── A2-Full (CH=8) inference benchmarks ──

/// Measures the processing time of an A2-Full (CH=8) WaveNet model.
/// A2-Full is the high-fidelity Criterion variant with 8 channels,
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
        let weights: AlignedVec<f32> = AlignedVec::new(in_size * out_size, 0.01);
        let bias: AlignedVec<f32> = AlignedVec::new(out_size, 0.0);
        let layer = DenseLayer::<16, 8> {
            weights: AlignedVec::new(0, 0u16),
            bias,
            do_bias: true,
            f32_weights: weights,
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
                    &layer.f32_weights,
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
            f32_weights: weights,
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
                    &layer.f32_weights,
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
            f32_weights: weights,
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
                    &layer.f32_weights,
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

// ── Gate FSM Benchmarks ──

/// Benchmarks the Gate FSM (`DynamicHysteresis::update()` + `multiplier()`) across
/// realistic DSP block sizes (64, 128, 256 samples). The gate is in the DSP hot-path
/// — every audio callback runs `update()` to determine whether to open/close the gate
/// based on the detected volume level.
///
/// Three steady-state scenarios are measured per block size:
/// - **Open**: Gate stays open (volume above open threshold). Most common path.
/// - **Closed**: Gate stays closed (volume below close threshold, post-hold+fade).
/// - **FadingOut**: Gate is actively ramping the multiplier down toward silence.
fn bench_gate_fsm(c: &mut Criterion) {
    use nam_rs::dsp::gate::{DynamicHysteresis, GateParams};

    let params = GateParams::default();
    let th_open = 10.0f32.powf(params.threshold_open_db / 20.0);
    let th_close = 10.0f32.powf(params.threshold_close_db / 20.0);

    let mut group = c.benchmark_group("Gate_FSM");

    for &n_samples in &[64, 128, 256] {
        // Steady Open: volume well above threshold, gate stays open.
        // Exercised every callback while the musician is actively playing.
        group.bench_function(format!("Open_{}samp", n_samples), |b| {
            let mut gate = DynamicHysteresis::new();
            b.iter(|| {
                gate.update(
                    std::hint::black_box(0.5),
                    th_open,
                    th_close,
                    &params,
                    n_samples,
                );
                std::hint::black_box(gate.multiplier());
            });
        });

        // Steady Closed: gate is already closed, volume stays below threshold.
        // Pre-condition: advance through hold + fade to reach Closed state.
        group.bench_function(format!("Closed_{}samp", n_samples), |b| {
            let mut gate = DynamicHysteresis::new();
            // Two calls with large blocks: hold_frames=2048 → FadingOut, then fade_frames ≤ 256 → Closed
            gate.update(0.0, th_open, th_close, &params, 2048);
            gate.update(0.0, th_open, th_close, &params, 256);
            b.iter(|| {
                gate.update(
                    std::hint::black_box(0.0),
                    th_open,
                    th_close,
                    &params,
                    n_samples,
                );
                std::hint::black_box(gate.multiplier());
            });
        });

        // FadingOut: gate is actively ramping down.
        // Pre-condition: advance hold_counter to just below hold_frames, then trigger FadingOut.
        group.bench_function(format!("FadingOut_{}samp", n_samples), |b| {
            b.iter_with_setup(
                || {
                    let mut gate = DynamicHysteresis::new();
                    // Advance to edge of hold expiry
                    gate.update(0.0, th_open, th_close, &params, params.hold_frames);
                    gate
                },
                |mut gate| {
                    gate.update(
                        std::hint::black_box(0.0),
                        th_open,
                        th_close,
                        &params,
                        n_samples,
                    );
                    std::hint::black_box(gate.multiplier());
                },
            );
        });
    }

    group.finish();
}

// ── LinearModel Dot Product Benchmarks ──

/// Benchmarks the LinearModel dot product kernel (AVX2/AVX-512 SIMD vs scalar).
/// With RF 256, the scalar path performs 16k FMAs per 64-sample block;
/// the SIMD path reduces this by 4-8×.
fn bench_linear_model_dot_product(c: &mut Criterion) {
    use nam_rs::models::linear::LinearModel;

    let rf = 256;
    let weights: Vec<f32> = (0..rf).map(|i| (i as f32 * 0.01).sin()).collect();
    let bias = 0.1;
    let mut model = LinearModel::new(weights, bias).unwrap();
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

// ── IR Cabsim Convolution Benchmarks ──

fn synth_ir(len: usize, freq: f32, decay: f32) -> Vec<f32> {
    let sample_rate = 48000u32;
    (0..len)
        .map(|n| {
            let t = n as f32 / sample_rate as f32;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}

fn bench_cabsim_process_block(
    c: &mut criterion::Criterion,
    ir_len: usize,
    partition_size: usize,
    label: &str,
) {
    use nam_rs::dsp::cabsim::conv::ConvEngine;

    let ir = synth_ir(ir_len, 440.0, 10.0);
    let mut engine = ConvEngine::new(&ir, partition_size);

    let mut input = vec![0.0f32; partition_size];
    let mut output = vec![0.0f32; partition_size];

    for (j, v) in input.iter_mut().enumerate() {
        *v = (j as f32 * 0.01).sin();
    }

    // Warm-up: fill FDL
    for _ in 0..engine.num_partitions().max(1) {
        engine.process(&input, &mut output);
    }

    c.bench_function(label, |b| {
        b.iter(|| {
            for (j, v) in input.iter_mut().enumerate() {
                *v = (j as f32 * 0.01).sin();
            }
            engine.process(
                std::hint::black_box(&input),
                std::hint::black_box(&mut output),
            );
        });
    });
}

fn bench_cabsim_short_ir_64samp(c: &mut criterion::Criterion) {
    bench_cabsim_process_block(c, 64, 64, "Cabsim_ShortIR_64samp");
}

fn bench_cabsim_medium_ir_64samp(c: &mut criterion::Criterion) {
    bench_cabsim_process_block(c, 2048, 64, "Cabsim_MediumIR_2048_64samp");
}

fn bench_cabsim_long_ir_64samp(c: &mut criterion::Criterion) {
    bench_cabsim_process_block(c, 16384, 64, "Cabsim_LongIR_16384_64samp");
}

fn bench_cabsim_256samp_block(c: &mut criterion::Criterion) {
    bench_cabsim_process_block(c, 2048, 256, "Cabsim_MediumIR_2048_256samp");
}

fn bench_cabsim_engine_construction(c: &mut criterion::Criterion) {
    use nam_rs::dsp::cabsim::conv::ConvEngine;

    let ir = synth_ir(2048, 440.0, 10.0);

    c.bench_function("Cabsim_Engine_Construction_2048_64", |b| {
        b.iter(|| {
            let engine = ConvEngine::new(&ir, 64);
            std::hint::black_box(engine);
        });
    });
}

fn bench_cabsim_engine_construction_long(c: &mut criterion::Criterion) {
    use nam_rs::dsp::cabsim::conv::ConvEngine;

    let ir = synth_ir(16384, 440.0, 10.0);

    c.bench_function("Cabsim_Engine_Construction_16384_64", |b| {
        b.iter(|| {
            let engine = ConvEngine::new(&ir, 64);
            std::hint::black_box(engine);
        });
    });
}

// cabsim_long_run → moved to benches/long_inference_bench.rs (T2.2.3)

// Main benchmark group definition (inference latency and DSP kernels)
criterion_group!(
    name = benches;
    // sample_size(50) is a balance between statistical accuracy and total runtime.
    config = criterion::Criterion::default().sample_size(50);
    targets = bench_wavenet_standard_process,
    bench_wavenet_p10_lofi_vs_hifi,
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
    bench_a2_full_process,
    bench_a2_full_block_sizes,
    bench_a2_lite_process,
    bench_a2_lite_block_sizes,
    bench_prewarm_wavenet_standard,
    bench_prewarm_lstm_2x16,
    bench_prewarm_a2_full,
    bench_prewarm_a2_lite,
    bench_head_rechannel_fp32,
    bench_clap_process_block_64samp,
    bench_cabsim_short_ir_64samp,
    bench_cabsim_medium_ir_64samp,
    bench_cabsim_long_ir_64samp,
    bench_cabsim_256samp_block,
    bench_cabsim_engine_construction,
    bench_cabsim_engine_construction_long,
    bench_gate_fsm,
    bench_linear_model_dot_product,
    bench_container_crossfade_64samp,
    bench_nondist_models
);

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
