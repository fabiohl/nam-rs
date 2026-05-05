// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Benchmarks formais de latência de inferência para o motor NAM-rs.
//!
//! Mede o tempo de processamento de 1 bloco DSP (64 amostras a 48 kHz = deadline
//! de 1.33 ms) para redes neurais WaveNet e LSTM, além dos kernels FastMath
//! que compõem as funções de ativação SIMD.
//!
//! ## Benchmarks disponíveis
//!
//! | ID                                      | Descrição                               | Contexto prático                         |
//! | --------------------------------------- | --------------------------------------- | ---------------------------------------- |
//! | `WaveNet_Standard_CH16_64samp_48kHz`    | Inferência WaveNet Standard completa    | Modelo ~284 KB, 10+10 layers dilatadas   |
//! | `LSTM_2x16_64samp_48kHz`                | Inferência LSTM 2 camadas × 16 hidden   | Rede recorrente mais pesada suportada    |
//! | `FastMath_tanh_AVX2_256elem`            | Ativação tanh Padé×rsqrt sobre 256 f32  | Kernel chamado N×layers/bloco no WaveNet |
//! | `FastMath_sigmoid_AVX2_256elem`         | Ativação sigmoid derivada de tanh       | Kernel chamado N×gates/bloco no LSTM     |
//! | `WaveNet_Dynamic_Standard_64samp_48kHz` | Inferência WaveNet Dynamic (fallback)   | Mede overhead do path sem const generics |
//! | `LSTM_Dynamic_1x16_64samp_48kHz`        | Inferência LSTM Dynamic 1×16 (fallback) | Mede overhead do path sem const generics |
//!
//! ## Interpretação dos resultados
//!
//! - O deadline de tempo-real a 48 kHz com buffer de 64 amostras é **1.33 ms**.
//! - Se qualquer benchmark de inferência exceder este deadline, o engine causará
//!   xruns (buffer underruns) em produção com esse tamanho de buffer.
//! - Os kernels FastMath são sub-componentes chamados centenas de vezes por bloco;
//!   seu tempo total contribui para a latência da inferência completa.
//!
//! ## Execução
//!
//! ```sh
//! cargo bench --bench inference_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{NamConfig, NamModelData, parse_nam_json};
use nam_rs::models::NamModel;

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
        weights_layout: nam_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Benchmark: WaveNet Standard (modelo real BossWN-standard.nam).
fn bench_wavenet_standard_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");

    if !path.exists() {
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model = build_model(&model_data).expect("Dispatcher falhou para benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("WaveNet_Standard_CH16_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Benchmark: LSTM 2×16 (sintético, 3345 pesos).
fn bench_lstm_2x16_process(c: &mut Criterion) {
    let data = make_lstm_data(2, 16, 3345);
    let mut model = build_model(&data).expect("Dispatcher falhou para LSTM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_2x16_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Benchmark Comparativo (T15): LSTM 1x8 Escalar vs SIMD (Gates Fundidos T3).
fn bench_lstm_1x8_comparison(c: &mut Criterion) {
    let data = make_lstm_data(1, 8, 345);
    let mut model_simd = build_model(&data).expect("Dispatcher falhou para LSTM 1x8 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher falhou para LSTM 1x8 benchmark");
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

    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::DynamicModel::Lstm1x8(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Modelo não é Lstm1x8"),
    });
    group.finish();
}

/// Benchmark Comparativo (T15): LSTM 2x16 Escalar vs SIMD (Gates Fundidos T3).
fn bench_lstm_2x16_comparison(c: &mut Criterion) {
    let data = make_lstm_data(2, 16, 3345);
    let mut model_simd = build_model(&data).expect("Dispatcher falhou para LSTM 2x16 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher falhou para LSTM 2x16 benchmark");
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

    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::DynamicModel::Lstm2x16(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Modelo não é Lstm2x16"),
    });
    group.finish();
}

/// Benchmark: kernel FastMath `tanh_slice_avx2` sobre 256 elementos f32.
fn bench_tanh_slice_256(c: &mut Criterion) {
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

    c.bench_function("FastMath_tanh_AVX2_256elem", |b| {
        let mut buf = base.clone();
        b.iter(|| {
            buf.copy_from_slice(&base);
            unsafe { nam_rs::math::fastmath::tanh_slice_avx2(&mut buf) };
        });
    });
}

/// Benchmark: kernel FastMath `sigmoid_slice_avx2` sobre 256 elementos f32.
fn bench_sigmoid_slice_256(c: &mut Criterion) {
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

    c.bench_function("FastMath_sigmoid_AVX2_256elem", |b| {
        let mut buf = base.clone();
        b.iter(|| {
            buf.copy_from_slice(&base);
            unsafe { nam_rs::math::fastmath::sigmoid_slice_avx2(&mut buf) };
        });
    });
}

/// Benchmark: WaveNet Dynamic Standard.
fn bench_wavenet_dynamic_standard(c: &mut Criterion) {
    use nam_rs::loader::dispatcher::build_wavenet_dynamic;
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model = build_wavenet_dynamic(&model_data).expect("Builder dinâmico falhou");
    model.prewarm(2048);
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    c.bench_function("WaveNet_Dynamic_Standard_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Benchmark: LSTM Dynamic 1×16.
fn bench_lstm_dynamic_1x16(c: &mut Criterion) {
    use nam_rs::loader::dispatcher::build_lstm_dynamic;
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossLSTM-1x16.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo LSTM");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model = build_lstm_dynamic(&model_data, 1, 16).expect("Builder dinâmico falhou");
    model.prewarm(2048);
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    c.bench_function("LSTM_Dynamic_1x16_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

fn bench_wavenet_standard_block_sizes(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model = build_model(&model_data).expect("Dispatcher falhou");
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

fn bench_lstm_2x16_block_sizes(c: &mut Criterion) {
    let data = make_lstm_data(2, 16, 3345);
    let mut model = build_model(&data).expect("Dispatcher falhou");
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

fn bench_dot_product_avx2_256(c: &mut Criterion) {
    let vec_a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
    let vec_b: Vec<u16> = (0..256)
        .map(|i| half::f16::from_f32((i as f32) * -0.1).to_bits())
        .collect();
    c.bench_function("DotProduct_AVX2_256elem", |b| {
        b.iter(|| unsafe {
            nam_rs::math::simd::dot_product_avx2(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

fn bench_dot_product_avx2_64(c: &mut Criterion) {
    let vec_a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let vec_b: Vec<u16> = (0..64)
        .map(|i| half::f16::from_f32((i as f32) * -0.1).to_bits())
        .collect();
    c.bench_function("DotProduct_AVX2_64elem", |b| {
        b.iter(|| unsafe {
            nam_rs::math::simd::dot_product_avx2(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

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
    group.bench_function("process_output", |b| {
        b.iter(|| {
            rs.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

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
    group.bench_function("process_output", |b| {
        b.iter(|| {
            rs.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

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

fn bench_tanh_avx512_256elem(c: &mut Criterion) {
    if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
        let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();
        c.bench_function("FastMath_tanh_AVX512_256elem", |b| {
            let mut buf = base.clone();
            b.iter(|| {
                buf.copy_from_slice(&base);
                unsafe { nam_rs::math::fastmath::tanh_slice_avx512(&mut buf) };
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
                unsafe { nam_rs::math::fastmath::sigmoid_slice_avx512(&mut buf) };
            });
        });
    }
}

fn bench_prewarm_wavenet_standard(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    c.bench_function("Prewarm_WaveNet_Standard_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&model_data).expect("Dispatcher falhou"),
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
            || build_model(&data).expect("Dispatcher falhou"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
}

// --- Long Benchmarks ---

#[cfg(feature = "long_bench")]
fn bench_wavenet_long_run(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");
    if !path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model = build_model(&model_data).expect("Dispatcher falhou");
    model.prewarm(4096);
    let size = 4096;
    let input = generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_WaveNet");
    group.measurement_time(std::time::Duration::from_secs(30));
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
    let mut model = build_model(&data).expect("Dispatcher falhou");
    model.prewarm(4096);
    let size = 4096;
    let input = generate_sine_440hz(size);
    let mut output = vec![0.0f32; size];
    let mut group = c.benchmark_group("Long_Run_LSTM");
    group.measurement_time(std::time::Duration::from_secs(30));
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
    group.measurement_time(std::time::Duration::from_secs(30));
    group.sample_size(100);
    group.bench_function("Long_Resampler_44100_to_48000_4096samp", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_wavenet_standard_process,
    bench_wavenet_standard_block_sizes,
    bench_lstm_2x16_process,
    bench_lstm_2x16_block_sizes,
    bench_lstm_1x8_comparison,
    bench_lstm_2x16_comparison,
    bench_tanh_slice_256,
    bench_sigmoid_slice_256,
    bench_wavenet_dynamic_standard,
    bench_lstm_dynamic_1x16,
    bench_dot_product_avx2_256,
    bench_dot_product_avx2_64,
    bench_resampler_44100_to_48000_256samp,
    bench_resampler_96000_to_48000_256samp,
    bench_resampler_48000_bypass,
    bench_tanh_avx512_256elem,
    bench_sigmoid_avx512_256elem,
    bench_prewarm_wavenet_standard,
    bench_prewarm_lstm_2x16
);

#[cfg(feature = "long_bench")]
criterion_group!(
    name = long_benches;
    config = Criterion::default();
    targets = bench_wavenet_long_run, bench_lstm_long_run, bench_resampler_long_run
);

#[cfg(not(feature = "long_bench"))]
criterion_main!(benches);

#[cfg(feature = "long_bench")]
criterion_main!(benches, long_benches);
