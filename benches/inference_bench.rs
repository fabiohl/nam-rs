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
///
/// Mede a latência de `process()` para 64 amostras (1 bloco DSP a 48 kHz).
/// O deadline de tempo-real para este buffer é 1.33 ms — se o benchmark
/// exceder esse valor, o engine causará xruns em produção.
fn bench_wavenet_standard_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");

    if !path.exists() {
        eprintln!(
            "SKIP bench: BossWN-standard.nam não encontrado em {path:?}. \
             Copie modelos para tests/fixtures/models/ (veja TODO.txt)."
        );
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
///
/// Mede a latência de `process()` para 64 amostras (1 bloco DSP a 48 kHz).
/// A rede LSTM 2×16 é a topologia recorrente mais pesada suportada pelo NAM-rs.
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

    group.bench_function("Scalar_Baseline", |b| {
        // Usamos downcast interno ou chamamos diretamente o método scalar se disponível via trait?
        // Como o DynamicModel não expõe process_scalar, vamos usar o tipo concreto se possível
        // ou adicionar process_scalar ao trait NamModel (menos limpo).
        // Melhor: adicionar process_scalar ao DynamicModel.
        match &mut *model_scalar {
            nam_rs::models::DynamicModel::Lstm1x8(m) => {
                b.iter(|| m.process_scalar(&input, &mut output));
            }
            _ => panic!("Modelo não é Lstm1x8"),
        }
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
///
/// Este kernel é chamado em cada layer×bloco do WaveNet e do LSTM para computar
/// a função de ativação tanh via polinômio Padé grau 5 + rsqrt_ps Newton-Raphson.
/// Processar 256 floats por invocação é representativo do workload interno.
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
///
/// O sigmoid é derivado via identidade `0.5*(1+tanh(0.5*x))` e é usado
/// nas portas i/f/o do LSTM. Processar 256 floats é representativo.
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

/// Benchmark: WaveNet Dynamic Standard (mesmo modelo BossWN-standard.nam via path dinâmico).
///
/// Constrói o modelo usando `build_wavenet_dynamic()` em vez do path estático
/// const-generic. Mede o overhead absoluto do fallback dinâmico que não possui
/// loop unrolling via const generics.
///
/// **Overhead esperado:** ≤ 50% mais lento que o estático para WaveNet,
/// pois as convoluções 1D dinâmicas não beneficiam de unrolling do compilador.
fn bench_wavenet_dynamic_standard(c: &mut Criterion) {
    use nam_rs::loader::dispatcher::build_wavenet_dynamic;

    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");

    if !path.exists() {
        eprintln!(
            "SKIP bench: BossWN-standard.nam não encontrado em {path:?}. \
             Copie modelos para tests/fixtures/models/."
        );
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    // Forçar construção pelo caminho dinâmico (sem const generics)
    let mut model =
        build_wavenet_dynamic(&model_data).expect("Builder dinâmico falhou para benchmark WaveNet");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("WaveNet_Dynamic_Standard_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Benchmark: LSTM Dynamic 1×16 (mesmo modelo BossLSTM-1x16.nam via path dinâmico).
///
/// Constrói o modelo usando `build_lstm_dynamic()` em vez do path estático
/// const-generic. Mede o overhead absoluto do fallback dinâmico.
///
/// **Overhead esperado:** ≤ 30% mais lento que o estático para LSTM,
/// pois o LSTM dinâmico realiza os mesmos dot products AVX2 mas sem
/// unrolling de loops proporcionado por const generics.
fn bench_lstm_dynamic_1x16(c: &mut Criterion) {
    use nam_rs::loader::dispatcher::build_lstm_dynamic;

    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossLSTM-1x16.nam");

    if !path.exists() {
        eprintln!(
            "SKIP bench: BossLSTM-1x16.nam não encontrado em {path:?}. \
             Copie modelos para tests/fixtures/models/."
        );
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo LSTM");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    // Forçar construção pelo caminho dinâmico (sem const generics)
    let mut model = build_lstm_dynamic(&model_data, 1, 16)
        .expect("Builder dinâmico falhou para benchmark LSTM");
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
        eprintln!(
            "SKIP bench: BossWN-standard.nam não encontrado em {:?}. \
             Copie modelos para tests/fixtures/models/ (veja TODO.txt).",
            path
        );
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model = build_model(&model_data).expect("Dispatcher falhou para benchmark");
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

/// Benchmark: LSTM 2×16 (sintético, 3345 pesos) com buffers variáveis.
fn bench_lstm_2x16_block_sizes(c: &mut Criterion) {
    let data = make_lstm_data(2, 16, 3345);
    let mut model = build_model(&data).expect("Dispatcher falhou para LSTM benchmark");
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

fn bench_resampler_44100_to_48000(c: &mut Criterion) {
    use nam_rs::dsp::resampler::NamResampler;
    let chunk = 1024;
    let mut rs = NamResampler::new(44_100, 48_000, chunk).expect("Failed to create NamResampler");
    let in_l = vec![0.0f32; chunk];
    let in_r = vec![0.0f32; chunk];
    let mut out_l = vec![0.0f32; chunk * 2];
    let mut out_r = vec![0.0f32; chunk * 2];

    c.bench_function("Resampler_44100_to_48k_1024samp", |b| {
        b.iter(|| {
            rs.process_input(
                std::hint::black_box(&in_l),
                std::hint::black_box(&in_r),
                std::hint::black_box(&mut out_l),
                std::hint::black_box(&mut out_r),
            );
        });
    });
}

fn bench_resampler_96000_to_48000(c: &mut Criterion) {
    use nam_rs::dsp::resampler::NamResampler;
    let chunk = 1024;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("Failed to create NamResampler");
    let in_l = vec![0.0f32; chunk];
    let in_r = vec![0.0f32; chunk];
    let mut out_l = vec![0.0f32; chunk * 2];
    let mut out_r = vec![0.0f32; chunk * 2];

    c.bench_function("Resampler_96000_to_48k_1024samp", |b| {
        b.iter(|| {
            rs.process_input(
                std::hint::black_box(&in_l),
                std::hint::black_box(&in_r),
                std::hint::black_box(&mut out_l),
                std::hint::black_box(&mut out_r),
            );
        });
    });
}

fn bench_resampler_48000_bypass(c: &mut Criterion) {
    use nam_rs::dsp::resampler::NamResampler;
    let chunk = 1024;
    let mut rs = NamResampler::new(48_000, 48_000, chunk).expect("Failed to create NamResampler");
    let in_l = vec![0.0f32; chunk];
    let in_r = vec![0.0f32; chunk];
    let mut out_l = vec![0.0f32; chunk * 2];
    let mut out_r = vec![0.0f32; chunk * 2];

    c.bench_function("Resampler_48000_bypass_1024samp", |b| {
        b.iter(|| {
            rs.process_input(
                std::hint::black_box(&in_l),
                std::hint::black_box(&in_r),
                std::hint::black_box(&mut out_l),
                std::hint::black_box(&mut out_r),
            );
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
    } else {
        eprintln!(
            "SKIP bench: Hardware does not support AVX-512 (avx512f/avx512vl) for FastMath_tanh_AVX512_256elem"
        );
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
    } else {
        eprintln!(
            "SKIP bench: Hardware does not support AVX-512 (avx512f/avx512vl) for FastMath_sigmoid_AVX512_256elem"
        );
    }
}

fn bench_prewarm_wavenet_standard(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");

    if !path.exists() {
        eprintln!("SKIP bench: BossWN-standard.nam não encontrado em {path:?}.");
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    c.bench_function("Prewarm_WaveNet_Standard_2048samp", |b| {
        b.iter_with_setup(
            || build_model(&model_data).expect("Dispatcher falhou para benchmark"),
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
            || build_model(&data).expect("Dispatcher falhou para benchmark"),
            |mut model| {
                model.prewarm(std::hint::black_box(2048));
            },
        );
    });
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
    bench_resampler_44100_to_48000,
    bench_resampler_96000_to_48000,
    bench_resampler_48000_bypass,
    bench_tanh_avx512_256elem,
    bench_sigmoid_avx512_256elem,
    bench_prewarm_wavenet_standard,
    bench_prewarm_lstm_2x16
);
criterion_main!(benches);
