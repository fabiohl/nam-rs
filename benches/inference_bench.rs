// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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

/// Gera sinal senoidal determinístico de 440 Hz a uma taxa de amostragem de 48 kHz.
/// Utilizado como entrada estável para garantir que a carga de processamento seja
/// consistente entre as iterações do benchmark, evitando flutuações por sinais aleatórios.
fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

/// Cria uma estrutura `NamModelData` sintética configurada como uma rede LSTM.
/// Permite testar diferentes topologias (camadas e tamanho oculto) sem depender
/// de arquivos externos, facilitando a validação de performance bruta do motor de inferência.
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
        // Pesos inicializados com valor pequeno (0.01) para evitar saturação/infinitos
        // prematuros durante execuções repetitivas de benchmark.
        weights: vec![0.01; total_weights],
        weights_layout: nam_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Mede o tempo de processamento de um modelo WaveNet real ("Standard").
/// Este benchmark é o mais representativo para o uso comum em guitarras,
/// testando a eficácia das convoluções dilatadas e kernels SIMD otimizados.
fn bench_wavenet_standard_process(c: &mut Criterion) {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-standard.nam");

    // Ignora silenciosamente se o modelo de fixture não estiver presente
    if !path.exists() {
        return;
    }

    let json_data = std::fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    // O dispatcher escolhe a implementação mais rápida (estática vs dinâmica)
    let mut model = build_model(&model_data).expect("Dispatcher falhou para benchmark");

    // O prewarm é CRÍTICO para preencher buffers de estado e evitar que a alocação
    // inicial de recursos dentro do modelo influencie na média do benchmark.
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("WaveNet_Standard_CH16_64samp_48kHz", |b| {
        b.iter(|| {
            // Executa a inferência de um bloco completo (64 amostras)
            model.process(&input, &mut output);
        });
    });
}

/// Mede o tempo de processamento de uma rede recorrente LSTM 2x16.
/// As LSTMs são conhecidas por sua alta carga computacional sequencial,
/// sendo o teste de estresse ideal para verificar a latência de loops de feedback.
fn bench_lstm_2x16_process(c: &mut Criterion) {
    let data = make_lstm_data(2, 16, 3345);
    let mut model = build_model(&data).expect("Dispatcher falhou para LSTM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_2x16_64samp_48kHz", |b| {
        b.iter(|| {
            // Processamento LSTM tende a ser mais pesado que WaveNet para blocos pequenos
            model.process(&input, &mut output);
        });
    });
}

/// Compara a performance entre a implementação Escalar (baseline) e SIMD para LSTM 1x8.
/// Este benchmark valida o ganho de performance obtido com o "T3: Fused Gates",
/// onde as 4 portas da LSTM são calculadas simultaneamente em registradores AVX2.
fn bench_lstm_1x8_comparison(c: &mut Criterion) {
    let data = make_lstm_data(1, 8, 345);
    let mut model_simd = build_model(&data).expect("Dispatcher falhou para LSTM 1x8 benchmark");
    let mut model_scalar = build_model(&data).expect("Dispatcher falhou para LSTM 1x8 benchmark");
    model_simd.prewarm(1024);
    model_scalar.prewarm(1024);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut group = c.benchmark_group("LSTM_1x8_Comparison");

    // Caminho otimizado (SIMD / Auto-vectorized)
    group.bench_function("SIMD_Fused_T3", |b| {
        b.iter(|| {
            model_simd.process(&input, &mut output);
        });
    });

    // Caminho escalar explícito para medir o "speedup" teórico
    #[cfg(any(test, feature = "long_bench"))]
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

    #[cfg(any(test, feature = "long_bench"))]
    group.bench_function("Scalar_Baseline", |b| match &mut *model_scalar {
        nam_rs::models::DynamicModel::Lstm2x16(m) => {
            b.iter(|| m.process_scalar(&input, &mut output));
        }
        _ => panic!("Modelo não é Lstm2x16"),
    });
    group.finish();
}

/// Mede a performance do kernel de ativação `tanh` otimizado para AVX2.
/// Este kernel utiliza aproximações de Padé e instruções rsqrt para maximizar
/// o throughput em detrimento de uma precisão sub-amostral irrelevante para áudio.
fn bench_tanh_slice_256(c: &mut Criterion) {
    // Range de entrada cobrindo a parte linear e de saturação da tanh
    let base: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05) - 6.4).collect();

    c.bench_function("FastMath_tanh_AVX2_256elem", |b| {
        let mut buf = base.clone();
        b.iter(|| {
            // Copiamos os dados originais para garantir que o kernel processe
            // sempre os mesmos valores, simulando a carga real de uma camada neural.
            buf.copy_from_slice(&base);
            unsafe { nam_rs::math::activations::tanh_slice_avx2(&mut buf) };
        });
    });
}

/// Mede a performance do kernel de ativação `sigmoid` otimizado para AVX2.
/// Essencial para modelos LSTM, este kernel converte a tanh aproximada em uma
/// função logística (0 a 1) para controlar as portas de memória.
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

/// Mede o overhead da implementação "Dynamic" da WaveNet.
/// Enquanto a versão standard usa const generics para tamanhos fixos, a dynamic
/// permite carregar qualquer configuração de camadas em tempo de execução,
/// servindo como fallback para modelos não-padrão.
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
            // Espera-se uma latência ligeiramente superior à versão estática
            model.process(&input, &mut output);
        });
    });
}

/// Mede a performance da LSTM Dinâmica 1x16.
/// Útil para validar o motor em cenários de treinamento customizado com
/// dimensões de hidden state fora dos padrões 8, 16 ou 32.
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

/// Avalia como o WaveNet escala com diferentes tamanhos de buffer DSP.
/// Buffers maiores permitem melhor aproveitamento do cache e prefetching,
/// mas aumentam a latência total percebida pelo músico.
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

    // Testa buffers de 32 a 512 amostras
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

/// Avalia o escalonamento da LSTM com diferentes tamanhos de buffer.
/// Diferente da WaveNet, a LSTM é puramente sequencial (amostra por amostra),
/// então o overhead por amostra tende a ser mais constante, independente do bloco.
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

/// Mede o throughput do produto escalar AVX2 com pesos em f16 (Half Precision).
/// Esta técnica reduz o uso de largura de banda de memória em 50% e melhora
/// a localidade do cache L1, crucial para as camadas densas da WaveNet.
fn bench_dot_product_avx2_256(c: &mut Criterion) {
    let vec_a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1).collect();
    let vec_b: Vec<u16> = (0..256)
        .map(|i| half::f16::from_f32((i as f32) * -0.1).to_bits())
        .collect();

    c.bench_function("DotProduct_AVX2_256elem", |b| {
        b.iter(|| unsafe {
            // O uso de black_box evita que o compilador otimize o loop inteiro,
            // garantindo que o cálculo matemático seja realmente executado.
            nam_rs::math::gemm::dot::dot_product_avx2(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

/// Versão do produto escalar para vetores pequenos (64 elementos).
/// Representa o tamanho típico das camadas intermediárias em modelos leves.
fn bench_dot_product_avx2_64(c: &mut Criterion) {
    let vec_a: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    let vec_b: Vec<u16> = (0..64)
        .map(|i| half::f16::from_f32((i as f32) * -0.1).to_bits())
        .collect();
    c.bench_function("DotProduct_AVX2_64elem", |b| {
        b.iter(|| unsafe {
            nam_rs::math::gemm::dot::dot_product_avx2(
                std::hint::black_box(&vec_a),
                std::hint::black_box(&vec_b),
            )
        });
    });
}

/// Mede o custo do resampler ao converter de 44.1 kHz para 48 kHz.
/// O resampler é um dos componentes mais sensíveis, pois envolve filtragem FIR.
/// `process_input` e `process_output` são medidos separadamente para identificar
/// gargalos na entrada (bufferização) vs saída (interpolação).
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

/// Mede a conversão de 96 kHz para 48 kHz (downsampling).
/// Geralmente mais leve que o upsampling, mas ainda exige filtragem anti-aliasing.
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

/// Mede o overhead do resampler quando as taxas de amostragem são iguais.
/// Serve para validar se o caminho de "bypass" é eficiente.
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

/// Benchmarks para processadores que suportam AVX-512 (ex: AMD Zen 4, Intel Ice Lake+).
/// O AVX-512 permite processar 16 floats simultaneamente (512 bits), teoricamente
/// dobrando o throughput em relação ao AVX2.
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

/// Mede o tempo gasto na função `prewarm`.
/// Embora o prewarm ocorra fora da thread de áudio, ele deve ser rápido o suficiente
/// para que a troca de modelos durante uma performance ao vivo seja imperceptível.
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

// --- Long Benchmarks (Soak Testing) ---
// Estes benchmarks só são executados se a feature "long_bench" estiver ativa.
// Destinam-se a validar a estabilidade térmica da CPU e detectar variações
// de performance ao longo do tempo (jitters, throttling).

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
    // Execução prolongada (30 segundos) para garantir convergência estatística
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
            // Validar se o resampler mantém estabilidade e não acumula erros de fase
            // ou latência variável durante longos períodos.
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

// Definição do grupo principal de benchmarks (latência de inferência e kernels DSP)
criterion_group!(
    name = benches;
    // sample_size(50) é um equilíbrio entre precisão estatística e tempo de execução total.
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

// Definição do grupo de benchmarks de longa duração (Soak Tests)
#[cfg(feature = "long_bench")]
criterion_group!(
    name = long_benches;
    config = Criterion::default();
    targets = bench_wavenet_long_run, bench_lstm_long_run, bench_resampler_long_run
);

// Ponto de entrada condicional dependendo da ativação de features de estresse
#[cfg(not(feature = "long_bench"))]
criterion_main!(benches);

#[cfg(feature = "long_bench")]
criterion_main!(benches, long_benches);
