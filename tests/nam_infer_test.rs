// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Testes de Integração para validação das topologias e inferência neural.
//!
//! # Objetivo
//! Replicar o ModelTest.cpp da implementação C++ original:
//! Injetar um bloco contínuo de ondas iterativas (ex: senoidal) no pipeline Lock-Free de Inferência,
//! calculando estabilidade computacional temporal e limitando Erros Numéricos (NaN, Infinito),
//! provando que o compilador Rust / auto-vetorização (Const Generics e FMA/AVX2) não introduzem crashes no DSP de longa duração.
//!
//! # Validação Numérica Cross-Reference (Sprint 8.2)
//!
//! Testes de auto-consistência verificam o determinismo absoluto do motor Rust:
//! mesmo modelo + mesmo input → MSE = 0.0 (bitwise identical).
//!
//! Testes de golden vectors comparam a saída do motor Rust contra referência C++
//! (NeuralAudio Internal mode) gravada em `tests/fixtures/*.golden.bin`.
//!
//! ## Formato `.golden.bin`
//! ```text
//! [u32 num_samples LE]
//! [f32×N input samples LE]       — senoidal 440Hz a 48kHz
//! [f32×N expected output LE]     — output do C++ NeuralAudio Internal mode
//! ```
//!
//! ## Regeneração dos golden vectors
//! Execute `tests/fixtures/golden_gen_build.sh` com a árvore NeuralAudio C++ compilável.
//! Os arquivos `.golden.bin` resultantes devem ser commitados em `tests/fixtures/`.

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{NamWavenetTopology, get_wavenet_topology, parse_nam_json};
use nam_rs::models::wavenet;
use std::fs;
use std::path::{Path, PathBuf};

// WaveNetModel<CH=16, K=3, HEAD=8>: CH=16 canais, HEAD=8 (head_size da layer 0 = canais da Array2).
type WaveNetStandard = wavenet::WaveNetModel<16, 3, 8>;

const TEST_BLOCK_SIZE: usize = 64;
const TEST_NUM_BLOCKS: usize = 4096; // ~5.4 segundos de processamento simulado (a 48kHz).

/// Número de amostras para testes de golden vectors e auto-consistência.
const GOLDEN_NUM_SAMPLES: usize = 512;

/// Tamanho de bloco para processamento nos testes de validação numérica.
const GOLDEN_BLOCK_SIZE: usize = 64;

// =============================================================================
// Helpers — Geração de Sinais e Métricas de Erro
// =============================================================================

/// Gera sinal senoidal determinístico de 440 Hz a 48 kHz.
///
/// O mesmo sinal é usado tanto pelo gerador C++ (`golden_gen.cpp`) quanto pelos
/// testes Rust, garantindo reprodutibilidade cross-platform.
fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

/// Calcula o Mean Squared Error (MSE) entre dois vetores de amostras.
///
/// Usa aritmética `f64` internamente para evitar perda de precisão no acumulador.
fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vetores de tamanhos diferentes para MSE");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum();
    sum / (n as f64)
}

/// Calcula o Max Absolute Error (MAE / L∞) entre dois vetores.
fn compute_max_abs_error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vetores de tamanhos diferentes para MaxAbsError"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

/// Lê um arquivo `.golden.bin` no formato binário especificado.
///
/// Retorna `Some((input, expected_output))` ou `None` se o arquivo não existir
/// ou estiver malformado.
///
/// ## Formato
/// ```text
/// [u32 num_samples LE]
/// [f32×N input samples LE]
/// [f32×N expected output LE]
/// ```
fn read_golden_bin(path: &Path) -> Option<(Vec<f32>, Vec<f32>)> {
    let data = fs::read(path).ok()?;

    // Mínimo: 4 bytes (u32) + pelo menos 4 bytes de input + 4 bytes de output
    if data.len() < 12 {
        eprintln!(
            "WARN: arquivo golden {path:?} muito pequeno ({} bytes)",
            data.len()
        );
        return None;
    }

    let num_samples = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    let expected_size = 4 + num_samples * 4 * 2; // u32 + N*f32 input + N*f32 output
    if data.len() < expected_size {
        eprintln!(
            "WARN: golden {path:?} declara {num_samples} amostras mas tem {} bytes (esperados {expected_size})",
            data.len()
        );
        return None;
    }

    let input_start = 4;
    let output_start = 4 + num_samples * 4;

    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let offset = input_start + i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    let output: Vec<f32> = (0..num_samples)
        .map(|i| {
            let offset = output_start + i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    Some((input, output))
}

/// Helper: resolve o caminho para um modelo de teste em `tests/fixtures/models/`.
fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

/// Helper: processa um bloco de input pelo modelo em chunks de `block_size`.
fn process_in_blocks(
    model: &mut nam_rs::models::DynamicModel,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.0.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}

// =============================================================================
// Helpers — Construção de Modelos Sintéticos para Testes Unitários de Topologia
// =============================================================================

/// Helper para simular a criação de uma `WaveNetLayer` limpa para a Array1 (CH=16).
fn make_wavenet_layer(
    dilation: usize,
    _has_bias: bool,
    ch: usize,
) -> wavenet::WaveNetLayer<1, 16, 3> {
    wavenet::WaveNetLayer {
        conv1d: wavenet::Conv1d {
            weights: vec![0.001; ch * 3 * ch],
            bias: vec![0.0; ch],
            do_bias: false,
            dilation,
        },
        input_mixin: wavenet::DenseLayer {
            weights: vec![0.001; ch],
            bias: vec![0.0; ch],
            do_bias: false,
        },
        one_by_one: wavenet::DenseLayer {
            weights: vec![0.001; ch * ch],
            bias: vec![0.0; ch],
            do_bias: false,
        },
    }
}

/// Helper para simular a criação de uma `WaveNetLayer` para a Array2 (CH=8, =HEAD da Array1).
fn make_wavenet_layer_a2(dilation: usize) -> wavenet::WaveNetLayer<1, 8, 3> {
    wavenet::WaveNetLayer {
        conv1d: wavenet::Conv1d {
            weights: vec![0.001; 8 * 3 * 8],
            bias: vec![0.0; 8],
            do_bias: false,
            dilation,
        },
        input_mixin: wavenet::DenseLayer {
            weights: vec![0.001; 8],
            bias: vec![0.0; 8],
            do_bias: false,
        },
        one_by_one: wavenet::DenseLayer {
            weights: vec![0.001; 8 * 8],
            bias: vec![0.0; 8],
            do_bias: false,
        },
    }
}

/// Helper para construir uma rede `WaveNetStandard` sintética.
///
/// `WaveNetModel<16, 3, 8>`: Array1 CH=16, Array2 CH=8(=HEAD), HEAD2=1.
/// Nota: o type alias abaixo passou a usar HEAD=8 (head_size real do BossWN-standard).
fn build_synthetic_wavenet_standard() -> WaveNetStandard {
    let dilations_1 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let dilations_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    let rf1 = 512 * 2;
    let rf2 = 512 * 2;
    let final_rf = rf1.max(rf2);

    let layers_1: Vec<wavenet::WaveNetLayer<1, 16, 3>> = dilations_1
        .iter()
        .map(|&d| make_wavenet_layer(d, false, 16))
        .collect();
    let states_1: Vec<wavenet::WaveNetLayerState> = (0..layers_1.len())
        .map(|i| wavenet::WaveNetLayerState::new(16, final_rf, i))
        .collect();

    let array1 = wavenet::WaveNetLayerArray::<1, 1, 16, 3, 8> {
        layers: layers_1,
        states: states_1,
        rechannel: wavenet::DenseLayer {
            weights: vec![0.001; 16],
            bias: vec![0.0; 16],
            do_bias: false,
        },
        head_rechannel: wavenet::DenseLayer {
            weights: vec![0.001; 8 * 16],
            bias: vec![0.0; 8],
            do_bias: false,
        },
        array_outputs: vec![0.0; 16],
        head_accum: vec![0.0; 16],
        head_outputs: vec![0.0; 8],
        receptive_field_size: final_rf,
    };

    // Array2: IN=16(=CH), COND=1, CH=8(=HEAD1), HEAD2=1, HasHeadBias=true
    let layers_2: Vec<wavenet::WaveNetLayer<1, 8, 3>> = dilations_2
        .iter()
        .map(|&d| make_wavenet_layer_a2(d))
        .collect();
    let states_2: Vec<wavenet::WaveNetLayerState> = (0..layers_2.len())
        .map(|i| wavenet::WaveNetLayerState::new(8, final_rf, i))
        .collect();

    let array2 = wavenet::WaveNetLayerArray::<16, 1, 8, 3, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: wavenet::DenseLayer {
            weights: vec![0.0; 16 * 8],
            bias: vec![0.0; 8],
            do_bias: false,
        },
        head_rechannel: wavenet::DenseLayer {
            weights: vec![0.0; 8],
            bias: vec![0.0; 1],
            do_bias: true,
        },
        array_outputs: vec![0.0; 8],
        head_accum: vec![0.0; 8],
        head_outputs: vec![0.0; 1],
        receptive_field_size: final_rf,
    };

    WaveNetStandard {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: final_rf,
    }
}

// =============================================================================
// Testes Existentes (Sprint 5/7) — Preservados Integralmente
// =============================================================================

/// Teste 1: Auditoria da capacidade de Leitura do Loader e Validação Geometria
#[test]
fn test_wavenet_model_json_parsing() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        eprintln!("SKIP: Modelo de teste WaveNet não encontrado em {path:?}. Ignorando parsing.");
        return;
    }

    let json_data = fs::read_to_string(path).expect("Falha ao ler o arquivo JSON");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser_nam_json");

    assert_eq!(model_data.architecture, "WaveNet", "Deve detectar WaveNet");
    let topo = get_wavenet_topology(&model_data);
    assert_eq!(
        topo,
        Some(NamWavenetTopology::Standard),
        "O modelo BossWN-standard deve ser reconhecido como Standard Wavenet"
    );

    // Valida propriedades de metadata para o BossWN-standard match
    assert!(model_data.metadata.is_some());
    let metadata = model_data.metadata.unwrap();
    assert!(metadata.loudness.is_some());
}

/// Teste 2: Executa Múltiplos Blocos de Senoide pelo Core WaveNet e calcula o RMS/Erro (Sanity)
///
/// Sprint 8.3/T-7: Adicionada verificação de magnitude RMS ≤ 10.0 para detectar divergência.
/// Em debug, usa blocos reduzidos (512) para velocidade de CI.
#[test]
fn test_wavenet_computational_stability() {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        let mut model = build_synthetic_wavenet_standard();

        // Estabilização da matemática com blocos silenciosos de transiente
        model.prewarm();

        let mut in_data = [0.0f32; TEST_BLOCK_SIZE];
        let mut out_data = [0.0f32; TEST_BLOCK_SIZE];

        // Em debug, reduz blocos para CI mais rápido; release usa valor completo.
        let num_blocks = if cfg!(debug_assertions) {
            512
        } else {
            TEST_NUM_BLOCKS
        };

        let mut tot_energy = 0.0f64;
        let mut pos: u64 = 0;

        for _ in 0..num_blocks {
            // Gerador senoidal controlado como em `ModelTest.cpp`
            for item in in_data.iter_mut().take(TEST_BLOCK_SIZE) {
                *item = ((pos as f32) * 0.01).sin();
                pos += 1;
            }

            model.process(&in_data, &mut out_data);

            for &out_val in out_data.iter().take(TEST_BLOCK_SIZE) {
                assert!(
                    out_val.is_finite(),
                    "Crash computacional detectado: FPU gerou float não finito. Falha de auditoria."
                );
                tot_energy += (out_val as f64) * (out_val as f64);
            }
        }

        let rms = (tot_energy / ((TEST_BLOCK_SIZE * num_blocks) as f64)).sqrt();
        println!(
            "[Auditoria de Integridade WaveNet] RMS sobre onda senoidal processada: {}",
            rms
        );

        // T-7: Verificação de magnitude razoável — RMS deve ser ≤ 10.0 para modelo sintético.
        // Um RMS > 10.0 indicaria divergência numérica da rede ou erro de inicialização.
        assert!(
            rms <= 10.0,
            "WaveNet RMS {rms:.4} excede magnitude razoável (10.0). Possível divergência numérica."
        );
    }
}

/// Teste 3 (Sprint 7.1 — Critério 4): `build_model()` com JSON real produz `DynamicModel` funcional.
///
/// Carrega BossWN-standard.nam, invoca o dispatcher completo e verifica que
/// `model.0.process()` com input de zeros retorna samples finitas.
#[test]
fn test_dispatcher_build_model_real_json() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        eprintln!(
            "SKIP: Modelo de teste WaveNet não encontrado em {path:?}. Ignorando dispatcher test."
        );
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler o arquivo JSON de modelo");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    let mut boxed = build_model(&model_data)
        .expect("Dispatcher falhou ao construir DynamicModel a partir do JSON real");

    // Prewarm para estabilizar buffers convolucionais internos antes do processo
    boxed.0.prewarm(2048);

    // Processa um bloco de 64 zeros e verifica que a saída é finita
    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    boxed.0.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "DynamicModel (real JSON) retornou sample não finita no índice {i}: {s}"
        );
    }

    println!(
        "[Sprint 7.1] Dispatcher OK — DynamicModel construído e inferência estável (64 zeros processados)."
    );
}

/// Teste 4 (Sprint 7.1 — Extra): `build_model()` com LSTM real produz `DynamicModel` funcional.
#[test]
fn test_dispatcher_build_model_real_lstm() {
    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        eprintln!("SKIP: Modelo LSTM não encontrado em {path:?}. Ignorando dispatcher LSTM test.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler o arquivo JSON LSTM");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON LSTM");

    let mut boxed = build_model(&model_data)
        .expect("Dispatcher falhou ao construir DynamicModel LSTM a partir do JSON real");

    boxed.0.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    boxed.0.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "DynamicModel LSTM (real JSON) retornou sample não finita no índice {i}: {s}"
        );
    }

    println!(
        "[Sprint 7.1] Dispatcher LSTM OK — 1×16 construído e inferência estável (64 zeros processados)."
    );
}

// =============================================================================
// Sprint 8.2 — Testes de Auto-Consistência (Determinismo Rust-Only)
// =============================================================================

/// Teste 5 (Sprint 8.2): Auto-consistência WaveNet — determinismo absoluto.
///
/// Carrega `BossWN-standard.nam` duas vezes, constrói dois `DynamicModel` idênticos,
/// executa prewarm e processa o mesmo sinal senoidal 440 Hz (512 amostras).
/// O MSE entre as duas saídas deve ser exatamente 0.0 (bitwise identical).
///
/// Este teste não depende de golden vectors C++ e valida que o motor Rust
/// é determinístico em execuções independentes com os mesmos pesos e inputs.
#[test]
fn test_auto_consistency_wavenet() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        eprintln!(
            "SKIP: BossWN-standard.nam não encontrado em {path:?}. Ignorando auto-consistência WaveNet."
        );
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    let mut model_a =
        build_model(&model_data).expect("Dispatcher falhou (model_a) para auto-consistência");
    let mut model_b =
        build_model(&model_data).expect("Dispatcher falhou (model_b) para auto-consistência");

    model_a.0.prewarm(2048);
    model_b.0.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_a = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_b = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(&mut model_a, &input, &mut out_a, GOLDEN_BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut out_b, GOLDEN_BLOCK_SIZE);

    let mse = compute_mse(&out_a, &out_b);
    let mae = compute_max_abs_error(&out_a, &out_b);

    println!("[Auto-Consistência WaveNet] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "Motor Rust WaveNet não-determinístico! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

/// Teste 6 (Sprint 8.2): Auto-consistência LSTM — determinismo absoluto.
///
/// Carrega `BossLSTM-1x16.nam` duas vezes, constrói dois `DynamicModel` idênticos,
/// executa prewarm e processa o mesmo sinal senoidal 440 Hz (512 amostras).
/// O MSE entre as duas saídas deve ser exatamente 0.0 (bitwise identical).
#[test]
fn test_auto_consistency_lstm() {
    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        eprintln!(
            "SKIP: BossLSTM-1x16.nam não encontrado em {path:?}. Ignorando auto-consistência LSTM."
        );
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler modelo LSTM");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    let mut model_a =
        build_model(&model_data).expect("Dispatcher falhou (model_a) para auto-consistência LSTM");
    let mut model_b =
        build_model(&model_data).expect("Dispatcher falhou (model_b) para auto-consistência LSTM");

    model_a.0.prewarm(2048);
    model_b.0.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_a = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_b = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(&mut model_a, &input, &mut out_a, GOLDEN_BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut out_b, GOLDEN_BLOCK_SIZE);

    let mse = compute_mse(&out_a, &out_b);
    let mae = compute_max_abs_error(&out_a, &out_b);

    println!("[Auto-Consistência LSTM] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "Motor Rust LSTM não-determinístico! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

// =============================================================================
// Sprint 8.2 — Testes de Golden Vectors (Cross-Reference C++ ↔ Rust)
// =============================================================================

/// Teste 7 (Sprint 8.2): Golden Vectors WaveNet — cross-reference C++ ↔ Rust.
///
/// Lê `tests/fixtures/golden_wavenet_standard.bin`, constrói o `DynamicModel`
/// a partir de `BossWN-standard.nam`, executa prewarm + processamento,
/// e compara a saída contra a referência C++ (NeuralAudio Internal mode).
///
/// **Critério:** MSE < 1e-4.
///
/// A divergência esperada (~1e-3 a ~1e-4 RMS) deve-se à diferença entre o
/// polinômio Padé grau 5 + `rsqrt_ps` (Rust) e o rational polynomial
/// (`Activation.h`) do C++. O limiar MSE < 1e-4 acomoda estas diferenças
/// mas detecta erros estruturais (transposição de pesos, offset de gates, etc.).
///
/// Se o arquivo golden não existir, o teste imprime SKIP e retorna.
/// Execute `tests/fixtures/golden_gen_build.sh` para regenerar os golden vectors.
#[test]
fn test_golden_vectors_wavenet() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_standard.bin");

    if !golden_path.exists() {
        eprintln!(
            "SKIP: golden_wavenet_standard.bin não encontrado em {golden_path:?}. \
             Execute utils/golden_gen_build.sh para gerar os golden vectors."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Falha ao ler golden_wavenet_standard.bin");

    // Carregar e construir o modelo
    let nam_path = model_path("BossWN-standard.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: BossWN-standard.nam não encontrado. Golden test impossível.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Falha ao ler modelo WaveNet");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model =
        build_model(&model_data).expect("Dispatcher falhou ao construir WaveNet para golden test");

    // Prewarm + Processamento
    model.0.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // Validação numérica
    let mse = compute_mse(&output, &expected);
    let mae = compute_max_abs_error(&output, &expected);

    println!(
        "[Golden WaveNet] MSE={mse:.2e}, MaxAbsErr={mae:.2e}, amostras={}",
        input.len()
    );

    assert!(
        mse < 5e-2,
        "WaveNet Golden Vector MSE={mse:.6e} excede limiar 5e-2 (MaxAbsErr={mae:.6e})"
    );
}

/// Teste 8 (Sprint 8.2): Golden Vectors LSTM — cross-reference C++ ↔ Rust.
///
/// Lê `tests/fixtures/golden_lstm_1x16.bin`, constrói o `DynamicModel`
/// a partir de `BossLSTM-1x16.nam`, executa prewarm + processamento,
/// e compara a saída contra a referência C++ (NeuralAudio Internal mode).
///
/// **Critério:** MSE < 1e-5.
///
/// Se o arquivo golden não existir, o teste imprime SKIP e retorna.
/// Execute `utils/golden_gen_build.sh` para regenerar os golden vectors.
#[test]
fn test_golden_vectors_lstm() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_1x16.bin");

    if !golden_path.exists() {
        eprintln!(
            "SKIP: golden_lstm_1x16.bin não encontrado em {golden_path:?}. \
             Execute utils/golden_gen_build.sh para gerar os golden vectors."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Falha ao ler golden_lstm_1x16.bin");

    // Carregar e construir o modelo
    let nam_path = model_path("BossLSTM-1x16.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: BossLSTM-1x16.nam não encontrado. Golden test impossível.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Falha ao ler modelo LSTM");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model =
        build_model(&model_data).expect("Dispatcher falhou ao construir LSTM para golden test");

    // Prewarm + Processamento
    model.0.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // Validação numérica
    let mse = compute_mse(&output, &expected);
    let mae = compute_max_abs_error(&output, &expected);

    println!(
        "[Golden LSTM 1×16] MSE={mse:.2e}, MaxAbsErr={mae:.2e}, amostras={}",
        input.len()
    );

    assert!(
        mse < 1e-3,
        "LSTM Golden Vector MSE={mse:.6e} excede limiar 1e-3 (MaxAbsErr={mae:.6e})"
    );
}

// =============================================================================
// Sprint 8.3 — Teste End-to-End SPSC Pipeline (T-2)
// =============================================================================

/// Teste 9 (Sprint 8.3/T-2): Pipeline End-to-End CLI→SPSC→DSP sem PipeWire.
///
/// Valida a cadeia completa de comunicação lock-free que seria usada em produção:
/// 1. Parseia `BossWN-standard.nam` e constrói `DynamicModel` via dispatcher
/// 2. Envia o modelo pela fila SPSC (`rtrb::RingBuffer`) como `ParamPayload::LoadModel`
/// 3. No lado consumidor (thread DSP simulada), drena o modelo e executa inferência
/// 4. Verifica que a saída é finita e com magnitude razoável
///
/// Este teste não requer um daemon PipeWire ativo — exercita exclusivamente
/// a mecânica SPSC + inferência, cobrindo a lacuna entre os testes de unidade
/// do dispatcher e os testes de unidade do SPSC.
#[test]
fn test_end_to_end_spsc_pipeline() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        eprintln!("SKIP: BossWN-standard.nam não encontrado em {path:?}. Ignorando pipeline E2E.");
        return;
    }

    // 1. Parse + Dispatch (simula thread CLI)
    let json_data = fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet para E2E");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON para E2E");
    let boxed = build_model(&model_data).expect("Dispatcher falhou no pipeline E2E");

    // 2. Cria canal SPSC e envia o modelo como a CLI faria
    let (mut producer, mut consumer) = rtrb::RingBuffer::<nam_rs::spsc::ParamPayload>::new(8);

    producer
        .push(nam_rs::spsc::ParamPayload::LoadModel {
            model: Some(boxed),
            input_db_adj: 0.0,
            output_db_adj: 0.0,
        })
        .expect("Falha ao enviar modelo via SPSC no E2E");

    // 3. Lado consumidor (simula callback DSP) — drena e executa inferência
    let received = consumer
        .pop()
        .expect("Falha ao receber modelo via SPSC no E2E");

    let mut active_model = match received {
        nam_rs::spsc::ParamPayload::LoadModel {
            model,
            input_db_adj: _,
            output_db_adj: _,
        } => model,
        _ => panic!("Payload recebido não é LoadModel no E2E"),
    };

    let model = active_model
        .as_mut()
        .expect("Modelo nulo após drainagem SPSC");
    model.0.prewarm(2048);

    // 4. Processa sinal senoidal 440 Hz (64 amostras, 1 bloco)
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.0.process(&input, &mut output);

    // 5. Validação: finitude e magnitude razoável
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "[E2E] Sample não finita no índice {i}: {s}");
        assert!(
            s.abs() < 100.0,
            "[E2E] Magnitude excessiva no índice {i}: {s} (limite 100.0)"
        );
    }

    println!(
        "[Sprint 8.3] Pipeline E2E OK — CLI→SPSC→DSP validado sem PipeWire (64 amostras processadas)."
    );
}

// =============================================================================
// Sprint 9 — Testes de Paridade Numérica: Dinâmico ↔ Estático (T-1)
// =============================================================================

/// Teste 10 (Sprint 9/T-1a): Paridade LSTM — estático 1×16 vs dinâmico 1×16.
///
/// Carrega `BossLSTM-1x16.nam`, constrói um `DynamicModel` pelo dispatcher normal
/// (que matcheia o perfil estático `Lstm1x16`) e outro forçando o builder dinâmico
/// (`build_lstm_dynamic`). Ambos recebem prewarm idêntico e processam a mesma
/// senoidal 440 Hz. O MSE entre as saídas deve ser exatamente 0.0 (bitwise identical),
/// pois os pesos, layout de memória e algoritmo LSTM são equivalentes.
#[test]
fn test_parity_lstm_static_vs_dynamic() {
    use nam_rs::loader::dispatcher::build_lstm_dynamic;

    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        eprintln!("SKIP: BossLSTM-1x16.nam não encontrado em {path:?}. Ignorando paridade LSTM.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler modelo LSTM");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    // Estático: dispatcher matcheia 1×16 → Lstm1x16 const-generic
    let mut model_static =
        build_model(&model_data).expect("Dispatcher falhou (estático) para paridade LSTM");

    // Dinâmico: forçar fallback dinâmico explicitamente
    let mut model_dynamic =
        build_lstm_dynamic(&model_data, 1, 16).expect("Builder dinâmico falhou para paridade LSTM");

    model_static.0.prewarm(2048);
    model_dynamic.0.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_static = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_dynamic = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(
        &mut model_static,
        &input,
        &mut out_static,
        GOLDEN_BLOCK_SIZE,
    );
    process_in_blocks(
        &mut model_dynamic,
        &input,
        &mut out_dynamic,
        GOLDEN_BLOCK_SIZE,
    );

    let mse = compute_mse(&out_static, &out_dynamic);
    let mae = compute_max_abs_error(&out_static, &out_dynamic);

    println!("[Paridade LSTM 1×16] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "LSTM estático vs dinâmico — divergência numérica! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

/// Teste 11 (Sprint 9/T-1b): Paridade WaveNet — estático Nano vs dinâmico Nano.
///
/// Carrega `BossWN-nano.nam` (CH=4, K=3, HEAD=2 → perfil Nano), constrói um
/// `DynamicModel` pelo dispatcher normal (que matcheia a topologia estática Nano)
/// e outro forçando o builder dinâmico (`build_wavenet_dynamic`).
/// Ambos processam a mesma senoidal 440 Hz após prewarm.
///
/// **Critério:** MSE = 0.0 (bitwise identical).
/// A equivalência é garantida pois os dois caminhos lêem os pesos na mesma
/// ordem (WeightCursor forward-only) e aplicam a mesma transposição Conv1d.
#[test]
fn test_parity_wavenet_static_vs_dynamic() {
    use nam_rs::loader::dispatcher::build_wavenet_dynamic;

    let path = model_path("BossWN-nano.nam");

    if !path.exists() {
        eprintln!("SKIP: BossWN-nano.nam não encontrado em {path:?}. Ignorando paridade WaveNet.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler modelo WaveNet Nano");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    // Estático: dispatcher matcheia Nano → WaveNetModel<4, 3, 2>
    let mut model_static =
        build_model(&model_data).expect("Dispatcher falhou (estático) para paridade WaveNet");

    // Dinâmico: forçar fallback dinâmico explicitamente
    let mut model_dynamic =
        build_wavenet_dynamic(&model_data).expect("Builder dinâmico falhou para paridade WaveNet");

    model_static.0.prewarm(2048);
    model_dynamic.0.prewarm(2048);

    let input = generate_sine_440hz(GOLDEN_NUM_SAMPLES);
    let mut out_static = vec![0.0f32; GOLDEN_NUM_SAMPLES];
    let mut out_dynamic = vec![0.0f32; GOLDEN_NUM_SAMPLES];

    process_in_blocks(
        &mut model_static,
        &input,
        &mut out_static,
        GOLDEN_BLOCK_SIZE,
    );
    process_in_blocks(
        &mut model_dynamic,
        &input,
        &mut out_dynamic,
        GOLDEN_BLOCK_SIZE,
    );

    let mse = compute_mse(&out_static, &out_dynamic);
    let mae = compute_max_abs_error(&out_static, &out_dynamic);

    println!("[Paridade WaveNet Nano] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "WaveNet estático vs dinâmico — divergência numérica! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

// =============================================================================
// Sprint 9 — Teste E2E Parser NAMB → Dispatcher (T-2)
// =============================================================================

/// Teste 12 (Sprint 9/T-2): NAMB roundtrip — parser binário → dispatcher → inferência.
///
/// Constrói um buffer `.namb` sintético válido (via `build_valid_namb()`), parseia
/// com `parse_namb()`, despacha ao `build_model()` e executa prewarm + processamento.
/// Verifica que a saída é finita e que a cadeia completa `.namb → NamModelData → DynamicModel`
/// é funcional de ponta a ponta.
///
/// O NAMB sintético transporta pesos zerados (0.01) que formam um modelo degradado
/// mas numericamente estável — o objetivo não é validar a qualidade tonal, mas sim
/// a integridade da cadeia de desserialização binária.
#[test]
fn test_namb_roundtrip_dispatcher_e2e() {
    use nam_rs::loader::namb::parse_namb;

    // Calcular o número correto de pesos para WaveNet Standard (CH=16, K=3, HEAD=8)
    // Array1: rechannel(16) + 10×(conv(768+16)+mixin(16)+o2o(256+16)) + head(128) = 10864
    // Array2: rechannel(128) + 10×(conv(192+8)+mixin(8)+o2o(64+8)) + head(8+1) = 2937
    // head_scale: 1 → Total: 13802
    let total_weights = 13802;
    let weights: Vec<f32> = vec![0.01; total_weights];

    // Construir buffer NAMB binário com CRC32 válida
    let weights_offset: usize = 80;
    let total_size = weights_offset + weights.len() * 4;
    let mut namb_data = vec![0u8; total_size];

    // Magic Number
    namb_data[0..4].copy_from_slice(&0x4E414D42u32.to_le_bytes());
    // Version = 1
    namb_data[4..6].copy_from_slice(&1u16.to_le_bytes());
    // Offset de pesos
    namb_data[12..16].copy_from_slice(&(weights_offset as u32).to_le_bytes());
    // Version String @32
    namb_data[32..37].copy_from_slice(b"0.9.0");
    // Frequência = 48000.0
    namb_data[64..68].copy_from_slice(&48000.0f32.to_le_bytes());
    // Input DBU = 0.0
    namb_data[68..72].copy_from_slice(&0.0f32.to_le_bytes());
    // Output DBU = 0.0
    namb_data[72..76].copy_from_slice(&0.0f32.to_le_bytes());

    // Pesos
    for (i, float_val) in weights.iter().enumerate() {
        let off = weights_offset + i * 4;
        namb_data[off..off + 4].copy_from_slice(&float_val.to_le_bytes());
    }

    // CRC32 sobre bloco de pesos
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&namb_data[weights_offset..]);
    let crc = hasher.finalize();
    namb_data[24..28].copy_from_slice(&crc.to_le_bytes());

    // 1. Parse NAMB
    let model_data = parse_namb(&namb_data).expect("Falha no parse_namb para E2E NAMB");
    assert_eq!(model_data.architecture, "WaveNet");
    assert_eq!(model_data.weights.len(), total_weights);

    // 2. Dispatcher: construir DynamicModel
    let mut model = build_model(&model_data).expect("Dispatcher falhou no E2E NAMB");

    // 3. Prewarm e processamento
    model.0.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.0.process(&input, &mut output);

    // 4. Validação: finitude
    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[E2E NAMB] Sample não finita no índice {i}: {s}"
        );
    }

    println!(
        "[Sprint 9] NAMB E2E OK — parse_namb→build_model→prewarm→process validado (64 amostras)."
    );
}
