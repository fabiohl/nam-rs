// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Testes de Integração para validação das topologias e inferência neural.
//!
//! # Objetivo
//! Replicar o ModelTest.cpp da implementação C++ original:
//! Injetar um bloco contínuo de ondas iterativas (ex: senoidal) no pipeline Lock-Free de Inferência,
//! calculando estabilidade computacional temporal e limitando Erros Numéricos (NaN, Infinito),
//! provando que o compilador Rust / auto-vetorização (Const Generics e FMA/AVX2) não introduzem crashes no DSP de longa duração.

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{NamWavenetTopology, get_wavenet_topology, parse_nam_json};
use nam_rs::models::wavenet;
use std::fs;
use std::path::PathBuf;

// WaveNetModel<CH=16, K=3, HEAD=8>: CH=16 canais, HEAD=8 (head_size da layer 0 = canais da Array2).
type WaveNetStandard = wavenet::WaveNetModel<16, 3, 8>;

const TEST_BLOCK_SIZE: usize = 64;
const TEST_NUM_BLOCKS: usize = 4096; // ~5.4 segundos de processamento simulado (a 48kHz).

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

/// Teste 1: Auditoria da capacidade de Leitura do Loader e Validação Geometria
#[test]
fn test_wavenet_model_json_parsing() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("github.com/mikeoliphant/NeuralAudio/Utils/Models/BossWN-standard.nam");

    if !path.exists() {
        println!("Aviso: Modelo de teste WaveNet não encontrado em {path:?}. Ignorando parsing.");
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
#[test]
fn test_wavenet_computational_stability() {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        let mut model = build_synthetic_wavenet_standard();

        // Estabilização da matemática com blocos silenciosos de transiente
        model.prewarm();

        let mut in_data = [0.0f32; TEST_BLOCK_SIZE];
        let mut out_data = [0.0f32; TEST_BLOCK_SIZE];

        let mut tot_energy = 0.0f64;
        let mut pos: u64 = 0;

        for _ in 0..TEST_NUM_BLOCKS {
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

        let rms = (tot_energy / ((TEST_BLOCK_SIZE * TEST_NUM_BLOCKS) as f64)).sqrt();
        println!(
            "[Auditoria de Integridade WaveNet] RMS sobre onda senoidal processada: {}",
            rms
        );
    }
}

/// Teste 3 (Sprint 7.1 — Critério 4): `build_model()` com JSON real produz `DynamicModel` funcional.
///
/// Carrega BossWN-standard.nam, invoca o dispatcher completo e verifica que
/// `model.0.process()` com input de zeros retorna samples finitas.
#[test]
fn test_dispatcher_build_model_real_json() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("github.com/mikeoliphant/NeuralAudio/Utils/Models/BossWN-standard.nam");

    if !path.exists() {
        println!(
            "Aviso: Modelo de teste WaveNet não encontrado em {path:?}. Ignorando dispatcher test."
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
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("github.com/mikeoliphant/NeuralAudio/Utils/Models/BossLSTM-1x16.nam");

    if !path.exists() {
        println!("Aviso: Modelo LSTM não encontrado em {path:?}. Ignorando dispatcher LSTM test.");
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
