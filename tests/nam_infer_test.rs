// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Testes de Integração para validação das topologias e inferência neural.
//!
//! # Objetivo
//! Replicar o ModelTest.cpp da implementação C++ original:
//! Injetar um bloco contínuo de ondas iterativas (ex: senoidal) no pipeline Lock-Free de Inferência,
//! calculando estabilidade computacional temporal e limitando Erros Numéricos (NaN, Infinito),
//! provando que o compilador Rust / auto-vetorização (Const Generics e FMA/AVX2) não introduzem crashes no DSP de longa duração.

use nam_rs::loader::nam_json::{NamWavenetTopology, get_wavenet_topology, parse_nam_json};
use nam_rs::models::wavenet;
use std::fs;
use std::path::PathBuf;

// Define as topologias compatíveis baseadas nos tipos de type alias do `models/mod.rs`
type WaveNetStandard = wavenet::WaveNetModel<16, 3, 2>;

const TEST_BLOCK_SIZE: usize = 64;
const TEST_NUM_BLOCKS: usize = 4096; // ~5.4 segundos de processamento simulado (a 48kHz).

/// Helper para simular a criação de uma `WaveNetLayer` limpa.
fn make_wavenet_layer(
    dilation: usize,
    has_bias: bool,
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
            do_bias: has_bias,
        },
    }
}

/// Helper para construir uma rede `WaveNetStandard` temporariamente estabilizada e zerada, uma vez
/// que o construtor completo via `weights` `.nam` será arquitetado na integração da GUI no futuro.
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

    let array1 = wavenet::WaveNetLayerArray::<1, 1, 16, 3, 2> {
        layers: layers_1,
        states: states_1,
        rechannel: wavenet::DenseLayer {
            weights: vec![0.001; 16],
            bias: vec![0.0; 16],
            do_bias: false,
        },
        head_rechannel: wavenet::DenseLayer {
            weights: vec![0.001; 2 * 16],
            bias: vec![0.0; 2],
            do_bias: false,
        },
        array_outputs: vec![0.0; 16],
        head_accum: vec![0.0; 16],
        head_outputs: vec![0.0; 2],
        receptive_field_size: final_rf,
    };

    let layers_2: Vec<wavenet::WaveNetLayer<1, 16, 3>> = dilations_2
        .iter()
        .map(|&d| make_wavenet_layer(d, true, 16))
        .collect();
    let states_2: Vec<wavenet::WaveNetLayerState> = (0..layers_2.len())
        .map(|i| wavenet::WaveNetLayerState::new(16, final_rf, i))
        .collect();

    let array2 = wavenet::WaveNetLayerArray::<16, 1, 16, 3, 2> {
        layers: layers_2,
        states: states_2,
        rechannel: wavenet::DenseLayer {
            weights: vec![0.0; 16 * 16],
            bias: vec![0.0; 16],
            do_bias: false,
        },
        head_rechannel: wavenet::DenseLayer {
            weights: vec![0.0; 2 * 16],
            bias: vec![0.0; 2],
            do_bias: false,
        },
        array_outputs: vec![0.0; 16],
        head_accum: vec![0.0; 16],
        head_outputs: vec![0.0; 2],
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
