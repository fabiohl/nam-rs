// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Teste de integração para verificar o round-trip de layout `GateMajorLstm` em modelos LSTM.
//!
//! Garante que a codificação (JSON -> NAMB v2) e a decodificação (NAMB v2) mantêm
//! a ordem correta dos pesos e a compatibilidade matemática (MSE próximo a zero)
//! para topologias LSTM de camada única e multi-camadas (multi-layer).

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{NamConfig, NamModelData, WeightsLayout, parse_nam_json};
use nam_rs::loader::namb::parse_namb;
use nam_rs::loader::namb_encoder::encode_namb;
use nam_rs::models::NamModel;
use std::fs;
use std::path::PathBuf;

/// Helper: resolve o caminho absoluto para as fixtures de teste.
fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

/// Gera um sinal de teste senoidal estável.
fn generate_sine(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

/// Computa o Erro Quadrático Médio (MSE).
fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
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

/// Helper para construir um modelo LSTM sintético com pesos previsíveis.
fn make_synthetic_lstm(num_layers: usize, hidden_size: usize) -> NamModelData {
    let mut weights = Vec::new();
    let mut current_input_size = 1;
    let mut val = 0.05f32;

    for _ in 0..num_layers {
        let ih = current_input_size + hidden_size;
        // 1. Pesos da camada
        let w_size = 4 * hidden_size * ih;
        for _ in 0..w_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        // 2. Bias
        let b_size = 4 * hidden_size;
        for _ in 0..b_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        // 3. hidden_init
        for _ in 0..hidden_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        // 4. cell_init
        for _ in 0..hidden_size {
            weights.push(val);
            val = (val + 0.007) % 0.3;
        }
        current_input_size = hidden_size;
    }

    // Head weights
    for _ in 0..hidden_size {
        weights.push(val);
        val = (val + 0.007) % 0.3;
    }
    // Head bias
    weights.push(0.01);

    NamModelData {
        version: Some("0.5.0".to_string()),
        architecture: "LSTM".to_string(),
        config: NamConfig {
            layers: vec![],
            head: None,
            head_scale: Some(1.0),
            num_layers: Some(num_layers),
            hidden_size: Some(hidden_size),
        },
        weights,
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

/// Testa a paridade do round-trip para uma dada topologia LSTM.
fn test_topology_roundtrip(num_layers: usize, hidden_size: usize) {
    let orig_data = make_synthetic_lstm(num_layers, hidden_size);

    // 1. Constrói modelo original (JSON/Original layout)
    let mut model_orig = build_model(&orig_data)
        .unwrap_or_else(|e| panic!("Erro ao construir original {}x{}: {:?}", num_layers, hidden_size, e));
    model_orig.prewarm(1024);

    // 2. Codifica para NAMB v2 (GateMajorLstm)
    let namb_v2 = encode_namb(&orig_data, 2, WeightsLayout::GateMajorLstm)
        .unwrap_or_else(|e| panic!("Erro ao codificar {}x{}: {:?}", num_layers, hidden_size, e));

    // 3. Decodifica para NAMModelData
    let v2_data = parse_namb(&namb_v2)
        .unwrap_or_else(|e| panic!("Erro ao decodificar {}x{}: {:?}", num_layers, hidden_size, e));
    assert_eq!(v2_data.weights_layout, WeightsLayout::GateMajorLstm);

    // 4. Constrói o modelo decodificado
    let mut model_v2 = build_model(&v2_data)
        .unwrap_or_else(|e| panic!("Erro ao construir v2 {}x{}: {:?}", num_layers, hidden_size, e));
    model_v2.prewarm(1024);

    // 5. Verifica a saída de inferência
    let input = generate_sine(512);
    let mut out_orig = vec![0.0f32; 512];
    let mut out_v2 = vec![0.0f32; 512];

    model_orig.process(&input, &mut out_orig);
    model_v2.process(&input, &mut out_v2);

    let mse = compute_mse(&out_orig, &out_v2);
    assert!(
        mse < 1e-12,
        "Divergência detectada na topologia {}x{}! MSE = {:e}",
        num_layers,
        hidden_size,
        mse
    );
}

#[test]
fn test_lstm_topologies_roundtrip() {
    let topologies = [
        (1, 8),
        (1, 12),
        (1, 16),
        (1, 24),
        (2, 8),
        (2, 12),
        (2, 16),
    ];

    for &(layers, hidden) in &topologies {
        test_topology_roundtrip(layers, hidden);
    }
}

#[test]
fn test_real_lstm_2x8_roundtrip() {
    let path = model_path("BossLSTM-2x8.nam");
    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).unwrap();
    let original_data = parse_nam_json(&json_data).unwrap();

    // 1. Constrói modelo original
    let mut model_orig = build_model(&original_data).unwrap();
    model_orig.prewarm(1024);

    // 2. Codifica para NAMB v2 Gate-Major
    let namb_v2 = encode_namb(&original_data, 2, WeightsLayout::GateMajorLstm).unwrap();

    // 3. Decodifica
    let v2_data = parse_namb(&namb_v2).unwrap();
    assert_eq!(v2_data.weights_layout, WeightsLayout::GateMajorLstm);

    // 4. Constrói modelo v2
    let mut model_v2 = build_model(&v2_data).unwrap();
    model_v2.prewarm(1024);

    // 5. Compara a saída
    let input = generate_sine(512);
    let mut out_orig = vec![0.0f32; 512];
    let mut out_v2 = vec![0.0f32; 512];

    model_orig.process(&input, &mut out_orig);
    model_v2.process(&input, &mut out_v2);

    let mse = compute_mse(&out_orig, &out_v2);
    println!("[BossLSTM-2x8 v2 Parity] MSE: {:.2e}", mse);
    assert!(
        mse < 1e-12,
        "Divergência no modelo real BossLSTM-2x8! MSE={:e}",
        mse
    );
}
