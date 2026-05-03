// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Teste de validação para o formato `.namb` v2 (Pesos Pré-transpostos).
//!
//! Verifica que a exportação v2 e o carregamento correspondente mantêm
//! paridade numérica absoluta com o carregamento via JSON (transposição em runtime).

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{WeightsLayout, parse_nam_json};
use nam_rs::loader::namb::parse_namb;
use nam_rs::loader::namb_encoder::encode_namb;
use nam_rs::models::NamModel;
use std::fs;
use std::path::PathBuf;

fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

fn generate_sine(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

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

#[test]
fn test_lstm_v2_gate_major_parity() {
    let path = model_path("BossLSTM-1x16.nam");
    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).unwrap();
    let original_data = parse_nam_json(&json_data).unwrap();

    // 1. Constrói modelo original (runtime transposition)
    let mut model_orig = build_model(&original_data).unwrap();
    model_orig.prewarm(1024);

    // 2. Codifica para NAMB v2 Gate-Major
    let namb_v2 = encode_namb(&original_data, 2, WeightsLayout::GateMajorLstm).unwrap();

    // 3. Carrega NAMB v2 (direct copy)
    let v2_data = parse_namb(&namb_v2).unwrap();
    assert_eq!(v2_data.weights_layout, WeightsLayout::GateMajorLstm);

    let mut model_v2 = build_model(&v2_data).unwrap();
    model_v2.prewarm(1024);

    // 4. Compara saídas
    let input = generate_sine(512);
    let mut out_orig = vec![0.0f32; 512];
    let mut out_v2 = vec![0.0f32; 512];

    model_orig.process(&input, &mut out_orig);
    model_v2.process(&input, &mut out_v2);

    let mse = compute_mse(&out_orig, &out_v2);
    println!("[LSTM v2 Parity] MSE: {:.2e}", mse);
    assert!(mse < 1e-12, "Paridade LSTM falhou! MSE={:e}", mse);
}

#[test]
fn test_wavenet_v2_interleaved4_parity() {
    let path = model_path("BossWN-nano.nam");
    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).unwrap();
    let original_data = parse_nam_json(&json_data).unwrap();

    // 1. Constrói modelo original (runtime transposition)
    let mut model_orig = build_model(&original_data).unwrap();
    model_orig.prewarm(2048);

    // 2. Codifica para NAMB v2 Interleaved-4
    let namb_v2 = encode_namb(&original_data, 2, WeightsLayout::Interleaved4WaveNet).unwrap();

    // 3. Carrega NAMB v2 (direct copy)
    let v2_data = parse_namb(&namb_v2).unwrap();
    assert_eq!(v2_data.weights_layout, WeightsLayout::Interleaved4WaveNet);

    let mut model_v2 = build_model(&v2_data).unwrap();
    model_v2.prewarm(2048);

    // 4. Compara saídas
    let input = generate_sine(512);
    let mut out_orig = vec![0.0f32; 512];
    let mut out_v2 = vec![0.0f32; 512];

    model_orig.process(&input, &mut out_orig);
    model_v2.process(&input, &mut out_v2);

    let mse = compute_mse(&out_orig, &out_v2);
    println!("[WaveNet v2 Parity] MSE: {:.2e}", mse);
    assert!(mse < 1e-12, "Paridade WaveNet falhou! MSE={:e}", mse);
}
