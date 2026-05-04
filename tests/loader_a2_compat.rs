// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Testes de Forward-Compatibility para o Loader NAM-rs.
//!
//! Garante que modelos com arquitetura A2 ou campos futuros não causem pânico
//! e façam o fallback gracioso para placeholders, enquanto modelos A1
//! continuam funcionando sem regressões.

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::{DynamicModel, NamModel};
use std::fs;
use std::path::PathBuf;

/// Helper: resolve o caminho para um modelo de teste em `tests/fixtures/models/`.
fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

/// Teste: Forward-compatibility WaveNet A2.
///
/// Verifica que o loader detecta um modelo A2 (via mock_a2.nam) e,
/// em vez de falhar por ativação não suportada, faz o fallback
/// gracioso para o `WavenetA2Placeholder` (que retorna silêncio).
#[test]
fn test_forward_compatibility_wavenet_a2() {
    let path = model_path("mock_a2.nam");

    if !path.exists() {
        panic!("Fixture mock_a2.nam não encontrada em {path:?}");
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler mock_a2.nam");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    assert!(
        model_data.is_wavenet_a2(),
        "mock_a2.nam deve ser detectado como A2"
    );

    let mut model = build_model(&model_data)
        .expect("Dispatcher falhou ao carregar A2 (deveria ter feito fallback)");

    // Verifica se o modelo construído é o variante WavenetA2
    match *model {
        DynamicModel::WavenetA2(_) => {
            println!("Fallback para WavenetA2Placeholder confirmado.");
        }
        _ => panic!("Esperado DynamicModel::WavenetA2, mas obteve outro variante"),
    }

    // Verifica se o processamento retorna silêncio (comportamento do placeholder)
    let input = [1.0f32; 64];
    let mut output = [1.0f32; 64];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert_eq!(
            s, 0.0,
            "Placeholder A2 deve retornar silêncio absoluto. Erro no índice {i}"
        );
    }
}

/// Teste: Regressão A1 - WaveNet Standard.
#[test]
fn test_regression_a1_wavenet_standard() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler o arquivo JSON");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    let mut model =
        build_model(&model_data).expect("Dispatcher falhou ao construir DynamicModel WaveNet A1");

    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for &s in output.iter() {
        assert!(s.is_finite());
    }
}

/// Teste: Regressão A1 - LSTM.
#[test]
fn test_regression_a1_lstm() {
    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler o arquivo JSON");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    let mut model =
        build_model(&model_data).expect("Dispatcher falhou ao construir DynamicModel LSTM A1");

    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for &s in output.iter() {
        assert!(s.is_finite());
    }
}
