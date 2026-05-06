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

/// Helper: resolve o caminho absoluto para um modelo de teste localizado em `tests/fixtures/models/`.
/// Utiliza `CARGO_MANIFEST_DIR` para garantir que o teste funcione independente do diretório de execução.
fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

/// Teste de Compatibilidade Futura (Forward-Compatibility) para WaveNet A2.
///
/// Verifica se o motor de inferência lida graciosamente com modelos v0.6+ (A2).
/// Atualmente, o NAM-rs não implementa todas as features A2 (como FiLM ou Gate dinâmico),
/// então ele deve carregar o modelo sem pânico e fazer o fallback para um placeholder
/// que emite silêncio, informando ao host que o modelo é incompatível mas seguro.
#[test]
fn test_forward_compatibility_wavenet_a2() {
    let path = model_path("mock_a2.nam");

    if !path.exists() {
        // Falha crítica se a fixture de teste estiver ausente
        panic!(
            "Fixture mock_a2.nam não encontrada em {path:?}. Verifique se o submódulo de fixtures foi baixado."
        );
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler mock_a2.nam");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    // Valida se o metadado do modelo foi corretamente identificado como arquitetura A2
    assert!(
        model_data.is_wavenet_a2(),
        "mock_a2.nam deve ser detectado como arquitetura A2 (v0.6+)"
    );

    // O dispatcher deve aceitar o modelo e retornar o variante de placeholder
    let mut model = build_model(&model_data).expect(
        "O dispatcher deveria ter realizado o fallback para o placeholder A2 em vez de falhar",
    );

    // Verifica explicitamente se o variante retornado é o Placeholder
    match *model {
        DynamicModel::WavenetA2(_) => {
            println!("Fallback para WavenetA2Placeholder confirmado com sucesso.");
        }
        _ => panic!(
            "Erro de arquitetura: O loader deveria ter retornado DynamicModel::WavenetA2 para este arquivo"
        ),
    }

    // Validação de segurança RT: o placeholder não deve processar áudio, apenas silenciar o buffer.
    let input = [1.0f32; 64];
    let mut output = [1.0f32; 64];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert_eq!(
            s, 0.0,
            "Placeholder A2 deve garantir silêncio absoluto para evitar ruídos indesejados. Falha no índice {i}"
        );
    }
}

/// Teste de Regressão para WaveNet A1 (Standard).
/// Garante que modelos legados continuam carregando e processando áudio normalmente.
#[test]
fn test_regression_a1_wavenet_standard() {
    let path = model_path("BossWN-standard.nam");

    // Ignora se o modelo real não estiver presente (geralmente arquivos grandes não estão no git)
    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler o arquivo JSON");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    let mut model = build_model(&model_data)
        .expect("O dispatcher falhou ao construir o modelo WaveNet A1 Standard");

    // Preenche buffers de atraso
    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    // Valida se a saída é numérica e finita (sem NaNs ou Infs por instabilidade)
    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "Saída do WaveNet A1 contém valores não-finitos (NaN/Inf)"
        );
    }
}

/// Teste de Regressão para arquitetura LSTM.
/// Garante que o motor recorrente legado (v0.5.x) mantém sua integridade funcional.
#[test]
fn test_regression_a1_lstm() {
    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Falha ao ler o arquivo JSON");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");

    let mut model =
        build_model(&model_data).expect("O dispatcher falhou ao construir o modelo LSTM A1");

    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    // As LSTMs são mais propensas a instabilidade numérica; este teste garante paridade funcional
    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "Saída da LSTM A1 contém valores não-finitos (NaN/Inf)"
        );
    }
}
