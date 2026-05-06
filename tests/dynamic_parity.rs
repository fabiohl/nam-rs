// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use nam_rs::loader::dispatcher::{build_model, build_wavenet_dynamic};
use nam_rs::loader::nam_json::{NamConfig, NamModelData};
use nam_rs::models::NamModel;

/// Gera um sinal senoidal para ser usado como entrada de teste.
/// A previsibilidade do sinal senoidal ajuda a diagnosticar drifts numéricos.
fn generate_sine(freq: f32, sr: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect()
}

/// Calcula o Erro Quadrático Médio (MSE) entre dois sinais.
/// Utilizado para validar a paridade numérica entre diferentes implementações
/// do mesmo algoritmo (ex: Escalar vs SIMD ou Estático vs Dinâmico).
fn calculate_mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Os buffers de áudio devem ter o mesmo tamanho para comparação"
    );
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    sum / a.len() as f32
}

/// Testa se a implementação WaveNet Dinâmica (usada como fallback) produz
/// resultados numericamente equivalentes à implementação Estática (otimizada).
/// A tolerância de erro (1e-10) garante que diferenças de acumulação de
/// ponto flutuante sejam mínimas.
#[test]
fn test_wavenet_dynamic_parity() {
    // Definimos uma topologia reduzida (3 dilatações por bloco) para manter o teste rápido.
    // O objetivo é testar a lógica de despacho e o loop de convolução, não a carga bruta.
    let dils_short = vec![1, 2, 4];

    // Camada 1: Rechannel de 1 para 8 canais
    let layer_s1 = nam_rs::loader::nam_json::NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        channels: Some(8),
        dilations: Some(dils_short.clone()),
        kernel_size: Some(3),
        head_size: Some(4),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(false),
    };

    // Camada 2: Processamento de 8 para 4 canais
    let layer_s2 = nam_rs::loader::nam_json::NamLayerConfig {
        input_size: Some(8),
        condition_size: Some(1),
        channels: Some(4),
        dilations: Some(dils_short),
        kernel_size: Some(3),
        head_size: Some(1),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(true),
    };

    // O cálculo de pesos deve ser exato para evitar falhas no dispatcher.
    // Detalhamento dos pesos (bias + kernel + linear):
    // Array 1: rechannel (8) + layers (3 * (8*3*8 + 8 + 8 + 64 + 8) = 840) + head (32) = 880
    // Array 2: rechannel (32) + layers (3 * (4*3*4 + 4 + 4 + 16 + 4) = 228) + head (5) = 265
    // Scale: 1 -> Total = 1146
    let total_weights = 1146;

    let data = NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![layer_s1, layer_s2],
            head: None,
            head_scale: None,
            num_layers: None,
            hidden_size: None,
        },
        weights: vec![0.01; total_weights],
        weights_layout: nam_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    };

    // Instancia ambos os motores de inferência
    let mut model_static = build_model(&data).expect("Falha ao construir modelo estático");
    let mut model_dyn = build_wavenet_dynamic(&data).expect("Falha ao construir modelo dinâmico");

    // Preaquecimento é necessário para estabilizar estados internos (delays)
    model_static.prewarm(1024);
    model_dyn.prewarm(1024);

    let input = generate_sine(440.0, 48000.0, 128);
    let mut out_static = vec![0.0f32; 128];
    let mut out_dyn = vec![0.0f32; 128];

    // Execução paralela dos modelos
    model_static.process(&input, &mut out_static);
    model_dyn.process(&input, &mut out_dyn);

    // Validação de paridade
    let mse = calculate_mse(&out_static, &out_dyn);
    println!("MSE Static vs Dynamic: {}", mse);
    assert!(
        mse < 1e-10,
        "Paridade numérica WaveNet falhou: MSE={} (Implementação Dinâmica divergiu da Estática)",
        mse
    );
}
