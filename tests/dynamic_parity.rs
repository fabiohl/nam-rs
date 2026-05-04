// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use nam_rs::loader::dispatcher::{build_model, build_wavenet_dynamic};
use nam_rs::loader::nam_json::{NamConfig, NamModelData};
use nam_rs::models::NamModel;

fn generate_sine(freq: f32, sr: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect()
}

fn calculate_mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    sum / a.len() as f32
}

#[test]
fn test_wavenet_dynamic_parity() {
    // Pesos: 0.01 fixo para tudo
    // CH16, K3: Array1.rechannel (1*16) + layers (10 * (16*3*16 + 16 + 1*16 + 16*16 + 16)) + head_rechan (16*8)
    // Array2... + head_scale (1)
    // Total aproximado de pesos para Standard: ~284,000 floats.
    // Vamos usar um modelo menor para o teste de paridade ser rápido.
    let dils_short = vec![1, 2, 4];
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

    // Calculo manual do total de pesos:
    // Array 1:
    //   rechannel: 1 * 8 = 8
    //   layers (3): 3 * (8*3*8 + 8 [conv] + 1*8 [mixin] + 8*8 + 8 [1x1]) = 3 * (192+8+8+64+8) = 3 * 280 = 840
    //   head_rechannel: 8 * 4 = 32
    // Array 2:
    //   rechannel: 8 * 4 = 32
    //   layers (3): 3 * (4*3*4 + 4 + 1*4 + 4*4 + 4) = 3 * (48+4+4+16+4) = 3 * 76 = 228
    //   head_rechannel: 4 * 1 + 1 (bias) = 5
    // Scale: 1
    // Total: 8 + 840 + 32 + 32 + 228 + 5 + 1 = 1146
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

    let mut model_static = build_model(&data).expect("Falha ao construir modelo estático");
    let mut model_dyn = build_wavenet_dynamic(&data).expect("Falha ao construir modelo dinâmico");

    model_static.prewarm(1024);
    model_dyn.prewarm(1024);

    let input = generate_sine(440.0, 48000.0, 128);
    let mut out_static = vec![0.0f32; 128];
    let mut out_dyn = vec![0.0f32; 128];

    model_static.process(&input, &mut out_static);
    model_dyn.process(&input, &mut out_dyn);

    let mse = calculate_mse(&out_static, &out_dyn);
    println!("MSE Static vs Dynamic: {}", mse);
    assert!(mse < 1e-10, "Paridade numérica WaveNet falhou: MSE={}", mse);
}
