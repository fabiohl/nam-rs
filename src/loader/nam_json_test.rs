// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::*;

#[test]
fn test_parse_feather_wavenet() {
    let json_str = r#"{
        "version": "0.5.4",
        "architecture": "WaveNet",
        "config": {
            "layers": [
                {
                    "input_size": 1, "condition_size": 1, "head_size": 4,
                    "channels": 8, "kernel_size": 3, "dilations": [1,2,4,8,16,32,64],
                    "activation": "Tanh", "gated": false, "head_bias": false
                },
                {
                    "input_size": 1, "condition_size": 1, "head_size": 4,
                    "channels": 8, "kernel_size": 3, "dilations": [128,256,512,1,2,4,8,16,32,64,128,256,512],
                    "activation": "Tanh", "gated": false, "head_bias": true
                }
            ],
            "head": null,
            "head_scale": 0.02
        },
        "weights": [0.0123, -0.456, 1.0, 2.0],
        "sample_rate": 48000,
        "metadata": {
            "name": "Super Twin",
            "modeled_by": "John Doe",
            "gear_make": "Fender",
            "input_level_dbu": 12.0,
            "output_level_dbu": 11.5,
            "loudness": -18.0
        }
    }"#;

    let parsed = parse_nam_json(json_str).expect("Falha ao efetuar parse do NAM JSON simulado");
    assert_eq!(parsed.architecture, "WaveNet");
    assert_eq!(parsed.weights.len(), 4);
    assert_eq!(parsed.sample_rate.unwrap(), 48000.0);
    let meta = parsed.metadata.as_ref().unwrap();
    assert_eq!(meta.input_level_dbu.unwrap(), 12.0);
    assert_eq!(meta.output_level_dbu.unwrap(), 11.5);
    assert_eq!(meta.loudness.unwrap(), -18.0);

    assert_eq!(meta.name.as_deref(), Some("Super Twin"));
    assert_eq!(meta.modeled_by.as_deref(), Some("John Doe"));
    assert_eq!(meta.gear_make.as_deref(), Some("Fender"));

    let topo = get_wavenet_topology(&parsed);
    assert_eq!(topo, Some(NamWavenetTopology::Feather));
}

#[test]
fn test_parse_lstm() {
    let json_str = r#"{
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": {
            "num_layers": 2,
            "hidden_size": 16,
            "layers": []
        },
        "weights": [0.1, 0.2]
    }"#;

    let parsed = parse_nam_json(json_str).expect("Falha ao efetuar parse de LSTM NAM JSON");
    assert_eq!(parsed.architecture, "LSTM");
    let topo = get_lstm_topology(&parsed);
    assert_eq!(topo, Some((2, 16)));
}

/// Helper: gera JSON mínimo de WaveNet com canais e dilatações fornecidos.
fn make_wavenet_json(channels: usize, dils_0: &[usize], dils_1: &[usize]) -> String {
    let d0: Vec<String> = dils_0.iter().map(|d| d.to_string()).collect();
    let d1: Vec<String> = dils_1.iter().map(|d| d.to_string()).collect();
    format!(
        r#"{{
            "architecture": "WaveNet",
            "config": {{
                "layers": [
                    {{
                        "channels": {channels}, "kernel_size": 3,
                        "dilations": [{}],
                        "gated": false, "head_bias": false
                    }},
                    {{
                        "channels": {channels}, "kernel_size": 3,
                        "dilations": [{}],
                        "gated": false, "head_bias": true
                    }}
                ],
                "head": null, "head_scale": 0.02
            }},
            "weights": [0.0]
        }}"#,
        d0.join(","),
        d1.join(",")
    )
}

#[test]
fn test_topology_standard() {
    let std_d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let json = make_wavenet_json(16, &std_d, &std_d);
    let parsed = parse_nam_json(&json).unwrap();
    assert_eq!(
        get_wavenet_topology(&parsed),
        Some(NamWavenetTopology::Standard)
    );
}

#[test]
fn test_topology_lite() {
    let d0 = [1, 2, 4, 8, 16, 32, 64];
    let d1 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let json = make_wavenet_json(12, &d0, &d1);
    let parsed = parse_nam_json(&json).unwrap();
    assert_eq!(
        get_wavenet_topology(&parsed),
        Some(NamWavenetTopology::Lite)
    );
}

#[test]
fn test_topology_nano() {
    let d0 = [1, 2, 4, 8, 16, 32, 64];
    let d1 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let json = make_wavenet_json(4, &d0, &d1);
    let parsed = parse_nam_json(&json).unwrap();
    assert_eq!(
        get_wavenet_topology(&parsed),
        Some(NamWavenetTopology::Nano)
    );
}

#[test]
fn test_topology_invalid_channels() {
    let std_d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let json = make_wavenet_json(10, &std_d, &std_d);
    let parsed = parse_nam_json(&json).unwrap();
    assert_eq!(
        get_wavenet_topology(&parsed),
        None,
        "Canais 10 não é uma topologia suportada"
    );
}

// =========================================================================
// Testes de Rejeição de JSON Malformado
// =========================================================================

/// JSON truncado no meio deve retornar `Err`.
#[test]
fn test_parse_truncated_json() {
    let truncated = r#"{"version": "0.5.4", "architecture": "WaveNet", "config": {"#;
    let result = parse_nam_json(truncated);
    assert!(
        result.is_err(),
        "JSON truncado deve retornar Err, mas obteve Ok"
    );
}

/// JSON válido sem o campo obrigatório `"architecture"` deve retornar `Err`.
#[test]
fn test_parse_missing_architecture() {
    let json = r#"{
        "version": "0.5.4",
        "config": { "layers": [] },
        "weights": [0.1, 0.2]
    }"#;
    let result = parse_nam_json(json);
    assert!(
        result.is_err(),
        "JSON sem 'architecture' deve retornar Err, mas obteve Ok"
    );
}

/// JSON válido sem o campo obrigatório `"weights"` deve retornar `Err`.
#[test]
fn test_parse_missing_weights() {
    let json = r#"{
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": { "num_layers": 1, "hidden_size": 8, "layers": [] }
    }"#;
    let result = parse_nam_json(json);
    assert!(
        result.is_err(),
        "JSON sem 'weights' deve retornar Err, mas obteve Ok"
    );
}

/// `"weights": []` deve ser aceito pelo parser (array vazia é JSON válido).
/// O dispatcher é responsável por rejeitar modelos com 0 pesos posteriormente.
#[test]
fn test_parse_empty_weights() {
    let json = r#"{
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": { "num_layers": 1, "hidden_size": 8, "layers": [] },
        "weights": []
    }"#;
    let result = parse_nam_json(json);
    assert!(
        result.is_ok(),
        "JSON com weights vazio deve ser aceito pelo parser (dispatcher rejeita depois)"
    );
    let data = result.unwrap();
    assert_eq!(data.weights.len(), 0);
}

/// `"config": "not_an_object"` deve retornar `Err` (tipo incorreto).
#[test]
fn test_parse_malformed_config() {
    let json = r#"{
        "version": "0.5.4",
        "architecture": "WaveNet",
        "config": "not_an_object",
        "weights": [0.1]
    }"#;
    let result = parse_nam_json(json);
    assert!(
        result.is_err(),
        "JSON com config como string deve retornar Err, mas obteve Ok"
    );
}
