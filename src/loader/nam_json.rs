// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Parser para o formato .nam (JSON)
//!
//! Realiza o carregamento dos tensores e metadados fora do caminho RT.

use serde::Deserialize;

/// Metadados opcionais contidos no fim do formato `.nam`.
#[derive(Deserialize, Debug, Clone)]
pub struct NamMetadata {
    /// Nível de entrada esperado pelo modelo (dBu). Usado no gain staging de entrada.
    pub input_level_dbu: Option<f32>,
    /// Nível de saída esperado pelo modelo (dBu). Usado no gain staging de saída.
    pub output_level_dbu: Option<f32>,
    /// Loudness geral gravado.
    pub loudness: Option<f32>,
}

/// A configuração estrutural de uma única camada (layer) da rede (seja WaveNet ou LSTM).
#[derive(Deserialize, Debug, Clone)]
pub struct NamLayerConfig {
    /// O tamanho da entrada.
    pub input_size: Option<usize>,
    /// Tamanho do condicionamento extra.
    pub condition_size: Option<usize>,
    /// O tamanho da dimensão de projeção de saída ("head" local).
    pub head_size: Option<usize>,
    /// A quantidade de canais convolutivos.
    pub channels: Option<usize>,
    /// Tamanho do kernel.
    pub kernel_size: Option<usize>,
    /// Lista de dilatações causais desta camada.
    pub dilations: Option<Vec<usize>>,
    /// Função de ativação (ex: "Tanh").
    pub activation: Option<String>,
    /// Se possui `gates` internamente.
    pub gated: Option<bool>,
    /// Se há bias acoplado à projeção dessa camada.
    pub head_bias: Option<bool>,
}

/// A configuração interna do nó da arquitetura no JSON.
#[derive(Deserialize, Debug, Clone)]
pub struct NamConfig {
    /// Lista das configurações das camadas empilhadas (presente em WaveNet, ausente em LSTM).
    #[serde(default)]
    pub layers: Vec<NamLayerConfig>,
    /// Uma possível string auxiliar pra head final. Se null no JSON, pode faltar.
    pub head: Option<std::option::Option<String>>,
    /// Escala fina sobre o somatório da rede.
    pub head_scale: Option<f32>,
    /// Número de layers (para LSTMs no C++ é count das layers, ou explícito)
    pub num_layers: Option<usize>,
    /// Tamanho oculto da célula LSTMs
    pub hidden_size: Option<usize>,
}

/// Estrutura raiz de mapeamento dos arquivos `.nam`.
#[derive(Deserialize, Debug, Clone)]
pub struct NamModelData {
    /// Versão no cabeçalho do JSON (ex: "0.5.4")
    pub version: Option<String>,
    /// Tipo da arquitetura declarada ("WaveNet" ou "LSTM")
    pub architecture: String,
    /// Configuração estrutural de hiperparâmetros
    pub config: NamConfig,
    /// Os imensos tensores Float32 planificados em formato SoA.
    pub weights: Vec<f32>,
    /// Frequência de amostragem original projetada pela modelagem (referência sempre ideal 48 kHz).
    pub sample_rate: Option<f32>,
    /// Propriedades físico-acústicas extras associadas.
    pub metadata: Option<NamMetadata>,
}

/// As Topologias fechadas e suportadas dentro da modelagem WaveNet nativa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamWavenetTopology {
    /// Canais: 16 (Standard)
    Standard,
    /// Canais: 12 (Lite)
    Lite,
    /// Canais: 8 (Feather)
    Feather,
    /// Canais: 4 (Nano)
    Nano,
}

static STD_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
static LITE_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
static LITE_DILATIONS_2: &[usize] = &[128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// Desserialização universal bruta da string do JSON via `serde_json`.
pub fn parse_nam_json(json_str: &str) -> anyhow::Result<NamModelData> {
    let data: NamModelData = serde_json::from_str(json_str)?;
    Ok(data)
}

/// Baseando-se no NeuralModel.cpp (`L:155-218`), verifica a identidade estática da topologia WaveNet.
pub fn get_wavenet_topology(data: &NamModelData) -> Option<NamWavenetTopology> {
    if data.architecture != "WaveNet" || data.config.layers.len() != 2 {
        return None;
    }

    let l0 = &data.config.layers[0];
    let l1 = &data.config.layers[1];

    let l0_gated = l0.gated.unwrap_or(false);
    let l1_gated = l1.gated.unwrap_or(false);
    let l0_head_bias = l0.head_bias.unwrap_or(false);
    let l1_head_bias = l1.head_bias.unwrap_or(false);

    if l0_gated || l1_gated || l0_head_bias || !l1_head_bias {
        return None;
    }

    let channels = l0.channels?;
    let dils_0 = l0.dilations.as_deref()?;
    let dils_1 = l1.dilations.as_deref()?;

    match channels {
        16 if dils_0 == STD_DILATIONS && dils_1 == STD_DILATIONS => {
            Some(NamWavenetTopology::Standard)
        }
        12 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Lite)
        }
        8 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Feather)
        }
        4 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Nano)
        }
        _ => None,
    }
}

/// Verifica e retorna a geometria do LSTM (num_layers, hidden_size).
pub fn get_lstm_topology(data: &NamModelData) -> Option<(usize, usize)> {
    if data.architecture != "LSTM" {
        return None;
    }

    let num_layers = data.config.num_layers?;
    let hidden_size = data.config.hidden_size?;
    Some((num_layers, hidden_size))
}

#[cfg(test)]
mod tests {
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
}
