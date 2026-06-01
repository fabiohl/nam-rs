// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Estruturas de dados e validação para o formato `.nam` (JSON).
//!
//! Contém as structs que modelam o arquivo de modelo neural, os erros tipados
//! do parser e os visitors customizados de serde para validação de limites.

use serde::{Deserialize, Deserializer, Serialize};

/// Tamanho máximo de floats no array `weights` (MAX_MODEL_BYTES / 4).
const MAX_WEIGHTS: usize = (256 * 1024 * 1024 / 4) as usize; // 64 Mi floats

/// Tamanho máximo do campo `metadata.training` em bytes.
const MAX_TRAINING_BYTES: usize = 1024 * 1024; // 1 MiB

/// Profundidade máxima da árvore JSON em `metadata.training`.
const MAX_TRAINING_DEPTH: usize = 16;

/// Erros tipados do parser JSON `.nam`.
#[derive(Debug)]
pub enum JsonError {
    /// O array `weights` excede o limite de floats.
    WeightsExceedLimit {
        /// Quantidade de floats recebida.
        got: usize,
        /// Limite máximo configurado.
        max: usize,
    },
    /// O campo `metadata.training` excede o limite de profundidade da árvore JSON.
    TrainingTooDeep {
        /// Profundidade encontrada.
        depth: usize,
        /// Profundidade máxima permitida.
        max_depth: usize,
    },
    /// O campo `metadata.training` excede o limite de tamanho.
    TrainingTooLarge {
        /// Tamanho aproximado em bytes.
        size: usize,
        /// Tamanho máximo permitido.
        max_size: usize,
    },
    /// Erro genérico de parse do serde_json.
    Serde(String),
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightsExceedLimit { got, max } => {
                write!(
                    f,
                    "weights array exceeds limit ({} floats, max is {})",
                    got, max
                )
            }
            Self::TrainingTooDeep { depth, max_depth } => {
                write!(
                    f,
                    "metadata.training JSON tree too deep (depth {}, max is {})",
                    depth, max_depth
                )
            }
            Self::TrainingTooLarge { size, max_size } => {
                write!(
                    f,
                    "metadata.training JSON too large ({} bytes, max is {} bytes)",
                    size, max_size
                )
            }
            Self::Serde(msg) => write!(f, "JSON parse error: {}", msg),
        }
    }
}

impl std::error::Error for JsonError {}

impl From<serde_json::Error> for JsonError {
    fn from(e: serde_json::Error) -> Self {
        JsonError::Serde(e.to_string())
    }
}

/// Visitor custom para `Vec<f32>` que aborta ao exceder MAX_WEIGHTS floats.
struct WeightsVisitor;

impl<'de> serde::de::Visitor<'de> for WeightsVisitor {
    type Value = Vec<f32>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence of f32 floats within the size limit")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Vec<f32>, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut weights = Vec::new();
        loop {
            match seq.next_element::<f32>() {
                Ok(Some(val)) => {
                    if weights.len() >= MAX_WEIGHTS {
                        return Err(serde::de::Error::custom(JsonError::WeightsExceedLimit {
                            got: weights.len() + 1,
                            max: MAX_WEIGHTS,
                        }));
                    }
                    weights.push(val);
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(weights)
    }
}

/// Custom deserializer para `weights: Vec<f32>` com cap em MAX_WEIGHTS.
fn deserialize_weights<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(WeightsVisitor)
}

/// Visitor da árvore JSON para `metadata.training` com limites de profundidade e tamanho.
/// Usa `std::cell::Cell<usize>` para que child visitors compartilhem o contador
/// de tamanho com o parent, evitando bypass do limite agregado de 1 MiB.
struct LimitedValueVisitor {
    depth: usize,
    max_depth: usize,
    max_size: usize,
    current_size: std::cell::Cell<usize>,
}

impl LimitedValueVisitor {
    fn root(max_depth: usize, max_size: usize) -> Self {
        Self {
            depth: 0,
            max_depth,
            max_size,
            current_size: std::cell::Cell::new(0),
        }
    }

    fn child(&self) -> Self {
        Self {
            depth: self.depth + 1,
            max_depth: self.max_depth,
            max_size: self.max_size,
            current_size: self.current_size.clone(),
        }
    }

    fn add_size(&self, bytes: usize) -> Result<(), serde_json::Error> {
        let new = self.current_size.get() + bytes;
        self.current_size.set(new);
        if new > self.max_size {
            return Err(serde::de::Error::custom(JsonError::TrainingTooLarge {
                size: new,
                max_size: self.max_size,
            }));
        }
        Ok(())
    }

    fn check_depth(&self) -> Result<(), serde_json::Error> {
        if self.depth > self.max_depth {
            return Err(serde::de::Error::custom(JsonError::TrainingTooDeep {
                depth: self.depth,
                max_depth: self.max_depth,
            }));
        }
        Ok(())
    }
}

impl<'de> serde::de::Visitor<'de> for LimitedValueVisitor {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON value within depth and size limits")
    }

    fn visit_bool<E>(self, v: bool) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(if v { 4 } else { 5 }).map_err(E::custom)?;
        Ok(serde_json::Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(serde_json::Number::from(v)))
    }

    fn visit_u64<E>(self, v: u64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(serde_json::Number::from(v)))
    }

    fn visit_f64<E>(self, v: f64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(
            serde_json::Number::from_f64(v).unwrap_or(serde_json::Number::from(0)),
        ))
    }

    fn visit_str<E>(self, v: &str) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(v.len() + 2).map_err(E::custom)?;
        Ok(serde_json::Value::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.add_size(v.len() + 2).map_err(E::custom)?;
        Ok(serde_json::Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<serde_json::Value, E> {
        Ok(serde_json::Value::Null)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<serde_json::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        self.check_depth().map_err(serde::de::Error::custom)?;
        self.add_size(2).map_err(serde::de::Error::custom)?; // [ ]
        let mut arr = Vec::new();
        loop {
            match seq.next_element_seed(self.child()) {
                Ok(Some(val)) => {
                    self.add_size(1).map_err(serde::de::Error::custom)?; // comma
                    arr.push(val);
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(serde_json::Value::Array(arr))
    }

    fn visit_map<A>(self, mut map: A) -> Result<serde_json::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        self.check_depth().map_err(serde::de::Error::custom)?;
        self.add_size(2).map_err(serde::de::Error::custom)?; // { }
        let mut obj = serde_json::Map::new();
        loop {
            match map.next_key::<String>() {
                Ok(Some(key)) => {
                    let key_len = key.len() + 4; // quotes and colon
                    self.add_size(key_len).map_err(serde::de::Error::custom)?;
                    let val: serde_json::Value = map.next_value_seed(self.child())?;
                    obj.insert(key, val);
                    self.add_size(1).map_err(serde::de::Error::custom)?; // comma
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(serde_json::Value::Object(obj))
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for LimitedValueVisitor {
    type Value = serde_json::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

/// Visitor externo para `Option<serde_json::Value>`: retorna `None` para null/ausente,
/// e `Some(value)` com limites de profundidade/tamanho para valores presentes.
struct TrainingOptionVisitor;

impl<'de> serde::de::Visitor<'de> for TrainingOptionVisitor {
    type Value = Option<serde_json::Value>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an optional JSON value")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(None)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(None)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let visitor = LimitedValueVisitor::root(MAX_TRAINING_DEPTH, MAX_TRAINING_BYTES);
        let value: serde_json::Value = deserializer.deserialize_any(visitor)?;
        Ok(Some(value))
    }
}

/// Custom deserializer para `metadata.training` com limites de profundidade e tamanho.
fn deserialize_training<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(TrainingOptionVisitor)
}

/// Estrutura que representa uma data e hora associada aos metadados do modelo.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct NamDate {
    /// Ano.
    pub year: Option<i32>,
    /// Mês.
    pub month: Option<i32>,
    /// Dia.
    pub day: Option<i32>,
    /// Hora.
    pub hour: Option<i32>,
    /// Minuto.
    pub minute: Option<i32>,
    /// Segundo.
    pub second: Option<i32>,
}

/// Metadados opcionais contidos no fim do formato `.nam`.
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct NamMetadata {
    /// Data de autoria ou exportação do modelo.
    pub date: Option<NamDate>,
    /// O nome do modelo.
    pub name: Option<String>,
    /// Quem fez/treinou o modelo.
    pub modeled_by: Option<String>,
    /// Fabricante do equipamento original (Ex: Fender).
    pub gear_make: Option<String>,
    /// O modelo do equipamento original (Ex: Deluxe Reverb).
    pub gear_model: Option<String>,
    /// Que tipo de equipamento é este? Opções: "amp", "pedal", "pedal_amp", "amp_cab", "amp_pedal_cab", "preamp" e "studio".
    pub gear_type: Option<String>,
    /// De qual estilo do equipamento? Opções: "clean", "overdrive", "crunch", "hi_gain" e "fuzz".
    pub tone_type: Option<String>,
    /// Informação opcional de documentação sobre configuração Pydantic de treinamento.
    #[serde(default, deserialize_with = "deserialize_training")]
    pub training: Option<serde_json::Value>,
    /// Nível de entrada esperado pelo modelo (dBu). Usado no gain staging de entrada.
    pub input_level_dbu: Option<f32>,
    /// Nível de saída esperado pelo modelo (dBu). Usado no gain staging de saída.
    pub output_level_dbu: Option<f32>,
    /// Loudness geral gravado.
    pub loudness: Option<f32>,
}

/// A configuração estrutural de uma única camada (layer) da rede (seja WaveNet ou LSTM).
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NamLayerConfig {
    /// Opcional: Tamanho do tensor de entrada.
    pub input_size: Option<usize>,
    /// Opcional: Tamanho do tensor de condicionamento (ex: parâmetros externos).
    pub condition_size: Option<usize>,
    /// Opcional: Tamanho do tensor de saída (head size).
    pub head_size: Option<usize>,
    /// Opcional: Quantidade de canais internos (ex: 16 ou 24).
    pub channels: Option<usize>,
    /// Opcional: Tamanho do kernel convolucional.
    pub kernel_size: Option<usize>,
    /// Opcional: Array de fatores de dilatação.
    pub dilations: Option<Vec<usize>>,
    /// Opcional: Função de ativação (ex: "Tanh").
    pub activation: Option<String>,
    /// Opcional: Se a arquitetura usa portas (gating).
    pub gated: Option<bool>,
    /// Opcional: Se a cabeça de processamento possui bias.
    pub head_bias: Option<bool>,
}

/// Opções de layout de pesos suportadas no formato `.namb`.
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WeightsLayout {
    /// Layout original (NAM padrão): [Gate][H][IH] para LSTM, [OUT][IN][K] para Conv1D.
    #[default]
    Original = 0,
    /// Layout otimizado para LSTM: [Gate][IH][H].
    GateMajorLstm = 1,
    /// Layout otimizado para WaveNet: Intercalado 4-Wide ([OUT/4][K][IN][4]).
    Interleaved4WaveNet = 2,
}

/// A configuração interna do nó da arquitetura no JSON.
#[derive(Deserialize, Serialize, Debug, Clone)]
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
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NamModelData {
    /// Versão no cabeçalho do JSON (ex: "0.5.4")
    pub version: Option<String>,
    /// Tipo da arquitetura declarada ("WaveNet" ou "LSTM")
    pub architecture: String,
    /// Configuração estrutural de hiperparâmetros
    pub config: NamConfig,
    /// Os imensos tensores Float32 planificados em formato SoA.
    #[serde(deserialize_with = "deserialize_weights")]
    pub weights: Vec<f32>,
    /// Frequência de amostragem original projetada pela modelagem (referência sempre ideal 48 kHz).
    pub sample_rate: Option<f32>,
    /// Propriedades físico-acústicas extras associadas.
    pub metadata: Option<NamMetadata>,
    /// Layout dos pesos (usado apenas no formato binário .namb v2+).
    #[serde(skip)]
    pub weights_layout: WeightsLayout,
}
