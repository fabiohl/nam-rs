// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Data structures and validation for the `.nam` format (JSON).
//!
//! Contains the structs that model the neural model file, typed parser errors,
//! and custom serde visitors for limit validation.

use serde::{Deserialize, Deserializer, Serialize};

/// Maximum number of floats in the `weights` array (MAX_MODEL_BYTES / 4).
const MAX_WEIGHTS: usize = (256 * 1024 * 1024 / 4) as usize; // 64 Mi floats

/// Maximum size of the `metadata.training` field in bytes.
const MAX_TRAINING_BYTES: usize = 1024 * 1024; // 1 MiB

/// Maximum depth of the JSON tree in `metadata.training`.
const MAX_TRAINING_DEPTH: usize = 16;

/// Typed errors of the `.nam` JSON parser.
#[derive(Debug)]
pub enum JsonError {
    /// The `weights` array exceeds the float limit.
    WeightsExceedLimit {
        /// Number of floats received.
        got: usize,
        /// Maximum configured limit.
        max: usize,
    },
    /// The `metadata.training` field exceeds the JSON tree depth limit.
    TrainingTooDeep {
        /// Depth found.
        depth: usize,
        /// Maximum allowed depth.
        max_depth: usize,
    },
    /// The `metadata.training` field exceeds the size limit.
    TrainingTooLarge {
        /// Approximate size in bytes.
        size: usize,
        /// Maximum allowed size.
        max_size: usize,
    },
    /// Generic serde_json parse error.
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

/// Custom visitor for `Vec<f32>` that aborts upon exceeding MAX_WEIGHTS floats.
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

/// Custom deserializer for `weights: Vec<f32>` with cap at MAX_WEIGHTS.
fn deserialize_weights<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(WeightsVisitor)
}

/// JSON tree visitor for `metadata.training` with depth and size limits.
/// Uses `std::cell::Cell<usize>` so that child visitors share the size counter
/// with the parent, avoiding bypass of the 1 MiB aggregate limit.
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

/// External visitor for `Option<serde_json::Value>`: returns `None` for null/absent,
/// and `Some(value)` with depth/size limits for present values.
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

/// Custom deserializer for `metadata.training` with depth and size limits.
fn deserialize_training<'de, D>(deserializer: D) -> Result<Option<serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(TrainingOptionVisitor)
}

/// Structure representing a date and time associated with the model's metadata.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, Eq, Default)]
pub struct NamDate {
    /// Year.
    pub year: Option<i32>,
    /// Month.
    pub month: Option<i32>,
    /// Day.
    pub day: Option<i32>,
    /// Hour.
    pub hour: Option<i32>,
    /// Minute.
    pub minute: Option<i32>,
    /// Second.
    pub second: Option<i32>,
}

/// Optional metadata contained at the end of the `.nam` format.
#[derive(Deserialize, Serialize, Debug, Clone, Default)]
pub struct NamMetadata {
    /// Model authorship or export date.
    pub date: Option<NamDate>,
    /// The model name.
    pub name: Option<String>,
    /// Who made/trained the model.
    pub modeled_by: Option<String>,
    /// Manufacturer of the original equipment (e.g. Fender).
    pub gear_make: Option<String>,
    /// The model of the original equipment (e.g. Deluxe Reverb).
    pub gear_model: Option<String>,
    /// What type of equipment is this? Options: "amp", "pedal", "pedal_amp", "amp_cab", "amp_pedal_cab", "preamp", and "studio".
    pub gear_type: Option<String>,
    /// What style of equipment? Options: "clean", "overdrive", "crunch", "hi_gain", and "fuzz".
    pub tone_type: Option<String>,
    /// Optional documentation about Pydantic training configuration.
    #[serde(default, deserialize_with = "deserialize_training")]
    pub training: Option<serde_json::Value>,
    /// Expected input level for the model (dBu). Used in input gain staging.
    pub input_level_dbu: Option<f32>,
    /// Expected output level for the model (dBu). Used in output gain staging.
    pub output_level_dbu: Option<f32>,
    /// Overall recorded loudness.
    pub loudness: Option<f32>,
}

/// The structural configuration of a single layer of the network (whether WaveNet or LSTM).
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NamLayerConfig {
    /// Optional: Input tensor size.
    pub input_size: Option<usize>,
    /// Optional: Conditioning tensor size (e.g. external parameters).
    pub condition_size: Option<usize>,
    /// Optional: Output tensor size (head size).
    pub head_size: Option<usize>,
    /// Optional: Number of internal channels (e.g. 16 or 24).
    pub channels: Option<usize>,
    /// Optional: Convolutional kernel size.
    pub kernel_size: Option<usize>,
    /// Optional: Array of dilation factors.
    pub dilations: Option<Vec<usize>>,
    /// Optional: Activation function (e.g. "Tanh").
    pub activation: Option<String>,
    /// Optional: Whether the architecture uses gating.
    pub gated: Option<bool>,
    /// Optional: Whether the processing head has bias.
    pub head_bias: Option<bool>,
}

/// Weight layout options supported in the `.namb` format.
#[derive(serde::Deserialize, serde::Serialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum WeightsLayout {
    /// Original layout (standard NAM): [Gate][H][IH] for LSTM, [OUT][IN][K] for Conv1D.
    #[default]
    Original = 0,
    /// Layout optimized for LSTM: [Gate][IH][H].
    GateMajorLstm = 1,
    /// Layout optimized for WaveNet: Interleaved 4-Wide ([OUT/4][K][IN][4]).
    Interleaved4WaveNet = 2,
}

/// The internal configuration of the architecture node in the JSON.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NamConfig {
    /// List of stacked layer configurations (present in WaveNet, absent in LSTM).
    #[serde(default)]
    pub layers: Vec<NamLayerConfig>,
    /// A possible auxiliary string for the final head. If null in JSON, it may be absent.
    pub head: Option<std::option::Option<String>>,
    /// Fine scale over the network summation.
    pub head_scale: Option<f32>,
    /// Number of layers (for LSTMs in C++ it is the layer count, or explicit)
    pub num_layers: Option<usize>,
    /// Hidden size of the LSTM cell
    pub hidden_size: Option<usize>,
}

/// Root mapping structure for `.nam` files.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NamModelData {
    /// Version in the JSON header (e.g. "0.5.4")
    pub version: Option<String>,
    /// Declared architecture type ("WaveNet" or "LSTM")
    pub architecture: String,
    /// Structural configuration of hyperparameters
    pub config: NamConfig,
    /// The huge Float32 tensors flattened in SoA format.
    #[serde(deserialize_with = "deserialize_weights")]
    pub weights: Vec<f32>,
    /// Original sample rate projected by the modeling (always ideal reference 48 kHz).
    pub sample_rate: Option<f32>,
    /// Extra physical-acoustic properties associated.
    pub metadata: Option<NamMetadata>,
    /// Weight layout (used only in the .namb v2+ binary format).
    #[serde(skip)]
    pub weights_layout: WeightsLayout,
}
