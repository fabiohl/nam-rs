// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Data structures for the `.nam` format (JSON).
//!
//! Contains the structs that model the neural model file.

use serde::{Deserialize, Serialize};

use super::validation::{deserialize_training, deserialize_weights};

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
