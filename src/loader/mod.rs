// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loading module for the NAM ecosystem.
//!
//! Contains parsers for .nam (JSON) and .namb (binary) formats.
//! The entire loading process occurs **outside** the RT thread to
//! avoid any unwanted allocation during audio processing.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};

use crate::models::{DynamicModel, NamModel};
use std::path::Path;

pub mod dispatcher;
pub mod nam_json;
pub mod namb;
pub mod namb_encoder;
pub mod transpose;

/// Default input level in dBu for models that do not specify metadata.
const DEFAULT_INPUT_LEVEL_DBU: f32 = 12.0;
/// Default reference loudness in dB for normalization.
const DEFAULT_LOUDNESS_DB: f32 = -18.0;
/// Default reference sample rate (NAM standard).
const DEFAULT_SAMPLE_RATE: f32 = 48000.0;
/// Maximum allowed model file size (256 MiB).
const MAX_MODEL_BYTES: u64 = 256 * 1024 * 1024;

/// Pair of loaded models with calibration metadata.
pub struct LoadedModelPair {
    /// Model for the left channel.
    pub model_l: Option<Box<DynamicModel>>,
    /// Model for the right channel.
    pub model_r: Option<Box<DynamicModel>>,
    /// Input gain adjustment multiplier.
    pub input_mult_adj: f32,
    /// Output gain adjustment multiplier.
    pub output_mult_adj: f32,
    /// Native sample rate of the model.
    pub sample_rate: u32,
    /// Model architecture (e.g. "LSTM", "WaveNet").
    pub architecture: String,
    /// Model topology (e.g. "Standard", "1x64").
    pub topology: String,
    /// Optional model metadata.
    pub metadata: Option<crate::loader::nam_json::NamMetadata>,
    /// Weights layout format.
    pub weights_layout: String,
}

impl std::fmt::Debug for LoadedModelPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModelPair")
            .field("model_l", &self.model_l.as_ref().map(|_| "DynamicModel"))
            .field("model_r", &self.model_r.as_ref().map(|_| "DynamicModel"))
            .field("input_mult_adj", &self.input_mult_adj)
            .field("output_mult_adj", &self.output_mult_adj)
            .field("sample_rate", &self.sample_rate)
            .field("architecture", &self.architecture)
            .field("topology", &self.topology)
            .field("metadata", &self.metadata)
            .field("weights_layout", &self.weights_layout)
            .finish()
    }
}

/// Loads and builds a pair of models (L+R) from a file.
pub fn load_and_build_model(path: &Path, sys: &SystemSnapshot) -> anyhow::Result<LoadedModelPair> {
    let path_str = path.to_string_lossy();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let ext_lower = ext.to_lowercase();

    // 1. Reading and Parsing
    let model_data = if ext_lower == "namb" {
        let len = std::fs::metadata(path)
            .map_err(|e| {
                NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                    .message(format!("Failed to read metadata of \"{}\".", path_str))
                    .hint("Please verify file access permissions.")
                    .param("file", &path_str)
                    .param("io_error", &e)
                    .emit();
                anyhow::Error::from(e)
            })?
            .len();
        if len > MAX_MODEL_BYTES {
            NamDiagnostic::new(NamErrorCode::ModelTooLarge, sys)
                .message(format!(
                    "Model file \"{}\" is too large ({} bytes, max is {} bytes).",
                    path_str, len, MAX_MODEL_BYTES
                ))
                .hint("Please check the file size and ensure it is a valid NAM model.")
                .param("file", &path_str)
                .param("size_bytes", len)
                .emit();
            return Err(anyhow::anyhow!(
                "Model file \"{}\" exceeds maximum allowed size of {} MiB.",
                path_str,
                MAX_MODEL_BYTES / (1024 * 1024)
            ));
        }
        let bytes = std::fs::read(path).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Failed to read the file \"{}\".", path_str))
                .hint("Please verify file access permissions.")
                .param("file", &path_str)
                .param("io_error", &e)
                .emit();
            anyhow::Error::from(e)
        })?;
        namb::parse_namb(&bytes).inspect_err(|e| {
            let code = match e.downcast_ref::<namb::NambError>() {
                Some(namb::NambError::Truncated { .. }) => NamErrorCode::NambTruncated,
                Some(namb::NambError::InvalidMagic(_)) => NamErrorCode::NambInvalidMagic,
                Some(namb::NambError::InvalidVersion(_)) => NamErrorCode::NambUnsupportedVersion,
                Some(namb::NambError::WeightsOffsetOutOfBounds { .. })
                | Some(namb::NambError::InvalidWeightsOffset { .. }) => NamErrorCode::NambTruncated,
                Some(namb::NambError::CrcMismatch { .. }) => NamErrorCode::NambCrc32Mismatch,
                Some(namb::NambError::CrcMissing { .. }) => NamErrorCode::NambCrc32Missing,
                None => NamErrorCode::ModelBuildFailed,
            };
            NamDiagnostic::new(code, sys)
                .message(format!("Invalid \".namb\" file: {}", path_str))
                .param("detail", e.to_string())
                .emit();
        })?
    } else if ext_lower == "nam" {
        let len = std::fs::metadata(path)
            .map_err(|e| {
                NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                    .message(format!("Failed to read metadata of \"{}\".", path_str))
                    .hint("Please verify file access permissions.")
                    .param("file", &path_str)
                    .param("io_error", &e)
                    .emit();
                anyhow::Error::from(e)
            })?
            .len();
        if len > MAX_MODEL_BYTES {
            NamDiagnostic::new(NamErrorCode::ModelTooLarge, sys)
                .message(format!(
                    "Model file \"{}\" is too large ({} bytes, max is {} bytes).",
                    path_str, len, MAX_MODEL_BYTES
                ))
                .hint("Please check the file size and ensure it is a valid NAM model.")
                .param("file", &path_str)
                .param("size_bytes", len)
                .emit();
            return Err(anyhow::anyhow!(
                "Model file \"{}\" exceeds maximum allowed size of {} MiB.",
                path_str,
                MAX_MODEL_BYTES / (1024 * 1024)
            ));
        }
        let json = std::fs::read_to_string(path).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Failed to read the file \"{}\".", path_str))
                .hint("Please verify file access permissions.")
                .param("file", &path_str)
                .param("io_error", &e)
                .emit();
            anyhow::Error::from(e)
        })?;
        nam_json::parse_nam_json(&json).inspect_err(|e| {
            let code = match e {
                nam_json::JsonError::WeightsExceedLimit { .. } => {
                    NamErrorCode::NamJsonWeightsExceedLimit
                }
                nam_json::JsonError::TrainingTooLarge { .. } => {
                    NamErrorCode::NamJsonTrainingTooLarge
                }
                nam_json::JsonError::TrainingTooDeep { .. } => NamErrorCode::NamJsonTrainingTooDeep,
                _ => NamErrorCode::NamJsonParseError,
            };
            NamDiagnostic::new(code, sys)
                .message(format!("Error parsing model JSON: {}", path_str))
                .param("detail", e)
                .emit();
        })?
    } else {
        return Err(anyhow::anyhow!("Unsupported file extension: {}", ext));
    };

    // 2. Metadata and Calibration Extraction
    let meta = model_data.metadata.clone().unwrap_or_default();
    let in_level = meta.input_level_dbu.unwrap_or(DEFAULT_INPUT_LEVEL_DBU);
    let loudness = meta.loudness.unwrap_or(DEFAULT_LOUDNESS_DB);

    let input_db_adj = DEFAULT_INPUT_LEVEL_DBU - in_level;
    let output_db_adj = DEFAULT_LOUDNESS_DB - loudness;
    let nam_rate = model_data.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE) as u32;

    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    let input_mult_adj = lut.db_to_linear(input_db_adj);
    let output_mult_adj = lut.db_to_linear(output_db_adj);

    // 3. Dispatcher (Build Model L/R)
    let mut model_l = dispatcher::build_model(&model_data)
        .inspect_err(|e| {
            NamDiagnostic::new(NamErrorCode::ModelBuildFailed, sys)
                .message(format!("Failed to build model (L): {}", path_str))
                .param("detail", e.to_string())
                .emit();
        })
        .ok();
    if let Some(ref mut m) = model_l {
        m.prewarm(m.prewarm_samples().max(2048));
    }

    let mut model_r = dispatcher::build_model(&model_data)
        .inspect_err(|e| {
            NamDiagnostic::new(NamErrorCode::ModelBuildFailed, sys)
                .message(format!("Failed to build model (R): {}", path_str))
                .param("detail", e.to_string())
                .emit();
        })
        .ok();
    if let Some(ref mut m) = model_r {
        m.prewarm(m.prewarm_samples().max(2048));
    }

    let architecture = model_data.architecture.clone();
    let topology = if architecture == "WaveNet" {
        match nam_json::get_wavenet_topology(&model_data) {
            Some(nam_json::NamWavenetTopology::Standard) => "Standard".to_string(),
            Some(nam_json::NamWavenetTopology::Lite) => "Lite".to_string(),
            Some(nam_json::NamWavenetTopology::Feather) => "Feather".to_string(),
            Some(nam_json::NamWavenetTopology::Nano) => "Nano".to_string(),
            None => "Custom".to_string(),
        }
    } else if architecture == "LSTM" {
        match nam_json::get_lstm_topology(&model_data) {
            Some((layers, hidden)) => format!("{}x{}", layers, hidden),
            None => "Custom".to_string(),
        }
    } else {
        "Unknown".to_string()
    };
    let metadata = model_data.metadata.clone();
    let weights_layout_str = match model_data.weights_layout {
        crate::loader::nam_json::WeightsLayout::Original => "Original".to_string(),
        crate::loader::nam_json::WeightsLayout::GateMajorLstm => "GateMajorLstm".to_string(),
        crate::loader::nam_json::WeightsLayout::Interleaved4WaveNet => {
            "Interleaved4WaveNet".to_string()
        }
    };

    Ok(LoadedModelPair {
        model_l,
        model_r,
        input_mult_adj,
        output_mult_adj,
        sample_rate: nam_rate,
        architecture,
        topology,
        metadata,
        weights_layout: weights_layout_str,
    })
}
