// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model loading and building — reads `.nam`/`.namb` files, parses, calibrates,
//! and dispatches to the appropriate architecture builder.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::loader::{dispatcher, nam_json, namb};
use crate::models::NamModel;
use std::path::Path;

use super::loaded_model_pair::{
    DEFAULT_INPUT_LEVEL_DBU, DEFAULT_LOUDNESS_DB, DEFAULT_SAMPLE_RATE, LoadedModelPair,
    MAX_MODEL_BYTES,
};

/// Loads and builds a model pair from a file.
///
/// When `stereo` is `false` only the left-channel model is built;
/// `model_r` is left as `None`, avoiding wasted build time and prewarming.
pub fn load_and_build_model(
    path: &Path,
    sys: &SystemSnapshot,
    stereo: bool,
) -> anyhow::Result<LoadedModelPair> {
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
                Some(namb::NambError::WeightsTooLarge { .. }) => NamErrorCode::ModelTooLarge,
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
                nam_json::JsonError::SubmodelsExceedLimit { .. } => {
                    NamErrorCode::NamJsonSubmodelsExceedLimit
                }
                nam_json::JsonError::SubmodelsTooDeep { .. } => {
                    NamErrorCode::NamJsonSubmodelsTooDeep
                }
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

    let model_r = if stereo {
        let mut m = dispatcher::build_model(&model_data)
            .inspect_err(|e| {
                NamDiagnostic::new(NamErrorCode::ModelBuildFailed, sys)
                    .message(format!("Failed to build model (R): {}", path_str))
                    .param("detail", e.to_string())
                    .emit();
            })
            .ok();
        if let Some(ref mut model) = m {
            model.prewarm(model.prewarm_samples().max(2048));
        }
        m
    } else {
        None
    };

    let architecture = model_data.architecture.clone();
    let topology = if architecture == "WaveNet" {
        match nam_json::get_wavenet_topology(&model_data) {
            Some(nam_json::NamWavenetTopology::Standard) => "Standard".to_string(),
            Some(nam_json::NamWavenetTopology::Lite) => "Lite".to_string(),
            Some(nam_json::NamWavenetTopology::Feather) => "Feather".to_string(),
            Some(nam_json::NamWavenetTopology::Nano) => "Nano".to_string(),
            None => {
                if let Some(ch) = nam_json::is_a2_shape(&model_data) {
                    match ch {
                        3 => "A2-Lite".to_string(),
                        8 => "A2-Full".to_string(),
                        _ => "A2-Custom".to_string(),
                    }
                } else if architecture == "SlimmableContainer" {
                    "Container".to_string()
                } else {
                    "Custom".to_string()
                }
            }
        }
    } else if architecture == "LSTM" {
        match nam_json::get_lstm_topology(&model_data) {
            Some((layers, hidden)) => format!("{}x{}", layers, hidden),
            None => "Custom".to_string(),
        }
    } else if architecture == "Linear" {
        match nam_json::get_linear_topology(&model_data) {
            Some((rf, has_bias)) => {
                if has_bias {
                    format!("RF{} (biased)", rf)
                } else {
                    format!("RF{}", rf)
                }
            }
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
