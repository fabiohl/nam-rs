// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Módulo de carregamento do ecossistema NAM.
//!
//! Contém os parsers dos formatos .nam (JSON) e .namb (Binário).
//! Todo o processo de carga ocorre **fora** da thread RT para
//! evitar qualquer alocação indesejada durante o processamento de áudio.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};

use crate::models::{DynamicModel, NamModel};
use std::path::Path;

pub mod dispatcher;
pub mod nam_json;
pub mod namb;
pub mod namb_encoder;

/// Nível de entrada padrão em dBu para modelos que não especificam metadados.
const DEFAULT_INPUT_LEVEL_DBU: f32 = 12.0;
/// Loudness de referência padrão em dB para normalização.
const DEFAULT_LOUDNESS_DB: f32 = -18.0;
/// Taxa de amostragem padrão de referência (NAM standard).
const DEFAULT_SAMPLE_RATE: f32 = 48000.0;
/// Tamanho máximo de arquivo de modelo permitido (256 MiB).
const MAX_MODEL_BYTES: u64 = 256 * 1024 * 1024;

/// Par de modelos carregados com metadados de calibração.
pub struct LoadedModelPair {
    /// Modelo para o canal esquerdo.
    pub model_l: Option<Box<DynamicModel>>,
    /// Modelo para o canal direito.
    pub model_r: Option<Box<DynamicModel>>,
    /// Multiplicador de ajuste de ganho de entrada.
    pub input_mult_adj: f32,
    /// Multiplicador de ajuste de ganho de saída.
    pub output_mult_adj: f32,
    /// Taxa de amostragem nativa do modelo.
    pub sample_rate: u32,
    /// Arquitetura do modelo (ex: "LSTM", "WaveNet").
    pub architecture: String,
    /// Topologia do modelo (ex: "Standard", "1x64").
    pub topology: String,
    /// Metadados opcionais do modelo.
    pub metadata: Option<crate::loader::nam_json::NamMetadata>,
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
            .finish()
    }
}

/// Carrega e constrói um par de modelos (L+R) a partir de um arquivo.
pub fn load_and_build_model(path: &Path, sys: &SystemSnapshot) -> anyhow::Result<LoadedModelPair> {
    let path_str = path.to_string_lossy();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let ext_lower = ext.to_lowercase();

    // 1. Leitura e Parsing
    let model_data = if ext_lower == "namb" {
        let len = std::fs::metadata(path).map_err(|e| {
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
                .param("size_bytes", &len)
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
            let msg = e.to_string();
            let code = if msg.contains("muito pequeno") {
                NamErrorCode::NambTruncated
            } else if msg.contains("mágica inválida") {
                NamErrorCode::NambInvalidMagic
            } else {
                NamErrorCode::ModelBuildFailed
            };
            NamDiagnostic::new(code, sys)
                .message(format!("Invalid \".namb\" file: {}", path_str))
                .param("detail", &msg)
                .emit();
        })?
    } else if ext_lower == "nam" {
        let len = std::fs::metadata(path).map_err(|e| {
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
                .param("size_bytes", &len)
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
            NamDiagnostic::new(NamErrorCode::NamJsonParseError, sys)
                .message(format!("Error parsing model JSON: {}", path_str))
                .param("detail", e)
                .emit();
        })?
    } else {
        return Err(anyhow::anyhow!("Unsupported file extension: {}", ext));
    };

    // 2. Extração de Metadados e Calibração
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
        m.prewarm(2048);
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
        m.prewarm(2048);
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

    Ok(LoadedModelPair {
        model_l,
        model_r,
        input_mult_adj,
        output_mult_adj,
        sample_rate: nam_rate,
        architecture,
        topology,
        metadata,
    })
}
