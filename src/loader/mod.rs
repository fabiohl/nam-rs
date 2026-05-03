// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de carregamento do ecossistema NAM.
//!
//! Contém os parsers dos formatos .nam (JSON) e .namb (Binário).
//! Todo o processo de carga ocorre **fora** da thread RT para
//! evitar qualquer alocação indesejada durante o processamento de áudio.

use crate::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::models::{DynamicModel, NamModel};
use std::path::Path;

pub mod dispatcher;
pub mod nam_json;
pub mod namb;
pub mod namb_encoder;

/// Resultado da carga de modelos (L, R, InAdj, OutAdj, Rate)
pub type LoadedModels = (
    Option<Box<DynamicModel>>,
    Option<Box<DynamicModel>>,
    f32,
    f32,
    u32,
);

/// Carrega e constrói um par de modelos (L+R) a partir de um arquivo.
pub fn load_and_build_model(path: &Path, sys: &SystemSnapshot) -> anyhow::Result<LoadedModels> {
    let path_str = path.to_string_lossy();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let ext_lower = ext.to_lowercase();

    // 1. Leitura e Parsing
    let model_data = if ext_lower == "namb" {
        let bytes = std::fs::read(path).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Não conseguimos ler o arquivo \"{}\".", path_str))
                .hint("Verifique as permissões de acesso ao arquivo.")
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
                .message(format!("Arquivo \".namb\" inválido: {}", path_str))
                .param("detail", &msg)
                .emit();
        })?
    } else if ext_lower == "nam" {
        let json = std::fs::read_to_string(path).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Não conseguimos ler o arquivo \"{}\".", path_str))
                .param("file", &path_str)
                .emit();
            anyhow::Error::from(e)
        })?;
        nam_json::parse_nam_json(&json).inspect_err(|e| {
            NamDiagnostic::new(NamErrorCode::NamJsonParseError, sys)
                .message(format!("Erro ao parsear JSON do modelo: {}", path_str))
                .param("detail", e)
                .emit();
        })?
    } else {
        return Err(anyhow::anyhow!(
            "Extensão de arquivo não suportada: {}",
            ext
        ));
    };

    // 2. Extração de Metadados e Calibração
    let meta = model_data
        .metadata
        .clone()
        .unwrap_or(nam_json::NamMetadata {
            date: None,
            name: None,
            modeled_by: None,
            gear_make: None,
            gear_model: None,
            gear_type: None,
            tone_type: None,
            training: None,
            input_level_dbu: None,
            output_level_dbu: None,
            loudness: None,
        });
    let in_level = meta.input_level_dbu.unwrap_or(12.0);
    let loudness = meta.loudness.unwrap_or(-18.0);

    let input_db_adj = 12.0 - in_level;
    let output_db_adj = -18.0 - loudness;
    let nam_rate = model_data.sample_rate.unwrap_or(48000.0) as u32;

    let lut = crate::math::fastmath::get_gain_lut();
    let input_mult_adj = lut.db_to_linear(input_db_adj);
    let output_mult_adj = lut.db_to_linear(output_db_adj);

    // 3. Dispatcher (Build Model L/R)
    let mut model_l = dispatcher::build_model(&model_data)
        .inspect_err(|e| {
            NamDiagnostic::new(NamErrorCode::ModelBuildFailed, sys)
                .message(format!("Falha ao construir modelo (L): {}", path_str))
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
                .message(format!("Falha ao construir modelo (R): {}", path_str))
                .param("detail", e.to_string())
                .emit();
        })
        .ok();
    if let Some(ref mut m) = model_r {
        m.prewarm(2048);
    }

    Ok((model_l, model_r, input_mult_adj, output_mult_adj, nam_rate))
}
