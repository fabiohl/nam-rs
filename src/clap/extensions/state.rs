// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão de estado (state) do CLAP.
//! Permite que o plugin salve e carregue sua configuração atual (parâmetros e modelo).
//!
//! Versionamento do payload:
//! - v0 (legacy, CLAP v1.5.x): `NamPluginParams` JSON puro, sem campo `version`.
//! - v1 (atual): envelope `StateEnvelope { version: 1, params: {...} }`.

use crate::clap::plugin::{ClapParamPayload, NamClapMainThread};
use crate::common::params::NamPluginParams;
use clack_common::stream::{InputStream, OutputStream};
use clack_extensions::log::{HostLog, LogSeverity};
use clack_extensions::params::{HostParams, ParamRescanFlags};
use clack_extensions::state::PluginStateImpl;
use clack_plugin::prelude::*;
use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::io::{Read, Write};

const CURRENT_STATE_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
enum StateError {
    #[error("Falha ao serializar estado: {0}")]
    Serialize(#[source] serde_json::Error),

    #[error("Falha ao escrever no stream de estado: {0}")]
    WriteStream(#[source] std::io::Error),

    #[error("Falha ao ler do stream de estado: {0}")]
    ReadStream(#[source] std::io::Error),

    #[error("Falha ao deserializar estado (envelope v1+ corrompido)")]
    CorruptedEnvelope,

    #[error("Falha ao deserializar estado (v0 legacy): {0}")]
    Deserialize(#[source] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StateEnvelope {
    version: u32,
    params: NamPluginParams,
}

#[allow(unused_mut)]
fn migrate(version: u32, mut params: NamPluginParams) -> NamPluginParams {
    if version < 1 {
        // v0 → v1: campos comuns copiados, novos campos com Default
        // (NamPluginParams já tem #[serde(default)] em todos os campos)
    }
    let _ = version;
    params
}

impl<'a> PluginStateImpl for NamClapMainThread<'a> {
    fn save(&mut self, output: &mut OutputStream) -> Result<(), PluginError> {
        self.params.input_gain_db = f32::from_bits(
            self.shared
                .param_input_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.output_gain_db = f32::from_bits(
            self.shared
                .param_output_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.gate_threshold_db = f32::from_bits(
            self.shared
                .param_gate_thresh
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.bypass = self
            .shared
            .param_bypass
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0;

        let envelope = StateEnvelope {
            version: CURRENT_STATE_VERSION,
            params: self.params.clone(),
        };
        let serialized = serde_json::to_vec(&envelope)
            .map_err(|e| PluginError::Error(Box::new(StateError::Serialize(e))))?;

        output
            .write_all(&serialized)
            .map_err(|e| PluginError::Error(Box::new(StateError::WriteStream(e))))?;

        Ok(())
    }

    fn load(&mut self, input: &mut InputStream) -> Result<(), PluginError> {
        let mut buffer = Vec::new();
        input
            .read_to_end(&mut buffer)
            .map_err(|e| PluginError::Error(Box::new(StateError::ReadStream(e))))?;

        let new_params = load_state(&buffer)?;

        self.params = new_params;
        self.shared.param_input_gain.store(
            self.params.input_gain_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.param_output_gain.store(
            self.params.output_gain_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.param_gate_thresh.store(
            self.params.gate_threshold_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.param_bypass.store(
            if self.params.bypass { 1 } else { 0 },
            std::sync::atomic::Ordering::Relaxed,
        );

        if let Some(path) = self.params.model_path.clone() {
            if path.exists() {
                if let Err(e) = self.load_model(&path)
                    && let Some(log) = self.host.get_extension::<HostLog>()
                {
                    let msg = format!(
                        "NAM-rs: Falha ao restaurar modelo salvo ({:?}): {}",
                        path, e
                    );
                    if let Ok(c_msg) = CString::new(msg) {
                        log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                    }
                }
                return Ok(());
            }

            // Fallback: path absoluto não existe, tenta busca portátil via basename
            if let Some(ref basename) = self.params.model_basename {
                let found = self
                    .params
                    .model_search_paths
                    .clone()
                    .into_iter()
                    .find_map(|dir| {
                        let candidate = dir.join(basename);
                        if candidate.exists() {
                            Some(candidate)
                        } else {
                            None
                        }
                    });
                if let Some(new_path) = found {
                    if let Some(log) = self.host.get_extension::<HostLog>() {
                        let msg = format!(
                            "NAM-rs: Modelo não encontrado no path original ({:?}), usando fallback portátil: {:?}",
                            path, new_path
                        );
                        if let Ok(c_msg) = CString::new(msg) {
                            log.log(&self.host.shared(), LogSeverity::Info, &c_msg);
                        }
                    }
                    if let Err(e) = self.load_model(&new_path)
                        && let Some(log) = self.host.get_extension::<HostLog>()
                    {
                        let msg = format!(
                            "NAM-rs: Falha ao restaurar modelo via fallback ({:?}): {}",
                            new_path, e
                        );
                        if let Ok(c_msg) = CString::new(msg) {
                            log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                        }
                    }
                } else if let Some(log) = self.host.get_extension::<HostLog>() {
                    let msg = format!(
                        "NAM-rs: Modelo salvo não encontrado no caminho: {:?} e basename {:?} não localizado nos search paths",
                        path, basename
                    );
                    if let Ok(c_msg) = CString::new(msg) {
                        log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                    }
                }
            } else if let Some(log) = self.host.get_extension::<HostLog>() {
                let msg = format!("NAM-rs: Modelo salvo não encontrado no caminho: {:?}", path);
                if let Ok(c_msg) = CString::new(msg) {
                    log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                }
            }
        }

        let _ = self
            .param_tx
            .push(ClapParamPayload::Params(self.params.clone()));

        if let Some(params_ext) = self.host.get_extension::<HostParams>() {
            params_ext.rescan(&mut self.host, ParamRescanFlags::VALUES);
        }

        Ok(())
    }
}

fn load_state(buffer: &[u8]) -> Result<NamPluginParams, PluginError> {
    if let Ok(envelope) = serde_json::from_slice::<StateEnvelope>(buffer) {
        return Ok(migrate(envelope.version, envelope.params));
    }

    // Se o buffer é um envelope v1+ (contém chave "version") que falhou parse,
    // não fazemos fallback v0 — propagamos erro como dados corrompidos
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(buffer)
        && value.get("version").is_some()
    {
        return Err(PluginError::Error(Box::new(StateError::CorruptedEnvelope)));
    }

    // Fallback: v0 legacy — NamPluginParams direto sem campo version
    let params: NamPluginParams = serde_json::from_slice(buffer)
        .map_err(|e| PluginError::Error(Box::new(StateError::Deserialize(e))))?;

    Ok(migrate(0, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_v0_legacy_load() {
        let v0_json = r#"{"input_gain_db": 3.0,"output_gain_db": -6.0,"gate_threshold_db": -50.0,"model_path": null,"bypass": false}"#;
        let params = load_state(v0_json.as_bytes()).expect("v0 payload deve carregar");
        assert!((params.input_gain_db - 3.0).abs() < f32::EPSILON);
        assert!((params.output_gain_db - (-6.0)).abs() < f32::EPSILON);
        assert!((params.gate_threshold_db - (-50.0)).abs() < f32::EPSILON);
        assert_eq!(params.model_path, None);
        assert!(!params.bypass);
    }

    #[test]
    fn test_v0_legacy_load_with_missing_fields() {
        // Simula payload v0 antigo que poderia ter campos ausentes
        let v0_json = r#"{"input_gain_db": 1.5}"#;
        let params = load_state(v0_json.as_bytes()).expect("v0 com campos ausentes deve carregar");
        assert!((params.input_gain_db - 1.5).abs() < f32::EPSILON);
        assert_eq!(params.output_gain_db, 0.0);
        assert_eq!(params.gate_threshold_db, -70.0);
        assert_eq!(params.model_path, None);
        assert!(!params.bypass);
    }

    #[test]
    fn test_v1_round_trip() {
        let original = NamPluginParams {
            input_gain_db: 2.5,
            output_gain_db: -3.0,
            gate_threshold_db: -40.0,
            model_path: Some(PathBuf::from("/tmp/test.nam")),
            model_basename: None,
            model_search_paths: Vec::new(),
            bypass: true,
        };

        let envelope = StateEnvelope {
            version: CURRENT_STATE_VERSION,
            params: original.clone(),
        };
        let json = serde_json::to_vec(&envelope).unwrap();

        let restored = load_state(&json).expect("v1 payload deve carregar");
        assert_eq!(restored, original, "round-trip v1 deve ser idempotente");
    }

    #[test]
    fn test_v1_save_format() {
        let params = NamPluginParams::default();
        let envelope = StateEnvelope {
            version: CURRENT_STATE_VERSION,
            params,
        };
        let json = serde_json::to_vec(&envelope).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(parsed["version"], 1, "envelope deve conter version: 1");
        assert!(parsed["params"].is_object(), "envelope deve conter params");
    }

    #[test]
    fn test_v0_legacy_load_new_fields_default() {
        let v0_json = r#"{"input_gain_db": 3.0,"output_gain_db": -6.0,"gate_threshold_db": -50.0,"model_path": null,"bypass": false}"#;
        let params = load_state(v0_json.as_bytes()).expect("v0 payload deve carregar");
        assert_eq!(params.model_basename, None);
        assert!(params.model_search_paths.is_empty());
    }

    #[test]
    fn test_v1_round_trip_with_search_fields() {
        let search_paths = vec![
            std::path::PathBuf::from("/usr/share/nam-models"),
            std::path::PathBuf::from("/home/user/models"),
        ];
        let original = NamPluginParams {
            input_gain_db: 2.5,
            output_gain_db: -3.0,
            gate_threshold_db: -40.0,
            model_path: Some(PathBuf::from("/tmp/test.nam")),
            model_basename: Some("test.nam".to_string()),
            model_search_paths: search_paths.clone(),
            bypass: true,
        };

        let envelope = StateEnvelope {
            version: CURRENT_STATE_VERSION,
            params: original.clone(),
        };
        let json = serde_json::to_vec(&envelope).unwrap();

        let restored = load_state(&json).expect("v1 payload deve carregar");
        assert_eq!(
            restored, original,
            "round-trip v1 com search fields deve ser idempotente"
        );
    }

    #[test]
    fn test_v1_search_fields_serialization_format() {
        let params = NamPluginParams {
            model_basename: Some("tone.nam".to_string()),
            model_search_paths: vec![PathBuf::from("/models")],
            ..Default::default()
        };
        let envelope = StateEnvelope {
            version: CURRENT_STATE_VERSION,
            params,
        };
        let json = serde_json::to_vec(&envelope).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(parsed["params"]["model_basename"], "tone.nam");
        assert_eq!(parsed["params"]["model_search_paths"][0], "/models");
    }
}
