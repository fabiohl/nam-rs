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
#[path = "state_test.rs"]
mod state_test;
