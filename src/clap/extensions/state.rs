// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the CLAP state extension.
//! Allows the plugin to save and load its current configuration (parameters and model).
//!
//! Payload versioning:
//! - v0 (legacy, CLAP v1.5.x): plain JSON `NamPluginParams`, without a `version` field.
//! - v1 (current): envelope `StateEnvelope { version: 1, params: {...} }`.

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
    #[error("Failed to serialize state: {0}")]
    Serialize(#[source] serde_json::Error),

    #[error("Failed to write to state stream: {0}")]
    WriteStream(#[source] std::io::Error),

    #[error("Failed to read from state stream: {0}")]
    ReadStream(#[source] std::io::Error),

    #[error("Failed to deserialize state (corrupted v1+ envelope)")]
    CorruptedEnvelope,

    #[error("Failed to deserialize state (v0 legacy): {0}")]
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
        // v0 → v1: common fields copied, new fields use Default
        // (NamPluginParams already has #[serde(default)] on all fields)
    }
    let _ = version;
    params
}

impl<'a> PluginStateImpl for NamClapMainThread<'a> {
    fn save(&mut self, output: &mut OutputStream) -> Result<(), PluginError> {
        self.params.input_gain_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_input_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.output_gain_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_output_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.gate_threshold_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_gate_thresh
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.bypass = self
            .shared
            .ui_to_rt
            .param_bypass
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0;
        self.params.adaptive_compute = crate::common::params::AdaptiveComputeMode::from_f32(
            self.shared
                .ui_to_rt
                .param_adaptive_compute
                .load(std::sync::atomic::Ordering::Relaxed) as f32,
        );

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
        self.shared.ui_to_rt.param_input_gain.store(
            self.params.input_gain_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.ui_to_rt.param_output_gain.store(
            self.params.output_gain_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.ui_to_rt.param_gate_thresh.store(
            self.params.gate_threshold_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.ui_to_rt.param_bypass.store(
            if self.params.bypass { 1 } else { 0 },
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.ui_to_rt.param_adaptive_compute.store(
            self.params.adaptive_compute as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared
            .ui_to_rt
            .gui_param_generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        if let Some(path) = self.params.model_path.clone() {
            if path.exists() {
                if let Err(e) = self.load_model(&path)
                    && let Some(log) = self.host.get_extension::<HostLog>()
                {
                    let msg = format!("NAM-rs: Failed to restore saved model ({:?}): {}", path, e);
                    if let Ok(c_msg) = CString::new(msg) {
                        log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                    }
                }
                return Ok(());
            }

            // Fallback: absolute path does not exist, try portable lookup via basename
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
                            "NAM-rs: Model not found at original path ({:?}), using portable fallback: {:?}",
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
                            "NAM-rs: Failed to restore model via fallback ({:?}): {}",
                            new_path, e
                        );
                        if let Ok(c_msg) = CString::new(msg) {
                            log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                        }
                    }
                } else if let Some(log) = self.host.get_extension::<HostLog>() {
                    let msg = format!(
                        "NAM-rs: Saved model not found at path: {:?} and basename {:?} not located in search paths",
                        path, basename
                    );
                    if let Ok(c_msg) = CString::new(msg) {
                        log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                    }
                }
            } else if let Some(log) = self.host.get_extension::<HostLog>() {
                let msg = format!("NAM-rs: Saved model not found at path: {:?}", path);
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

    // If the buffer is a v1+ envelope (contains a "version" key) that failed to parse,
    // we don't fall back to v0 — we propagate the error as corrupted data
    if let Ok(value) = serde_json::from_slice::<serde_json::Value>(buffer)
        && value.get("version").is_some()
    {
        return Err(PluginError::Error(Box::new(StateError::CorruptedEnvelope)));
    }

    // Fallback: v0 legacy — NamPluginParams directly, without a version field
    let params: NamPluginParams = serde_json::from_slice(buffer)
        .map_err(|e| PluginError::Error(Box::new(StateError::Deserialize(e))))?;

    Ok(migrate(0, params))
}

#[cfg(test)]
#[path = "state_test.rs"]
mod state_test;
