// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the CLAP state extension.
//! Allows the plugin to save and load its current configuration (parameters and model).
//!
//! Payload versioning:
//! - v0 (legacy, CLAP v1.5.x): plain JSON `NamPluginParams`, without a `version` field.
//! - v1 (current): envelope `StateEnvelope { version: 1, params: {...} }`.

use crate::clap::extensions::params::bypass_bool_to_u32;
use crate::clap::plugin::debug_assert_main_thread;
use crate::clap::plugin::{ClapParamPayload, NamClapMainThread};
use crate::common::params::{NamPluginParams, RtPluginParams};
use clack_common::stream::{InputStream, OutputStream};
use clack_extensions::params::{HostParams, ParamRescanFlags};
use clack_extensions::state::PluginStateImpl;
use clack_plugin::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub(crate) const CURRENT_STATE_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub(crate) enum StateError {
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
pub(crate) struct StateEnvelope {
    pub(crate) version: u32,
    pub(crate) params: NamPluginParams,
}

/// Shared helper: serialises `NamPluginParams` wrapped in a v1 `StateEnvelope`.
pub(crate) fn serialize_envelope(params: &NamPluginParams) -> Result<Vec<u8>, PluginError> {
    let envelope = StateEnvelope {
        version: CURRENT_STATE_VERSION,
        params: params.clone(),
    };
    serde_json::to_vec(&envelope)
        .map_err(|e| PluginError::Error(Box::new(StateError::Serialize(e))))
}

#[expect(
    clippy::single_match,
    reason = "Match kept for exhaustiveness — future extensions expected at this dispatch site"
)]
fn migrate(version: u32, params: NamPluginParams) -> NamPluginParams {
    match version {
        0 => {
            // v0 → v1: common fields copied, new fields use Default
            // (NamPluginParams already has #[serde(default)] on all fields)
        }
        _ => {}
    }
    params
}

impl<'a> PluginStateImpl for NamClapMainThread<'a> {
    fn save(&mut self, output: &mut OutputStream) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        self.snapshot_params();

        let serialized = serialize_envelope(&self.params)?;

        output
            .write_all(&serialized)
            .map_err(|e| PluginError::Error(Box::new(StateError::WriteStream(e))))?;

        Ok(())
    }

    fn load(&mut self, input: &mut InputStream) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        let mut buffer = Vec::new();
        input
            .read_to_end(&mut buffer)
            .map_err(|e| PluginError::Error(Box::new(StateError::ReadStream(e))))?;

        if buffer.is_empty() {
            log::debug!("Empty state buffer, returning false");
            return Err(PluginError::Message("Empty state buffer"));
        }

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
            bypass_bool_to_u32(self.params.bypass),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.ui_to_rt.param_adaptive_compute.store(
            self.params.adaptive_compute as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.ui_to_rt.param_slim_override.store(
            self.params.slim_override as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.ui_to_rt.param_oversample.store(
            self.params.oversample.to_f32() as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.ui_to_rt.param_activation.store(
            self.params.activation_precision as u32,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.bump_generation();

        if let Some(path) = self.params.model_path.clone() {
            if path.exists() {
                if let Err(e) = self.load_model(&path) {
                    log::warn!("Failed to restore saved model ({path:?}): {e}");
                }
            } else {
                // Fallback: absolute path does not exist, try portable lookup via basename
                if let Some(ref basename) = self.params.model_basename {
                    let found =
                        self.params
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
                        log::info!(
                            "Model not found at original path ({path:?}), using portable fallback: {new_path:?}"
                        );
                        if let Err(e) = self.load_model(&new_path) {
                            log::warn!("Failed to restore model via fallback ({new_path:?}): {e}");
                        }
                    } else {
                        log::warn!(
                            "Saved model not found at path: {path:?} and basename {basename:?} not located in search paths"
                        );
                    }
                } else {
                    log::warn!("Saved model not found at path: {path:?}");
                }
            }
        }

        // Restore cab-sim IR if present in state.
        // Follows the same SPSC pattern as model path load.
        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        {
            let ir_path_opt = self.params.ir_path.clone();
            if let Some(ref ir_path) = ir_path_opt {
                if ir_path.exists() {
                    if let Err(e) = self.load_cabsim(ir_path) {
                        log::warn!("Failed to restore saved IR ({ir_path:?}): {e}");
                    }
                } else {
                    log::warn!("Saved IR not found at path: {ir_path:?}");
                }
            } else {
                // No IR in saved state: bypass cabsim by sending None engine.
                let _ = self
                    .param_tx
                    .push(ClapParamPayload::LoadCabIr { engine: None });
            }
        }

        let _ = self.param_tx.push(ClapParamPayload::Params(
            RtPluginParams::from_plugin_params(&self.params),
        ));

        if let Some(params_ext) = self.host.get_extension::<HostParams>() {
            params_ext.rescan(&mut self.host, ParamRescanFlags::VALUES);
        }

        Ok(())
    }
}

pub(crate) fn load_state(buffer: &[u8]) -> Result<NamPluginParams, PluginError> {
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
