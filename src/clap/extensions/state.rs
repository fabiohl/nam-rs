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
use crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED;
use crate::dsp::resampler::NamResampler;
use clack_common::stream::{InputStream, OutputStream};
use clack_extensions::params::{HostParams, ParamRescanFlags};
use clack_extensions::state::PluginStateImpl;
use clack_plugin::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::sync::atomic::Ordering;

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
        let blob_len = serialized.len();

        output
            .write_all(&serialized)
            .map_err(|e| PluginError::Error(Box::new(StateError::WriteStream(e))))?;

        log::debug!("[State] Save completed: {} bytes serialized.", blob_len);

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

        // ══════ Phase 1: PREPARE — validate and load assets ══════

        let mut model_load_failed = false;

        let model_resolved = if let Some(ref path) = new_params.model_path {
            if path.exists() {
                if let Err(e) = self.load_model(path) {
                    log::warn!("Failed to restore saved model ({path:?}): {e}");
                    model_load_failed = true;
                    false
                } else {
                    true
                }
            } else if let Some(ref basename) = new_params.model_basename {
                let found = new_params
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
                        model_load_failed = true;
                        false
                    } else {
                        true
                    }
                } else {
                    log::warn!(
                        "Saved model not found at path: {path:?} and basename {basename:?} not located in search paths"
                    );
                    model_load_failed = true;
                    false
                }
            } else {
                log::warn!("Saved model not found at path: {path:?}");
                model_load_failed = true;
                false
            }
        } else {
            // No model in saved state — keep current model
            true
        };

        // Restore cab-sim IR if present in state (prepare phase).
        // A failed IR load must also clean the old IR to avoid hybrid state.
        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        let ir_load_failed = {
            let mut failed = false;
            if let Some(ref ir_path) = new_params.ir_path {
                if ir_path.exists() {
                    if let Err(e) = self.load_cabsim(ir_path) {
                        log::warn!("Failed to restore saved IR ({ir_path:?}): {e}");
                        failed = true;
                    }
                } else {
                    log::warn!("Saved IR not found at path: {ir_path:?}");
                    failed = true;
                }
            }
            // If IR not present in saved state and old IR was loaded,
            // the "else { push LoadCabIr{None} }" branch below handles cleanup.
            failed
        };

        // ══════ Phase 2: COMMIT — publish or rollback ══════

        if model_load_failed {
            // Unload the old model so no stale DSP persists silently.
            // Construct a bypass resampler (no I/O needed on main thread)
            // so the RT thread can safely replace its resampler during
            // the cold-load swap.
            let host_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);
            let host_rate = if host_rate == 0 { 48000 } else { host_rate };
            let bypass_resampler =
                Box::new(NamResampler::new(host_rate, host_rate, 0).map_err(|e| {
                    PluginError::Message(Box::leak(
                        format!("Failed to create bypass resampler during unload: {e:?}")
                            .into_boxed_str(),
                    ))
                })?);
            let _ = self.param_tx.push(ClapParamPayload::LoadModel {
                model_l: None,
                new_resampler: bypass_resampler,
                input_mult_adj: 1.0,
                output_mult_adj: 1.0,
            });
            self.shared
                .cold
                .rt_status
                .set_flag(RT_STATUS_MODEL_LOAD_FAILED);
            // Clear model metadata so the UI shows the error state
            if let Ok(mut name_guard) = self.shared.cold.ui_model_name.lock() {
                name_guard.clear();
            }
            log::error!("NAM-rs: State restore failed — model not found. Old model unloaded.");
        }

        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        if ir_load_failed {
            // Clear any old IR from the RT thread
            let _ = self
                .param_tx
                .push(ClapParamPayload::LoadCabIr { adapter: None });
            log::error!("NAM-rs: State restore failed — IR not found. Old IR unloaded.");
        }

        // Commit params to main thread state
        self.params = new_params.clone();

        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        {
            if !model_resolved || ir_load_failed {
                // IR must be cleared from cold state if load failed or
                // was not present in new params. The RT thread was already
                // told via LoadCabIr{None} above.
                if let Ok(mut ir_guard) = self.shared.cold.ir_path.lock() {
                    *ir_guard = None;
                }
                if let Ok(mut raw_guard) = self.shared.cold.ir_raw_samples.lock() {
                    *raw_guard = None;
                }
                self.params.ir_path = None;
            }
        }

        if !model_resolved && new_params.model_path.is_some() {
            // Model was requested but could not be loaded. Clear the stale
            // model metadata so the UI doesn't show the old model name.
            self.params.model_path = None;
            self.params.model_basename = None;
        }

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

        // IR cleanup when no IR in saved state and model may have been unloaded
        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        {
            if new_params.ir_path.is_none() && !ir_load_failed {
                // No IR requested in new state — bypass cabsim
                let _ = self
                    .param_tx
                    .push(ClapParamPayload::LoadCabIr { adapter: None });
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
