// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the CLAP state-context extension.
//!
//! Allows the plugin to distinguish between saving state for a preset
//! (portable, without absolute paths) vs for a project or duplication
//! (full state, with absolute model paths).
//!
//! See: `clap/ext/state-context.h`

use crate::clap::extensions::params::bypass_bool_to_u32;
use crate::clap::plugin::ClapParamPayload;
use crate::clap::plugin::NamClapMainThread;
use crate::clap::plugin::debug_assert_main_thread;
use crate::common::params::RtPluginParams;
use crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED;
use crate::dsp::resampler::NamResampler;
use clack_common::stream::{InputStream, OutputStream};
use clack_extensions::state_context::{
    PluginStateContext, PluginStateContextImpl, StateContextType,
};
use clack_plugin::prelude::*;
use std::io::{Read, Write};
use std::sync::atomic::Ordering;

/// Type alias for the CLAP state-context extension registration.
pub type NamPluginStateContext = PluginStateContext;

impl<'a> PluginStateContextImpl for NamClapMainThread<'a> {
    fn save(
        &mut self,
        output: &mut OutputStream,
        context_type: StateContextType,
    ) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        self.snapshot_params();

        let save_params = if context_type == StateContextType::ForPreset {
            let mut preset_params = self.params.clone();
            preset_params.model_path = None;
            preset_params
        } else {
            self.params.clone()
        };

        let serialized = super::state::serialize_envelope(&save_params)?;

        output
            .write_all(&serialized)
            .map_err(|e| PluginError::Error(Box::new(e)))?;

        Ok(())
    }

    fn load(
        &mut self,
        input: &mut InputStream,
        context_type: StateContextType,
    ) -> Result<(), PluginError> {
        debug_assert_main_thread(&self.host);
        let mut buffer = Vec::new();
        input
            .read_to_end(&mut buffer)
            .map_err(|e| PluginError::Error(Box::new(e)))?;

        let loaded_params = super::state::load_state(&buffer)?;

        if context_type == StateContextType::ForPreset {
            // ══════ ForPreset: partial param merge + portable model lookup ══════
            self.params.input_gain_db = loaded_params.input_gain_db;
            self.params.output_gain_db = loaded_params.output_gain_db;
            self.params.gate_threshold_db = loaded_params.gate_threshold_db;
            self.params.bypass = loaded_params.bypass;
            self.params.adaptive_compute = loaded_params.adaptive_compute;
            self.params.slim_override = loaded_params.slim_override;

            let mut model_load_failed = false;
            if let Some(ref basename) = loaded_params.model_basename {
                let found = loaded_params
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
                    if let Err(e) = self.load_model(&new_path) {
                        log::warn!("ForPreset: failed to restore model ({new_path:?}): {e}");
                        model_load_failed = true;
                    }
                } else {
                    log::warn!("ForPreset: model basename {basename:?} not found in search paths");
                    model_load_failed = true;
                }
            }

            if model_load_failed {
                let host_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);
                let host_rate = if host_rate == 0 { 48000 } else { host_rate };
                let bypass_resampler =
                    Box::new(NamResampler::new(host_rate, host_rate, 0).map_err(|e| {
                        PluginError::Message(Box::leak(
                            format!("Failed to create bypass resampler: {e:?}").into_boxed_str(),
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
                self.params.model_path = None;
                self.params.model_basename = None;
                if let Ok(mut name_guard) = self.shared.cold.ui_model_name.lock() {
                    name_guard.clear();
                }
                log::error!(
                    "NAM-rs: ForPreset restore failed — model not found. Old model unloaded."
                );
            }
        } else {
            // ══════ Non-ForPreset: full state restore (project, duplicate) ══════
            let mut model_load_failed = false;

            if let Some(ref path) = loaded_params.model_path.clone() {
                if path.exists() {
                    if let Err(e) = self.load_model(path) {
                        log::warn!("Failed to restore model ({path:?}): {e}");
                        model_load_failed = true;
                    }
                } else if let Some(ref basename) = loaded_params.model_basename {
                    let found = loaded_params
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
                        }
                    } else {
                        log::warn!(
                            "Saved model not found at path: {path:?} and basename {basename:?} not located in search paths"
                        );
                        model_load_failed = true;
                    }
                } else {
                    log::warn!("Saved model not found at path: {path:?}");
                    model_load_failed = true;
                }
            }

            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            let ir_load_failed = {
                let mut failed = false;
                if let Some(ref ir_path) = loaded_params.ir_path {
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
                failed
            };

            // ══════ Commit non-ForPreset ══════

            if model_load_failed {
                let host_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);
                let host_rate = if host_rate == 0 { 48000 } else { host_rate };
                let bypass_resampler =
                    Box::new(NamResampler::new(host_rate, host_rate, 0).map_err(|e| {
                        PluginError::Message(Box::leak(
                            format!("Failed to create bypass resampler: {e:?}").into_boxed_str(),
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
                if let Ok(mut name_guard) = self.shared.cold.ui_model_name.lock() {
                    name_guard.clear();
                }
                log::error!(
                    "NAM-rs: State-context restore failed — model not found. Old model unloaded."
                );
            }

            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            if ir_load_failed {
                let _ = self
                    .param_tx
                    .push(ClapParamPayload::LoadCabIr { adapter: None });
                log::error!(
                    "NAM-rs: State-context restore failed — IR not found. Old IR unloaded."
                );
            }

            self.params = loaded_params.clone();

            if model_load_failed && self.params.model_path.is_some() {
                self.params.model_path = None;
                self.params.model_basename = None;
            }

            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            if ir_load_failed {
                if let Ok(mut ir_guard) = self.shared.cold.ir_path.lock() {
                    *ir_guard = None;
                }
                if let Ok(mut raw_guard) = self.shared.cold.ir_raw_samples.lock() {
                    *raw_guard = None;
                }
                self.shared.cold.ir_raw_sample_rate.store(0, Ordering::Relaxed);
                self.params.ir_path = None;
            } else if loaded_params.ir_path.is_none() {
                let _ = self
                    .param_tx
                    .push(ClapParamPayload::LoadCabIr { adapter: None });
            }
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
        self.shared.bump_generation();

        let _ = self.param_tx.push(ClapParamPayload::Params(
            RtPluginParams::from_plugin_params(&self.params),
        ));

        if let Some(params_ext) = self
            .host
            .get_extension::<clack_extensions::params::HostParams>()
        {
            params_ext.rescan(
                &mut self.host,
                clack_extensions::params::ParamRescanFlags::VALUES,
            );
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "state_context_test.rs"]
mod tests;
