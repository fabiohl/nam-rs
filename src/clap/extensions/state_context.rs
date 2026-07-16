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
use crate::clap::plugin::NamClapMainThread;
use crate::clap::plugin::debug_assert_main_thread;
use crate::common::diagnostics::NamDiagnostic;
use crate::common::params::RtPluginParams;
use clack_common::stream::{InputStream, OutputStream};
use clack_extensions::log::{HostLog, LogSeverity};
use clack_extensions::state_context::{
    PluginStateContext, PluginStateContextImpl, StateContextType,
};
use clack_plugin::prelude::*;
use std::ffi::CString;
use std::io::{Read, Write};

#[derive(Debug, thiserror::Error)]
enum StateContextError {
    #[error("Failed to restore model from state: {0}")]
    ModelRestore(#[source] NamDiagnostic),
}

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
            self.params.input_gain_db = loaded_params.input_gain_db;
            self.params.output_gain_db = loaded_params.output_gain_db;
            self.params.gate_threshold_db = loaded_params.gate_threshold_db;
            self.params.bypass = loaded_params.bypass;
            self.params.adaptive_compute = loaded_params.adaptive_compute;
            self.params.slim_override = loaded_params.slim_override;

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
                if let Some(new_path) = found
                    && let Err(e) = self.load_model(&new_path)
                {
                    return Err(PluginError::Error(Box::new(
                        StateContextError::ModelRestore(*e),
                    )));
                }
            }
        } else {
            self.params = loaded_params;
            if let Some(ref path) = self.params.model_path.clone() {
                if path.exists() {
                    if let Err(e) = self.load_model(path) {
                        return Err(PluginError::Error(Box::new(
                            StateContextError::ModelRestore(*e),
                        )));
                    }
                } else {
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
                            if let Some(log) = self.host.get_extension::<HostLog>() {
                                let msg = format!(
                                    "NAM-rs: Model not found at original path ({:?}), using portable fallback: {:?}",
                                    path, new_path
                                );
                                if let Ok(c_msg) = CString::new(msg) {
                                    log.log(&self.host.shared(), LogSeverity::Info, &c_msg);
                                }
                            }
                            if let Err(e) = self.load_model(&new_path) {
                                return Err(PluginError::Error(Box::new(
                                    StateContextError::ModelRestore(*e),
                                )));
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
            }

            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            {
                let ir_path_opt = self.params.ir_path.clone();
                if let Some(ref ir_path) = ir_path_opt {
                    if ir_path.exists() {
                        if let Err(e) = self.load_cabsim(ir_path)
                            && let Some(log) = self.host.get_extension::<HostLog>()
                        {
                            let msg = format!(
                                "NAM-rs: Failed to restore saved IR ({:?}): {}",
                                ir_path, e
                            );
                            if let Ok(c_msg) = CString::new(msg) {
                                log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                            }
                        }
                    } else if let Some(log) = self.host.get_extension::<HostLog>() {
                        let msg = format!("NAM-rs: Saved IR not found at path: {:?}", ir_path);
                        if let Ok(c_msg) = CString::new(msg) {
                            log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                        }
                    }
                } else {
                    use crate::clap::plugin::ClapParamPayload;
                    let _ = self
                        .param_tx
                        .push(ClapParamPayload::LoadCabIr { engine: None });
                }
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

        use crate::clap::plugin::ClapParamPayload;
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
