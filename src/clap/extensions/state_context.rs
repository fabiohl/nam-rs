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
mod tests {
    use crate::common::params::{AdaptiveComputeMode, NamPluginParams};
    use std::path::PathBuf;

    fn make_test_params() -> NamPluginParams {
        NamPluginParams {
            input_gain_db: 3.0,
            output_gain_db: -6.0,
            gate_threshold_db: -50.0,
            model_path: Some(PathBuf::from("/tmp/test.nam")),
            model_basename: Some("test.nam".to_string()),
            model_search_paths: vec![PathBuf::from("/tmp")],
            bypass: false,
            adaptive_compute: AdaptiveComputeMode::Off,
            slim_override: Default::default(),
            ir_path: None,
        }
    }

    fn deserialize_from_context(buf: &[u8]) -> NamPluginParams {
        crate::clap::extensions::state::load_state(buf).unwrap()
    }

    #[test]
    fn test_serialized_format_is_v1_envelope() {
        let params = make_test_params();
        let buf = crate::clap::extensions::state::serialize_envelope(&params).unwrap();

        let parsed: serde_json::Value = serde_json::from_slice(&buf).expect("should be valid JSON");
        assert_eq!(parsed["version"], 1, "envelope should have version: 1");
        assert!(
            parsed["params"].is_object(),
            "envelope should contain params object"
        );
        assert_eq!(parsed["params"]["model_basename"], "test.nam");
    }

    #[test]
    fn test_preset_save_strips_model_path_from_envelope() {
        let params = make_test_params();

        let mut preset_params = params.clone();
        preset_params.model_path = None;
        let buf = crate::clap::extensions::state::serialize_envelope(&preset_params).unwrap();

        let loaded = deserialize_from_context(&buf);
        assert_eq!(
            loaded.model_path, None,
            "preset should not contain model_path"
        );
        assert_eq!(loaded.model_basename, Some("test.nam".to_string()));
        assert!((loaded.input_gain_db - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_project_save_preserves_model_path_in_envelope() {
        let params = make_test_params();
        let buf = crate::clap::extensions::state::serialize_envelope(&params).unwrap();

        let loaded = deserialize_from_context(&buf);
        assert_eq!(
            loaded.model_path,
            Some(PathBuf::from("/tmp/test.nam")),
            "project save should preserve model_path"
        );
    }

    #[test]
    fn test_duplicate_save_preserves_model_path_in_envelope() {
        let params = make_test_params();
        let buf = crate::clap::extensions::state::serialize_envelope(&params).unwrap();

        let loaded = deserialize_from_context(&buf);
        assert_eq!(
            loaded.model_path,
            Some(PathBuf::from("/tmp/test.nam")),
            "duplicate save should preserve model_path"
        );
    }

    #[test]
    fn test_preset_load_without_model_path_restores_audio_params() {
        let preset_json = r#"{"input_gain_db":2.5,"output_gain_db":-3.0,"gate_threshold_db":-40.0,"model_path":null,"model_basename":"test.nam","model_search_paths":[],"bypass":false,"adaptive_compute":"Off"}"#;
        let loaded = deserialize_from_context(preset_json.as_bytes());

        assert!((loaded.input_gain_db - 2.5).abs() < f32::EPSILON);
        assert!((loaded.output_gain_db - (-3.0)).abs() < f32::EPSILON);
        assert!((loaded.gate_threshold_db - (-40.0)).abs() < f32::EPSILON);
        assert_eq!(loaded.model_path, None);
        assert_eq!(loaded.model_basename, Some("test.nam".to_string()));
    }

    #[test]
    fn test_project_load_preserves_full_state_from_v0_legacy() {
        let project_json = r#"{"input_gain_db":2.5,"output_gain_db":-3.0,"gate_threshold_db":-40.0,"model_path":"/tmp/test.nam","model_basename":"test.nam","model_search_paths":[],"bypass":false,"adaptive_compute":"Off"}"#;
        let loaded = deserialize_from_context(project_json.as_bytes());

        assert_eq!(
            loaded.model_path,
            Some(PathBuf::from("/tmp/test.nam")),
            "project load from v0 should preserve model_path"
        );
    }

    #[test]
    fn test_v1_envelope_load_preserves_full_state() {
        let project_json = serde_json::json!({
            "version": 1,
            "params": {
                "input_gain_db": 2.5,
                "output_gain_db": -3.0,
                "gate_threshold_db": -40.0,
                "model_path": "/tmp/test.nam",
                "model_basename": "test.nam",
                "model_search_paths": [],
                "bypass": false,
                "adaptive_compute": "Off"
            }
        })
        .to_string();

        let loaded = deserialize_from_context(project_json.as_bytes());

        assert_eq!(
            loaded.model_path,
            Some(PathBuf::from("/tmp/test.nam")),
            "v1 envelope load should preserve model_path"
        );
    }

    #[test]
    fn test_v1_envelope_round_trip_preserves_ir_path() {
        let params = NamPluginParams {
            ir_path: Some(PathBuf::from("/tmp/cab.wav")),
            ..make_test_params()
        };

        let buf = crate::clap::extensions::state::serialize_envelope(&params).unwrap();
        let loaded = deserialize_from_context(&buf);

        assert_eq!(
            loaded.ir_path,
            Some(PathBuf::from("/tmp/cab.wav")),
            "ir_path should be preserved in v1 envelope round-trip"
        );
    }

    #[test]
    fn test_v0_legacy_load_preserves_ir_path_null() {
        let v0_json = r#"{"input_gain_db": 3.0, "output_gain_db": -6.0, "gate_threshold_db": -50.0, "model_path": null, "bypass": false}"#;
        let loaded = deserialize_from_context(v0_json.as_bytes());
        assert_eq!(
            loaded.ir_path, None,
            "v0 legacy ir_path should default to None"
        );
    }

    #[test]
    fn test_serialize_envelope_produces_valid_v1_json() {
        let params = make_test_params();
        let buf = crate::clap::extensions::state::serialize_envelope(&params).unwrap();

        let envelope: serde_json::Value =
            serde_json::from_slice(&buf).expect("should deserialize as JSON value");
        assert!(envelope.get("version").is_some());
        assert!(envelope.get("params").is_some());
        assert!(envelope.get("params").unwrap().get("ir_path").is_some());
    }
}
