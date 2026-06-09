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
use crate::common::params::{NamPluginParams, RtPluginParams};
use clack_common::stream::{InputStream, OutputStream};
use clack_extensions::state_context::{
    PluginStateContext, PluginStateContextImpl, StateContextType,
};
use clack_plugin::prelude::*;
use std::io::{Read, Write};

/// Type alias for the CLAP state-context extension registration.
pub type NamPluginStateContext = PluginStateContext;

impl<'a> PluginStateContextImpl for NamClapMainThread<'a> {
    fn save(
        &mut self,
        output: &mut OutputStream,
        context_type: StateContextType,
    ) -> Result<(), PluginError> {
        let save_params = if context_type == StateContextType::ForPreset {
            let mut preset_params = self.params.clone();
            preset_params.model_path = None;
            preset_params
        } else {
            self.params.clone()
        };

        let serialized = serde_json::to_vec(&save_params)
            .map_err(|e| PluginError::Error(Box::new(std::io::Error::other(e))))?;

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

        let loaded_params: NamPluginParams = serde_json::from_slice(&buffer)
            .map_err(|e| PluginError::Error(Box::new(std::io::Error::other(e))))?;

        if context_type == StateContextType::ForPreset {
            self.params.input_gain_db = loaded_params.input_gain_db;
            self.params.output_gain_db = loaded_params.output_gain_db;
            self.params.gate_threshold_db = loaded_params.gate_threshold_db;
            self.params.bypass = loaded_params.bypass;
            self.params.adaptive_compute = loaded_params.adaptive_compute;

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
                    let msg = format!(
                        "NAM-rs: Failed to restore model from preset ({:?}): {}",
                        new_path, e
                    );
                    return Err(PluginError::Message(Box::leak(msg.into_boxed_str())));
                }
            }
        } else {
            let model_path = self.params.model_path.clone();
            self.params = loaded_params;
            if let Some(ref path) = model_path
                && path.exists()
                && let Err(e) = self.load_model(path)
            {
                let msg = format!("NAM-rs: Failed to restore model ({:?}): {}", path, e);
                return Err(PluginError::Message(Box::leak(msg.into_boxed_str())));
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
    use super::*;
    use crate::common::params::AdaptiveComputeMode;
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
        }
    }

    /// Helper: serializes params via save-like logic for a given context type.
    fn serialize_for_context(params: &NamPluginParams, context_type: StateContextType) -> Vec<u8> {
        let save_params = if context_type == StateContextType::ForPreset {
            let mut preset_params = params.clone();
            preset_params.model_path = None;
            preset_params
        } else {
            params.clone()
        };
        serde_json::to_vec(&save_params).unwrap()
    }

    #[test]
    fn test_preset_save_omits_model_path() {
        let params = make_test_params();
        let buf = serialize_for_context(&params, StateContextType::ForPreset);

        let loaded: NamPluginParams = serde_json::from_slice(&buf).expect("should deserialize");
        assert_eq!(
            loaded.model_path, None,
            "preset should not contain model_path"
        );
        assert_eq!(loaded.model_basename, Some("test.nam".to_string()));
        assert!((loaded.input_gain_db - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_project_save_preserves_model_path() {
        let params = make_test_params();
        let buf = serialize_for_context(&params, StateContextType::ForProject);

        let loaded: NamPluginParams = serde_json::from_slice(&buf).expect("should deserialize");
        assert_eq!(
            loaded.model_path,
            Some(PathBuf::from("/tmp/test.nam")),
            "project save should preserve model_path"
        );
    }

    #[test]
    fn test_duplicate_save_preserves_model_path() {
        let params = make_test_params();
        let buf = serialize_for_context(&params, StateContextType::ForDuplicate);

        let loaded: NamPluginParams = serde_json::from_slice(&buf).expect("should deserialize");
        assert_eq!(
            loaded.model_path,
            Some(PathBuf::from("/tmp/test.nam")),
            "duplicate save should preserve model_path"
        );
    }

    #[test]
    fn test_preset_load_without_model_path_restores_audio_params() {
        let preset_json = r#"{"input_gain_db":2.5,"output_gain_db":-3.0,"gate_threshold_db":-40.0,"model_path":null,"model_basename":"test.nam","model_search_paths":[],"bypass":false,"adaptive_compute":"Off"}"#;
        let loaded: NamPluginParams =
            serde_json::from_slice(preset_json.as_bytes()).expect("should deserialize");

        assert!((loaded.input_gain_db - 2.5).abs() < f32::EPSILON);
        assert!((loaded.output_gain_db - (-3.0)).abs() < f32::EPSILON);
        assert!((loaded.gate_threshold_db - (-40.0)).abs() < f32::EPSILON);
        assert_eq!(loaded.model_path, None);
        assert_eq!(loaded.model_basename, Some("test.nam".to_string()));
    }

    #[test]
    fn test_project_load_preserves_full_state() {
        let project_json = r#"{"input_gain_db":2.5,"output_gain_db":-3.0,"gate_threshold_db":-40.0,"model_path":"/tmp/test.nam","model_basename":"test.nam","model_search_paths":[],"bypass":false,"adaptive_compute":"Off"}"#;
        let loaded: NamPluginParams =
            serde_json::from_slice(project_json.as_bytes()).expect("should deserialize");

        assert_eq!(
            loaded.model_path,
            Some(PathBuf::from("/tmp/test.nam")),
            "project load should preserve model_path"
        );
    }
}
