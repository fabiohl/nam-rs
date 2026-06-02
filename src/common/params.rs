// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host-agnostic parameters for NAM-rs.
//!
//! This module defines the complete processing configuration state,
//! allowing different hosts (CLI/PipeWire or CLAP) to manage and
//! synchronize parameters consistently.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const GATE_THRESHOLD_DB_DEFAULT: f32 = -70.0;

/// Global processing parameters for the plugin/application.
///
/// This structure encapsulates all controls available to the user,
/// from basic gains to the path of the loaded neural model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamPluginParams {
    /// Input gain in decibels (dB). Default: 0.0.
    #[serde(default)]
    pub input_gain_db: f32,
    /// Output gain in decibels (dB). Default: 0.0.
    #[serde(default)]
    pub output_gain_db: f32,
    /// Noise Gate threshold in decibels (dB). Default: -70.0.
    /// This value maps to the `threshold_open_db` of the gate engine.
    #[serde(default = "default_gate_threshold_db")]
    pub gate_threshold_db: f32,
    /// Path to the loaded `.nam` or `.namb` model.
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    /// Base name of the model (filename only), used for portable lookup
    /// when the absolute path does not exist (cross-machine / cross-user).
    #[serde(default)]
    pub model_basename: Option<String>,
    /// Directories to search for the model if the absolute `model_path` does not exist.
    #[serde(default)]
    pub model_search_paths: Vec<PathBuf>,
    /// Bypass state (if `true`, audio passes without neural processing).
    #[serde(default)]
    pub bypass: bool,
}

fn default_gate_threshold_db() -> f32 {
    GATE_THRESHOLD_DB_DEFAULT
}

impl Default for NamPluginParams {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            gate_threshold_db: GATE_THRESHOLD_DB_DEFAULT,
            model_path: None,
            model_basename: None,
            model_search_paths: Vec::new(),
            bypass: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_default() {
        let params = NamPluginParams::default();
        assert_eq!(params.input_gain_db, 0.0);
        assert_eq!(params.output_gain_db, 0.0);
        assert_eq!(params.gate_threshold_db, -70.0);
        assert_eq!(params.model_path, None);
        assert_eq!(params.model_basename, None);
        assert!(params.model_search_paths.is_empty());
        assert!(!params.bypass);
    }
}
