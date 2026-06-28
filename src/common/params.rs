// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host-agnostic parameters for NAM-rs.
//!
//! This module defines the complete processing configuration state,
//! allowing different hosts (CLI/PipeWire or CLAP) to manage and
//! synchronize parameters consistently.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub use crate::dsp::adaptive::AdaptiveComputeMode;
pub use crate::dsp::adaptive::SlimOverride;
use crate::dsp::oversample::OversampleFactor;

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
    /// Adaptive compute mode for soft-degrade under CPU pressure.
    /// Default: `Off` for standalone; `Conservative` for CLAP plugin.
    #[serde(default)]
    pub adaptive_compute: AdaptiveComputeMode,
    /// Manual slim override quality level. Default: `Auto` (FSM decides).
    #[serde(default)]
    pub slim_override: SlimOverride,
    /// Oversampling factor for the neural stage (anti-aliasing).
    /// Default: `Off` (lowest latency).
    #[serde(default)]
    pub oversample: OversampleFactor,
    /// Path to the loaded cab-sim impulse response (.wav).
    #[serde(default)]
    pub ir_path: Option<PathBuf>,
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
            adaptive_compute: AdaptiveComputeMode::Off,
            slim_override: SlimOverride::Auto,
            oversample: OversampleFactor::Off,
            ir_path: None,
        }
    }
}

/// Simplified parameter snapshot for the Real-Time audio thread.
/// Contains no heap-allocated fields (like PathBuf or String) to guarantee RT-safety when dropped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtPluginParams {
    /// Input gain in decibels (dB).
    pub input_gain_db: f32,
    /// Output gain in decibels (dB).
    pub output_gain_db: f32,
    /// Noise Gate threshold in decibels (dB).
    pub gate_threshold_db: f32,
    /// Bypass state.
    pub bypass: bool,
    /// Adaptive compute mode.
    pub adaptive_compute: AdaptiveComputeMode,
    /// Manual slim override quality level.
    pub slim_override: SlimOverride,
    /// Oversampling factor for the neural stage (anti-aliasing).
    pub oversample: OversampleFactor,
}

impl RtPluginParams {
    /// Extract the RT-safe parameters from a full NamPluginParams.
    pub fn from_plugin_params(params: &NamPluginParams) -> Self {
        Self {
            input_gain_db: params.input_gain_db,
            output_gain_db: params.output_gain_db,
            gate_threshold_db: params.gate_threshold_db,
            bypass: params.bypass,
            adaptive_compute: params.adaptive_compute,
            slim_override: params.slim_override,
            oversample: params.oversample,
        }
    }
}

impl Default for RtPluginParams {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            gate_threshold_db: GATE_THRESHOLD_DB_DEFAULT,
            bypass: false,
            adaptive_compute: AdaptiveComputeMode::Off,
            slim_override: SlimOverride::Auto,
            oversample: OversampleFactor::Off,
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
        assert_eq!(params.ir_path, None);
    }
}
