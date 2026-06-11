// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of WaveNet and LSTM topologies from model data.

use super::data::NamModelData;

/// The closed and supported topologies within native WaveNet modeling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamWavenetTopology {
    /// Channels: 16 (Standard)
    Standard,
    /// Channels: 12 (Lite)
    Lite,
    /// Channels: 8 (Feather)
    Feather,
    /// Channels: 4 (Nano)
    Nano,
}

static STD_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
static LITE_DILATIONS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
static LITE_DILATIONS_2: &[usize] = &[128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

/// Parses a version string in minimal SemVer format.
/// Supports pre-release or metadata suffixes and 'v' or 'V' prefix.
/// Returns `Some((major, minor, patch))` or `None` on parsing failure.
pub(crate) fn parse_semver(version: &str) -> Option<(u16, u16, u16)> {
    let clean = version.trim().trim_start_matches(['v', 'V']);
    let clean = clean.split('-').next()?.split('+').next()?;
    let mut parts = clean.split('.');
    let major = parts.next()?.trim().parse::<u16>().ok()?;
    let minor = parts.next().unwrap_or("0").trim().parse::<u16>().ok()?;
    let patch = parts.next().unwrap_or("0").trim().parse::<u16>().ok()?;
    Some((major, minor, patch))
}

impl NamModelData {
    /// Detects whether the model uses the WaveNet A2 architecture.
    ///
    /// A model is considered A2 if the architecture is "WaveNet" and:
    /// 1. The declared version is >= 0.6.0.
    /// 2. It uses activation functions other than "Tanh".
    pub fn is_wavenet_a2(&self) -> bool {
        if self.architecture != "WaveNet" {
            return false;
        }

        if let Some(ref v) = self.version
            && let Some(ver) = parse_semver(v)
            && ver >= (0, 6, 0)
        {
            return true;
        }

        for layer in &self.config.layers {
            if layer.activation.as_deref().is_some_and(|a| a != "Tanh") {
                return true;
            }
        }

        false
    }
}

/// Based on NeuralModel.cpp (`L:155-218`), checks the static identity of the WaveNet topology.
pub fn get_wavenet_topology(data: &NamModelData) -> Option<NamWavenetTopology> {
    if data.architecture != "WaveNet" || data.config.layers.len() != 2 {
        return None;
    }

    let l0 = &data.config.layers[0];
    let l1 = &data.config.layers[1];

    let l0_gated = l0.gated.unwrap_or(false);
    let l1_gated = l1.gated.unwrap_or(false);
    let l0_head_bias = l0.head_bias.unwrap_or(false);
    let l1_head_bias = l1.head_bias.unwrap_or(false);

    if l0_gated || l1_gated || l0_head_bias || !l1_head_bias {
        return None;
    }

    let channels = l0.channels?;
    let dils_0 = l0.dilations.as_deref()?;
    let dils_1 = l1.dilations.as_deref()?;

    match channels {
        16 if dils_0 == STD_DILATIONS && dils_1 == STD_DILATIONS => {
            Some(NamWavenetTopology::Standard)
        }
        12 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Lite)
        }
        8 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Feather)
        }
        4 if dils_0 == LITE_DILATIONS && dils_1 == LITE_DILATIONS_2 => {
            Some(NamWavenetTopology::Nano)
        }
        _ => None,
    }
}

/// Checks and returns the LSTM geometry (num_layers, hidden_size).
pub fn get_lstm_topology(data: &NamModelData) -> Option<(usize, usize)> {
    if data.architecture != "LSTM" {
        return None;
    }

    let num_layers = data.config.num_layers?;
    let hidden_size = data.config.hidden_size?;
    Some((num_layers, hidden_size))
}

/// Checks and returns the Linear geometry (receptive_field, has_bias).
pub fn get_linear_topology(data: &NamModelData) -> Option<(usize, bool)> {
    if data.architecture != "Linear" {
        return None;
    }

    let receptive_field = data.config.receptive_field?;
    let has_bias = data.config.bias.unwrap_or(false);
    Some((receptive_field, has_bias))
}

// =============================================================================
// A2 Shape Detection — Mirror of C++ is_a2_shape (a2_fast.cpp:754-885)
// =============================================================================

/// Shape-based A2 detector: returns `Some(channels)` if the parsed model data
/// matches the A2 architectural signature (single layer array, channels ∈ {3,8},
/// dilations matching kDilations). Returns `None` if the shape does not match.
///
/// This check is limited to fields available in `NamLayerConfig`. Full C++
/// equivalence requires additional fields (per-layer kernel_sizes, per-layer
/// activations with negative_slope, bottleneck, FiLM flags, etc.) not yet
/// captured by the Rust deserialization structs. Once those fields are added,
/// this function should be extended to match a2_fast.cpp:754-885 exactly.
pub fn is_a2_shape(data: &NamModelData) -> Option<u8> {
    use crate::models::a2::{A2_DILATIONS, A2_NUM_LAYERS, A2_VALID_CHANNELS};

    if data.architecture != "WaveNet" {
        return None;
    }

    let layers = &data.config.layers;

    // Exactly one layer array (C++ a2_fast.cpp:757-759)
    if layers.len() != 1 {
        return None;
    }

    let l0 = &layers[0];
    let ch = l0.channels? as u8;

    // Channels must be exactly 3 or 8 (C++ a2_fast.cpp:785-788)
    if !A2_VALID_CHANNELS.contains(&ch) {
        return None;
    }

    // Dilations must match kDilations exactly (C++ a2_fast.cpp:800-808)
    let dils = l0.dilations.as_deref()?;
    if dils.len() != A2_NUM_LAYERS {
        return None;
    }
    if dils.iter().zip(A2_DILATIONS.iter()).any(|(a, b)| *a != *b) {
        return None;
    }

    Some(ch)
}
