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
    /// A2 detection is layered:
    /// 1. **Primary — shape-based:** `is_a2_shape()` checks channels, dilations,
    ///    and single-layer structure. If shape matches, the model is A2.
    /// 2. **Secondary — activation:** any non-Tanh activation implies A2 semantics.
    /// 3. **Tertiary — version telemetry:** a SemVer >= 0.6.0 is logged as a
    ///    warning when the shape is not recognised, but version alone is NOT
    ///    sufficient to classify the model as A2. This prevents a WaveNet A1
    ///    model with a high version string from being silently misrouted.
    pub fn is_wavenet_a2(&self) -> bool {
        if self.architecture != "WaveNet" {
            return false;
        }

        // Primary: shape-based detection (channels + dilations + kernel sizes)
        if is_a2_shape(self).is_some() {
            return true;
        }

        // Secondary: activation-based detection
        for layer in &self.config.layers {
            if layer.activation.as_deref().is_some_and(|a| a != "Tanh") {
                return true;
            }
        }

        // Tertiary: version telemetry (warning only, does NOT classify)
        if let Some(ref v) = self.version
            && let Some(ver) = parse_semver(v)
            && ver >= (0, 6, 0)
        {
            log::warn!(
                "WaveNet model declares version {v} (>= 0.6.0), but its shape \
                 does not match any known A2 topology. Treating as non-A2. \
                 Channels/dilations may indicate an unsupported A2 variant \
                 or an A1 model with an unusually high version string."
            );
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
// A2 Shape Detection — Mirror of C++ is_a2_shape (a2_fast.cpp:875-908)
// =============================================================================

/// Shape-based A2 detector: returns `Some(channels)` if the parsed model data
/// matches the A2 architectural signature exactly (18 criteria from the C++
/// reference `a2_fast.cpp:875-908`). Returns `None` if the shape does not match.
///
/// Checks that require raw JSON (activation arrays, FiLM keys, etc.) are
/// skipped when `layer_raw` is `None` (structs constructed directly, not via
/// JSON deserialization). This keeps existing test helpers working while
/// enforcing full parity for models loaded from actual `.nam` files.
pub fn is_a2_shape(data: &NamModelData) -> Option<u8> {
    use crate::models::a2::{
        A2_DILATIONS, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS, A2_VALID_CHANNELS,
    };

    // 1. Architecture must be WaveNet
    if data.architecture != "WaveNet" {
        return None;
    }

    // 2. Exactly one layer array (a2_fast.cpp:876-879)
    let layers = &data.config.layers;
    if layers.len() != 1 {
        return None;
    }

    // 3. No post-stack head (a2_fast.cpp:881-884)
    //    Already enforced by validate_wavenet_features(), but check here for shape purity.
    if let Some(ref head) = data.config.head
        && head.is_some()
    {
        return None;
    }

    // 4. head_scale present and numeric (a2_fast.cpp:886-890)
    data.config.head_scale.as_ref()?;

    // 5. in_channels defaults to 1, must be 1 (a2_fast.cpp:892-894)
    if data.config.in_channels.unwrap_or(1) != 1 {
        return None;
    }

    let l0 = &layers[0];

    // 6. input_size must be 1 (a2_fast.cpp:898)
    if l0.input_size != Some(1) {
        return None;
    }

    // 7. condition_size must be 1 (a2_fast.cpp:900)
    if l0.condition_size != Some(1) {
        return None;
    }

    // 8. channels and bottleneck must match (a2_fast.cpp:903-906)
    let ch = l0.channels? as u8;
    let bn = l0.bottleneck.unwrap_or(0);
    if ch as usize != bn {
        return None;
    }

    // 9. Channels must be exactly 3 or 8 (a2_fast.cpp:907-908)
    if !A2_VALID_CHANNELS.contains(&ch) {
        return None;
    }

    // 10. kernel_sizes must match A2_KERNEL_SIZES exactly (a2_fast.cpp:910-918)
    let ks = l0.kernel_sizes.as_deref()?;
    if ks.len() != A2_NUM_LAYERS {
        return None;
    }
    if ks.iter().zip(A2_KERNEL_SIZES.iter()).any(|(a, b)| *a != *b) {
        return None;
    }

    // 11. Dilations must match A2_DILATIONS exactly (a2_fast.cpp:920-928)
    let dils = l0.dilations.as_deref()?;
    if dils.len() != A2_NUM_LAYERS {
        return None;
    }
    if dils.iter().zip(A2_DILATIONS.iter()).any(|(a, b)| *a != *b) {
        return None;
    }

    // ── Checks 12-19 require raw JSON (a2_fast.cpp:930-1000) ──
    if let Some(ref raw) = l0.layer_raw {
        // 12. All activations must be LeakyReLU(negative_slope ≈ 0.01) (a2_fast.cpp:930-940)
        if !check_activations_are_leaky_relu(raw, A2_NUM_LAYERS, A2_LEAKY_SLOPE as f64) {
            return None;
        }

        // 13. gating_mode: all "none" or absent (a2_fast.cpp:942-948)
        if !check_gating_mode_all_none(raw, A2_NUM_LAYERS) {
            return None;
        }

        // 14. secondary_activation: all null or absent (a2_fast.cpp:950-956)
        if !check_secondary_activation_all_null(raw) {
            return None;
        }

        // 15. head1x1 must be inactive (a2_fast.cpp:958-961)
        if !check_head1x1_inactive(raw) {
            return None;
        }

        // 16. layer1x1 active with groups=1 (a2_fast.cpp:963-970)
        if !check_layer1x1_active_groups1(raw) {
            return None;
        }

        // 17. Layer-array head: out_channels=1, kernel_size=16, bias=true
        //     (a2_fast.cpp:972-981)
        if !check_layer_array_head(raw) {
            return None;
        }

        // 18. No FiLM anywhere (a2_fast.cpp:983-989)
        if !check_film_all_inactive(raw) {
            return None;
        }

        // 19. groups_input == 1 AND groups_input_mixin == 1 (a2_fast.cpp:991-995)
        if !check_groups_are_1(raw) {
            return None;
        }

        // 20. Not slimmable (a2_fast.cpp:997-1000)
        if !check_not_slimmable(raw) {
            return None;
        }
    }

    Some(ch)
}

// ── Raw JSON helper functions for strict A2 shape checks ──

fn get_f64(val: &serde_json::Value) -> Option<f64> {
    val.as_f64().or_else(|| val.as_i64().map(|i| i as f64))
}

/// Checks that every entry in the `activation` array is a LeakyReLU with the
/// expected negative_slope.
fn check_activations_are_leaky_relu(
    raw: &serde_json::Value,
    num_layers: usize,
    slope: f64,
) -> bool {
    let arr = match raw.get("activation").and_then(|v| v.as_array()) {
        Some(a) if a.len() == num_layers => a,
        _ => return false,
    };
    for entry in arr {
        let obj = match entry.as_object() {
            Some(o) => o,
            None => return false,
        };
        if obj.get("type").and_then(|v| v.as_str()) != Some("LeakyReLU") {
            return false;
        }
        let ns = match obj.get("negative_slope").and_then(get_f64) {
            Some(n) => n,
            None => return false,
        };
        if (ns - slope).abs() > 1e-6 {
            return false;
        }
    }
    true
}

/// Checks that `gating_mode` is absent, null, or an array of all "none".
fn check_gating_mode_all_none(raw: &serde_json::Value, num_layers: usize) -> bool {
    match raw.get("gating_mode") {
        None | Some(serde_json::Value::Null) => true,
        Some(arr) => {
            let elems = match arr.as_array() {
                Some(a) if a.len() == num_layers => a,
                _ => return false,
            };
            elems
                .iter()
                .all(|v| v.as_str() == Some("none") || v.is_null())
        }
    }
}

/// Checks that `secondary_activation` is absent, null, or an array of all null.
fn check_secondary_activation_all_null(raw: &serde_json::Value) -> bool {
    match raw.get("secondary_activation") {
        None | Some(serde_json::Value::Null) => true,
        Some(arr) => match arr.as_array() {
            Some(a) => a.iter().all(|v| v.is_null()),
            None => false,
        },
    }
}

/// Checks that `head1x1` is absent, null, not an object, or active==false.
fn check_head1x1_inactive(raw: &serde_json::Value) -> bool {
    match raw.get("head1x1") {
        None | Some(serde_json::Value::Null) => true,
        Some(v) => match v.as_object() {
            Some(obj) => obj.get("active").and_then(|a| a.as_bool()) != Some(true),
            None => true,
        },
    }
}

/// Checks that `layer1x1` is present as an object, active==true, groups==1.
fn check_layer1x1_active_groups1(raw: &serde_json::Value) -> bool {
    let obj = match raw.get("layer1x1").and_then(|v| v.as_object()) {
        Some(o) => o,
        None => return false,
    };
    if obj.get("active").and_then(|v| v.as_bool()) != Some(true) {
        return false;
    }
    obj.get("groups").and_then(|g| g.as_u64()) == Some(1)
}

/// Checks the layer-array head: out_channels==1, kernel_size==16, bias==true.
fn check_layer_array_head(raw: &serde_json::Value) -> bool {
    use crate::models::a2::A2_HEAD_KERNEL_SIZE;
    let obj = match raw.get("head").and_then(|v| v.as_object()) {
        Some(o) => o,
        None => return false,
    };
    if obj.get("out_channels").and_then(|c| c.as_u64()) != Some(1) {
        return false;
    }
    if obj.get("kernel_size").and_then(|k| k.as_u64()) != Some(A2_HEAD_KERNEL_SIZE as u64) {
        return false;
    }
    obj.get("bias").and_then(|b| b.as_bool()) == Some(true)
}

/// Checks that all 8 FiLM keys are inactive (absent, null, not an object, or active==false).
fn check_film_all_inactive(raw: &serde_json::Value) -> bool {
    const FILM_KEYS: &[&str] = &[
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    ];
    for key in FILM_KEYS {
        match raw.get(key) {
            None | Some(serde_json::Value::Null) => continue,
            Some(v) => match v.as_object() {
                Some(obj) => {
                    if obj.get("active").and_then(|a| a.as_bool()) == Some(true) {
                        return false;
                    }
                }
                None => continue,
            },
        }
    }
    true
}

/// Checks that `groups_input == 1` AND `groups_input_mixin == 1`.
fn check_groups_are_1(raw: &serde_json::Value) -> bool {
    let gi = raw
        .get("groups_input")
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    let gim = raw
        .get("groups_input_mixin")
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    gi == 1 && gim == 1
}

/// Checks that `slimmable` is absent or null.
fn check_not_slimmable(raw: &serde_json::Value) -> bool {
    matches!(raw.get("slimmable"), None | Some(serde_json::Value::Null))
}

// =============================================================================
// WaveNet feature validation — condition_size and head (post-stack)
// =============================================================================

/// Validates that WaveNet models use only supported feature combinations.
///
/// The current NAM-rs implementation hardcodes `COND=1` as a const generic and
/// does not implement a post-stack head sub-object. Models that require
/// `condition_size != 1` or a non-null `head` would load and produce incorrect
/// output silently — the worst failure mode.
///
/// This validator **rejects** such models with a clear diagnostic, preventing
/// silent misbehavior. If official models requiring these features are found in
/// circulation (Tone3000/ToneHunt), they can be supported in a future release.
pub fn validate_wavenet_features(data: &NamModelData) -> Result<(), String> {
    // Check condition_size on all layers
    for (i, layer) in data.config.layers.iter().enumerate() {
        match layer.condition_size {
            None | Some(1) => {}
            Some(cs) => {
                return Err(format!(
                    "WaveNet layer {} has condition_size={}, but only condition_size=1 is \
                     supported. Multi-condition WaveNet is an official NAMCore feature not \
                     yet implemented in NAM-rs.",
                    i, cs
                ));
            }
        }
    }

    // Check head: must be absent (None) or null (Some(None))
    if let Some(ref head) = data.config.head
        && head.is_some()
    {
        return Err("WaveNet 'head' (post-stack sub-object) is not supported. \
             Only head=null is accepted for A1 WaveNet topologies. \
             Post-stack heads are an official NAMCore feature not yet \
             implemented in NAM-rs."
            .to_string());
    }

    Ok(())
}
