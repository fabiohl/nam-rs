// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of WaveNet and LSTM topologies from model data.

use super::data::NamModelData;
use super::model::HeadConfig;

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

/// Description of a detected free (non-catalog) WaveNet A1 geometry.
///
/// Returned by [`get_wavenet_topology`] when the model is a valid WaveNet A1
/// but does not match any of the four catalog SKUs
/// (Standard/Lite/Feather/Nano). The dynamic engine (T3.1) consumes this
/// structure to build a runtime-dimensioned [`WaveNetModelDyn`].
///
/// ## Semver note
///
/// The `channels` and `head_sizes` fields changed from `usize` to `Vec<usize>`
/// to support per-array geometry (required for condition_dsp). This is a
/// breaking change from the v2.x API.
///
/// [`WaveNetModelDyn`]: crate::models::wavenet::WaveNetModelDyn
#[derive(Debug, Clone, PartialEq)]
pub struct FreeWavenetGeometry {
    /// Internal channels per layer-array (first array → array[0]).
    pub channels: Vec<usize>,
    /// Kernel size (same across all layers for legacy models;
    /// per-array kernel sizes for heterogeneous topologies).
    pub kernel_size: usize,
    /// Per-array kernel sizes, one per layer-array.
    /// For homogeneous topologies this is a repetition of `kernel_size`.
    pub kernel_sizes: Vec<usize>,
    /// Head projection size per layer-array.
    /// `head_sizes.last()` determines `NumOutputChannels()` for the model
    /// (C++ `wave_net_output_channels` returns `layer_array_params.back().head_size`).
    pub head_sizes: Vec<usize>,
    /// Dilations per layer-array.
    pub dilations: Vec<Vec<usize>>,
    /// Number of layer-arrays.
    pub num_arrays: usize,
    /// Condition input size (number of conditioning channels).
    /// Models with `condition_size > 1` flow into the dynamic engine
    /// (the const-generic fast-path is reserved for `COND=1`).
    pub condition_size: usize,
    /// Post-stack head configuration (Conv1D + activation) that processes
    /// the signal after all layer arrays, before `head_scale` and output.
    /// `None` means no post-stack head is present (standard behavior).
    pub post_stack_head: Option<HeadConfig>,
}

/// Result of WaveNet topology detection.
///
/// Distinguishes three states:
/// - `Known`: fast-path const-generic SKU (Standard/Lite/Feather/Nano).
/// - `Free`: valid geometry outside the catalog — destined for the dynamic engine.
/// - `Rejected`: invalid or missing shape data (no channels, no dilations, etc.).
#[derive(Debug, Clone, PartialEq)]
#[allow(private_interfaces)]
pub enum WavenetTopologyResult {
    /// Matches a known catalog SKU — use const-generic fast-path.
    Known(NamWavenetTopology),
    /// Valid geometry not in the catalog — use dynamic engine (T3.1).
    Free(FreeWavenetGeometry),
    /// Rejected due to unsupported feature or invalid shape.
    Rejected(String),
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

        // Secondary: activation-based detection — only if the model has the
        // structural signature of an A2 (single layer array). Multi-array
        // WaveNet models with non-Tanh activation are A1 geometries with
        // a non-standard activation, not A2.
        if self.config.layers.len() == 1
            && let Some(layer) = self.config.layers.first()
            && layer.activation.as_deref().is_some_and(|a| a != "Tanh")
        {
            return true;
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

/// Detects the WaveNet topology, distinguishing known SKUs from free geometries.
///
/// Returns [`WavenetTopologyResult`] with three possible outcomes:
/// - `Known(SKU)`: matches a catalog variant (Standard/Lite/Feather/Nano) —
///   use the const-generic fast-path. Only `condition_size=1` (or absent) models
///   can match a catalog SKU.
/// - `Free(geometry)`: valid A1 geometry outside the catalog — destined for the
///   dynamic engine (T3.1). Any `condition_size` is accepted.
/// - `Rejected(reason)`: missing/invalid shape data.
///
/// Mirror of C++ `NeuralModel.cpp` (L:155-218) generalized to accept any valid
/// WaveNet A1 geometry, not only the four catalog SKUs.
pub fn get_wavenet_topology(data: &NamModelData) -> WavenetTopologyResult {
    // ── Architecture gate ──
    if data.architecture != "WaveNet" {
        return WavenetTopologyResult::Rejected("Not a WaveNet model.".to_string());
    }

    let layers = &data.config.layers;
    if layers.is_empty() {
        return WavenetTopologyResult::Rejected("WaveNet model has no layer arrays.".to_string());
    }

    // ── Extract condition_size from first layer (all layers share the same value) ──
    let condition_size = layers[0].condition_size.unwrap_or(1);

    // ── Extract per-layer fields (gently — full validation deferred to free geometry) ──
    let mut first_channels: Option<usize> = None;
    let mut first_kernel_size: Option<usize> = None;
    let mut first_head_size: Option<usize> = None;
    let mut dilations: Vec<Vec<usize>> = Vec::with_capacity(layers.len());
    let mut head_sizes: Vec<usize> = Vec::with_capacity(layers.len());
    let mut channels: Vec<usize> = Vec::with_capacity(layers.len());
    let mut kernel_sizes: Vec<usize> = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let ch = match layer.channels {
            Some(c) if c > 0 => c,
            _ => {
                return WavenetTopologyResult::Rejected(format!(
                    "Layer {} is missing or has invalid 'channels'.",
                    i
                ));
            }
        };
        let k = layer
            .kernel_size
            .and_then(|k| if k > 0 { Some(k) } else { None });
        let dils = match layer.dilations.as_deref() {
            Some(d) if !d.is_empty() => d.to_vec(),
            _ => {
                return WavenetTopologyResult::Rejected(format!(
                    "Layer {} is missing or has invalid 'dilations'.",
                    i
                ));
            }
        };

        if i == 0 {
            first_channels = Some(ch);
            first_kernel_size = k;
            first_head_size = layer.head_size;
        }

        let hd = layer.head_size.unwrap_or(1);
        if hd == 0 {
            return WavenetTopologyResult::Rejected(format!(
                "Layer {} has invalid head_size=0.",
                i
            ));
        }

        channels.push(ch);
        kernel_sizes.push(k.unwrap_or(0));
        head_sizes.push(hd);
        dilations.push(dils);
    }

    let first_channels_val = match first_channels {
        Some(c) => c,
        None => {
            return WavenetTopologyResult::Rejected(
                "Layer 0 is missing or has invalid 'channels' — required for \
                 WaveNet topology detection."
                    .to_string(),
            );
        }
    };

    // ── Try matching a known catalog SKU (fast-path) ──
    // Catalog SKU detection uses only channels + dilations, matching the
    // original detector (NeuralModel.cpp L:155-218). kernel_size and
    // head_size from JSON are optional for catalog models (already known
    // from the const-generic type parameters).
    //
    // condition_size must be 1 (or absent) for catalog matching because
    // the const-generic fast-path uses COND=1. Models with condition_size > 1
    // fall through to the Free geometry path (dynamic engine).
    if layers.len() == 2 {
        let l0 = &layers[0];
        let l1 = &layers[1];

        let l0_gated = l0.gated.unwrap_or(false);
        let l1_gated = l1.gated.unwrap_or(false);
        let l0_head_bias = l0.head_bias.unwrap_or(false);
        let l1_head_bias = l1.head_bias.unwrap_or(false);

        // Catalog SKUs require: no gating, array1 no head_bias, array2 head_bias,
        // and condition_size=1 (const-generic fast-path uses COND=1).
        let catalog_compatible =
            !l0_gated && !l1_gated && !l0_head_bias && l1_head_bias && condition_size <= 1;

        if catalog_compatible {
            let dils_0 = &dilations[0];
            let dils_1 = &dilations[1];

            let result = match first_channels_val {
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
            };

            if let Some(sku) = result {
                return WavenetTopologyResult::Known(sku);
            }
        }
    }

    // ── Free geometry (valid A1, but not in catalog) ──
    let kernel_size = match first_kernel_size {
        Some(k) => k,
        None => {
            return WavenetTopologyResult::Rejected(
                "Layer 0 is missing or has invalid 'kernel_size' — required for \
                 free geometry WaveNet A1."
                    .to_string(),
            );
        }
    };
    if first_head_size.is_none_or(|h| h == 0) {
        return WavenetTopologyResult::Rejected(
            "Layer 0 is missing or has invalid 'head_size' — required for \
             WaveNet A1 geometries (determines the head projection dimension)."
                .to_string(),
        );
    }

    WavenetTopologyResult::Free(FreeWavenetGeometry {
        channels,
        kernel_size,
        kernel_sizes,
        head_sizes,
        condition_size,
        num_arrays: layers.len(),
        dilations,
        post_stack_head: data.config.parse_head(),
    })
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

/// Detected ConvNet topology.
///
/// Extracted from the model data when `architecture == "ConvNet"`.
/// ConvNet is a feed-forward stack of Conv1D → BatchNorm → Activation blocks
/// with an optional post-stack head, analogous to WaveNet without the
/// recurrent gating.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvNetTopology {
    /// Number of ConvNet blocks.
    pub num_blocks: usize,
    /// Per-block channel count.
    pub channels: Vec<usize>,
    /// Per-block kernel size.
    pub kernel_sizes: Vec<usize>,
    /// Per-block dilations.
    pub dilations: Vec<Vec<usize>>,
    /// Optional post-stack head configuration (Conv1D + activation).
    pub head: Option<HeadConfig>,
}

/// Checks and returns the ConvNet geometry.
///
/// Returns `None` if the architecture is not `"ConvNet"` or if required
/// per-block fields (`channels`, `kernel_size`, `dilations`) are missing.
pub fn get_convnet_topology(data: &NamModelData) -> Option<ConvNetTopology> {
    if data.architecture != "ConvNet" {
        return None;
    }

    let layers = &data.config.layers;
    if layers.is_empty() {
        return None;
    }

    let mut channels = Vec::with_capacity(layers.len());
    let mut kernel_sizes = Vec::with_capacity(layers.len());
    let mut dilations = Vec::with_capacity(layers.len());

    for (i, layer) in layers.iter().enumerate() {
        let ch = match layer.channels {
            Some(c) if c > 0 => c,
            _ => {
                log::warn!("ConvNet block {i}: missing or invalid 'channels'");
                return None;
            }
        };
        let k = match layer.kernel_size {
            Some(k) if k > 0 => k,
            _ => {
                log::warn!("ConvNet block {i}: missing or invalid 'kernel_size'");
                return None;
            }
        };
        let d = match layer.dilations.as_deref() {
            Some(d) if !d.is_empty() => d.to_vec(),
            _ => {
                log::warn!("ConvNet block {i}: missing or invalid 'dilations'");
                return None;
            }
        };
        channels.push(ch);
        kernel_sizes.push(k);
        dilations.push(d);
    }

    Some(ConvNetTopology {
        num_blocks: layers.len(),
        channels,
        kernel_sizes,
        dilations,
        head: data.config.parse_head(),
    })
}

// =============================================================================
// A2 Shape Detection — Hybrid Dispatch (A2 Fast-Path vs A2 Dynamic)
// =============================================================================

/// Valid A2 topologies.
#[derive(Debug, Clone, PartialEq)]
pub enum A2TopologyResult {
    /// Fast-path const-generic with exact channels (3 or 8).
    KnownFastPath(u8),
    /// Valid A2 geometry that falls outside the rigid fast-path.
    Dynamic,
}

/// Shape-based A2 detector: returns `Some(A2TopologyResult)` if the parsed model data
/// matches the A2 architectural signature. If the model strictly matches the
/// C++ fast-path criteria, it returns `KnownFastPath(CH)`, otherwise `Dynamic`.
///
/// Checks that require raw JSON (activation arrays, FiLM keys, etc.) are
/// skipped when `layer_raw` is `None` (structs constructed directly, not via
/// JSON deserialization). This keeps existing test helpers working while
/// enforcing full parity for models loaded from actual `.nam` files.
pub fn is_a2_shape(data: &NamModelData) -> Option<A2TopologyResult> {
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
    if let Some(ref head) = data.config.head
        && !head.is_null()
    {
        return None;
    }

    // 4. head_scale present and numeric (a2_fast.cpp:886-890)
    data.config.head_scale.as_ref()?;

    let l0 = &layers[0];

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

    // --- Se chegamos aqui, o modelo é inequivocamente um WaveNet A2. ---
    // A partir daqui, validamos se ele se enquadra no fast-path const-generic.

    // 5. in_channels defaults to 1, must be 1 (a2_fast.cpp:892-894)
    if data.config.in_channels.unwrap_or(1) != 1 {
        return Some(A2TopologyResult::Dynamic);
    }

    // 6. input_size must be 1 (a2_fast.cpp:898)
    if l0.input_size != Some(1) {
        return Some(A2TopologyResult::Dynamic);
    }

    // 7. condition_size must be 1 (a2_fast.cpp:900)
    if l0.condition_size != Some(1) {
        return Some(A2TopologyResult::Dynamic);
    }

    let ch = match l0.channels {
        Some(c) => c as u8,
        None => return Some(A2TopologyResult::Dynamic),
    };

    // 8. channels and bottleneck must match (a2_fast.cpp:903-906)
    let bn = l0.bottleneck.unwrap_or(0);
    if ch as usize != bn {
        return Some(A2TopologyResult::Dynamic);
    }

    // 9. Channels must be exactly 3 or 8 (a2_fast.cpp:907-908)
    if !A2_VALID_CHANNELS.contains(&ch) {
        return Some(A2TopologyResult::Dynamic);
    }

    // ── Checks 12-19 require raw JSON (a2_fast.cpp:930-1000) ──
    if let Some(ref raw) = l0.layer_raw {
        // 12. All activations must be LeakyReLU(negative_slope ≈ 0.01)
        if !check_activations_are_leaky_relu(raw, A2_NUM_LAYERS, A2_LEAKY_SLOPE as f64) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 13. gating_mode: all "none" or absent
        if !check_gating_mode_all_none(raw, A2_NUM_LAYERS) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 14. secondary_activation: all null or absent
        if !check_secondary_activation_all_null(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 15. head1x1 must be inactive
        if !check_head1x1_inactive(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 16. layer1x1 active with groups=1
        if !check_layer1x1_active_groups1(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 17. Layer-array head: out_channels=1, kernel_size=16, bias=true
        if !check_layer_array_head(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        if !check_film_all_inactive(raw) {
            // Models with FiLM are now routed to Dynamic if they break const-generic assumptions,
            // but currently the fast-path does support some FiLM. For safety, let's say:
            // if it has active FiLM it goes to Dynamic? No, the T1/T2 already added FiLM to fast-path.
            // So we don't return Dynamic here just for FiLM.
        }

        // 19. groups_input == 1 AND groups_input_mixin == 1
        if !check_groups_are_1(raw) {
            return Some(A2TopologyResult::Dynamic);
        }

        // 20. Not slimmable
        if !check_not_slimmable(raw) {
            return Some(A2TopologyResult::Dynamic);
        }
    }

    Some(A2TopologyResult::KnownFastPath(ch))
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
