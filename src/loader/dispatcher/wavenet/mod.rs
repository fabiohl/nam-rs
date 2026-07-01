// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::loader::nam_json::{
    A2TopologyResult, NamModelData, WavenetTopologyResult, get_wavenet_topology, is_a2_shape,
};
use crate::models::StaticModel;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::GatingMode;
use crate::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES, A2_LEAKY_SLOPE};
use crate::models::a2::{WaveNetA2, WaveNetA2Dyn};
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;
use anyhow::bail;
use log::info;

pub(crate) mod dynamic;
pub(crate) mod feather;
pub(crate) mod layout;
pub(crate) mod lite;
pub(crate) mod nano;
pub(crate) mod standard;
pub(crate) mod traits;

pub use layout::transpose_conv1d_interleaved_4wide;

// =============================================================================
// Validation
// =============================================================================

/// Validates the `activation` field in all layers of a WaveNet A1 model.
///
/// **Scope: A1 topologies only** (Standard, Lite, Feather, Nano). A2 models use
/// `LeakyReLU` (not `Tanh`) and are dispatched by `is_a2_shape` before this branch
/// is ever reached — so this function is never called for A2.
///
/// Called from `build_wavenet_typed` (which backs all A1 builders).
/// Returns an error if any layer declares an unsupported activation function.
pub(crate) fn validate_layer_activations(data: &NamModelData) -> anyhow::Result<()> {
    for (idx, layer) in data.config.layers.iter().enumerate() {
        let act = layer.activation.as_deref().unwrap_or("Tanh");
        if act != "Tanh" {
            bail!(
                "Activation '{}' in layer {} is not supported. Only 'Tanh' is implemented.",
                act,
                idx
            );
        }
    }
    Ok(())
}

// =============================================================================
// WaveNet dispatcher entry point
// =============================================================================

/// Detects the WaveNet topology and branches to the correct const-generic builder.
pub(crate) fn build_wavenet(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    for layer in &data.config.layers {
        if let Some(ref raw) = layer.layer_raw
            && raw
                .as_object()
                .is_some_and(|obj| {
                    obj.get("slimmable")
                        .is_some_and(|v| !v.is_null())
                })
        {
            bail!(
                "slimmable single-net weight slicing is not supported; use SlimmableContainer instead"
            );
        }
    }

    // ── A2: first-class branch (detected by shape) ──
    if let Some(topo) = is_a2_shape(data) {
        let layer_raw = data.config.layers.first().and_then(|l| l.layer_raw.clone());
        match topo {
            A2TopologyResult::KnownFastPath(3) => {
                let mut model = WaveNetA2::<3>::new()?;
                model.set_layer_raw(layer_raw);
                model
                    .set_weights(&data.weights)
                    .map_err(|e| anyhow::anyhow!("A2-Lite weight load failed: {e}"))?;
                info!(
                    "[Dispatcher] WaveNet A2-Lite built — CH=3, layers=23, weights={}",
                    data.weights.len()
                );
                return Ok(Box::new(StaticModel::WavenetA2Lite(Box::new(model))));
            }
            A2TopologyResult::KnownFastPath(8) => {
                let mut model = WaveNetA2::<8>::new()?;
                model.set_layer_raw(layer_raw);
                model
                    .set_weights(&data.weights)
                    .map_err(|e| anyhow::anyhow!("A2-Full weight load failed: {e}"))?;
                info!(
                    "[Dispatcher] WaveNet A2-Full built — CH=8, layers=23, weights={}",
                    data.weights.len()
                );
                return Ok(Box::new(StaticModel::WavenetA2Full(Box::new(model))));
            }
            A2TopologyResult::KnownFastPath(unexpected) => {
                bail!(
                    "Unexpected A2 channels in KnownFastPath: {}. Only 3 (Lite) and 8 (Full) are supported.",
                    unexpected
                )
            }
            A2TopologyResult::Dynamic => {
                use crate::loader::nam_json::validation::{
                    MAX_A2_DYN_BOTTLENECK, MAX_A2_DYN_CHANNELS,
                };

                let l0 = &data.config.layers[0];
                let channels = l0.channels.unwrap_or(0);
                let bottleneck = l0.bottleneck.unwrap_or(channels);
                let kernel_sizes = if let Some(ks) = l0.kernel_sizes.clone() {
                    ks
                } else if let Some(ks_scalar) = l0.kernel_size {
                    // A2 generic with scalar kernel_size — repeat for each layer
                    // defined by the dilations array.
                    let dils = l0
                        .dilations
                        .clone()
                        .unwrap_or_else(|| A2_DILATIONS.to_vec());
                    vec![ks_scalar; dils.len()]
                } else {
                    A2_KERNEL_SIZES.to_vec()
                };
                let dilations = l0
                    .dilations
                    .clone()
                    .unwrap_or_else(|| A2_DILATIONS.to_vec());
                let num_layers = kernel_sizes.len();

                if channels > MAX_A2_DYN_CHANNELS {
                    bail!(
                        "A2-Dynamic channels ({}) exceeds maximum {} (OOM/DoS protection)",
                        channels,
                        MAX_A2_DYN_CHANNELS
                    );
                }
                if bottleneck > MAX_A2_DYN_BOTTLENECK {
                    bail!(
                        "A2-Dynamic bottleneck ({}) exceeds maximum {} (OOM/DoS protection)",
                        bottleneck,
                        MAX_A2_DYN_BOTTLENECK
                    );
                }

                // Parse activations from raw JSON using actual layer count.
                let act_cfg = l0
                    .parse_activation_config(num_layers)
                    .map(|cfg| (cfg.activations, cfg.gating_modes, cfg.secondary_activations));
                let (activations, gating_modes, secondary_activations) = match act_cfg {
                    Some((a, g, s)) => (a, g, s),
                    None => {
                        // Fallback: standard LeakyReLU, no gating.
                        (
                            vec![
                                ActivationType::LeakyReLU {
                                    negative_slope: A2_LEAKY_SLOPE,
                                };
                                num_layers
                            ],
                            vec![GatingMode::None; num_layers],
                            vec![None; num_layers],
                        )
                    }
                };

                // Detect head1x1 from raw JSON.
                let head1x1_active = l0
                    .layer_raw
                    .as_ref()
                    .and_then(|raw| raw.get("head1x1"))
                    .and_then(|h| h.get("active"))
                    .and_then(|a| a.as_bool())
                    .unwrap_or(false);

                let condition_size = l0.condition_size.unwrap_or(1);

                let mut model = WaveNetA2Dyn::new(
                    channels,
                    bottleneck,
                    &kernel_sizes,
                    &dilations,
                    activations,
                    gating_modes,
                    secondary_activations,
                    head1x1_active,
                )?;
                model.set_layer_raw(layer_raw);
                model.condition_size = condition_size;
                model
                    .set_weights(&data.weights)
                    .map_err(|e| anyhow::anyhow!("A2-Dynamic weight load failed: {e}"))?;

                // Build condition_dsp sub-model if present in JSON config.
                if let Some(ref cond_dsp_json) = data.config.condition_dsp {
                    let cond_dsp_data: NamModelData =
                        serde_json::from_value(cond_dsp_json.clone())?;
                    let cond_model = crate::loader::dispatcher::build_model(&cond_dsp_data)?;
                    let cond_out = cond_model.num_output_channels();
                    if cond_out != condition_size {
                        bail!(
                            "condition_dsp output channels ({}) must match A2 condition_size ({})",
                            cond_out,
                            condition_size
                        );
                    }
                    info!(
                        "[Dispatcher] A2-Dynamic condition_dsp built — architecture={}, output_channels={}",
                        cond_dsp_data.architecture, cond_out
                    );
                    model.set_condition_dsp(Some(cond_model), WAVENET_MAX_NUM_FRAMES);
                }

                info!(
                    "[Dispatcher] WaveNet A2-Dynamic built — CH={}, BN={}, layers={}, weights={}",
                    channels,
                    bottleneck,
                    num_layers,
                    data.weights.len()
                );
                return Ok(Box::new(StaticModel::WavenetA2Dyn(Box::new(model))));
            }
        }
    }

    // ── A2: activation-based detection (secondary) — reject before A1 validation ──
    if data.is_wavenet_a2() {
        let layer_info: Vec<(usize, usize)> = data
            .config
            .layers
            .iter()
            .map(|l| {
                let ch = l.channels.unwrap_or(0);
                let k = l.kernel_size.unwrap_or(0);
                (ch, k)
            })
            .collect();
        bail!(
            "WaveNet A2 model detected but architecture shape not recognized — \
             channels or dilations do not match any known A2 topology. \
             Real A2 inference requires channels=3 (Lite) or 8 (Full) with the \
             canonical 23-layer dilation pattern. \
             Geometry: {:?}",
            layer_info
        );
    }

    // ── A1: topology detection (3-way) ──
    let topo = get_wavenet_topology(data);

    match topo {
        WavenetTopologyResult::Known(crate::loader::nam_json::NamWavenetTopology::Standard) => {
            let model = standard::build_wavenet_standard(data)?;
            Ok(Box::new(StaticModel::WavenetStandard(Box::new(model))))
        }
        WavenetTopologyResult::Known(crate::loader::nam_json::NamWavenetTopology::Lite) => {
            let model = lite::build_wavenet_lite(data)?;
            Ok(Box::new(StaticModel::WavenetLite(Box::new(model))))
        }
        WavenetTopologyResult::Known(crate::loader::nam_json::NamWavenetTopology::Feather) => {
            let model = feather::build_wavenet_feather(data)?;
            Ok(Box::new(StaticModel::WavenetFeather(Box::new(model))))
        }
        WavenetTopologyResult::Known(crate::loader::nam_json::NamWavenetTopology::Nano) => {
            let model = nano::build_wavenet_nano(data)?;
            Ok(Box::new(StaticModel::WavenetNano(Box::new(model))))
        }
        WavenetTopologyResult::Free(ref geom) => {
            let model = dynamic::build_wavenet_dynamic(data, geom)?;
            Ok(Box::new(StaticModel::WavenetDyn(Box::new(model))))
        }
        WavenetTopologyResult::Rejected(reason) => {
            let layer_info: Vec<(usize, usize)> = data
                .config
                .layers
                .iter()
                .map(|l| {
                    let ch = l.channels.unwrap_or(0);
                    let k = l.kernel_size.unwrap_or(0);
                    (ch, k)
                })
                .collect();
            bail!(
                "WaveNet model rejected: {reason}. \
                 Detected: {} layer(s) with geometry {layer_info:?}",
                data.config.layers.len(),
            );
        }
    }
}
