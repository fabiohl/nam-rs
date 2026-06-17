// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::loader::nam_json::{
    NamModelData, WavenetTopologyResult, get_wavenet_topology, is_a2_shape,
    validate_wavenet_features,
};
use crate::models::StaticModel;
use crate::models::a2::WaveNetA2;
use anyhow::bail;
use log::info;

mod bias_tune;
pub(crate) mod dynamic;
pub(crate) mod feather;
pub(crate) mod layout;
pub(crate) mod lite;
pub(crate) mod nano;
pub(crate) mod standard;
mod traits;

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
    // ── Feature validation: reject unsupported WaveNet features early ──
    if let Err(e) = validate_wavenet_features(data) {
        bail!("{e}");
    }

    // ── A2: first-class branch (detected by shape) ──
    if let Some(ch) = is_a2_shape(data) {
        return match ch {
            3 => {
                let mut model = WaveNetA2::<3>::new();
                model
                    .set_weights(&data.weights)
                    .map_err(|e| anyhow::anyhow!("A2-Lite weight load failed: {e}"))?;
                info!(
                    "[Dispatcher] WaveNet A2-Lite built — CH=3, layers=23, weights={}",
                    data.weights.len()
                );
                Ok(Box::new(StaticModel::WavenetA2Lite(Box::new(model))))
            }
            8 => {
                let mut model = WaveNetA2::<8>::new();
                model
                    .set_weights(&data.weights)
                    .map_err(|e| anyhow::anyhow!("A2-Full weight load failed: {e}"))?;
                info!(
                    "[Dispatcher] WaveNet A2-Full built — CH=8, layers=23, weights={}",
                    data.weights.len()
                );
                Ok(Box::new(StaticModel::WavenetA2Full(Box::new(model))))
            }
            _ => unreachable!("is_a2_shape only returns 3 or 8"),
        };
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
