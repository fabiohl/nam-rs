// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::loader::nam_json::{
    NamModelData, NamWavenetTopology, get_wavenet_topology, is_a2_shape,
};
use crate::models::DynamicModel;
use crate::models::a2::{WaveNetA2, WavenetA2Placeholder};
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

pub use dynamic::build_wavenet_dynamic;
pub use layout::transpose_conv1d_interleaved_4wide;

// =============================================================================
// Validation
// =============================================================================

/// Validates the `activation` field in all layers of a WaveNet model.
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
pub(crate) fn build_wavenet(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    // ── A2: first-class branch (detected by shape, not fallback) ──
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
                Ok(Box::new(DynamicModel::WavenetA2Lite(Box::new(model))))
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
                Ok(Box::new(DynamicModel::WavenetA2Full(Box::new(model))))
            }
            _ => unreachable!("is_a2_shape only returns 3 or 8"),
        };
    }

    // ── A1: static topology detection ──
    let topo_opt = get_wavenet_topology(data);

    match topo_opt {
        Some(NamWavenetTopology::Standard) => {
            let model = standard::build_wavenet_standard(data)?;
            Ok(Box::new(DynamicModel::WavenetStandard(Box::new(model))))
        }
        Some(NamWavenetTopology::Lite) => {
            let model = lite::build_wavenet_lite(data)?;
            Ok(Box::new(DynamicModel::WavenetLite(Box::new(model))))
        }
        Some(NamWavenetTopology::Feather) => {
            let model = feather::build_wavenet_feather(data)?;
            Ok(Box::new(DynamicModel::WavenetFeather(Box::new(model))))
        }
        Some(NamWavenetTopology::Nano) => {
            let model = nano::build_wavenet_nano(data)?;
            Ok(Box::new(DynamicModel::WavenetNano(Box::new(model))))
        }
        None => {
            // ── A2 heuristic fallback (SemVer / activation-based) ──
            if data.is_wavenet_a2() {
                let channels = data
                    .config
                    .layers
                    .first()
                    .and_then(|l| l.channels)
                    .map(|c| c as u8)
                    .unwrap_or(0);
                info!(
                    "[Dispatcher] WaveNet A2 model detected (SemVer heuristic, channels={}). Using temporary placeholder...",
                    channels
                );
                return Ok(Box::new(DynamicModel::WavenetA2(Box::new(
                    WavenetA2Placeholder::new(channels),
                ))));
            }

            // ── Dynamic fallback (pending Sprint 1.5 removal) ──
            dynamic::build_wavenet_dynamic(data)
        }
    }
}
