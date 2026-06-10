// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::loader::nam_json::{
    NamModelData, NamWavenetTopology, get_wavenet_topology, is_a2_shape,
};
use crate::models::DynamicModel;
use crate::models::a2::WaveNetA2;
use anyhow::bail;
use log::info;

mod bias_tune;
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

            if data.is_wavenet_a2() {
                bail!(
                    "WaveNet A2 model detected but architecture shape not recognized — \
                     channels or dilations do not match any known A2 or A1 topology. \
                     Real A2 inference requires channels=3 (Lite) or 8 (Full) with the \
                     canonical 23-layer dilation pattern. \
                     Geometry: {:?}",
                    layer_info
                );
            }

            let num_layers = data.config.layers.len();
            let head_scale = data.config.head_scale.unwrap_or(0.0);
            bail!(
                "WaveNet topology not in catalog and dynamic fallback is no longer available. \
                 Only Standard (16ch/k3/d8), Lite (12ch/k3/d6), Feather (8ch/k3/d4), \
                 Nano (4ch/k3/d2), A2-Full (8ch), and A2-Lite (3ch) are supported. \
                 Detected: {} layer(s) with geometry {:?}, head_scale={}",
                num_layers,
                layer_info,
                head_scale
            );
        }
    }
}
