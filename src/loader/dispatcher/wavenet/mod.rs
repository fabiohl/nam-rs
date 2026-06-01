// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::loader::nam_json::{NamModelData, NamWavenetTopology, get_wavenet_topology};
use crate::models::DynamicModel;
use anyhow::bail;
use log::info;

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
// Validação
// =============================================================================

/// Valida o campo `activation` em todas as layers de um modelo WaveNet.
pub(crate) fn validate_layer_activations(data: &NamModelData) -> anyhow::Result<()> {
    for (idx, layer) in data.config.layers.iter().enumerate() {
        let act = layer.activation.as_deref().unwrap_or("Tanh");
        if act != "Tanh" {
            bail!(
                "Ativação '{}' na layer {} não é suportada. Apenas 'Tanh' é implementado.",
                act,
                idx
            );
        }
    }
    Ok(())
}

// =============================================================================
// Ponto de entrada do dispatcher WaveNet
// =============================================================================

/// Detecta a topologia do WaveNet e bifurca para o construtor const-generic correto.
pub(crate) fn build_wavenet(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    let topo_opt = get_wavenet_topology(data);

    let res = match topo_opt {
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
        None => dynamic::build_wavenet_dynamic(data),
    };

    if res.is_err() && data.is_wavenet_a2() {
        info!("[Dispatcher] Modelo WaveNet A2 detectado. Utilizando placeholder temporário...");
        return Ok(Box::new(DynamicModel::WavenetA2(Box::default())));
    }

    res
}
