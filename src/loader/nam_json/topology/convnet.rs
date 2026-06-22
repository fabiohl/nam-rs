// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of ConvNet topologies from model data.

use super::super::data::NamModelData;
use super::super::model::HeadConfig;

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
