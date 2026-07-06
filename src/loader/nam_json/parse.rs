// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! High-level parsing function for the `.nam` format (JSON).

use super::data::{JsonError, NamModelData};
use super::validation::{MAX_HIDDEN_SIZE, MAX_LAYERS};

/// Raw universal deserialization of the JSON string via `serde_json`.
///
/// After deserialization, performs post-parse OOM-safety validation of the
/// declared topology: layer count must not exceed [`MAX_LAYERS`] and
/// `hidden_size` must not exceed [`MAX_HIDDEN_SIZE`]. Returns
/// `JsonError::UnsupportedTopology` immediately if limits are breached.
pub fn parse_nam_json(json_str: &str) -> Result<NamModelData, JsonError> {
    let data: NamModelData = serde_json::from_str(json_str)?;
    validate_topology_limits(&data)?;
    Ok(data)
}

fn validate_topology_limits(data: &NamModelData) -> Result<(), JsonError> {
    let layers = data.config.layers.len();
    if layers > MAX_LAYERS {
        return Err(JsonError::UnsupportedTopology {
            architecture: data.architecture.clone(),
            issue: format!(
                "config.layers has {} entries (max is {})",
                layers, MAX_LAYERS
            ),
            limit: MAX_LAYERS,
        });
    }

    if let Some(hidden_size) = data.config.hidden_size
        && hidden_size > MAX_HIDDEN_SIZE
    {
        return Err(JsonError::UnsupportedTopology {
            architecture: data.architecture.clone(),
            issue: format!(
                "config.hidden_size={} exceeds maximum {}",
                hidden_size, MAX_HIDDEN_SIZE
            ),
            limit: MAX_HIDDEN_SIZE,
        });
    }

    Ok(())
}
