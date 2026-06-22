// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of LSTM topologies from model data.

use super::super::data::NamModelData;

/// Checks and returns the LSTM geometry (num_layers, hidden_size).
///
/// Rejects topologies that exceed safe bounds to prevent DoS/OOM:
/// - `num_layers > MAX_LSTM_LAYERS` (16)
/// - `hidden_size > MAX_LSTM_HIDDEN_SIZE` (1024)
pub fn get_lstm_topology(data: &NamModelData) -> Option<(usize, usize)> {
    use super::super::validation::{MAX_LSTM_HIDDEN_SIZE, MAX_LSTM_LAYERS};

    if data.architecture != "LSTM" {
        return None;
    }

    let num_layers = data.config.num_layers?;
    let hidden_size = data.config.hidden_size?;

    if num_layers > MAX_LSTM_LAYERS {
        log::warn!("LSTM num_layers={num_layers} exceeds maximum {MAX_LSTM_LAYERS} — rejected");
        return None;
    }
    if hidden_size > MAX_LSTM_HIDDEN_SIZE {
        log::warn!(
            "LSTM hidden_size={hidden_size} exceeds maximum {MAX_LSTM_HIDDEN_SIZE} — rejected"
        );
        return None;
    }
    Some((num_layers, hidden_size))
}
