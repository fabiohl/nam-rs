// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LSTM Gate-Major weight transposition.
//!
//! **Cross-module dependency:** The output layout must remain in lock-step with
//! the decoder in `dispatcher/lstm.rs` (weight reading order, bias/state layout).
//! See `dispatcher/lstm.rs` L154 (state order: hidden_init + cell_init) and
//! L265 (head_weights + head_bias append).

use crate::loader::nam_json::NamModelData;
use anyhow::{Context, Result};

/// Specialized for LSTM: rearranges the data to optimize matrix operations.
pub(crate) fn transpose_lstm_gate_major(data: &NamModelData) -> Result<Vec<f32>> {
    if data.architecture != "LSTM" {
        anyhow::bail!("Layout GateMajorLstm requires LSTM architecture");
    }

    let hidden_size = data
        .config
        .hidden_size
        .context("LSTM without hidden_size")?;
    let num_layers = data.config.num_layers.unwrap_or(1);

    let mut cursor = 0;
    let mut out_weights = Vec::with_capacity(data.weights.len());

    let mut current_input_size = 1;
    for l in 0..num_layers {
        let ih = current_input_size + hidden_size;
        let h = hidden_size;
        let layer_size = 4 * h * ih;

        if cursor + layer_size > data.weights.len() {
            anyhow::bail!("Insufficient weights for LSTM at layer {l}");
        }

        let raw = &data.weights[cursor..cursor + layer_size];
        let mut transposed = vec![0.0f32; layer_size];

        // Triple loop: we rearrange the rows and columns of the data (transposition).
        // This allows the program to perform mathematical calculations much faster.
        for k in 0..4 {
            for i in 0..h {
                for j in 0..ih {
                    let val = raw[k * h * ih + i * ih + j];
                    transposed[k * h * ih + j * h + i] = val;
                }
            }
        }
        out_weights.extend(transposed);
        cursor += layer_size;

        let bias_size = 4 * h;
        if cursor + bias_size > data.weights.len() {
            anyhow::bail!("Insufficient bias for LSTM at layer {l}");
        }
        out_weights.extend_from_slice(&data.weights[cursor..cursor + bias_size]);
        cursor += bias_size;

        // Processing of the layer's initial states (hidden_init and cell_init).
        // This is necessary to maintain parity with the order the decoder expects.
        // MUST match dispatcher/lstm.rs: L154 state read order.
        let state_size = 2 * h; // hidden_init (h) + cell_init (h)
        if cursor + state_size > data.weights.len() {
            anyhow::bail!("Insufficient initial states for LSTM at layer {l}");
        }
        out_weights.extend_from_slice(&data.weights[cursor..cursor + state_size]);
        cursor += state_size;

        current_input_size = hidden_size;
    }

    // If there are extra weights at the end of the file (such as the output layer),
    // we add them without modification (head_weights and head_bias).
    if cursor < data.weights.len() {
        out_weights.extend_from_slice(&data.weights[cursor..]);
    }

    Ok(out_weights)
}
