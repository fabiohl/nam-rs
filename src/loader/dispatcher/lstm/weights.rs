// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use crate::math::common::quantize_weight;
use crate::models::lstm::LstmLayer;

/// Fills `output` (buffer of `u16` with size `4 * hidden * (input + hidden)`)
/// with the LSTM weights quantized from `raw` (`f32` slice already read from the cursor).
///
/// Applies layout transposition when needed (Original → GateMajor).
pub(crate) fn read_lstm_weights_into(
    raw: &[f32],
    output: &mut [u16],
    is_gate_major: bool,
    hidden: usize,
    input: usize,
    is_bf16: bool,
) {
    let ih = input + hidden;
    if is_gate_major {
        for i in 0..output.len() {
            output[i] = quantize_weight(raw[i], is_bf16);
        }
    } else {
        for k in 0..4 {
            let gate_offset = k * hidden * ih;
            for i in 0..hidden {
                for j in 0..ih {
                    let v = raw[k * ih * hidden + i * ih + j];
                    output[gate_offset + j * hidden + i] = quantize_weight(v, is_bf16);
                }
            }
        }
    }
}

/// Reads the weights of an `LstmLayer<I, H, IH, H4>`.
///
/// Layout NAM JSON (C++ `LSTMLayerT::SetNAMWeights`):
/// ```text
/// input_hidden_weights: [H4 rows × IH cols] — row-major, direct mapping
/// bias:                 [H4]
/// initial_hidden:       [H]  → state[I..I+H]
/// initial_cell_state:   [H]  → cell_state[0..H]
/// ```
pub(crate) fn read_lstm_layer<const I: usize, const H: usize, const IH: usize, const H4: usize>(
    cursor: &mut WeightCursor<'_>,
    is_bf16: bool,
) -> anyhow::Result<LstmLayer<I, H, IH, H4>> {
    let mut layer = LstmLayer::<I, H, IH, H4>::new();

    let raw_weights = cursor.read_slice(H4 * IH)?;
    let is_gate_major = cursor.is_gate_major_lstm();

    let dst = unsafe {
        core::slice::from_raw_parts_mut(
            layer.input_hidden_weights.as_mut_ptr() as *mut u16,
            H4 * IH,
        )
    };
    read_lstm_weights_into(raw_weights, dst, is_gate_major, H, I, is_bf16);

    let bias_data = cursor.read_slice(H4)?;
    layer.bias.copy_from_slice(bias_data);

    let hidden_init = cursor.read_slice(H)?;
    layer.state[I..I + H].copy_from_slice(hidden_init);

    let cell_init = cursor.read_slice(H)?;
    layer.cell_state.copy_from_slice(cell_init);

    Ok(layer)
}
