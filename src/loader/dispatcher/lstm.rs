// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::WeightCursor;
use crate::loader::nam_json::{NamModelData, get_lstm_topology};
use crate::math::common::quantize_weight;
use crate::models::DynamicModel;
use crate::models::lstm::{LstmDynLayer, LstmDynModel, LstmLayer, LstmModel1, LstmModel2};
use anyhow::Context;
use log::info;

/// Detects the LSTM geometry (num_layers × hidden_size) and dispatches to the correct constructor.
pub(crate) fn build_lstm(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    let (num_layers, hidden_size) = get_lstm_topology(data)
        .context("LSTM geometry not detectable (check num_layers and hidden_size)")?;

    match (num_layers, hidden_size) {
        (1, 8) => {
            let model = build_lstm_1layer::<8, 9, 32>(data, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm1x8(Box::new(model))))
        }
        (1, 12) => {
            let model = build_lstm_1layer::<12, 13, 48>(data, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm1x12(Box::new(model))))
        }
        (1, 16) => {
            let model = build_lstm_1layer::<16, 17, 64>(data, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm1x16(Box::new(model))))
        }
        (1, 24) => {
            let model = build_lstm_1layer::<24, 25, 96>(data, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm1x24(Box::new(model))))
        }
        (2, 8) => {
            let model = build_lstm_2layer::<8, 9, 16, 32>(data, num_layers, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm2x8(Box::new(model))))
        }
        (2, 12) => {
            let model = build_lstm_2layer::<12, 13, 24, 48>(data, num_layers, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm2x12(Box::new(model))))
        }
        (2, 16) => {
            let model = build_lstm_2layer::<16, 17, 32, 64>(data, num_layers, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm2x16(Box::new(model))))
        }
        (1, 40) => {
            let model = build_lstm_1layer::<40, 41, 160>(data, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm1x40(Box::new(model))))
        }
        (2, 24) => {
            let model = build_lstm_2layer::<24, 25, 48, 96>(data, num_layers, hidden_size)?;
            Ok(Box::new(DynamicModel::Lstm2x24(Box::new(model))))
        }
        _ => build_lstm_dynamic(data, num_layers, hidden_size),
    }
}

/// Builds an `LstmModel1<H, H1_IH, H_H4>` with sequentially read weights.
///
/// LSTM NAM Layout (C++ `LSTMLayerT::SetNAMWeights`):
/// ```text
/// layer.input_hidden_weights[H4 * IH]  (row-major)
/// layer.bias[H4]
/// layer.initial_hidden_state[H]
/// layer.initial_cell_state[H]
/// head_weights[H]
/// head_bias
/// ```
pub(crate) fn build_lstm_1layer<const H: usize, const H1_IH: usize, const H_H4: usize>(
    data: &NamModelData,
    hidden_size: usize,
) -> anyhow::Result<LstmModel1<H, H1_IH, H_H4>> {
    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;

    // Layer 1: input_size=1
    let layer = read_lstm_layer::<1, H, H1_IH, H_H4>(&mut cursor, is_bf16)?;

    // Head: output linear projection weights
    let head_weights_data = cursor.read_slice(H)?;
    let mut head_weights = [0u16; H];
    for i in 0..H {
        head_weights[i] = quantize_weight(head_weights_data[i], is_bf16);
    }
    let head_bias = cursor.read_f32()?;

    cursor.verify_exhausted()?;

    let model = LstmModel1::<H, H1_IH, H_H4> {
        layer,
        head_weights,
        head_bias,
    };

    info!(
        "[Dispatcher] LSTM 1×{} built — weights={}",
        hidden_size,
        data.weights.len()
    );

    Ok(model)
}

/// Builds an `LstmModel2<H, H1_IH, H2_IH, H_H4>` with sequentially read weights.
pub(crate) fn build_lstm_2layer<
    const H: usize,
    const H1_IH: usize,
    const H2_IH: usize,
    const H_H4: usize,
>(
    data: &NamModelData,
    num_layers: usize,
    hidden_size: usize,
) -> anyhow::Result<LstmModel2<H, H1_IH, H2_IH, H_H4>> {
    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;

    // Layer 1: input_size=1
    let layer1 = read_lstm_layer::<1, H, H1_IH, H_H4>(&mut cursor, is_bf16)?;

    // Layer 2: input_size=H (hidden state from previous layer)
    let layer2 = read_lstm_layer::<H, H, H2_IH, H_H4>(&mut cursor, is_bf16)?;

    // Head: final projection weights
    let head_weights_data = cursor.read_slice(H)?;
    let mut head_weights = [0u16; H];
    for i in 0..H {
        head_weights[i] = quantize_weight(head_weights_data[i], is_bf16);
    }
    let head_bias = cursor.read_f32()?;

    cursor.verify_exhausted()?;

    let model = LstmModel2::<H, H1_IH, H2_IH, H_H4> {
        layer1,
        layer2,
        head_weights,
        head_bias,
    };

    info!(
        "[Dispatcher] LSTM {}×{} built — weights={}",
        num_layers,
        hidden_size,
        data.weights.len()
    );

    Ok(model)
}

// =============================================================================
// LSTM — Dynamic Builder (Fallback)
// =============================================================================

/// Builds an `LstmDynModel` with sequentially read weights (dynamic fallback).
///
/// Publicly visible for dynamic ↔ static numerical parity tests.
pub fn build_lstm_dynamic(
    data: &NamModelData,
    num_layers: usize,
    hidden_size: usize,
) -> anyhow::Result<Box<DynamicModel>> {
    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let mut layers = Vec::with_capacity(num_layers);

    let mut current_input_size = 1; // The first signal that enters has size 1 (a single audio value)

    // We process each "layer" of the model. Think of them as stages of an assembly line.
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;

    for _ in 0..num_layers {
        // We read all the weights (the trained "intelligence") of this layer at once.
        let raw_weights =
            cursor.read_slice(hidden_size * 4 * (current_input_size + hidden_size))?;

        let mut input_hidden_weights = vec![0u16; raw_weights.len()];
        read_lstm_weights_into(
            raw_weights,
            &mut input_hidden_weights,
            cursor.is_gate_major_lstm(),
            hidden_size,
            current_input_size,
            is_bf16,
        );

        // The 'bias' is a fixed adjustment added at the end of each calculation, like a "calibration".
        let bias = cursor.read_slice(hidden_size * 4)?.to_vec();

        // The 'state' (hidden state) and the 'cell_state' (cell state) are the memory of the network.
        // They store information about sounds that passed milliseconds ago to
        // help predict the current sound.
        let hidden_init = cursor.read_slice(hidden_size)?;
        let mut state = vec![0.0; current_input_size + hidden_size];
        state[current_input_size..current_input_size + hidden_size].copy_from_slice(hidden_init);

        let cell_init = cursor.read_slice(hidden_size)?;
        let mut cell_state = vec![0.0; hidden_size];
        cell_state.copy_from_slice(cell_init);

        layers.push(LstmDynLayer {
            input_hidden_weights,
            bias,
            state: state.clone(),
            state_bf16: vec![0u16; current_input_size + hidden_size],
            cell_state,
            gates: vec![0.0; hidden_size * 4],
            tanh_cs: vec![0.0; hidden_size],
            input_size: current_input_size,
            hidden_size,
        });

        current_input_size = hidden_size;
    }

    // The "Head" is the final stage. It takes all the accumulated memory
    // and transforms it back into a single sound volume value (audio sample).
    let raw_head_weights = cursor.read_slice(hidden_size)?;
    let mut head_weights = vec![0u16; hidden_size];
    for i in 0..hidden_size {
        head_weights[i] = quantize_weight(raw_head_weights[i], is_bf16);
    }
    let head_bias = cursor.read_f32()?;

    // Checks whether we read exactly everything we needed, with nothing left over.
    cursor.verify_exhausted()?;

    let model = LstmDynModel {
        layers,
        head_weights,
        head_bias,
    };

    info!(
        "[Dispatcher] LSTM Dynamic {}×{} built — weights={}",
        num_layers,
        hidden_size,
        data.weights.len()
    );

    Ok(Box::new(DynamicModel::LstmDyn(Box::new(model))))
}

// =============================================================================
// LSTM — Shared Function for Reading Weights
// =============================================================================

/// Fills `output` (buffer of `u16` with size `4 * hidden * (input + hidden)`)
/// with the LSTM weights quantized from `raw` (`f32` slice already read from the cursor).
///
/// Applies layout transposition when needed (Original → GateMajor).
fn read_lstm_weights_into(
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
fn read_lstm_layer<const I: usize, const H: usize, const IH: usize, const H4: usize>(
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
