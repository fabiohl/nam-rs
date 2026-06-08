// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::weights::read_lstm_weights_into;
use crate::loader::nam_json::NamModelData;
use crate::math::common::quantize_weight;
use crate::math::common::{InstructionSet, SimdMathConfig};
use crate::models::DynamicModel;
use crate::models::lstm::{LstmDynLayer, LstmDynModel};
use log::info;

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
    let is_bf16 = SimdMathConfig::get().instruction_set == InstructionSet::Avx512VnniBf16;

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
    let mut head_weights_f32 = vec![0.0f32; hidden_size];
    for i in 0..hidden_size {
        head_weights[i] = quantize_weight(raw_head_weights[i], is_bf16);
        head_weights_f32[i] = raw_head_weights[i];
    }
    let head_bias = cursor.read_f32()?;

    // Checks whether we read exactly everything we needed, with nothing left over.
    cursor.verify_exhausted()?;

    let model = LstmDynModel {
        layers,
        head_weights,
        head_weights_f32,
        head_bias,
        use_f32_head: true,
    };

    info!(
        "[Dispatcher] LSTM Dynamic {}×{} built — weights={}",
        num_layers,
        hidden_size,
        data.weights.len()
    );

    Ok(Box::new(DynamicModel::LstmDyn(Box::new(model))))
}
