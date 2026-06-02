// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Utility for exporting models in the `.namb` v2 binary format.
//!
//! Allows converting JSON models or in-memory models to the optimized
//! binary format with pre-transposed weights.

use super::nam_json::{NamModelData, WeightsLayout};
use super::namb::{FLAG_HAS_CRC32, NambHeader, crc32_ieee};
use anyhow::{Context, Result};
use std::io::Write;

/// Encodes a `NamModelData` into the `.namb` binary format.
pub fn encode_namb(
    data: &NamModelData,
    version: u16,
    target_layout: WeightsLayout,
) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();

    // 1. Prepares the transposed weights
    let transposed_weights = if target_layout != WeightsLayout::Original {
        transpose_weights(data, target_layout)?
    } else {
        data.weights.clone()
    };

    let weights_bytes: Vec<u8> = transposed_weights
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    // 2. Prepares the metadata JSON
    let mut metadata_only = data.clone();
    metadata_only.weights = Vec::new();
    let json_str = serde_json::to_string(&metadata_only)?;
    let json_bytes = json_str.as_bytes();

    let header_flags = if version >= 2 { FLAG_HAS_CRC32 } else { 0 };

    // 3. Builds the Header
    let mut header = NambHeader {
        magic: 0x4E414D42,
        version,
        layout_type: target_layout as u8,
        flags: header_flags,
        reserved_v2: [0; 4],
        weights_offset: (std::mem::size_of::<NambHeader>() + json_bytes.len()) as u32,
        reserved1: [0; 2],
        crc32: crc32_ieee(&weights_bytes),
        reserved2: 0,
        version_str: [0; 32],
        sample_rate: data.sample_rate.unwrap_or(48000.0),
        input_level_dbu: data
            .metadata
            .as_ref()
            .and_then(|m| m.input_level_dbu)
            .unwrap_or(12.0),
        output_level_dbu: data
            .metadata
            .as_ref()
            .and_then(|m| m.output_level_dbu)
            .unwrap_or(-6.0),
        reserved3: [0; 1],
    };

    let ver_str = format!("NAMB v{} ({:?})", version, target_layout);
    let ver_bytes = ver_str.as_bytes();
    let copy_len = ver_bytes.len().min(32);
    header.version_str[..copy_len].copy_from_slice(&ver_bytes[..copy_len]);

    // 4. Writes everything into the buffer
    let header_bytes = unsafe {
        std::slice::from_raw_parts(
            (&header as *const NambHeader) as *const u8,
            std::mem::size_of::<NambHeader>(),
        )
    };

    buffer.write_all(header_bytes)?;
    buffer.write_all(json_bytes)?;
    buffer.write_all(&weights_bytes)?;

    Ok(buffer)
}

/// Rearranges the "weights" so that the processor can
/// read them more efficiently during audio execution.
fn transpose_weights(data: &NamModelData, layout: WeightsLayout) -> Result<Vec<f32>> {
    match layout {
        // For LSTM, we use an organization where the logical "gates" are grouped.
        WeightsLayout::GateMajorLstm => transpose_lstm_gate_major(data),
        // For WaveNet, we interleave the data to facilitate batch calculations (SIMD).
        WeightsLayout::Interleaved4WaveNet => transpose_wavenet_interleaved4(data),
        // Default case: simply copies the original weights without changes.
        _ => Ok(data.weights.clone()),
    }
}

/// Specialized for LSTM: rearranges the data to optimize matrix operations.
fn transpose_lstm_gate_major(data: &NamModelData) -> Result<Vec<f32>> {
    if data.architecture != "LSTM" {
        anyhow::bail!("Layout GateMajorLstm requires LSTM architecture");
    }

    // Internal memory size (hidden) and number of model layers.
    let hidden_size = data
        .config
        .hidden_size
        .context("LSTM without hidden_size")?;
    let num_layers = data.config.num_layers.unwrap_or(1);

    let mut cursor = 0; // "Bookmark" to know where we are in the original data.
    let mut out_weights = Vec::with_capacity(data.weights.len());

    let mut current_input_size = 1;
    for l in 0..num_layers {
        let ih = current_input_size + hidden_size;
        let h = hidden_size;
        let layer_size = 4 * h * ih; // Each LSTM layer has 4 main "gates".

        if cursor + layer_size > data.weights.len() {
            anyhow::bail!("Insufficient weights for LSTM at layer {l}");
        }

        let raw = &data.weights[cursor..cursor + layer_size];
        let mut transposed = vec![0.0f32; layer_size];

        // Triple loop: we rearrange the rows and columns of the data (transposition).
        // This allows the program to perform mathematical calculations much faster.
        for k in 0..4 {
            // Iterates over the 4 LSTM gates
            for i in 0..h {
                for j in 0..ih {
                    let val = raw[k * h * ih + i * ih + j];
                    // We swap the reading order to the format the "engine" expects.
                    transposed[k * h * ih + j * h + i] = val;
                }
            }
        }
        out_weights.extend(transposed);
        cursor += layer_size;

        // Bias processing (adjustment/calibration values for each neuron).
        let bias_size = 4 * h;
        if cursor + bias_size > data.weights.len() {
            anyhow::bail!("Insufficient bias for LSTM at layer {l}");
        }
        out_weights.extend_from_slice(&data.weights[cursor..cursor + bias_size]);
        cursor += bias_size;

        // Processing of the layer's initial states (hidden_init and cell_init).
        // This is necessary to maintain parity with the order the decoder expects.
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

/// Specialized for WaveNet: rearranges the data so that the processor can
/// process 4 channels simultaneously (technique called SIMD Interleaved).
fn transpose_wavenet_interleaved4(data: &NamModelData) -> Result<Vec<f32>> {
    if data.architecture != "WaveNet" {
        anyhow::bail!("Layout Interleaved4WaveNet requires WaveNet architecture");
    }

    let mut cursor = 0;
    let mut out_weights = Vec::with_capacity(data.weights.len());

    for (li, layer_cfg) in data.config.layers.iter().enumerate() {
        // We extract the "brain" size configurations for each layer.
        let in_ch = layer_cfg.input_size.unwrap_or(1);
        let ch = layer_cfg.channels.unwrap_or(16);
        let cond_ch = layer_cfg.condition_size.unwrap_or(1);
        let k = layer_cfg.kernel_size.unwrap_or(3);
        let head_ch = layer_cfg.head_size.unwrap_or(8);
        let dilations = layer_cfg
            .dilations
            .as_ref()
            .context("WaveNet without dilations")?;
        let gated = layer_cfg.gated.unwrap_or(false);
        let conv_out_ch = if gated { 2 * ch } else { ch };

        // 1. Rechannel: Adjusts the input to the number of internal channels.
        // Transposes [Output Channels][Input Channels] -> [Input][Output].
        let size = ch * in_ch;
        ensure_capacity(
            &data.weights,
            cursor,
            size,
            format!("Array {} Rechannel Weights", li),
        )?;
        let raw = &data.weights[cursor..cursor + size];
        for in_c in 0..in_ch {
            for out_c in 0..ch {
                out_weights.push(raw[out_c * in_ch + in_c]);
            }
        }
        cursor += size;

        // 2. Convolution Layers (the "filters" that shape the sound).
        for (di, _) in dilations.iter().enumerate() {
            // Conv1D: We rearrange to the "Interleaved 4" format.
            // This groups data into blocks of 4, allowing modern processors
            // to perform 4 calculations in the time of 1.
            let size = conv_out_ch * ch * k;
            ensure_capacity(
                &data.weights,
                cursor,
                size,
                format!("Array {} Layer {} Conv1D Weights", li, di),
            )?;
            let raw = &data.weights[cursor..cursor + size];
            let num_blocks = conv_out_ch.div_ceil(4);
            for b in 0..num_blocks {
                for ki in 0..k {
                    for in_c in 0..ch {
                        for lane in 0..4 {
                            let out_c = b * 4 + lane;
                            if out_c < conv_out_ch {
                                out_weights.push(raw[(out_c * ch + in_c) * k + ki]);
                            } else {
                                out_weights.push(0.0);
                            }
                        }
                    }
                }
            }
            cursor += size;

            // Bias: Fine-tuning values for the convolution filters.
            ensure_capacity(
                &data.weights,
                cursor,
                conv_out_ch,
                format!("Array {} Layer {} Conv1D Bias", li, di),
            )?;
            out_weights.extend_from_slice(&data.weights[cursor..cursor + conv_out_ch]);
            cursor += conv_out_ch;

            // Input Mixin: Combines the audio signal with external controls (if any).
            let size = ch * cond_ch;
            ensure_capacity(
                &data.weights,
                cursor,
                size,
                format!("Array {} Layer {} Input Mixin Weights", li, di),
            )?;
            let raw = &data.weights[cursor..cursor + size];
            for in_c in 0..cond_ch {
                for out_c in 0..ch {
                    out_weights.push(raw[out_c * cond_ch + in_c]);
                }
            }
            cursor += size;

            // 1x1: Internal adjustment layer that mixes the processed channels.
            let size = ch * ch;
            ensure_capacity(
                &data.weights,
                cursor,
                size,
                format!("Array {} Layer {} 1x1 Weights", li, di),
            )?;
            let raw = &data.weights[cursor..cursor + size];
            for in_c in 0..ch {
                for out_c in 0..ch {
                    out_weights.push(raw[out_c * ch + in_c]);
                }
            }
            cursor += size;

            // 1x1 Bias: Fine-tuning for the channel mixing.
            ensure_capacity(
                &data.weights,
                cursor,
                ch,
                format!("Array {} Layer {} 1x1 Bias", li, di),
            )?;
            out_weights.extend_from_slice(&data.weights[cursor..cursor + ch]);
            cursor += ch;
        }

        // 3. Head Rechannel: Prepares the final signal to exit the WaveNet layer.
        let size = head_ch * ch;
        ensure_capacity(
            &data.weights,
            cursor,
            size,
            format!("Array {} Head Rechannel Weights", li),
        )?;
        let raw = &data.weights[cursor..cursor + size];
        for in_c in 0..ch {
            for out_c in 0..head_ch {
                out_weights.push(raw[out_c * ch + in_c]);
            }
        }
        cursor += size;

        // Head Rechannel Bias: Final output adjustment.
        if layer_cfg.head_bias.unwrap_or(false) {
            ensure_capacity(
                &data.weights,
                cursor,
                head_ch,
                format!("Array {} Head Rechannel Bias", li),
            )?;
            out_weights.extend_from_slice(&data.weights[cursor..cursor + head_ch]);
            cursor += head_ch;
        }
    }

    // If there are extra weights (such as final scale), we add them at the end.
    if cursor < data.weights.len() {
        out_weights.extend_from_slice(&data.weights[cursor..]);
    }

    Ok(out_weights)
}

/// Safety function: ensures we won't try to read data beyond what exists in the file.
/// If the model is corrupted or incomplete, the program warns instead of crashing.
fn ensure_capacity(weights: &[f32], cursor: usize, needed: usize, label: String) -> Result<()> {
    if cursor + needed > weights.len() {
        anyhow::bail!(
            "Insufficient weights for {}: needs index {}..{}, but total length is {}",
            label,
            cursor,
            cursor + needed,
            weights.len()
        );
    }
    Ok(())
}
