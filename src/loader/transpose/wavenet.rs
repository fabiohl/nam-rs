// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet Interleaved-4 weight transposition.
//!
//! **Cross-module dependency:** The output layout (interleave order, padding,
//! layer sub-component sequence) must remain in lock-step with the decoder in
//! `dispatcher/wavenet/`. See `dispatcher/wavenet/mod.rs` comments on the
//! interleaved-4 block read pattern.

use super::super::namb_encoder::ensure_capacity;
use crate::loader::nam_json::NamModelData;
use anyhow::{Context, Result};

/// Specialized for WaveNet: rearranges the data so that the processor can
/// process 4 channels simultaneously (technique called SIMD Interleaved).
pub(crate) fn transpose_wavenet_interleaved4(data: &NamModelData) -> Result<Vec<f32>> {
    if data.architecture != "WaveNet" {
        anyhow::bail!("Layout Interleaved4WaveNet requires WaveNet architecture");
    }

    let mut cursor = 0;
    let mut out_weights = Vec::with_capacity(data.weights.len());

    for (li, layer_cfg) in data.config.layers.iter().enumerate() {
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
