// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Utilitário para exportação de modelos no formato binário `.namb` v2.
//!
//! Permite converter modelos JSON ou modelos em memória para o formato
//! binário otimizado com pesos pré-transpostos.

use super::nam_json::{NamModelData, WeightsLayout};
use super::namb::{NambHeader, crc32_ieee};
use anyhow::{Context, Result};
use std::io::Write;

/// Codifica um `NamModelData` para o formato binário `.namb`.
pub fn encode_namb(
    data: &NamModelData,
    version: u16,
    target_layout: WeightsLayout,
) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();

    // 1. Prepara os pesos transpostos
    let transposed_weights = if target_layout != WeightsLayout::Original {
        transpose_weights(data, target_layout)?
    } else {
        data.weights.clone()
    };

    let weights_bytes: Vec<u8> = transposed_weights
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    // 2. Prepara o JSON de metadados
    let mut metadata_only = data.clone();
    metadata_only.weights = Vec::new();
    let json_str = serde_json::to_string(&metadata_only)?;
    let json_bytes = json_str.as_bytes();

    // 3. Constrói o Header
    let mut header = NambHeader {
        magic: 0x4E414D42,
        version,
        layout_type: target_layout as u8,
        reserved_v2: [0; 5],
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

    // 4. Escreve tudo no buffer
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

fn transpose_weights(data: &NamModelData, layout: WeightsLayout) -> Result<Vec<f32>> {
    match layout {
        WeightsLayout::GateMajorLstm => transpose_lstm_gate_major(data),
        WeightsLayout::Interleaved4WaveNet => transpose_wavenet_interleaved4(data),
        _ => Ok(data.weights.clone()),
    }
}

fn transpose_lstm_gate_major(data: &NamModelData) -> Result<Vec<f32>> {
    if data.architecture != "LSTM" {
        anyhow::bail!("Layout GateMajorLstm requer arquitetura LSTM");
    }

    let hidden_size = data.config.hidden_size.context("LSTM sem hidden_size")?;
    let num_layers = data.config.num_layers.unwrap_or(1);

    let mut cursor = 0;
    let mut out_weights = Vec::with_capacity(data.weights.len());

    let mut current_input_size = 1;
    for l in 0..num_layers {
        let ih = current_input_size + hidden_size;
        let h = hidden_size;
        let layer_size = 4 * h * ih;

        if cursor + layer_size > data.weights.len() {
            anyhow::bail!("Pesos insuficientes para LSTM na camada {l}");
        }

        let raw = &data.weights[cursor..cursor + layer_size];
        let mut transposed = vec![0.0f32; layer_size];

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

        // Bias [4 * H]
        let bias_size = 4 * h;
        if cursor + bias_size > data.weights.len() {
            anyhow::bail!("Bias insuficiente para LSTM na camada {l}");
        }
        out_weights.extend_from_slice(&data.weights[cursor..cursor + bias_size]);
        cursor += bias_size;

        current_input_size = hidden_size;
    }

    // Remaining weights (head weights, head bias, initial states if any)
    if cursor < data.weights.len() {
        out_weights.extend_from_slice(&data.weights[cursor..]);
    }

    Ok(out_weights)
}

fn transpose_wavenet_interleaved4(data: &NamModelData) -> Result<Vec<f32>> {
    if data.architecture != "WaveNet" {
        anyhow::bail!("Layout Interleaved4WaveNet requer arquitetura WaveNet");
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
            .context("WaveNet sem dilatações")?;
        let gated = layer_cfg.gated.unwrap_or(false);
        let conv_out_ch = if gated { 2 * ch } else { ch };

        // 1. Rechannel: [CH][IN_CH] -> [IN_CH][CH] (NO BIAS em NAM)
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

        // 2. Conv Layers
        for (di, _) in dilations.iter().enumerate() {
            // Conv1D: [CONV_OUT][CH][K] -> Interleaved [CONV_OUT/4][K][CH][4]
            let size = conv_out_ch * ch * k;
            ensure_capacity(
                &data.weights,
                cursor,
                size,
                format!("Array {} Layer {} Conv1D Weights", li, di),
            )?;
            let raw = &data.weights[cursor..cursor + size];
            let num_blocks = conv_out_ch / 4;
            for b in 0..num_blocks {
                for ki in 0..k {
                    for in_c in 0..ch {
                        for lane in 0..4 {
                            let out_c = b * 4 + lane;
                            out_weights.push(raw[(out_c * ch + in_c) * k + ki]);
                        }
                    }
                }
            }
            // Tail channels if not multiple of 4
            let tail_start = num_blocks * 4;
            for out_c in tail_start..conv_out_ch {
                for ki in 0..k {
                    for in_c in 0..ch {
                        out_weights.push(raw[(out_c * ch + in_c) * k + ki]);
                    }
                }
            }
            cursor += size;

            // Conv1D Bias [CONV_OUT]
            ensure_capacity(
                &data.weights,
                cursor,
                conv_out_ch,
                format!("Array {} Layer {} Conv1D Bias", li, di),
            )?;
            out_weights.extend_from_slice(&data.weights[cursor..cursor + conv_out_ch]);
            cursor += conv_out_ch;

            // Input Mixin: [CH][COND_CH] -> [COND_CH][CH]
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

            // 1x1: [CH][CH] -> [CH][CH] (transposed)
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

            // 1x1 Bias [CH]
            ensure_capacity(
                &data.weights,
                cursor,
                ch,
                format!("Array {} Layer {} 1x1 Bias", li, di),
            )?;
            out_weights.extend_from_slice(&data.weights[cursor..cursor + ch]);
            cursor += ch;
        }

        // 3. Head Rechannel: [HEAD_CH][CH] -> [CH][HEAD_CH]
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

        // Head Rechannel Bias [HEAD_CH]
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

    // Remaining weights (e.g. head_scale for WaveNet)
    if cursor < data.weights.len() {
        out_weights.extend_from_slice(&data.weights[cursor..]);
    }

    Ok(out_weights)
}

fn ensure_capacity(weights: &[f32], cursor: usize, needed: usize, label: String) -> Result<()> {
    if cursor + needed > weights.len() {
        anyhow::bail!(
            "Pesos insuficientes para {}: necessita de index {}..{}, mas comprimento total é {}",
            label,
            cursor,
            cursor + needed,
            weights.len()
        );
    }
    Ok(())
}
