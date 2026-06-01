// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::layout;
use crate::loader::nam_json::NamModelData;
use crate::math::common::AlignedVec;
use crate::models::DynamicModel;
use crate::models::wavenet::{Conv1dDyn, DenseLayerDyn, MAX_KERNEL, WAVENET_MAX_NUM_FRAMES};
use crate::models::wavenet::{WaveNetDynModel, WaveNetLayerArrayDyn, WaveNetLayerDyn};
use crate::models::wavenet::WaveNetLayerState;
use anyhow::{Context, bail};
use log::info;

// =============================================================================
// Construtor dinâmico (fallback)
// =============================================================================

/// Constrói um `WaveNetDynModel` com pesos lidos sequencialmente (fallback dinâmico).
pub fn build_wavenet_dynamic(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    if data.config.layers.len() != 2 {
        bail!("WaveNet dinâmico exige 2 arrays");
    }

    super::validate_layer_activations(data)?;

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);

    let l0 = &data.config.layers[0];
    let l1 = &data.config.layers[1];

    let ch1 = l0.channels.context("Layer 0: sem channels")?;
    let k1 = l0.kernel_size.unwrap_or(3);
    let head1 = l0.head_size.context("Layer 0: sem head_size")?;
    let dils_0 = l0.dilations.as_deref().context("Layer 0: sem dilations")?;
    let b1 = l0.head_bias.unwrap_or(false);

    let dils_1 = l1.dilations.as_deref().context("Layer 1: sem dilations")?;
    let b2 = l1.head_bias.unwrap_or(true);

    let mut alloc_num = 0usize;

    let array1 = build_wavenet_array_dyn(WaveNetArrayDynConfig {
        cursor: &mut cursor,
        in_size: 1,
        cond_size: 1,
        ch: ch1,
        k: k1,
        head: head1,
        dilations: dils_0,
        has_head_bias: b1,
        gated: l0.gated.unwrap_or(false),
        alloc_num: &mut alloc_num,
    })?;

    let array2 = build_wavenet_array_dyn(WaveNetArrayDynConfig {
        cursor: &mut cursor,
        in_size: ch1,
        cond_size: 1,
        ch: head1,
        k: k1,
        head: 1,
        dilations: dils_1,
        has_head_bias: b2,
        gated: l1.gated.unwrap_or(false),
        alloc_num: &mut alloc_num,
    })?;

    let head_scale = cursor.read_f32()?;

    cursor.verify_exhausted()?;

    let rf = array1.receptive_field_size.max(array2.receptive_field_size);

    let model = WaveNetDynModel {
        array1,
        array2,
        head_scale,
        receptive_field_size: rf,
        head: head1,
    };

    info!(
        "[Dispatcher] WaveNet Dinâmico construído — CH={}, K={}, HEAD={}, PESOS={}",
        ch1,
        k1,
        head1,
        data.weights.len()
    );

    Ok(Box::new(DynamicModel::WavenetDyn(Box::new(model))))
}

// =============================================================================
// Construtor de array dinâmico
// =============================================================================

/// Configurações para construção de um WaveNetLayerArrayDyn.
pub(crate) struct WaveNetArrayDynConfig<'a, 'b, 'c> {
    pub cursor: &'a mut WeightCursor<'b>,
    pub in_size: usize,
    pub cond_size: usize,
    pub ch: usize,
    pub k: usize,
    pub head: usize,
    pub dilations: &'c [usize],
    pub has_head_bias: bool,
    pub gated: bool,
    pub alloc_num: &'a mut usize,
}

pub(crate) fn build_wavenet_array_dyn(
    config: WaveNetArrayDynConfig<'_, '_, '_>,
) -> anyhow::Result<WaveNetLayerArrayDyn> {
    let WaveNetArrayDynConfig {
        cursor,
        in_size,
        cond_size,
        ch,
        k,
        head,
        dilations,
        has_head_bias,
        gated,
        alloc_num,
    } = config;

    let conv_out_ch = if gated { 2 * ch } else { ch };

    if conv_out_ch * WAVENET_MAX_NUM_FRAMES > 4096 {
        bail!(
            "Dimensões da WaveNet não suportadas: conv_out_ch ({}) * MAX_FRAMES ({}) > 4096",
            conv_out_ch,
            WAVENET_MAX_NUM_FRAMES
        );
    }

    let rechannel =
        layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, in_size, ch, false)?;

    let mut layers = Vec::with_capacity(dilations.len());
    let mut states = Vec::with_capacity(dilations.len());

    for &dilation in dilations {
        if k > MAX_KERNEL {
            bail!(
                "Tamanho do kernel {} excede o máximo suportado ({})",
                k,
                MAX_KERNEL
            );
        }
        let conv1d =
            layout::read_conv1d_weights_typed::<Conv1dDyn>(
                cursor, ch, conv_out_ch, k, dilation, true,
            )?;
        let input_mixin =
            layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, cond_size, ch, false)?;
        let one_by_one =
            layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, ch, ch, true)?;

        layers.push(WaveNetLayerDyn {
            conv1d,
            input_mixin,
            one_by_one,
            ch,
            gated,
        });

        let rf = (k - 1) * dilation;
        states.push(WaveNetLayerState::new(ch, rf, *alloc_num)?);
        *alloc_num += 1;
    }

    let head_rechannel =
        layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, ch, head, has_head_bias)?;

    let receptive_field_size: usize = dilations.iter().map(|&d| (k - 1) * d).sum();

    let block_size = if gated { 2 * ch } else { ch };

    Ok(WaveNetLayerArrayDyn {
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0),
        head_accum: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0),
        head_outputs: AlignedVec::new(head * WAVENET_MAX_NUM_FRAMES, 0.0),
        block_buffer: AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0),
        block_size,
        receptive_field_size,
        ch,
        head,
        last_condition: AlignedVec::new(cond_size, 0.0),
        last_condition_bf16: AlignedVec::new(cond_size, 0),
        condition_init: false,
    })
}
