// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ConvNet model builder — reads weights from `NamModelData` and constructs a `ConvNetModel`.

use super::WeightCursor;
use crate::loader::nam_json::{NamModelData, get_convnet_topology};
use crate::math::common::AlignedVec;
use crate::models::StaticModel;
use crate::models::convnet::{ConvNetBlock, ConvNetModel};
use crate::models::wavenet::PostStackHead;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;
use crate::models::wavenet::post_stack_head::parse_activation;
use anyhow::Context;
use log::info;

use crate::loader::dispatcher::checked_arith;
use crate::loader::dispatcher::wavenet::layout;

/// Builds a `Box<StaticModel::ConvNet>` from the parsed model data.
///
/// Weight layout (per-block):
/// ```text
/// for block in blocks:
///     Conv1D.weights[IN*OUT*K] + Conv1D.bias[OUT]?
///     BatchNorm.scale[OUT]
///     BatchNorm.offset[OUT]
/// [head.weights + head.bias]?
/// [head_scale: f32]
/// ```
pub(crate) fn build_convnet(data: &NamModelData) -> anyhow::Result<Box<StaticModel>> {
    let topo = get_convnet_topology(data).context("ConvNet topology not detectable")?;

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);
    let mut blocks = Vec::with_capacity(topo.num_blocks);

    let mut total_rf = 0usize;

    for (alloc_num, i) in (0..topo.num_blocks).enumerate() {
        let in_ch = if i == 0 { 1 } else { topo.channels[i - 1] };
        let out_ch = topo.channels[i];
        let kernel = topo.kernel_sizes[i];
        let dilations = &topo.dilations[i];

        let dilation = *dilations
            .first()
            .context(format!("ConvNet block {i}: empty dilations list"))?;

        let do_bias = true;

        let activation_str = data
            .config
            .layers
            .get(i)
            .and_then(|l| l.activation.as_deref())
            .unwrap_or("Tanh");
        let activation = parse_activation(activation_str);

        let mut block = ConvNetBlock::new(
            in_ch, out_ch, kernel, dilation, do_bias, activation, alloc_num,
        )
        .map_err(|e| anyhow::anyhow!("ConvNet block {i}: failed to create: {e}"))?;

        let interleaved = cursor.is_interleaved4();
        let num_blocks = out_ch.div_ceil(4);
        let padded_total = checked_arith::checked_conv_padded_total(num_blocks, 4, in_ch, kernel)?;

        if interleaved {
            let raw = cursor.read_slice(padded_total)?;
            block.set_conv_weights(raw);
        } else {
            let total = checked_arith::checked_conv_total(out_ch, in_ch, kernel)?;
            let raw = cursor.read_slice(total)?;
            let mut interleaved_weights = AlignedVec::new(padded_total, 0.0f32);
            layout::transpose_conv1d_interleaved_4wide(
                raw,
                &mut interleaved_weights,
                in_ch,
                out_ch,
                kernel,
            );
            block.set_conv_weights(&interleaved_weights);
        }

        if do_bias {
            let bias_raw = cursor.read_slice(out_ch)?;
            block.set_conv_bias(bias_raw);
        }

        let bn_scale = cursor.read_slice(out_ch)?;
        let bn_offset = cursor.read_slice(out_ch)?;
        block.set_bn_params(bn_scale, bn_offset);

        for &d in dilations {
            total_rf += (kernel - 1) * d;
        }

        blocks.push(block);
    }

    let post_stack_head = if let Some(ref head_config) = topo.head {
        let in_ch = *topo
            .channels
            .last()
            .context("ConvNet: no blocks, cannot build head")?;

        let mut head = PostStackHead::from_config(head_config, in_ch)
            .map_err(|e| anyhow::anyhow!("Failed to build post-stack head: {}", e))?;

        let ch = head.in_channels();
        let out_ch = head.out_channels();
        let kernel = head.conv.kernel;
        let num_blocks = out_ch.div_ceil(4);
        let padded_total = checked_arith::checked_conv_padded_total(num_blocks, 4, ch, kernel)?;

        if cursor.is_interleaved4() {
            let raw = cursor.read_slice(padded_total)?;
            head.set_weights(raw);
        } else {
            let total = checked_arith::checked_conv_total(out_ch, ch, kernel)?;
            let raw = cursor.read_slice(total)?;
            let mut interleaved = AlignedVec::new(padded_total, 0.0f32);
            layout::transpose_conv1d_interleaved_4wide(raw, &mut interleaved, ch, out_ch, kernel);
            head.set_weights(&interleaved);
        }

        if head.conv.do_bias {
            let bias_raw = cursor.read_slice(out_ch)?;
            head.set_bias(bias_raw);
        }

        total_rf += head.receptive_field() - 1;

        Some(head)
    } else {
        None
    };

    let head_scale = cursor.read_f32_finite()?;

    cursor.verify_exhausted()?;

    let head_out_ch = post_stack_head
        .as_ref()
        .map(|h| h.out_channels())
        .unwrap_or(1);
    let head_output_scratch = AlignedVec::new(head_out_ch * WAVENET_MAX_NUM_FRAMES, 0.0);

    let max_scratch_ch = blocks.iter().map(|b| b.conv.out_ch).max().unwrap_or(1);
    let scratch_a = AlignedVec::new(max_scratch_ch * WAVENET_MAX_NUM_FRAMES, 0.0);
    let scratch_b = AlignedVec::new(max_scratch_ch * WAVENET_MAX_NUM_FRAMES, 0.0);

    let model = ConvNetModel {
        blocks,
        head_scale,
        receptive_field_size: total_rf,
        post_stack_head,
        head_output_scratch,
        scratch_a,
        scratch_b,
        prewarm_on_reset: true,
    };

    info!(
        "[Dispatcher] ConvNet built — num_blocks={}, head_scale={:.6}, weights={}",
        topo.num_blocks,
        head_scale,
        data.weights.len()
    );

    Ok(Box::new(StaticModel::ConvNet(Box::new(model))))
}
