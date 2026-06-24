// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Weight loading for the dynamic A2 model.
//!
//! Parses a flat f32 weight stream in NAM JSON order and populates
//! the model layers with runtime-dimensioned weights. Supports
//! gating/blending (2× bottleneck), head1x1, and per-layer FiLM.

use crate::math::common::{AlignedVec, PrefetchFn};
use crate::models::a2::gating::GatingMode;
use crate::models::a2::head::A2HeadConv;
use crate::models::a2::layer::A2Layer;
use crate::models::a2::weights_layout::{
    transpose_conv1d_interleaved_4wide, transpose_dense_f32, transpose_head_w,
};

use super::WaveNetA2Dyn;

impl WaveNetA2Dyn {
    /// Loads weights from a flat f32 slice in A2 stream order.
    ///
    /// ## Weight order
    ///
    /// 1. `_rechannel`: weights `channels` f32
    /// 2. Per layer 0..num_layers-1:
    ///    - `_conv`: weights `channels*bottleneck*kernel_size` f32 + bias `bottleneck` f32
    ///      (or `channels*2*bottleneck*kernel_size` + bias `2*bottleneck` if gating/blending)
    ///    - `_input_mixin`: weights `bottleneck` f32
    ///    - `_layer1x1`: weights `bottleneck*channels` f32 + bias `channels` f32
    /// 3. Head1x1 (if active): weights `bottleneck*channels` f32 + bias `channels` f32
    /// 4. `_head_rechannel`: conv k=16 weights `16*channels` f32 + bias `1` f32
    /// 5. `head_scale`: last f32
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), String> {
        let total = weights.len();
        let mut pos: usize = 0;

        self.load_rechannel_weights(weights, &mut pos, total)?;

        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let layer = self.load_per_layer_weights(weights, &mut pos, total, i)?;
            layers.push(layer);
        }

        self.load_head1x1_weights(weights, &mut pos, total)?;

        let (head_w, head_b, head_scale) =
            self.load_head_conv_and_scale(weights, &mut pos, total)?;

        if pos != total {
            return Err(format!(
                "set_weights: stream has {} unconsumed f32 (consumed {}, total {})",
                total - pos,
                pos,
                total
            ));
        }

        self.layers = layers;
        self.head_conv = Some(A2HeadConv::new(head_w, head_b, head_scale, self.channels));

        Ok(())
    }

    /// Loads rechannel weights from the stream: `Conv1x1(1 → channels)` (no bias).
    fn load_rechannel_weights(
        &mut self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
    ) -> Result<(), String> {
        let channels = self.channels;
        let rw_f32 =
            super::super::set_weights::read_slice(weights, pos, channels, total, "rechannel_w")?;
        self.rechannel_w_f32.copy_from_slice(rw_f32);
        Ok(())
    }

    /// Loads a single layer's weights (conv, mixin, l1x1, optional FiLM).
    #[allow(clippy::too_many_lines)]
    fn load_per_layer_weights(
        &mut self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
        i: usize,
    ) -> Result<A2Layer, String> {
        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let ksize = self.kernel_sizes[i];
        let dilation = self.dilations[i];
        let use_gating = self.gating_modes[i] == GatingMode::Gated
            || self.gating_modes[i] == GatingMode::Blended;
        let conv_out = if use_gating {
            bottleneck * 2
        } else {
            bottleneck
        };

        // 2a. Dilated conv weights — interleave-4-wide.
        let conv_w_count = channels * conv_out * ksize;
        let conv_w_padded = conv_out.div_ceil(4) * 4 * channels * ksize;
        let conv_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            conv_w_count,
            total,
            &format!("layer[{i}].conv_w"),
        )?;
        let mut conv_w = AlignedVec::new(conv_w_padded, 0.0f32);
        transpose_conv1d_interleaved_4wide(conv_w_f32, &mut conv_w, channels, conv_out, ksize);

        // 2b. Conv bias.
        let conv_b_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            conv_out,
            total,
            &format!("layer[{i}].conv_b"),
        )?;
        let conv_b = AlignedVec::from(conv_b_f32.to_vec());

        let prefetch_fn: PrefetchFn = if dilation >= 128 {
            crate::math::common::prefetch_strategy_2stage
        } else {
            crate::math::common::prefetch_strategy_simple
        };

        let conv = crate::models::a2::conv1d::A2Conv1d::new(
            conv_w,
            conv_b,
            true,
            dilation,
            channels,
            conv_out,
            ksize,
            prefetch_fn,
        );

        // 2c. Mixin (conv_out elements, applied after conv).
        let mixin_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            conv_out,
            total,
            &format!("layer[{i}].mixin_w"),
        )?;
        let mixin_w = AlignedVec::from(mixin_w_f32.to_vec());

        // 2d. L1x1: bottleneck×channels + channels bias.
        let l1x1_w_count = bottleneck * channels;
        let l1x1_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            l1x1_w_count,
            total,
            &format!("layer[{i}].l1x1_w"),
        )?;
        let mut l1x1_w = AlignedVec::new(l1x1_w_count, 0.0f32);
        transpose_dense_f32(l1x1_w_f32, &mut l1x1_w, bottleneck, channels);

        let l1x1_b_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            channels,
            total,
            &format!("layer[{i}].l1x1_b"),
        )?;
        let l1x1_b = AlignedVec::from(l1x1_b_f32.to_vec());

        let mut layer = A2Layer::new_dyn(conv, mixin_w, l1x1_w, l1x1_b, channels, bottleneck);

        // FiLM layers (if active in layer_raw JSON) — read weights after l1x1 bias.
        if let Some(ref raw) = self.layer_raw {
            let configs = super::super::set_weights::parse_film_configs(raw);
            super::super::set_weights::load_film_for_layer(
                &mut layer,
                &configs,
                channels,
                self.condition_size,
                weights,
                pos,
                total,
                i,
            )?;
        }

        Ok(layer)
    }

    /// Loads head1x1 projection weights from the stream (if active).
    fn load_head1x1_weights(
        &mut self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
    ) -> Result<(), String> {
        if !self.head1x1_active {
            return Ok(());
        }
        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let h1_w_count = bottleneck * channels;
        let h1_w_f32 =
            super::super::set_weights::read_slice(weights, pos, h1_w_count, total, "head1x1_w")?;
        let mut h1_w = AlignedVec::new(h1_w_count, 0.0f32);
        transpose_dense_f32(h1_w_f32, &mut h1_w, bottleneck, channels);

        let h1_b_f32 =
            super::super::set_weights::read_slice(weights, pos, channels, total, "head1x1_b")?;
        let mut h1_b = AlignedVec::new(channels, 0.0f32);
        h1_b.copy_from_slice(h1_b_f32);

        self.head1x1_w = h1_w;
        self.head1x1_b = h1_b;
        Ok(())
    }

    /// Loads head conv weights (K=16), bias, and head scale from the stream.
    fn load_head_conv_and_scale(
        &self,
        weights: &[f32],
        pos: &mut usize,
        total: usize,
    ) -> Result<(AlignedVec<f32>, f32, f32), String> {
        let channels = self.channels;
        let head_k = crate::models::a2::params::A2_HEAD_KERNEL_SIZE;
        let head_w_f32 = super::super::set_weights::read_slice(
            weights,
            pos,
            head_k * channels,
            total,
            "head_w",
        )?;
        let mut head_w = AlignedVec::new(head_k * channels, 0.0f32);
        transpose_head_w(head_w_f32, &mut head_w, channels, head_k);

        let head_b = {
            let s = super::super::set_weights::read_slice(weights, pos, 1, total, "head_b")?;
            if !s[0].is_finite() {
                return Err(format!(
                    "set_weights: head_b is not finite (value: {:e})",
                    s[0]
                ));
            }
            s[0]
        };

        let head_scale = {
            let s = super::super::set_weights::read_slice(weights, pos, 1, total, "head_scale")?;
            if !s[0].is_finite() {
                return Err(format!(
                    "set_weights: head_scale is not finite (value: {:e})",
                    s[0]
                ));
            }
            s[0]
        };

        Ok((head_w, head_b, head_scale))
    }
}
