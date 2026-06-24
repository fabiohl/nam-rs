// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Dynamic model (`WaveNetA2Dyn`).
//!
//! Runtime-dimensioned A2 engine supporting `bottleneck != channels`,
//! heterogeneous activations, gating/blending, head1x1, grouped layer1x1,
//! and the full A2 topology spectrum.

use crate::dsp::mirror_buf::MirroredBuffer;
use crate::math::common::AlignedVec;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::{BlendingActivationConfig, GatingActivationConfig, GatingMode};
use crate::models::a2::head::A2HeadConv;
use crate::models::a2::layer::A2Layer;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;
use serde_json::Value;

/// Weight-loading routines (`set_weights` and helpers).
pub mod build;
/// Pre-warm implementation (fills receptive field with silence).
pub mod prewarm;
/// Forward-pass methods (`process`, `process_internal`, per-frame loops).
pub mod process;

/// Runtime-dimensioned WaveNet A2 model.
///
/// Supports the full A2 topology spectrum: arbitrary channel counts,
/// bottleneck ≠ channels, heterogeneous activations, gating/blending,
/// grouped convolutions, head1x1, and FiLM.
///
/// Uses per-frame processing for maximum flexibility. Each frame's conv
/// dispatch (standard, grouped, depthwise) is handled by the polymorphic
/// `A2Conv1d::process_single_frame`.
pub struct WaveNetA2Dyn {
    /// Inter-layer channel count.
    pub channels: usize,
    /// Internal bottleneck channel count (conv output size).
    pub bottleneck: usize,
    /// Number of layers.
    pub num_layers: usize,

    /// A2 layers (conv, mixin, l1x1, FiLM). Conv is `channels → bottleneck`
    /// (or `channels → 2*bottleneck` if gating/blending).
    /// mixin_w has `bottleneck` elements.
    /// l1x1_w has `bottleneck × channels` elements, l1x1_b has `channels`.
    pub layers: Vec<A2Layer>,

    /// Input rechannel weights: `Conv1x1(1 → channels)` (no bias), u16 quantized.
    pub rechannel_w: AlignedVec<u16>,
    /// Input rechannel weights (pre-decoded f32).
    pub rechannel_w_f32: AlignedVec<f32>,

    /// Head convolution (K=16 over head accumulator, bias, head_scale).
    pub head_conv: Option<A2HeadConv>,

    /// Head accumulator ring buffer (channels-wide, pow2 size).
    pub head_accum: AlignedVec<f32>,
    /// Write position in head_accum (in columns, wraps via head_ring_mask).
    pub head_write_pos: usize,
    /// Ring mask for head_accum (pow2 ring, mask = capacity - 1).
    pub head_ring_mask: usize,

    /// Per-layer history buffers: one MirroredBuffer per layer.
    /// Each stores `channels`-wide frames (the layer_in data).
    pub layer_buffers: Vec<MirroredBuffer<f32>>,
    /// Per-layer ring sizes in elements (pow2 page-aligned).
    pub layer_ring_sizes: Vec<usize>,
    /// Per-layer maximum dilation lookback = (kernel-1) * dilation * channels.
    pub layer_lookbacks: Vec<usize>,
    /// Per-layer buffer starts (advanced with each written frame).
    pub layer_buffer_starts: Vec<usize>,

    /// Inter-layer data buffer: `channels × max_buffer_size` f32.
    pub layer_in: AlignedVec<f32>,

    /// Per-layer kernel sizes.
    pub kernel_sizes: Vec<usize>,
    /// Per-layer dilation factors.
    pub dilations: Vec<usize>,

    /// Primary activation per layer.
    pub activations: Vec<ActivationType>,
    /// Gating mode per layer.
    pub gating_modes: Vec<GatingMode>,
    /// Secondary activation per layer (for gating/blending).
    pub secondary_activations: Vec<Option<ActivationType>>,

    /// Pre-allocated gating configs per layer (None when gating_mode != Gated).
    pub gating_configs: Vec<Option<GatingActivationConfig>>,
    /// Pre-allocated blending configs per layer (None when gating_mode != Blended).
    pub blending_configs: Vec<Option<BlendingActivationConfig>>,

    /// Whether head1x1 projection (`bottleneck → channels`) is active.
    pub head1x1_active: bool,
    /// Head1x1 weights: `[channels][bottleneck]` row-major (channels rows × bottleneck cols).
    pub head1x1_w: AlignedVec<f32>,
    /// Head1x1 bias: `channels` elements.
    pub head1x1_b: AlignedVec<f32>,

    /// Total receptive field: sum of `(kernel-1)*dilation` + head kernel - 1.
    pub receptive_field_size: usize,
    /// Maximum frames per processing block.
    pub max_buffer_size: usize,

    /// Raw JSON for the single layer array.
    pub layer_raw: Option<Value>,

    /// Condition vector size (default 1 for A2 fast-path equivalence).
    pub condition_size: usize,

    /// Per-frame scratch buffer for conv output (2*bottleneck elements).
    /// When gating/blending is active, the conv output is 2× wide.
    pub z_scratch: AlignedVec<f32>,

    /// Scratch buffer for head1x1 projection output (channels elements).
    pub head1x1_scratch: AlignedVec<f32>,
}

impl WaveNetA2Dyn {
    /// Creates a new uninitialized dynamic A2 model.
    ///
    /// Allocates ring buffers (MirroredBuffer per layer, pow2 head accumulator)
    /// sized for the architecture and computes the receptive field.
    ///
    /// `kernel_sizes` and `dilations` must have length `num_layers`.
    /// The activation config vectors must also have length `num_layers`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        channels: usize,
        bottleneck: usize,
        kernel_sizes: &[usize],
        dilations: &[usize],
        activations: Vec<ActivationType>,
        gating_modes: Vec<GatingMode>,
        secondary_activations: Vec<Option<ActivationType>>,
        head1x1_active: bool,
    ) -> anyhow::Result<Self> {
        let num_layers = kernel_sizes.len();
        assert_eq!(dilations.len(), num_layers);
        assert_eq!(activations.len(), num_layers);
        assert_eq!(gating_modes.len(), num_layers);
        assert_eq!(secondary_activations.len(), num_layers);

        let max_buf = WAVENET_MAX_NUM_FRAMES;

        let mut rf = 0usize;
        for i in 0..num_layers {
            rf += (kernel_sizes[i] - 1) * dilations[i];
        }
        rf += super::super::params::A2_HEAD_KERNEL_SIZE - 1;

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        let head_ring_mask = head_ring_size - 1;

        let mut layer_buffers = Vec::with_capacity(num_layers);
        let mut layer_ring_sizes = Vec::with_capacity(num_layers);
        let mut layer_lookbacks = Vec::with_capacity(num_layers);
        let mut layer_buffer_starts = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let max_lookback = (kernel_sizes[i] - 1) * dilations[i];
            let cap = max_lookback + max_buf + 1;
            let mb = MirroredBuffer::<f32>::new(cap * channels)?;
            let ring_size = mb.size();
            layer_buffers.push(mb);
            layer_ring_sizes.push(ring_size);
            layer_lookbacks.push(max_lookback * channels);
            layer_buffer_starts.push(ring_size);
        }

        // Pre-allocate gating/blending configs.
        let mut gating_configs = Vec::with_capacity(num_layers);
        let mut blending_configs = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let gc = match gating_modes[i] {
                GatingMode::Gated => {
                    let sec = secondary_activations[i]
                        .clone()
                        .unwrap_or(ActivationType::Sigmoid);
                    Some(GatingActivationConfig::new(activations[i].clone(), sec))
                }
                _ => None,
            };
            let bc = match gating_modes[i] {
                GatingMode::Blended => {
                    let sec = secondary_activations[i]
                        .clone()
                        .unwrap_or(ActivationType::Sigmoid);
                    Some(BlendingActivationConfig::new(
                        activations[i].clone(),
                        sec,
                        bottleneck,
                    ))
                }
                _ => None,
            };
            gating_configs.push(gc);
            blending_configs.push(bc);
        }

        let head1x1_w = if head1x1_active {
            AlignedVec::new(bottleneck * channels, 0.0f32)
        } else {
            AlignedVec::new(0, 0.0f32)
        };
        let head1x1_b = if head1x1_active {
            AlignedVec::new(channels, 0.0f32)
        } else {
            AlignedVec::new(0, 0.0f32)
        };
        let head1x1_scratch = if head1x1_active {
            AlignedVec::new(channels, 0.0f32)
        } else {
            AlignedVec::new(0, 0.0f32)
        };

        Ok(Self {
            channels,
            bottleneck,
            num_layers,
            layers: Vec::with_capacity(num_layers),
            rechannel_w: AlignedVec::new(channels, 0u16),
            rechannel_w_f32: AlignedVec::new(channels, 0.0f32),
            head_conv: None,
            head_accum: AlignedVec::new(head_ring_size * channels, 0.0f32),
            head_write_pos: rf,
            head_ring_mask,
            layer_buffers,
            layer_ring_sizes,
            layer_lookbacks,
            layer_buffer_starts,
            layer_in: AlignedVec::new(channels * max_buf, 0.0f32),
            kernel_sizes: kernel_sizes.to_vec(),
            dilations: dilations.to_vec(),
            activations,
            gating_modes,
            secondary_activations,
            gating_configs,
            blending_configs,
            head1x1_active,
            head1x1_w,
            head1x1_b,
            receptive_field_size: rf,
            max_buffer_size: max_buf,
            layer_raw: None,
            condition_size: 1,
            z_scratch: AlignedVec::new(bottleneck * 2, 0.0f32),
            head1x1_scratch,
        })
    }

    /// Returns the channel count.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Returns the total receptive field size.
    #[inline(always)]
    pub fn receptive_field(&self) -> usize {
        self.receptive_field_size
    }

    /// Stores the raw layer JSON for FiLM config parsing during weight loading.
    pub fn set_layer_raw(&mut self, raw: Option<Value>) {
        self.layer_raw = raw;
    }

    /// Returns whether weights have been loaded.
    #[inline(always)]
    pub fn has_weights(&self) -> bool {
        !self.layers.is_empty()
    }

    /// Reallocates internal buffers to support the given maximum block size.
    pub fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        if max_buf <= self.max_buffer_size {
            return Ok(());
        }
        self.max_buffer_size = max_buf;
        let rf = self.receptive_field_size;
        let channels = self.channels;

        self.layer_buffers.clear();
        self.layer_ring_sizes.clear();
        self.layer_lookbacks.clear();
        self.layer_buffer_starts.clear();

        for i in 0..self.num_layers {
            let max_lookback = (self.kernel_sizes[i] - 1) * self.dilations[i];
            let cap = max_lookback + max_buf + 1;
            let mb = MirroredBuffer::<f32>::new(cap * channels)?;
            let ring_size = mb.size();
            self.layer_buffers.push(mb);
            self.layer_ring_sizes.push(ring_size);
            self.layer_lookbacks.push(max_lookback * channels);
            self.layer_buffer_starts.push(ring_size);
        }

        self.layer_in = AlignedVec::new(channels * max_buf, 0.0f32);

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        self.head_ring_mask = head_ring_size - 1;
        self.head_accum = AlignedVec::new(head_ring_size * channels, 0.0f32);
        self.head_write_pos = rf;

        Ok(())
    }

    /// Resets internal state for a new sample rate and max buffer size.
    pub fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buffer_size)?;
        self.prewarm();
        Ok(())
    }
}

#[cfg(test)]
#[path = "../dynamic_test.rs"]
mod tests;
