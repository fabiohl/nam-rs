// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Dynamic model (`WaveNetA2Dyn`).
//!
//! Runtime-dimensioned A2 engine supporting `bottleneck != channels`,
//! heterogeneous activations, gating/blending, head1x1, grouped layer1x1,
//! and the full A2 topology spectrum.
//!
//! ## Architecture
//!
//! 1. Input rechannel: `Conv1x1(1 → channels)` (bias, no activation)
//! 2. Per-layer (per-frame):
//!    - Dilated causal conv: `channels → bottleneck` (or `2*bottleneck` if gating/blending)
//!    - FiLM post-conv (optional)
//!    - Input mixin: `+ mixin_w[c] * input_cond`
//!    - FiLM post-mixin (optional)
//!    - Activation (heterogeneous) or Gating/Blending
//!    - FiLM post-activation (optional)
//!    - Head accumulator: direct or via head1x1 projection `bottleneck → channels`
//!    - L1x1 residual: `bottleneck → channels` added to `layer_in` (skip last layer)
//!    - FiLM post-l1x1 (optional)
//! 3. Head conv: `Conv1D(channels → 1, K=16, bias)` × head_scale
//!
//! ## Ring buffer architecture
//!
//! Same MirroredBuffer + pow2 head ring as `WaveNetA2<CH>`. Per-layer history
//! stores `channels`-wide data. The dilated conv reads `channels`-wide history
//! and produces `bottleneck` (or `2*bottleneck`) outputs.
//!
//! ## RT-Safety
//!
//! All scratch buffers (z_scratch, gating_scratch) and gating/blending configs
//! are pre-allocated at construction time. Zero heap alloc on the hot-path.

use crate::dsp::mirror_buf::MirroredBuffer;
use crate::math::common::AlignedVec;
use crate::models::a2::activations::{ActivationFn, ActivationType};
use crate::models::a2::gating::{BlendingActivationConfig, GatingActivationConfig, GatingMode};
use crate::models::a2::head::A2HeadConv;
use crate::models::a2::layer::A2Layer;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;
use serde_json::Value;

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
    #[allow(clippy::too_many_lines)]
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), String> {
        let total = weights.len();
        let mut pos: usize = 0;
        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let num_layers = self.num_layers;

        // ── 1. Rechannel: Conv1x1(1 → channels) (no bias) ─────────────
        let rw_f32 = read_slice_dyn(weights, &mut pos, channels, total, "rechannel_w")?;
        self.rechannel_w_f32.copy_from_slice(rw_f32);

        // ── 2. Per-layer weights ──────────────────────────────────────
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
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
            let conv_w_f32 = read_slice_dyn(
                weights,
                &mut pos,
                conv_w_count,
                total,
                &format!("layer[{i}].conv_w"),
            )?;
            let mut conv_w = AlignedVec::new(conv_w_padded, 0.0f32);
            transpose_conv1d_interleaved_4wide(conv_w_f32, &mut conv_w, channels, conv_out, ksize);

            // 2b. Conv bias.
            let conv_b_f32 = read_slice_dyn(
                weights,
                &mut pos,
                conv_out,
                total,
                &format!("layer[{i}].conv_b"),
            )?;
            let conv_b = AlignedVec::from(conv_b_f32.to_vec());

            let prefetch_fn: crate::math::common::PrefetchFn = if dilation >= 128 {
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
            let mixin_w_f32 = read_slice_dyn(
                weights,
                &mut pos,
                conv_out,
                total,
                &format!("layer[{i}].mixin_w"),
            )?;
            let mixin_w = AlignedVec::from(mixin_w_f32.to_vec());

            // 2d. L1x1: bottleneck×channels + channels bias.
            let l1x1_w_count = bottleneck * channels;
            let l1x1_w_f32 = read_slice_dyn(
                weights,
                &mut pos,
                l1x1_w_count,
                total,
                &format!("layer[{i}].l1x1_w"),
            )?;
            let mut l1x1_w = AlignedVec::new(l1x1_w_count, 0.0f32);
            transpose_dense_f32(l1x1_w_f32, &mut l1x1_w, bottleneck, channels);

            let l1x1_b_f32 = read_slice_dyn(
                weights,
                &mut pos,
                channels,
                total,
                &format!("layer[{i}].l1x1_b"),
            )?;
            let l1x1_b = AlignedVec::from(l1x1_b_f32.to_vec());

            let mut layer = A2Layer::new_dyn(conv, mixin_w, l1x1_w, l1x1_b, channels, bottleneck);

            // FiLM layers (if active in layer_raw JSON) — read weights after l1x1 bias.
            if let Some(ref raw) = self.layer_raw {
                let configs = super::set_weights::parse_film_configs(raw);
                super::set_weights::load_film_for_layer(
                    &mut layer,
                    &configs,
                    channels,
                    self.condition_size,
                    weights,
                    &mut pos,
                    total,
                    i,
                )?;
            }

            layers.push(layer);
        }

        // ── 3. Head1x1 (if active) ─────────────────────────────────
        if self.head1x1_active {
            let h1_w_count = bottleneck * channels;
            let h1_w_f32 = read_slice_dyn(weights, &mut pos, h1_w_count, total, "head1x1_w")?;
            let mut h1_w = AlignedVec::new(h1_w_count, 0.0f32);
            transpose_dense_f32(h1_w_f32, &mut h1_w, bottleneck, channels);

            let h1_b_f32 = read_slice_dyn(weights, &mut pos, channels, total, "head1x1_b")?;
            let mut h1_b = AlignedVec::new(channels, 0.0f32);
            h1_b.copy_from_slice(h1_b_f32);

            self.head1x1_w = h1_w;
            self.head1x1_b = h1_b;
        }

        // ── 4. Head conv ───────────────────────────────────────────
        let head_k = crate::models::a2::params::A2_HEAD_KERNEL_SIZE;
        let head_w_f32 = read_slice_dyn(weights, &mut pos, head_k * channels, total, "head_w")?;
        let mut head_w = AlignedVec::new(head_k * channels, 0.0f32);
        transpose_head_w(head_w_f32, &mut head_w, channels, head_k);

        let head_b = {
            let s = read_slice_dyn(weights, &mut pos, 1, total, "head_b")?;
            s[0]
        };

        // ── 5. Head scale ──────────────────────────────────────────
        let head_scale = {
            let s = read_slice_dyn(weights, &mut pos, 1, total, "head_scale")?;
            s[0]
        };

        // ── 6. Exhaustion check ────────────────────────────────────
        if pos != total {
            return Err(format!(
                "set_weights: stream has {} unconsumed f32 (consumed {}, total {})",
                total - pos,
                pos,
                total
            ));
        }

        // ── 7. Commit ──────────────────────────────────────────────
        self.layers = layers;
        self.head_conv = Some(A2HeadConv::new(head_w, head_b, head_scale, channels));

        Ok(())
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

    /// Full forward pass through the dynamic A2 model.
    ///
    /// Uses per-frame processing with the polymorphic `A2Conv1d::process_single_frame`
    /// for maximum flexibility. Each layer applies activation or gating/blending
    /// according to its per-layer config.
    ///
    /// # Block Size Contract
    /// Any input size ≤ `max_buffer_size` is safe: processing is internally chunked
    /// into sub-blocks of ≤ `WAVENET_MAX_NUM_FRAMES` (64).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let total = input.len();
        if total == 0 {
            return;
        }

        output[..total].fill(0.0);

        if self.layers.is_empty() {
            self.head_write_pos += total;
            return;
        }

        debug_assert!(
            total <= self.max_buffer_size,
            "process: input ({total}) > max_buffer_size ({})",
            self.max_buffer_size
        );
        let nf_total = total.min(self.max_buffer_size);

        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let num_layers = self.num_layers;
        let head_keep = super::super::params::A2_HEAD_KERNEL_SIZE - 1;

        let mut pos = 0;
        while pos < nf_total {
            let nf = (nf_total - pos).min(WAVENET_MAX_NUM_FRAMES);

            // ── Phase 0: rechannel pre-scaling: input × rechannel_w_f32 → layer_in ──
            for (f, &x) in input[pos..pos + nf].iter().enumerate() {
                let base = f * channels;
                for c in 0..channels {
                    self.layer_in[base + c] = self.rechannel_w_f32[c] * x;
                }
            }

            // Head ring wrap.
            let head_cap = self.head_ring_mask + 1;
            if self.head_write_pos + nf > head_cap {
                let keep_start = self.head_write_pos - head_keep;
                let keep_bytes = head_keep * channels;
                let src = keep_start * channels;
                self.head_accum.copy_within(src..src + keep_bytes, 0);
                self.head_write_pos = head_keep;
            }
            let head_wp = self.head_write_pos;

            // ── Phase 1: per-layer processing ──
            for li in 0..num_layers {
                let is_first = li == 0;
                let is_last = li == num_layers - 1;
                let ring_size = self.layer_ring_sizes[li];
                let lookback = self.layer_lookbacks[li];
                let max_lookback_cols = lookback / channels;
                let bs = self.layer_buffer_starts[li];
                let use_gating = self.gating_modes[li] == GatingMode::Gated;
                let use_blending = self.gating_modes[li] == GatingMode::Blended;
                let z_out_ch = if use_gating || use_blending {
                    bottleneck * 2
                } else {
                    bottleneck
                };

                // Copy layer_in → history buffer.
                {
                    let buf = &mut self.layer_buffers[li];
                    buf[bs..bs + nf * channels].copy_from_slice(&self.layer_in[..nf * channels]);
                    // Apply conv_pre_film on new frames.
                    for f in 0..nf {
                        if let Some(ref mut film) = self.layers[li].conv_pre_film {
                            unsafe {
                                film.process(
                                    &mut buf[bs + f * channels..bs + (f + 1) * channels],
                                    &input[pos + f..pos + f + 1],
                                );
                            }
                        }
                    }
                }

                // Advance buffer start with wrap.
                if bs + nf * channels + self.max_buffer_size * channels > ring_size * 2 {
                    self.layer_buffer_starts[li] = bs + nf * channels - ring_size;
                } else {
                    self.layer_buffer_starts[li] = bs + nf * channels;
                }

                {
                    let history = &self.layer_buffers[li][bs - lookback..bs + nf * channels];
                    let layer = &mut self.layers[li];

                    for f in 0..nf {
                        let frame_idx = max_lookback_cols + f;
                        let cond = input[pos + f];
                        let cond_slice = &input[pos + f..pos + f + 1];

                        #[allow(unused_assignments)]
                        let mut z_len = z_out_ch;

                        // 1. Dilated conv → z_scratch.
                        unsafe {
                            layer.conv.process_single_frame(
                                history,
                                &mut self.z_scratch[..z_out_ch],
                                frame_idx,
                                None,
                            );
                        }

                        // FiLM post-conv + pre-mixin.
                        if let Some(ref mut film) = layer.conv_post_film {
                            unsafe {
                                film.process(&mut self.z_scratch[..z_out_ch], cond_slice);
                            }
                        }
                        if let Some(ref mut film) = layer.input_mixin_pre_film {
                            unsafe {
                                film.process(&mut self.z_scratch[..z_out_ch], cond_slice);
                            }
                        }

                        // 2. Input mixin.
                        // SAFETY: bounds-checked by z_out_ch .min(mixin_w.len()).
                        for c in 0..z_out_ch.min(layer.mixin_w.len()) {
                            self.z_scratch[c] += layer.mixin_w[c] * cond;
                        }

                        // FiLM post-mixin + pre-activation.
                        if let Some(ref mut film) = layer.input_mixin_post_film {
                            unsafe {
                                film.process(&mut self.z_scratch[..z_out_ch], cond_slice);
                            }
                        }
                        if let Some(ref mut film) = layer.activation_pre_film {
                            unsafe {
                                film.process(&mut self.z_scratch[..z_out_ch], cond_slice);
                            }
                        }

                        // 3. Activation or Gating/Blending.
                        if use_gating {
                            if let Some(ref gc) = self.gating_configs[li] {
                                gc.apply_gating(&mut self.z_scratch[..z_out_ch]);
                            }
                            z_len = bottleneck;
                        } else if use_blending {
                            if let Some(ref mut bc) = self.blending_configs[li] {
                                bc.apply_blending(&mut self.z_scratch[..z_out_ch]);
                            }
                            z_len = bottleneck;
                        } else {
                            self.activations[li].apply(&mut self.z_scratch[..bottleneck]);
                            z_len = bottleneck;
                        }

                        // FiLM post-activation.
                        if let Some(ref mut film) = layer.activation_post_film {
                            unsafe {
                                film.process(&mut self.z_scratch[..z_len], cond_slice);
                            }
                        }

                        // 4. Head accumulator.
                        let head_off = (head_wp + f) * channels;
                        if self.head1x1_active {
                            // Apply head1x1: bottleneck → channels projection.
                            let h1_w = &self.head1x1_w;
                            let h1_b = &self.head1x1_b;
                            for oc in 0..channels {
                                let mut sum = h1_b[oc];
                                for ic in 0..bottleneck {
                                    sum += h1_w[ic * channels + oc] * self.z_scratch[ic];
                                }
                                self.head1x1_scratch[oc] = sum;
                            }
                            if is_first {
                                self.head_accum[head_off..head_off + channels]
                                    .copy_from_slice(&self.head1x1_scratch[..channels]);
                            } else {
                                for c in 0..channels {
                                    self.head_accum[head_off + c] += self.head1x1_scratch[c];
                                }
                            }
                        } else {
                            debug_assert_eq!(
                                bottleneck, channels,
                                "head1x1 must be active when bottleneck != channels"
                            );
                            if is_first {
                                self.head_accum[head_off..head_off + bottleneck]
                                    .copy_from_slice(&self.z_scratch[..bottleneck]);
                            } else {
                                for c in 0..bottleneck {
                                    self.head_accum[head_off + c] += self.z_scratch[c];
                                }
                            }
                        }

                        // 5. L1x1 residual (skip on last layer).
                        if !is_last {
                            let base = f * channels;
                            let l1x1_w = &layer.l1x1_w;
                            let l1x1_b = &layer.l1x1_b;
                            for oc in 0..channels {
                                let mut sum = l1x1_b[oc];
                                for ic in 0..bottleneck {
                                    sum += l1x1_w[ic * channels + oc] * self.z_scratch[ic];
                                }
                                self.layer_in[base + oc] += sum;
                            }
                            // FiLM post-l1x1.
                            if let Some(ref mut film) = layer.layer1x1_post_film {
                                unsafe {
                                    film.process(
                                        &mut self.layer_in[base..base + channels],
                                        cond_slice,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            self.head_write_pos = (head_wp + nf) & self.head_ring_mask;

            if let Some(ref head) = self.head_conv {
                head.process(
                    &self.head_accum,
                    self.head_write_pos,
                    self.head_ring_mask,
                    nf,
                    &mut output[pos..pos + nf],
                );
            }

            pos += nf;
        }
    }

    /// Pre-warms the model by filling the receptive field with silence.
    #[cold]
    pub fn prewarm(&mut self) {
        for buf in &mut self.layer_buffers {
            let len = buf.size();
            buf[..len].fill(0.0);
        }
        for i in 0..self.num_layers {
            self.layer_buffer_starts[i] = self.layer_ring_sizes[i];
        }
        self.layer_in.fill(0.0);
        self.head_accum.fill(0.0);
        self.head_write_pos = self.receptive_field_size;

        if self.has_weights() {
            let prewarm_samples = self.receptive_field_size;
            let block = WAVENET_MAX_NUM_FRAMES;
            let zeros = vec![0.0f32; block];
            let mut discard = vec![0.0f32; block];
            let mut remaining = prewarm_samples;
            while remaining > 0 {
                let nf = remaining.min(block);
                self.process(&zeros[..nf], &mut discard[..nf]);
                remaining -= nf;
            }
        }
    }

    /// Resets internal state for a new sample rate and max buffer size.
    pub fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.set_max_buffer_size(max_buffer_size)?;
        self.prewarm();
        Ok(())
    }
}

// ── Private helpers for set_weights ───────────────────────────────────

/// Reads a contiguous slice of `n` f32 values from `weights[pos..]`,
/// advancing `pos`. Returns an error with the label if out of bounds.
#[inline]
fn read_slice_dyn<'a>(
    weights: &'a [f32],
    pos: &mut usize,
    n: usize,
    total: usize,
    label: &str,
) -> Result<&'a [f32], String> {
    if *pos + n > total {
        return Err(format!(
            "set_weights: stream exhausted at position {} (need {} for \"{}\", total {})",
            *pos, n, label, total
        ));
    }
    let slice = &weights[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

/// Rearranges dense layer weights from row-major to col-major.
///
/// Input:  `raw[out * in_size + in_c]` (row-major)
/// Output: `weights[in_c * out_size + out_c]` (col-major)
fn transpose_dense_f32(raw: &[f32], weights: &mut [f32], in_size: usize, out_size: usize) {
    for out_c in 0..out_size {
        for in_c in 0..in_size {
            weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c];
        }
    }
}

/// Rearranges conv1d weights into "Interleaved 4-Wide" format.
///
/// Groups output channels in blocks of 4 for SIMD processing.
fn transpose_conv1d_interleaved_4wide(
    raw: &[f32],
    weights: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    let num_blocks = out_ch.div_ceil(4);
    for b in 0..num_blocks {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    let target_idx = b * (kernel * in_ch * 4) + k * (in_ch * 4) + in_c * 4 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * kernel + k;
                        weights[target_idx] = raw[raw_idx];
                    }
                }
            }
        }
    }
}

/// Transposes head weights from [channel][tap] to [tap][channel].
fn transpose_head_w(raw: &[f32], head: &mut [f32], channels: usize, kernel: usize) {
    for tap in 0..kernel {
        for ch in 0..channels {
            head[tap * channels + ch] = raw[ch * kernel + tap];
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::a2::activations::ActivationType;
    use crate::models::a2::gating::GatingMode;
    use crate::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS};

    fn make_standard_activations(num: usize) -> Vec<ActivationType> {
        vec![
            ActivationType::LeakyReLU {
                negative_slope: A2_LEAKY_SLOPE,
            };
            num
        ]
    }

    fn make_standard_gating(num: usize) -> Vec<GatingMode> {
        vec![GatingMode::None; num]
    }

    fn make_standard_secondary(num: usize) -> Vec<Option<ActivationType>> {
        vec![None; num]
    }

    #[test]
    fn test_wavenet_a2_dyn_new_ch3() {
        let model = WaveNetA2Dyn::new(
            3,
            3,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();

        assert_eq!(model.channels, 3);
        assert_eq!(model.bottleneck, 3);
        assert_eq!(model.num_layers, A2_NUM_LAYERS);
        assert!(model.receptive_field_size > 0);
        assert!(!model.head_accum.is_empty());
        assert!(!model.layer_buffers.is_empty());
        assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
        assert_eq!(model.rechannel_w.len(), 3);
        assert_eq!(model.layer_in.len(), 3 * model.max_buffer_size);
        assert!(!model.head1x1_active);
    }

    #[test]
    fn test_wavenet_a2_dyn_new_ch8() {
        let model = WaveNetA2Dyn::new(
            8,
            8,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();

        assert_eq!(model.channels, 8);
        assert_eq!(model.bottleneck, 8);
        assert_eq!(model.num_layers, A2_NUM_LAYERS);
        assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
    }

    #[test]
    fn test_wavenet_a2_dyn_bottleneck_neq_channels() {
        let model = WaveNetA2Dyn::new(
            8,
            4,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            true,
        )
        .unwrap();

        assert_eq!(model.channels, 8);
        assert_eq!(model.bottleneck, 4);
        assert!(model.head1x1_active);
        assert_eq!(model.head1x1_w.len(), 4 * 8);
        assert_eq!(model.head1x1_b.len(), 8);
    }

    #[test]
    fn test_wavenet_a2_dyn_gating_prealloc() {
        let num = A2_NUM_LAYERS;
        let mut gating = vec![GatingMode::None; num];
        gating[0] = GatingMode::Gated;
        gating[1] = GatingMode::Blended;
        let mut sec = vec![None; num];
        sec[0] = Some(ActivationType::Sigmoid);
        sec[1] = Some(ActivationType::Tanh);

        let model = WaveNetA2Dyn::new(
            3,
            3,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(num),
            gating,
            sec,
            false,
        )
        .unwrap();

        assert!(model.gating_configs[0].is_some());
        assert!(model.blending_configs[1].is_some());
        assert!(model.gating_configs[2].is_none());
        assert!(model.blending_configs[0].is_none());
    }

    #[test]
    fn test_wavenet_a2_dyn_process_empty_input() {
        let mut model = WaveNetA2Dyn::new(
            3,
            3,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        model.process(&input, &mut output);
    }

    #[test]
    fn test_wavenet_a2_dyn_process_silence_no_weights() {
        let mut model = WaveNetA2Dyn::new(
            3,
            3,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();
        let input = vec![0.5f32; 64];
        let mut output = vec![1.0f32; 64];
        model.process(&input, &mut output);
        for v in &output {
            assert!(v.abs() < 1e-9, "expected silence, got {}", v);
        }
    }

    #[test]
    fn test_wavenet_a2_dyn_prewarm_zeroes() {
        let mut model = WaveNetA2Dyn::new(
            3,
            3,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();
        for buf in &mut model.layer_buffers {
            let len = buf.size();
            buf[..len].fill(0.5);
        }
        model.head_accum.fill(0.5);
        model.layer_in.fill(0.5);
        model.prewarm();
        for buf in &model.layer_buffers {
            let len = buf.size();
            for &v in buf[..len].iter() {
                assert!(v.abs() < 1e-9, "layer_buffer not zeroed");
            }
        }
        for v in model.head_accum.iter() {
            assert!(v.abs() < 1e-9, "head_accum not zeroed");
        }
        for v in model.layer_in.iter() {
            assert!(v.abs() < 1e-9, "layer_in not zeroed");
        }
    }

    #[test]
    fn test_wavenet_a2_dyn_set_max_buffer_size_noop() {
        let mut model = WaveNetA2Dyn::new(
            3,
            3,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();
        let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
        model.set_max_buffer_size(32).unwrap();
        assert_eq!(model.layer_ring_sizes, orig_sizes);
    }

    #[test]
    fn test_wavenet_a2_dyn_set_max_buffer_size_grows() {
        let mut model = WaveNetA2Dyn::new(
            8,
            8,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();
        let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
        model.set_max_buffer_size(256).unwrap();
        assert!(model.max_buffer_size == 256);
        let any_grew = orig_sizes
            .iter()
            .zip(model.layer_ring_sizes.iter())
            .any(|(a, b)| b > a);
        assert!(any_grew, "at least one ring should grow");
    }

    #[test]
    fn test_wavenet_a2_dyn_has_weights_false_initially() {
        let model = WaveNetA2Dyn::new(
            3,
            3,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();
        assert!(!model.has_weights());
    }

    #[test]
    fn test_wavenet_a2_dyn_receptive_field() {
        let model = WaveNetA2Dyn::new(
            3,
            3,
            &A2_KERNEL_SIZES,
            &A2_DILATIONS,
            make_standard_activations(A2_NUM_LAYERS),
            make_standard_gating(A2_NUM_LAYERS),
            make_standard_secondary(A2_NUM_LAYERS),
            false,
        )
        .unwrap();
        let expected = {
            let mut sum = 0usize;
            for i in 0..A2_NUM_LAYERS {
                sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            }
            sum + (crate::models::a2::A2_HEAD_KERNEL_SIZE - 1)
        };
        assert_eq!(model.receptive_field_size, expected);
        assert_eq!(model.receptive_field(), expected);
    }
}
