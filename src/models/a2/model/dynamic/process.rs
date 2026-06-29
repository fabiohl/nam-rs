// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 Dynamic model — processing methods.
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

use crate::math::common::SimdMath;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::{BlendingActivationConfig, GatingActivationConfig, GatingMode};
use crate::models::a2::layer::A2Layer;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::WaveNetA2Dyn;

impl WaveNetA2Dyn {
    /// Full forward pass through the dynamic A2 model.
    ///
    /// Uses per-frame processing with the polymorphic `A2Conv1d::process_single_frame`
    /// for maximum flexibility. Each layer applies activation or gating/blending
    /// according to its per-layer config.
    ///
    /// # Block Size Contract
    /// Any input size ≤ `max_buffer_size` is safe: processing is internally chunked
    /// into sub-blocks of ≤ `WAVENET_MAX_NUM_FRAMES` (64).
    ///
    /// **SIMD Dispatch:** The `dispatch_simd!` macro evaluates the hardware once
    /// and monomorphizes `process_internal` to the detected ISA (AVX2/AVX-512),
    /// eliminating per-frame `is_x86_feature_detected` branches.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::common::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Monomorphized inner loop — see [`process`](Self::process) for contract.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total = input.len();
        if total == 0 {
            return;
        }

        output[..total].fill(0.0);

        if self.layers.is_empty() {
            self.head_write_pos = (self.head_write_pos + total) & self.head_ring_mask;
            return;
        }

        debug_assert!(
            total <= self.max_buffer_size,
            "process: input ({total}) > max_buffer_size ({})",
            self.max_buffer_size
        );
        let nf_total = total.min(self.max_buffer_size);

        let mut pos = 0;
        while pos < nf_total {
            let nf = (nf_total - pos).min(WAVENET_MAX_NUM_FRAMES);

            self.rechannel_prescale(input, pos, nf);
            let head_wp = self.advance_head_ring(nf);

            for li in 0..self.num_layers {
                self.layer_forward_dispatch::<M>(li, nf, input, pos, head_wp);
            }

            self.head_finalize(head_wp, nf, &mut output[pos..pos + nf]);
            pos += nf;
        }
    }

    /// Phase 0: rechannel pre-scaling — `input × rechannel_w_f32 → layer_in`.
    #[inline(always)]
    fn rechannel_prescale(&mut self, input: &[f32], pos: usize, nf: usize) {
        let channels = self.channels;
        for (f, &x) in input[pos..pos + nf].iter().enumerate() {
            let base = f * channels;
            for c in 0..channels {
                self.layer_in[base + c] = self.rechannel_w_f32[c] * x;
            }
        }
    }

    /// Advances the head accumulator ring buffer.
    ///
    /// When the write cursor plus `nf` would overflow the ring capacity,
    /// the tail `K-1` samples are memmove'd to the start and the write
    /// position wraps around. Returns the (possibly wrapped) write position
    /// for use by the layer loop.
    #[inline(always)]
    fn advance_head_ring(&mut self, nf: usize) -> usize {
        let head_keep = super::super::super::params::A2_HEAD_KERNEL_SIZE - 1;
        let head_cap = self.head_ring_mask + 1;
        if self.head_write_pos + nf > head_cap {
            let keep_start = self.head_write_pos - head_keep;
            let keep_bytes = head_keep * self.channels;
            let src = keep_start * self.channels;
            self.head_accum.copy_within(src..src + keep_bytes, 0);
            self.head_write_pos = head_keep;
        }
        self.head_write_pos
    }

    /// Per-layer forward dispatch for a single layer index.
    ///
    /// # Safety
    ///
    /// Caller must ensure `li < self.num_layers` and that `nf` frames of valid
    /// data are available at `input[pos..pos+nf]`. Internal conv/film/head
    /// accesses assume caller-verified buffer capacities.
    #[inline(always)]
    fn layer_forward_dispatch<M: SimdMath>(
        &mut self,
        li: usize,
        nf: usize,
        input: &[f32],
        pos: usize,
        head_wp: usize,
    ) {
        let channels = self.channels;
        let bottleneck = self.bottleneck;
        let is_first = li == 0;
        let is_last = li == self.num_layers - 1;
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

            let z_scratch = &mut self.z_scratch;
            let head_accum = &mut self.head_accum;
            let layer_in = &mut self.layer_in;
            let head1x1_scratch = &mut self.head1x1_scratch;
            let head1x1_w = &self.head1x1_w;
            let head1x1_b = &self.head1x1_b;
            let gating_config = self.gating_configs[li].as_ref();
            let mut blending_config = self.blending_configs[li].as_mut();
            let activation = &self.activations[li];

            for f in 0..nf {
                let bc = blending_config.as_deref_mut();
                unsafe {
                    process_frame_dyn::<M>(
                        layer,
                        history,
                        f,
                        max_lookback_cols,
                        input,
                        pos,
                        head_wp,
                        z_out_ch,
                        use_gating,
                        use_blending,
                        is_first,
                        is_last,
                        self.channels,
                        self.bottleneck,
                        self.head1x1_active,
                        z_scratch,
                        head_accum,
                        layer_in,
                        head1x1_scratch,
                        head1x1_w,
                        head1x1_b,
                        gating_config,
                        bc,
                        activation,
                    );
                }
            }
        }
    }

    /// Finalizes the head convolution and advances the head write position.
    #[inline(always)]
    fn head_finalize(&mut self, head_wp: usize, nf: usize, output: &mut [f32]) {
        self.head_write_pos = (head_wp + nf) & self.head_ring_mask;

        if let Some(ref head) = self.head_conv {
            head.process(
                &self.head_accum,
                self.head_write_pos,
                self.head_ring_mask,
                nf,
                output,
            );
        }
    }
}

// ── Private helpers ────────────────────────────────────────────

/// Per-frame inner core: conv, FiLM, mixin, activation/gating/blending,
/// head accumulation, and l1x1 residual for a single frame in one layer.
///
/// `M` is the ISA monomorphization type propagated from the top-level
/// `dispatch_simd!` in [`WaveNetA2Dyn::process`].
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn process_frame_dyn<M: SimdMath>(
    layer: &mut A2Layer,
    history: &[f32],
    f: usize,
    max_lookback_cols: usize,
    input: &[f32],
    pos: usize,
    head_wp: usize,
    z_out_ch: usize,
    use_gating: bool,
    use_blending: bool,
    is_first: bool,
    is_last: bool,
    channels: usize,
    bottleneck: usize,
    head1x1_active: bool,
    z_scratch: &mut [f32],
    head_accum: &mut [f32],
    layer_in: &mut [f32],
    head1x1_scratch: &mut [f32],
    head1x1_w: &[f32],
    head1x1_b: &[f32],
    gating_config: Option<&GatingActivationConfig>,
    blending_config: Option<&mut BlendingActivationConfig>,
    activation: &ActivationType,
) {
    let frame_idx = max_lookback_cols + f;
    let cond_slice = &input[pos + f..pos + f + 1];
    let cond = cond_slice[0];

    #[allow(unused_assignments)]
    let mut z_len = z_out_ch;

    // 1. Dilated conv → z_scratch.
    unsafe {
        layer
            .conv
            .process_single_frame::<M>(history, &mut z_scratch[..z_out_ch], frame_idx, None);
    }

    // FiLM post-conv + pre-mixin.
    if let Some(ref mut film) = layer.conv_post_film {
        unsafe {
            film.process(&mut z_scratch[..z_out_ch], cond_slice);
        }
    }
    if let Some(ref mut film) = layer.input_mixin_pre_film {
        unsafe {
            film.process(&mut z_scratch[..z_out_ch], cond_slice);
        }
    }

    // 2. Input mixin.
    for (c, z) in z_scratch
        .iter_mut()
        .enumerate()
        .take(z_out_ch.min(layer.mixin_w.len()))
    {
        *z += layer.mixin_w[c] * cond;
    }

    // FiLM post-mixin + pre-activation.
    if let Some(ref mut film) = layer.input_mixin_post_film {
        unsafe {
            film.process(&mut z_scratch[..z_out_ch], cond_slice);
        }
    }
    if let Some(ref mut film) = layer.activation_pre_film {
        unsafe {
            film.process(&mut z_scratch[..z_out_ch], cond_slice);
        }
    }

    // 3. Activation or Gating/Blending.
    if use_gating {
        if let Some(gc) = gating_config {
            unsafe {
                gc.apply_gating_simd::<M>(&mut z_scratch[..z_out_ch]);
            }
        }
        z_len = bottleneck;
    } else if use_blending {
        if let Some(bc) = blending_config {
            unsafe {
                bc.apply_blending_simd::<M>(&mut z_scratch[..z_out_ch]);
            }
        }
        z_len = bottleneck;
    } else {
        unsafe {
            activation.apply_simd::<M>(&mut z_scratch[..bottleneck]);
        }
        z_len = bottleneck;
    }

    // FiLM post-activation.
    if let Some(ref mut film) = layer.activation_post_film {
        unsafe {
            film.process(&mut z_scratch[..z_len], cond_slice);
        }
    }

    // 4. Head accumulator.
    let head_off = (head_wp + f) * channels;
    if head1x1_active {
        for oc in 0..channels {
            let mut sum = head1x1_b[oc];
            for ic in 0..bottleneck {
                sum += head1x1_w[ic * channels + oc] * z_scratch[ic];
            }
            head1x1_scratch[oc] = sum;
        }
        if is_first {
            head_accum[head_off..head_off + channels].copy_from_slice(&head1x1_scratch[..channels]);
        } else {
            for c in 0..channels {
                head_accum[head_off + c] += head1x1_scratch[c];
            }
        }
    } else {
        debug_assert_eq!(
            bottleneck, channels,
            "head1x1 must be active when bottleneck != channels"
        );
        if is_first {
            head_accum[head_off..head_off + bottleneck].copy_from_slice(&z_scratch[..bottleneck]);
        } else {
            for c in 0..bottleneck {
                head_accum[head_off + c] += z_scratch[c];
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
                sum += l1x1_w[ic * channels + oc] * z_scratch[ic];
            }
            layer_in[base + oc] += sum;
        }
        if let Some(ref mut film) = layer.layer1x1_post_film {
            unsafe {
                film.process(&mut layer_in[base..base + channels], cond_slice);
            }
        }
    }
}
