// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 model struct (`WaveNetA2<const CH: usize>`).
//!
//! Single layer-array of 23 dilated causal layers with skip-connection accumulator
//! and head rechannel convolution, matching the fast-path from `a2_fast.cpp`.
//!
//! ## Architecture
//!
//! 1. Input rechannel: `Conv1x1(1 → CH)` (bias, no activation)
//! 2. 23 layers: dilated conv → input-mixin → LeakyReLU → head_accum += out → layer1x1 → residual
//! 3. Head conv: `Conv1D(CH → 1, K=16, bias)` over head_accum ring → × head_scale
//!
//! Processing is chunked by `WAVENET_MAX_NUM_FRAMES` (64) with zero allocation on the hot-path.

use super::head::A2HeadConv;
use super::layer::A2Layer;
use super::params::{A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_NUM_LAYERS};
use crate::math::common::AlignedVec;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

/// Computes the receptive field size for the A2 architecture.
///
/// Sum of `(kernel_size - 1) * dilation` across all 23 layers,
/// plus `(head_kernel_size - 1)` for the head convolution lookback.
#[inline]
pub const fn a2_receptive_field() -> usize {
    let mut rf = 0usize;
    let mut i = 0;
    while i < A2_NUM_LAYERS {
        rf += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
        i += 1;
    }
    rf + (A2_HEAD_KERNEL_SIZE - 1)
}

/// Complete WaveNet A2 Model.
///
/// `CH` = channel count (3 for Lite/Nano, 8 for Full/Standard).
///
/// ## Source of truth
/// - `a2_fast.cpp`: class `A2FastModel` (members, process, prewarm, reset)
/// - `detail.h`: `LayerArray::Process` (per-layer sequence)
/// - `docs/wavenet_walkthrough.rst:278-351`
pub struct WaveNetA2<const CH: usize> {
    /// 23 A2 layers (one per layer index). Populated by `set_weights` (T1.6).
    pub layers: Vec<A2Layer>,

    /// Input rechannel weights: `Conv1x1(1 → CH)`, u16 quantized.
    pub rechannel_w: AlignedVec<u16>,

    /// Input rechannel bias, f32.
    pub rechannel_b: AlignedVec<f32>,

    /// Head convolution (K=16 over skip-connection accumulator, bias, head_scale).
    pub head_conv: Option<A2HeadConv>,

    /// Head accumulator ring buffer (skip-connection sum, column-major).
    pub head_accum: AlignedVec<f32>,

    /// Write position in `head_accum` (in columns, wraps via `head_ring_mask`).
    pub head_write_pos: usize,

    /// Ring mask for `head_accum` (pow2 ring, mask = capacity - 1).
    pub head_ring_mask: usize,

    /// Combined history arena for all 23 layers' linear ring buffers (column-major).
    pub layer_buffer: AlignedVec<f32>,

    /// Offsets into `layer_buffer` for each layer's ring (byte-based for direct slicing).
    pub layer_offsets: Vec<usize>,

    /// Per-layer linear ring capacities (in columns).
    pub layer_ring_capacities: Vec<usize>,

    /// Per-layer max lookback = (kernel-1)*dilation.
    pub layer_lookbacks: Vec<usize>,

    /// Per-layer write positions in their ring buffers (in columns).
    pub layer_write_poses: Vec<usize>,

    /// Inter-layer data buffer: `CH × max_buffer_size` f32, reused across layers.
    /// Each layer reads from it, then writes its l1x1 residual back (in-place update).
    pub layer_in: AlignedVec<f32>,

    /// Total receptive field: sum of `(kernel-1)*dilation` + head kernel - 1.
    pub receptive_field_size: usize,

    /// Maximum frames per processing block (= `WAVENET_MAX_NUM_FRAMES`).
    pub max_buffer_size: usize,
}

impl<const CH: usize> WaveNetA2<CH> {
    /// Creates a new uninitialized WaveNet A2 model.
    ///
    /// Allocates ring buffers sized for the architecture and computes
    /// the receptive field. Weight-bearing fields start empty and are
    /// populated by the weight loader (T1.6).
    pub fn new() -> Self {
        let rf = a2_receptive_field();
        let max_buf = WAVENET_MAX_NUM_FRAMES;

        // Head ring buffer: powers-of-2 above total (for efficient wrapping).
        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        let head_ring_mask = head_ring_size - 1;

        // Compute per-layer linear ring sizes and total arena.
        let mut layer_offsets = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_ring_capacities = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_lookbacks = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_write_poses = Vec::with_capacity(A2_NUM_LAYERS);
        let mut arena_total = 0usize;
        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            // Linear ring: 2*max_lookback + max_buffer_size columns (RING_MODE == 0).
            let cap = 2 * max_lookback + max_buf;
            layer_offsets.push(arena_total);
            layer_ring_capacities.push(cap);
            layer_lookbacks.push(max_lookback);
            layer_write_poses.push(max_lookback); // initial write_pos = max_lookback
            arena_total += CH * cap;
        }

        Self {
            layers: Vec::with_capacity(A2_NUM_LAYERS),
            rechannel_w: AlignedVec::new(CH, 0u16),
            rechannel_b: AlignedVec::new(CH, 0.0f32),
            head_conv: None,
            head_accum: AlignedVec::new(head_ring_size * CH, 0.0f32),
            head_write_pos: rf,
            head_ring_mask,
            layer_buffer: AlignedVec::new(arena_total, 0.0f32),
            layer_offsets,
            layer_ring_capacities,
            layer_lookbacks,
            layer_write_poses,
            layer_in: AlignedVec::new(CH * max_buf, 0.0f32),
            receptive_field_size: rf,
            max_buffer_size: max_buf,
        }
    }

    /// Returns the channel count.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        CH
    }

    /// Returns the total receptive field size.
    #[inline(always)]
    pub fn receptive_field(&self) -> usize {
        self.receptive_field_size
    }

    /// Reallocates internal buffers to support the given maximum block size.
    pub fn set_max_buffer_size(&mut self, max_buf: usize) {
        if max_buf <= self.max_buffer_size {
            return;
        }
        self.max_buffer_size = max_buf;
        let rf = self.receptive_field_size;

        // Recompute per-layer linear ring sizes.
        let mut layer_offsets = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_ring_capacities = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_lookbacks = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_write_poses = Vec::with_capacity(A2_NUM_LAYERS);
        let mut arena_total = 0usize;
        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            let cap = 2 * max_lookback + max_buf;
            layer_offsets.push(arena_total);
            layer_ring_capacities.push(cap);
            layer_lookbacks.push(max_lookback);
            layer_write_poses.push(max_lookback);
            arena_total += CH * cap;
        }

        self.layer_buffer = AlignedVec::new(arena_total, 0.0f32);
        self.layer_offsets = layer_offsets;
        self.layer_ring_capacities = layer_ring_capacities;
        self.layer_lookbacks = layer_lookbacks;
        self.layer_write_poses = layer_write_poses;

        self.layer_in = AlignedVec::new(CH * max_buf, 0.0f32);

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        self.head_ring_mask = head_ring_size - 1;
        self.head_accum = AlignedVec::new(head_ring_size * CH, 0.0f32);
        self.head_write_pos = rf;
    }

    /// Full forward pass through the A2 model.
    ///
    /// Processes `input` samples and writes to `output`.
    /// Requires layers to be populated via `set_weights` (T1.6).
    /// Outputs silence until weights are loaded.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let num_frames = input.len();
        if num_frames == 0 {
            return;
        }

        output[..num_frames].fill(0.0);

        // If layers/history haven't been loaded yet (pre-T1.6), just track positions.
        if self.layers.is_empty() {
            self.head_write_pos += num_frames;
            return;
        }

        let nf = num_frames.min(self.max_buffer_size);
        let ch = CH;

        // 1. Rechannel and prepare cond buffer from input.
        // layer_in[c + f*CH] = rechannel_w[c] * input[f]
        let rechannel = &self.rechannel_w;
        for (f, x) in input.iter().take(nf).enumerate() {
            let base = f * ch;
            for c in 0..ch {
                let rw = half::f16::from_bits(rechannel[c]).to_f32();
                self.layer_in[base + c] = rw * x;
            }
        }

        // 2. Head ring management: rewind if overflow.
        let head_keep = A2_HEAD_KERNEL_SIZE - 1;
        let head_cap = self.head_ring_mask + 1;
        if self.head_write_pos + nf > head_cap {
            let keep_start = self.head_write_pos - head_keep;
            let keep_bytes = head_keep * ch;
            let src = keep_start * ch;
            self.head_accum.copy_within(src..src + keep_bytes, 0);
            self.head_write_pos = head_keep;
        }
        let head_wp = self.head_write_pos;

        // 3. Per-layer forward pass.
        for li in 0..A2_NUM_LAYERS {
            let is_first = li == 0;
            let is_last = li == A2_NUM_LAYERS - 1;
            let cap = self.layer_ring_capacities[li];
            let lookback = self.layer_lookbacks[li];
            let wp = self.layer_write_poses[li];
            let offset = self.layer_offsets[li];

            // Linear ring rewind if overflow (RING_MODE == 0).
            let wp = if wp + nf > cap {
                let keep = lookback;
                let keep_bytes = keep * ch;
                let src_start = offset + (wp - keep) * ch;
                // memmove the last `keep` columns to the start of this layer's ring.
                self.layer_buffer
                    .copy_within(src_start..src_start + keep_bytes, offset);
                lookback
            } else {
                wp
            };

            // Ring-write: copy layer_in into this layer's ring at wp.
            let ring_dst = offset + wp * ch;
            self.layer_buffer[ring_dst..ring_dst + nf * ch]
                .copy_from_slice(&self.layer_in[..nf * ch]);

            self.layer_write_poses[li] = wp + nf;

            // Phase B: process frames (immutable borrow of layer_buffer).
            {
                let history = &self.layer_buffer[offset..offset + (wp + nf) * ch];
                let layer = &self.layers[li];

                for (f, x) in input.iter().take(nf).enumerate() {
                    let head_col = head_wp + f;
                    let lin_slice = &mut self.layer_in[f * ch..(f + 1) * ch];
                    let mut frame_z = [0.0f32; 8];
                    let z_slice = &mut frame_z[..ch];

                    // frame_idx = wp + f (post-ring-write, so wp points to the start of this block).
                    let frame_idx = wp + f;

                    unsafe {
                        layer
                            .conv
                            .process_single_frame::<crate::math::common::Avx2Math>(
                                history, z_slice, frame_idx, None,
                            );
                    }

                    let mixin: &[f32] = &layer.mixin_w;
                    for c in 0..ch {
                        z_slice[c] += mixin[c] * x;
                    }
                    for z in z_slice.iter_mut().take(ch) {
                        if *z < 0.0 {
                            *z *= 0.01;
                        }
                    }
                    let head_off = head_col * ch;
                    if is_first {
                        self.head_accum[head_off..head_off + ch].copy_from_slice(z_slice);
                    } else {
                        for (c, z_val) in z_slice.iter().enumerate().take(ch) {
                            self.head_accum[head_off + c] += *z_val;
                        }
                    }
                    if !is_last {
                        let l1x1: &[f32] = &layer.l1x1_w;
                        let l1x1_b: &[f32] = &layer.l1x1_b;
                        for c in 0..ch {
                            let mut sum = l1x1_b[c];
                            for u in 0..ch {
                                sum += l1x1[u * ch + c] * z_slice[u];
                            }
                            lin_slice[c] += sum;
                        }
                    }
                }
            }
        }

        // 4. Advance head write position.
        self.head_write_pos = (head_wp + nf) & self.head_ring_mask;

        // 5. Head convolution → output.
        if let Some(ref head) = self.head_conv {
            head.process(
                &self.head_accum,
                self.head_write_pos,
                self.head_ring_mask,
                nf,
                &mut output[..nf],
            );
        }
    }

    /// Pre-warms the model by filling the receptive field with silence.
    #[cold]
    pub fn prewarm(&mut self) {
        let rf = self.receptive_field_size;

        // Zero the entire layer history arena.
        self.layer_buffer.fill(0.0);

        // Reset each layer's write position to max_lookback.
        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            self.layer_write_poses[i] = max_lookback;
        }

        // Zero inter-layer buffer.
        self.layer_in.fill(0.0);

        // Fill head accumulator with zeros.
        self.head_accum.fill(0.0);
        self.head_write_pos = rf;
    }

    /// Resets internal state for a new sample rate and max buffer size.
    pub fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) {
        self.set_max_buffer_size(max_buffer_size);
        self.prewarm();
    }
}

impl<const CH: usize> Default for WaveNetA2<CH> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenet_a2_receptive_field_ch3() {
        let model = WaveNetA2::<3>::new();
        // Reference: computed from A2_KERNEL_SIZES and A2_DILATIONS arrays.
        let expected = {
            let mut sum = 0usize;
            for i in 0..A2_NUM_LAYERS {
                sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            }
            sum + (A2_HEAD_KERNEL_SIZE - 1)
        };
        assert_eq!(model.receptive_field_size, expected);
        assert_eq!(model.receptive_field(), expected);
        assert_eq!(model.channels(), 3);
    }

    #[test]
    fn test_wavenet_a2_receptive_field_ch8() {
        let model = WaveNetA2::<8>::new();
        let expected = {
            let mut sum = 0usize;
            for i in 0..A2_NUM_LAYERS {
                sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            }
            sum + (A2_HEAD_KERNEL_SIZE - 1)
        };
        assert_eq!(model.receptive_field_size, expected);
        assert_eq!(model.channels(), 8);
    }

    #[test]
    fn test_wavenet_a2_process_stub_output_silence() {
        let mut model = WaveNetA2::<3>::new();
        let input = vec![0.5f32; 64];
        let mut output = vec![1.0f32; 64];
        model.process(&input, &mut output);
        for v in &output {
            assert!(v.abs() < 1e-9, "expected silence, got {}", v);
        }
    }

    #[test]
    fn test_wavenet_a2_process_empty_input() {
        let mut model = WaveNetA2::<3>::new();
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        model.process(&input, &mut output);
        // Empty input should be a no-op.
    }

    #[test]
    fn test_wavenet_a2_prewarm_fills_buffers() {
        let mut model = WaveNetA2::<3>::new();
        // Pre-fill with non-zero to verify overwrite.
        model.layer_buffer.fill(0.5);
        model.head_accum.fill(0.5);
        model.layer_in.fill(0.5);
        model.prewarm();
        for v in model.layer_buffer.iter() {
            assert!(v.abs() < 1e-9, "layer_buffer not zeroed");
        }
        for v in model.head_accum.iter() {
            assert!(v.abs() < 1e-9, "head_accum not zeroed");
        }
        for v in model.layer_in.iter() {
            assert!(v.abs() < 1e-9, "layer_in not zeroed");
        }
        assert_eq!(model.head_write_pos, model.receptive_field_size);
    }

    #[test]
    fn test_wavenet_a2_reset_reallocates_and_prewarms() {
        let mut model = WaveNetA2::<3>::new();
        let orig_layer_len = model.layer_buffer.len();
        model.reset(48000, 128);
        assert!(model.layer_buffer.len() > orig_layer_len);
        assert_eq!(model.max_buffer_size, 128);
        for v in model.layer_buffer.iter() {
            assert!(v.abs() < 1e-9, "reset layer_buffer not zeroed");
        }
    }

    #[test]
    fn test_wavenet_a2_set_max_buffer_size_noop_on_smaller() {
        let mut model = WaveNetA2::<3>::new();
        let orig_len = model.layer_buffer.len();
        model.set_max_buffer_size(32);
        assert_eq!(model.layer_buffer.len(), orig_len);
        assert_eq!(model.max_buffer_size, WAVENET_MAX_NUM_FRAMES);
    }

    #[test]
    fn test_wavenet_a2_set_max_buffer_size_grows() {
        let mut model = WaveNetA2::<8>::new();
        let orig_len = model.layer_buffer.len();
        model.set_max_buffer_size(256);
        assert!(model.layer_buffer.len() > orig_len);
        assert_eq!(model.max_buffer_size, 256);
        // Verify per-layer offsets increase monotonically.
        assert_eq!(model.layer_offsets.len(), A2_NUM_LAYERS);
        for i in 1..A2_NUM_LAYERS {
            assert!(model.layer_offsets[i] > model.layer_offsets[i - 1]);
        }
        assert_eq!(model.layer_ring_capacities.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_write_poses.len(), A2_NUM_LAYERS);
    }

    #[test]
    fn test_wavenet_a2_default_creates_valid_model() {
        let model = WaveNetA2::<3>::default();
        assert_eq!(model.channels(), 3);
        assert!(model.receptive_field_size > 0);
        assert!(!model.head_accum.is_empty());
        assert!(!model.layer_buffer.is_empty());
        assert_eq!(model.rechannel_w.len(), 3);
        assert_eq!(model.rechannel_b.len(), 3);
        assert_eq!(model.layer_offsets.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_ring_capacities.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_write_poses.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_in.len(), 3 * model.max_buffer_size);
    }

    #[test]
    fn test_wavenet_a2_const_receptive_field_matches_runtime() {
        let rf_const = a2_receptive_field();
        let model3 = WaveNetA2::<3>::new();
        let model8 = WaveNetA2::<8>::new();
        assert_eq!(model3.receptive_field_size, rf_const);
        assert_eq!(model8.receptive_field_size, rf_const);
    }
}
