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

use super::conv1d::A2Conv1d;
use super::head::A2HeadConv;
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
    /// 23 dilated causal convolutions (one per layer).
    pub layer_convs: Vec<A2Conv1d>,

    /// 23 input-mixin weight vectors (`Conv1x1: condition(1) → CH`), u16 quantized.
    pub mixin_ws: Vec<AlignedVec<u16>>,

    /// 23 input-mixin bias vectors, f32.
    pub mixin_bs: Vec<AlignedVec<f32>>,

    /// 23 layer-1x1 weight vectors (`Conv1x1: CH → CH`), u16 quantized.
    pub l1x1_ws: Vec<AlignedVec<u16>>,

    /// 23 layer-1x1 bias vectors, f32.
    pub l1x1_bs: Vec<AlignedVec<f32>>,

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

    /// Shared layer processing ring buffer (column-major: `Channels` × cols).
    pub layer_buffer: AlignedVec<f32>,

    /// Write position in `layer_buffer` (in columns, wraps via layer ring logic).
    pub layer_write_pos: usize,

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
        let total = rf + max_buf + 1;

        // Head ring buffer: powers-of-2 above total (for efficient wrapping).
        let head_ring_size = total.next_power_of_two();
        let head_ring_mask = head_ring_size - 1;

        Self {
            layer_convs: Vec::with_capacity(A2_NUM_LAYERS),
            mixin_ws: Vec::with_capacity(A2_NUM_LAYERS),
            mixin_bs: Vec::with_capacity(A2_NUM_LAYERS),
            l1x1_ws: Vec::with_capacity(A2_NUM_LAYERS),
            l1x1_bs: Vec::with_capacity(A2_NUM_LAYERS),
            rechannel_w: AlignedVec::new(CH, 0u16),
            rechannel_b: AlignedVec::new(CH, 0.0f32),
            head_conv: None,
            head_accum: AlignedVec::new(head_ring_size * CH, 0.0f32),
            head_write_pos: 0,
            head_ring_mask,
            layer_buffer: AlignedVec::new(total * CH, 0.0f32),
            layer_write_pos: 0,
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
        let total = rf + max_buf + 1;

        self.layer_buffer = AlignedVec::new(total * CH, 0.0f32);
        self.layer_write_pos = 0;

        let head_ring_size = total.next_power_of_two();
        self.head_ring_mask = head_ring_size - 1;
        self.head_accum = AlignedVec::new(head_ring_size * CH, 0.0f32);
        self.head_write_pos = 0;
    }

    /// Full forward pass through the A2 model.
    ///
    /// Processes `input` samples and writes to `output`.
    /// Currently a stub — outputs silence until T1.5 (A2Layer) is implemented.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 {
            return;
        }

        // Stub: output silence. Full forward pass wired in T1.5.
        output[..total_frames].fill(0.0);

        // Track write position for buffer management even in stub mode.
        let num_frames = total_frames.min(self.max_buffer_size);
        self.layer_write_pos = self.layer_write_pos.wrapping_add(num_frames);
        self.head_write_pos = self.head_write_pos.wrapping_add(num_frames);
    }

    /// Pre-warms the model by filling the receptive field with silence.
    #[cold]
    pub fn prewarm(&mut self) {
        let rf = self.receptive_field_size;

        // Fill layer buffer with zeros for the receptive field.
        self.layer_buffer.fill(0.0);
        self.layer_write_pos = rf;

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
        model.prewarm();
        for v in model.layer_buffer.iter() {
            assert!(v.abs() < 1e-9, "layer_buffer not zeroed");
        }
        for v in model.head_accum.iter() {
            assert!(v.abs() < 1e-9, "head_accum not zeroed");
        }
        assert_eq!(model.layer_write_pos, model.receptive_field_size);
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
    }

    #[test]
    fn test_wavenet_a2_default_creates_valid_model() {
        let model = WaveNetA2::<3>::default();
        assert_eq!(model.channels(), 3);
        assert!(model.receptive_field_size > 0);
        assert!(!model.head_accum.is_empty());
        assert!(!model.layer_buffer.is_empty());
        assert!(model.rechannel_w.len() == 3);
        assert!(model.rechannel_b.len() == 3);
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
