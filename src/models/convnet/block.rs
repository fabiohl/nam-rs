// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ConvNet feed-forward block: Conv1D → BatchNorm → Activation.
//!
//! Each block is simpler than a WaveNet layer — there is no gating mechanism
//! (no input_mixin / 1x1) and no rechannel projection. Output feeds directly
//! into the next block or the post-stack head.

use crate::math::common::{AlignedVec, SimdMath};
use crate::models::a2::activations::{ActivationFn, ActivationType};
use crate::models::wavenet::Conv1dDyn;
use crate::models::wavenet::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};

use super::batch_norm::BatchNorm1D;

/// A single ConvNet feed-forward block.
///
/// Processing pipeline:
/// 1. Write input frames into the causal ring buffer
/// 2. Run causal Conv1D on the ring buffer → scratch buffer
/// 3. Apply BatchNorm (affine transform) in-place on scratch
/// 4. Apply activation function in-place on scratch
/// 5. Copy scratch to output
#[derive(Clone)]
#[repr(align(64))]
pub struct ConvNetBlock {
    /// Causal 1D convolution (runtime dimensions).
    pub conv: Conv1dDyn,
    /// Batch normalization (pre-fused affine transform).
    pub bn: BatchNorm1D,
    /// Activation function applied after batch norm.
    pub activation: ActivationType,
    /// Ring buffer state for causal convolution lookback.
    pub state: WaveNetLayerState,
    /// Scratch buffer for convolution output (out_ch * WAVENET_MAX_NUM_FRAMES).
    scratch: AlignedVec<f32>,
}

impl ConvNetBlock {
    /// Creates a new `ConvNetBlock` with zero-initialized weights.
    ///
    /// Weights must be populated via `set_conv_weights` / `set_conv_bias`
    /// and `set_bn_params` by the dispatcher.
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        do_bias: bool,
        activation: ActivationType,
        alloc_num: usize,
    ) -> std::io::Result<Self> {
        let num_blocks = out_ch.div_ceil(4);
        let weights_len = num_blocks * kernel * in_ch * 4;

        let conv = Conv1dDyn {
            weights: AlignedVec::new(weights_len, 0.0f32),
            bias: AlignedVec::new(out_ch, 0.0f32),
            do_bias,
            dilation,
            in_ch,
            out_ch,
            num_blocks,
            kernel,
            prefetch_fn: crate::math::common::prefetch_strategy_simple,
        };

        let bn = BatchNorm1D::from_fused(out_ch, &vec![0.0f32; out_ch], &vec![0.0f32; out_ch]);

        let receptive_field = (kernel - 1) * dilation;
        let state = WaveNetLayerState::new(in_ch, receptive_field, alloc_num)?;

        let scratch = AlignedVec::new(out_ch * WAVENET_MAX_NUM_FRAMES, 0.0f32);

        Ok(Self {
            conv,
            bn,
            activation,
            state,
            scratch,
        })
    }

    /// Returns the receptive field contribution of this block.
    pub fn receptive_field(&self) -> usize {
        (self.conv.kernel - 1) * self.conv.dilation
    }

    /// Loads convolution weights from a flat f32 slice.
    pub fn set_conv_weights(&mut self, weights: &[f32]) {
        let len = self.conv.weights.len().min(weights.len());
        self.conv.weights[..len].copy_from_slice(&weights[..len]);
    }

    /// Loads convolution bias from a flat f32 slice.
    pub fn set_conv_bias(&mut self, bias: &[f32]) {
        let len = self.conv.bias.len().min(bias.len());
        self.conv.bias[..len].copy_from_slice(&bias[..len]);
    }

    /// Loads pre-fused batch norm parameters.
    pub fn set_bn_params(&mut self, scale: &[f32], offset: &[f32]) {
        self.bn = BatchNorm1D::from_fused(self.conv.out_ch, scale, offset);
    }

    /// Public dispatch wrapper that selects the optimal SIMD path.
    ///
    /// # Safety
    /// Input and output slices must have sizes compatible with the block dimensions:
    /// `input.len() == num_frames * in_ch`, `output.len() == num_frames * out_ch`.
    #[inline(always)]
    pub unsafe fn process_block(&mut self, input: &[f32], output: &mut [f32], num_frames: usize) {
        unsafe {
            crate::math::common::dispatch_simd!(
                self,
                process_block_internal,
                input,
                output,
                num_frames
            )
        };
    }

    /// SIMD-dispatched processing kernel.
    ///
    /// # Safety
    /// `input.len()` must equal `num_frames * in_ch`.
    /// `output.len()` must equal `num_frames * out_ch`.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        let in_ch = self.conv.in_ch;
        let out_ch = self.conv.out_ch;
        let input_len = num_frames * in_ch;

        let buf_start = self.state.buffer_start * in_ch;
        self.state.layer_buffer[buf_start..buf_start + input_len]
            .copy_from_slice(&input[..input_len]);

        let scratch_slice = &mut self.scratch[..num_frames * out_ch];
        unsafe {
            self.conv.process_block::<M>(
                &self.state.layer_buffer,
                scratch_slice,
                self.state.buffer_start,
                num_frames,
                None,
            );
        }

        unsafe {
            self.bn.process(scratch_slice, num_frames);
        }

        self.activation.apply(scratch_slice);

        output[..num_frames * out_ch].copy_from_slice(scratch_slice);

        self.state.advance_frames(num_frames, in_ch);
    }

    /// Public prewarm wrapper with SIMD dispatch.
    #[cold]
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Fills the conv state buffer with a single frame of silence replicated
    /// backward to cover the entire receptive field.
    ///
    /// # Safety
    /// Must be called via `dispatch_simd!` macro. The state buffer must be
    /// properly allocated and the ring buffer start pointer must be valid.
    #[inline(always)]
    pub unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let in_ch = self.conv.in_ch;
        let out_ch = self.conv.out_ch;
        let kernel = self.conv.kernel;
        let dilation = self.conv.dilation;

        let silence = vec![0.0f32; in_ch];
        let buf_start = self.state.buffer_start * in_ch;

        self.state.layer_buffer[buf_start..buf_start + in_ch].copy_from_slice(&silence);

        let start_idx = buf_start;
        let src_range = start_idx..start_idx + in_ch;
        let max_offset = (kernel - 1) * dilation + 1;
        for offset in 1..=max_offset {
            let dst_idx = (self.state.buffer_start - offset) * in_ch;
            self.state
                .layer_buffer
                .copy_within(src_range.clone(), dst_idx);
        }

        let scratch_slice = &mut self.scratch[..out_ch];
        unsafe {
            self.conv.process_single_frame::<M>(
                &self.state.layer_buffer,
                scratch_slice,
                self.state.buffer_start,
                None,
            );
        }
        unsafe {
            self.bn.process(scratch_slice, 1);
        }
        self.activation.apply(scratch_slice);

        self.state.advance_frames(1, in_ch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::a2::activations::ActivationType;

    fn build_identity_block() -> ConvNetBlock {
        ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block")
    }

    #[test]
    fn test_construction_defaults() {
        let block = build_identity_block();
        assert_eq!(block.conv.in_ch, 1);
        assert_eq!(block.conv.out_ch, 1);
        assert_eq!(block.conv.kernel, 1);
        assert_eq!(block.receptive_field(), 0);
    }

    #[test]
    fn test_receptive_field_computation() {
        let block =
            ConvNetBlock::new(1, 4, 3, 8, false, ActivationType::ReLU, 0).expect("create block");
        assert_eq!(block.receptive_field(), (3 - 1) * 8);
    }

    #[test]
    fn test_process_identity_tanh() {
        let mut block =
            ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block");

        let weights = vec![1.0f32, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);

        let bn_scale = vec![1.0f32];
        let bn_offset = vec![0.0f32];
        block.set_bn_params(&bn_scale, &bn_offset);

        let input = [0.5f32];
        let mut output = [0.0f32];

        unsafe {
            block.process_block(&input, &mut output, 1);
        }

        let expected = 0.5f32.tanh();
        assert!((output[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_process_with_relu() {
        let mut block =
            ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::ReLU, 0).expect("create block");

        let weights = vec![1.0f32, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);

        let bn_scale = vec![1.0f32];
        let bn_offset = vec![0.0f32];
        block.set_bn_params(&bn_scale, &bn_offset);

        let input = [0.5f32, -0.3, 0.8, -1.0];
        let mut output = [0.0f32; 4];

        unsafe {
            block.process_block(&input, &mut output, 4);
        }

        assert!((output[0] - 0.5).abs() < 1e-4);
        assert!((output[1] - 0.0).abs() < 1e-4); // RELU clamps
        assert!((output[2] - 0.8).abs() < 1e-4);
        assert!((output[3] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_batch_norm_integration() {
        let mut block =
            ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block");

        let weights = vec![1.0f32, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);

        let bn_scale = vec![2.0f32];
        let bn_offset = vec![1.0f32];
        block.set_bn_params(&bn_scale, &bn_offset);

        let input = [0.5f32];
        let mut output = [0.0f32];

        unsafe {
            block.process_block(&input, &mut output, 1);
        }

        let conv_out: f32 = 0.5;
        let bn_out: f32 = conv_out * 2.0 + 1.0;
        let expected = bn_out.tanh();
        assert!((output[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_multi_frame_causal() {
        let mut block =
            ConvNetBlock::new(1, 1, 2, 1, false, ActivationType::Tanh, 0).expect("create block");

        let weights = vec![0.5f32, 0.0, 0.0, 0.0, 0.5f32, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);

        let bn_scale = vec![1.0f32];
        let bn_offset = vec![0.0f32];
        block.set_bn_params(&bn_scale, &bn_offset);

        let input = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 3];

        unsafe {
            block.process_block(&input, &mut output, 3);
        }

        let v0: f32 = 0.5 * 1.0;
        let expected_0 = v0.tanh();
        assert!((output[0] - expected_0).abs() < 1e-4);

        let v1: f32 = 0.5 * 1.0 + 0.5 * 2.0;
        let expected_1 = v1.tanh();
        assert!((output[1] - expected_1).abs() < 1e-4);

        let v2: f32 = 0.5 * 2.0 + 0.5 * 3.0;
        let expected_2 = v2.tanh();
        assert!((output[2] - expected_2).abs() < 2e-4);
    }

    #[test]
    fn test_prewarm_no_nan() {
        let mut block =
            ConvNetBlock::new(1, 1, 2, 1, false, ActivationType::Tanh, 0).expect("create block");

        let weights = vec![0.5f32, 0.0, 0.0, 0.0, 0.5f32, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);

        block.prewarm();

        let input = [1.0f32];
        let mut output = [0.0f32];
        unsafe {
            block.process_block(&input, &mut output, 1);
        }
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_conv_bias() {
        let mut block =
            ConvNetBlock::new(1, 1, 1, 1, true, ActivationType::Tanh, 0).expect("create block");

        let weights = vec![2.0f32, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);

        let bias = vec![0.5f32];
        block.set_conv_bias(&bias);

        let bn_scale = vec![1.0f32];
        let bn_offset = vec![0.0f32];
        block.set_bn_params(&bn_scale, &bn_offset);

        let input = [1.0f32];
        let mut output = [0.0f32];

        unsafe {
            block.process_block(&input, &mut output, 1);
        }

        let v: f32 = 1.0 * 2.0 + 0.5;
        let expected = v.tanh();
        assert!((output[0] - expected).abs() < 2e-4);
    }

    #[test]
    fn test_struct_alignment() {
        assert_eq!(std::mem::align_of::<ConvNetBlock>(), 64);
    }
}
