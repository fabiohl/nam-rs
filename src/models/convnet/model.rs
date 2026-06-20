// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ConvNet feed-forward model — chains ConvNetBlock layers sequentially
//! with an optional post-stack head.

use crate::math::common::{AlignedVec, SimdMath};
use crate::models::wavenet::PostStackHead;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::block::ConvNetBlock;

/// ConvNet feed-forward model.
///
/// Composed of a sequence of [`ConvNetBlock`]s chained sequentially,
/// followed by an optional [`PostStackHead`] and a final `head_scale` gain.
///
/// Unlike WaveNet, ConvNet has no gating, no rechannel projections,
/// no condition_dsp, and no dual-array architecture. Each block's
/// output is the input of the next block directly.
#[repr(align(64))]
pub struct ConvNetModel {
    /// Sequential ConvNet blocks.
    pub blocks: Vec<ConvNetBlock>,
    /// Final voltage compensation scale (Target Output Scale).
    pub head_scale: f32,
    /// Total receptive field (sum of all block RFs + head RF contribution).
    pub receptive_field_size: usize,
    /// Optional post-stack head sub-object (Conv1D + activation).
    pub post_stack_head: Option<PostStackHead>,
    /// Scratch buffer for post-stack head output.
    pub head_output_scratch: AlignedVec<f32>,
    /// Ping-pong scratch buffers for block-to-block signal relay.
    pub(crate) scratch_a: AlignedVec<f32>,
    pub(crate) scratch_b: AlignedVec<f32>,
}

impl ConvNetModel {
    /// Returns the number of input channels expected by the first block.
    pub fn in_channels(&self) -> usize {
        self.blocks.first().map(|b| b.conv.in_ch).unwrap_or(1)
    }

    /// Returns the number of output channels produced by the model.
    pub fn out_channels(&self) -> usize {
        self.post_stack_head
            .as_ref()
            .map(|h| h.out_channels())
            .unwrap_or_else(|| self.blocks.last().map(|b| b.conv.out_ch).unwrap_or(1))
    }

    /// Resolves the full forward pass and produces waveform samples in zero allocation (DSP).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { crate::math::common::dispatch_simd!(self, process_internal, input, output) };
    }

    /// SIMD-dispatched processing kernel.
    ///
    /// Chains blocks sequentially: block 0 receives raw input, each subsequent
    /// block receives the output of the previous block. After all blocks,
    /// the result is passed through the optional post-stack head and
    /// scaled by `head_scale`.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 || self.blocks.is_empty() {
            output[..total_frames].fill(0.0);
            return;
        }

        let out_ch = self.out_channels();
        let mut pos = 0;

        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            let num_blocks = self.blocks.len();
            let blocks_ptr = self.blocks.as_mut_ptr();

            // Block 0 writes to scratch_a
            let first_out_ch = unsafe { (*blocks_ptr).conv.out_ch };
            let dst_a = &mut self.scratch_a[..num_frames * first_out_ch];
            unsafe {
                (*blocks_ptr).process_block_internal::<M>(in_slice, dst_a, num_frames);
            }

            let mut src_is_a = true;

            for i in 1..num_blocks {
                let curr = unsafe { &mut *blocks_ptr.add(i) };
                let curr_out_ch = curr.conv.out_ch;

                if src_is_a {
                    let src = &self.scratch_a
                        [..num_frames * unsafe { (*blocks_ptr.add(i - 1)).conv.out_ch }];
                    let dst = &mut self.scratch_b[..num_frames * curr_out_ch];
                    unsafe {
                        curr.process_block_internal::<M>(src, dst, num_frames);
                    }
                } else {
                    let src = &self.scratch_b
                        [..num_frames * unsafe { (*blocks_ptr.add(i - 1)).conv.out_ch }];
                    let dst = &mut self.scratch_a[..num_frames * curr_out_ch];
                    unsafe {
                        curr.process_block_internal::<M>(src, dst, num_frames);
                    }
                }

                src_is_a = !src_is_a;
            }

            // After the loop, if number of blocks is odd, result is in scratch_a.
            // If number of blocks is even (n>=2), result alternates: block0→a, block1→b, block2→a...
            // Final: if (num_blocks-1) % 2 == 0, result is in scratch_a.
            let last_result_in_a = (num_blocks - 1).is_multiple_of(2);
            let last_out_ch = unsafe { (*blocks_ptr.add(num_blocks - 1)).conv.out_ch };
            let last_slice = if last_result_in_a {
                &self.scratch_a[..num_frames * last_out_ch]
            } else {
                &self.scratch_b[..num_frames * last_out_ch]
            };

            if let Some(ref mut head_proc) = self.post_stack_head {
                let head_out_ch = head_proc.out_channels();
                let head_scratch = &mut self.head_output_scratch[..num_frames * head_out_ch];
                unsafe {
                    head_proc.process_block(last_slice, head_scratch, num_frames);
                }
                let out_start = pos * out_ch;
                let out_slice = &mut output[out_start..out_start + num_frames * out_ch];
                out_slice.copy_from_slice(head_scratch);
                unsafe {
                    M::apply_gain(out_slice, self.head_scale);
                }
            } else {
                let out_start = pos * out_ch;
                let out_slice = &mut output[out_start..out_start + num_frames * out_ch];
                out_slice.copy_from_slice(last_slice);
                unsafe {
                    M::apply_gain(out_slice, self.head_scale);
                }
            }

            pos += num_frames;
        }
    }

    /// Stabilizes the model by processing silence (Zero Input) for pre-warm.
    #[cold]
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    #[inline(always)]
    #[cold]
    unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let num_blocks = self.blocks.len();
        if num_blocks == 0 {
            return;
        }

        let blocks_ptr = self.blocks.as_mut_ptr();

        for i in 0..num_blocks {
            unsafe {
                (*blocks_ptr.add(i)).prewarm_internal::<M>();
            }
        }

        if let Some(ref mut head_proc) = self.post_stack_head {
            head_proc.prewarm();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::a2::activations::ActivationType;

    fn build_single_block_model() -> ConvNetModel {
        let mut block =
            ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block");

        let weights = vec![1.0f32, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);

        let bn_scale = vec![1.0f32];
        let bn_offset = vec![0.0f32];
        block.set_bn_params(&bn_scale, &bn_offset);

        ConvNetModel {
            blocks: vec![block],
            head_scale: 1.0,
            receptive_field_size: 0,
            post_stack_head: None,
            head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
        }
    }

    #[test]
    fn test_single_block_process() {
        let mut model = build_single_block_model();

        let input = [0.5f32];
        let mut output = [0.0f32];

        model.process(&input, &mut output);

        assert!((output[0] - 0.5f32.tanh()).abs() < 1e-4);
    }

    #[test]
    fn test_head_scale() {
        let mut model = build_single_block_model();
        model.head_scale = 2.0;

        let input = [0.5f32];
        let mut output = [0.0f32];

        model.process(&input, &mut output);

        let expected = 2.0 * 0.5f32.tanh();
        assert!((output[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn test_empty_model_outputs_silence() {
        let mut model = ConvNetModel {
            blocks: vec![],
            head_scale: 1.0,
            receptive_field_size: 0,
            post_stack_head: None,
            head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
        };

        let input = [0.5f32, -0.3];
        let mut output = [1.0f32; 2];
        model.process(&input, &mut output);
        assert_eq!(output, [0.0, 0.0]);
    }

    #[test]
    fn test_empty_input_noop() {
        let mut model = build_single_block_model();

        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];

        model.process(&input, &mut output);
    }

    #[test]
    fn test_prewarm_no_panic() {
        let mut model = build_single_block_model();
        model.prewarm();

        let input = [0.0f32];
        let mut output = [0.0f32];
        model.process(&input, &mut output);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_two_block_chain() {
        let mut block0 =
            ConvNetBlock::new(1, 2, 1, 1, false, ActivationType::ReLU, 0).expect("block 0");
        let weights0 = vec![1.0f32, 2.0, 0.0, 0.0];
        block0.set_conv_weights(&weights0);
        block0.set_bn_params(&[1.0f32, 1.0], &[0.0f32, 0.0]);

        let mut block1 =
            ConvNetBlock::new(2, 1, 1, 1, false, ActivationType::Tanh, 1).expect("block 1");
        let weights1 = vec![0.5f32, 0.0, 0.0, 0.0, 0.5f32, 0.0, 0.0, 0.0];
        block1.set_conv_weights(&weights1);
        block1.set_bn_params(&[1.0f32], &[0.0f32]);

        let model = ConvNetModel {
            blocks: vec![block0, block1],
            head_scale: 1.0,
            receptive_field_size: 0,
            post_stack_head: None,
            head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_a: AlignedVec::new(2 * WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
        };

        let mut model = model;
        let input = [2.0f32];
        let mut output = [0.0f32];

        model.process(&input, &mut output);

        let _b0_c0: f32 = 2.0 * 1.0;
        let _b0_c1: f32 = 2.0 * 2.0;
        let b1_out: f32 = 4.0 * 0.5 + 2.0 * 0.5;
        let expected = b1_out.tanh();
        assert!(
            (output[0] - expected).abs() < 5e-4,
            "output[0]={}, expected={}, diff={}",
            output[0],
            expected,
            (output[0] - expected).abs()
        );
    }

    #[test]
    fn test_post_stack_head_integration() {
        use crate::loader::nam_json::model::HeadConfig;
        use crate::models::wavenet::PostStackHead;

        let mut block =
            ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::ReLU, 0).expect("block");
        let weights = vec![1.0f32, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);
        block.set_bn_params(&[1.0f32], &[0.0f32]);

        let head_config = HeadConfig {
            channels: Some(1),
            bias: Some(false),
            out_channels: Some(1),
            activation: Some("Tanh".to_string()),
            kernel_size: Some(1),
        };
        let head = PostStackHead::from_config(&head_config, 1).expect("head");

        let model = ConvNetModel {
            blocks: vec![block],
            head_scale: 1.0,
            receptive_field_size: 0,
            post_stack_head: Some(head),
            head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
        };

        let mut model = model;
        model
            .post_stack_head
            .as_mut()
            .unwrap()
            .set_weights(&[1.0, 0.0, 0.0, 0.0]);

        let input = [0.5f32];
        let mut output = [0.0f32];
        model.process(&input, &mut output);
        assert!((output[0] - 0.5f32.tanh()).abs() < 1e-4);
    }

    #[test]
    fn test_prewarm_with_head() {
        use crate::loader::nam_json::model::HeadConfig;

        let mut block =
            ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::ReLU, 0).expect("block");
        block.set_conv_weights(&[1.0, 0.0, 0.0, 0.0]);
        block.set_bn_params(&[1.0f32], &[0.0f32]);

        let head_config = HeadConfig {
            channels: Some(1),
            bias: Some(false),
            out_channels: Some(1),
            activation: None,
            kernel_size: Some(1),
        };
        let head = PostStackHead::from_config(&head_config, 1).expect("head");

        let mut model = ConvNetModel {
            blocks: vec![block],
            head_scale: 1.0,
            receptive_field_size: 0,
            post_stack_head: Some(head),
            head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
        };

        model
            .post_stack_head
            .as_mut()
            .unwrap()
            .set_weights(&[1.0, 0.0, 0.0, 0.0]);
        model.prewarm();

        let input = [0.0f32];
        let mut output = [0.0f32];
        model.process(&input, &mut output);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_struct_alignment() {
        assert_eq!(std::mem::align_of::<ConvNetModel>(), 64);
    }
}
