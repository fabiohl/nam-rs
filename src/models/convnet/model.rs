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
#[path = "convnet_model_test.rs"]
mod tests;
