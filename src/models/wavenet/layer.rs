// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::WavenetProcessContext;
use super::conv1d::Conv1d;
use super::dense::DenseLayer;
use crate::math::common::{AlignedVec, SimdMath};

/// Complete Convolutional Cell (WaveNet Layer).
#[derive(Clone)]
pub struct WaveNetLayer<const COND: usize, const CH: usize, const K: usize> {
    /// This layer's parametric dilated Causal 1D convolution mesh.
    pub conv1d: Conv1d<CH, CH, K>,
    /// Conditional injection mixing network.
    pub input_mixin: DenseLayer<COND, CH>,
    /// 1x1 decompression linear affine transform of the layer.
    pub one_by_one: DenseLayer<CH, CH>,
    /// Pre-allocated scratch buffer for conditioning mixin output
    /// (size: `CH * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_mixin: AlignedVec<f32>,
    /// Pre-allocated scratch buffer for Conv1D + mixin intermediate results
    /// (size: `CH * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_conv: AlignedVec<f32>,
}

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processes a full WaveNet layer, iterating `FastMath` in AVX2.
    ///
    /// Conv1D uses full-precision f32 weights; DenseLayer uses standard quantized paths.
    ///
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    #[inline]
    pub unsafe fn process_block_internal<M: SimdMath>(&mut self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            head_input,
            output,
            layer_buffer,
            buffer_start,
            num_frames,
            seed,
            is_first_layer,
            ..
        } = ctx;

        unsafe {
            debug_assert!(
                num_frames * CH <= self.scratch_mixin.len(),
                "process_block_internal: num_frames*CH ({}) exceeds scratch_mixin capacity ({})",
                num_frames * CH,
                self.scratch_mixin.len(),
            );
            assert!(
                num_frames * CH <= self.scratch_mixin.len(),
                "process_block_internal: num_frames*CH ({}) exceeds scratch_mixin capacity ({})",
                num_frames * CH,
                self.scratch_mixin.len(),
            );
            assert!(
                num_frames * CH <= self.scratch_conv.len(),
                "process_block_internal: num_frames*CH ({}) exceeds scratch_conv capacity ({})",
                num_frames * CH,
                self.scratch_conv.len(),
            );

            let mixin_out = self.scratch_mixin.get_unchecked_mut(..num_frames * CH);
            self.input_mixin
                .process_block::<M>(condition, mixin_out, num_frames);

            let conv_slice = self.scratch_conv.get_unchecked_mut(..num_frames * CH);

            let mut i = 0;
            let mut chunks = conv_slice.chunks_exact_mut(2 * CH);
            for chunk in chunks.by_ref() {
                let (out_frame_f0, out_frame_f1) = chunk.split_at_mut(CH);

                let mix_idx_f0 = i * CH;
                let mix_idx_f1 = (i + 1) * CH;
                let mixin_f0 = &mixin_out[mix_idx_f0..mix_idx_f0 + CH];
                let mixin_f1 = &mixin_out[mix_idx_f1..mix_idx_f1 + CH];

                self.conv1d.process_dual_frame_with_mixin::<M>(
                    layer_buffer,
                    out_frame_f0,
                    out_frame_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    mixin_f0,
                    mixin_f1,
                );
                i += 2;
            }

            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                let mix_idx = i * CH;
                let mixin_slice = &mixin_out[mix_idx..mix_idx + CH];

                self.conv1d.process_single_frame_with_mixin::<M>(
                    layer_buffer,
                    rem,
                    buffer_start + i,
                    mixin_slice,
                );
            }

            if let Some(s) = seed {
                M::tanh_and_accumulate_with_seed(head_input, conv_slice, s);
            } else if is_first_layer {
                M::tanh_and_overwrite_block(head_input, conv_slice);
            } else {
                M::tanh_and_accumulate_block(head_input, conv_slice);
            }

            let lb_offset = buffer_start * CH;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * CH);

            self.one_by_one.process_residual_batch::<M>(
                conv_slice,
                residual_slice,
                output,
                num_frames,
            );
        }
    }
}
