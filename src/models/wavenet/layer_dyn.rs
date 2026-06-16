// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::{WAVENET_MAX_NUM_FRAMES, WavenetProcessContext};
use super::conv1d_dyn::Conv1dDyn;
use super::dense_dyn::DenseLayerDyn;
use crate::math::common::{AlignedVec, SimdMath};

/// Complete Convolutional Cell (WaveNet Layer) with runtime dimensions.
#[derive(Clone)]
pub struct WaveNetLayerDyn {
    /// This layer's parametric dilated Causal 1D convolution mesh.
    pub conv1d: Conv1dDyn,
    /// Conditional injection mixing network.
    pub input_mixin: DenseLayerDyn,
    /// 1x1 decompression linear affine transform of the layer.
    pub one_by_one: DenseLayerDyn,
    /// Pre-allocated scratch buffer for conditioning mixin output
    /// (size: `ch * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_mixin: AlignedVec<f32>,
    /// Pre-allocated scratch buffer for Conv1D + mixin intermediate results
    /// (size: `ch * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_conv: AlignedVec<f32>,
}

impl WaveNetLayerDyn {
    /// Creates a new dynamic WaveNet layer.
    ///
    /// Scratch buffers are pre-allocated with size `ch * WAVENET_MAX_NUM_FRAMES`,
    /// ensuring zero allocation on the hot-path and supporting CH > 16
    /// (vs the const-generic stack buffer `[f32; 1024]` which is limited to CH ≤ 16).
    pub fn new(
        ch: usize,
        conv1d: Conv1dDyn,
        input_mixin: DenseLayerDyn,
        one_by_one: DenseLayerDyn,
    ) -> Self {
        let scratch_size = ch * WAVENET_MAX_NUM_FRAMES;
        Self {
            conv1d,
            input_mixin,
            one_by_one,
            scratch_mixin: AlignedVec::new(scratch_size, 0.0f32),
            scratch_conv: AlignedVec::new(scratch_size, 0.0f32),
        }
    }

    /// Processes a full WaveNet layer with runtime-dimensional scratch buffers.
    ///
    /// Uses single-frame Conv1D processing (dual-frame tiling to be added in
    /// T2.2/SIMD vectorization). Scratch buffers are pre-allocated `AlignedVec<f32>`
    /// on the heap, replacing the const-generic stack buffer `[f32; 1024]` to
    /// support CH > 16.
    ///
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(
        &mut self,
        mut ctx: WavenetProcessContext<'_>,
    ) {
        let ch = self.conv1d.in_ch;
        let scratch_len = ctx.num_frames * ch;
        debug_assert!(
            scratch_len <= self.scratch_mixin.len(),
            "process_block_internal: num_frames*ch ({}) exceeds scratch buffer ({})",
            scratch_len,
            self.scratch_mixin.len()
        );

        unsafe {
            let mixin_out = &mut self.scratch_mixin[..scratch_len];

            if M::IS_BF16 {
                self.input_mixin
                    .process_bf16::<M>(ctx.condition_bf16, mixin_out);
            } else {
                self.input_mixin
                    .process_block::<M>(ctx.condition, mixin_out, ctx.num_frames);
            }

            let conv_slice = &mut self.scratch_conv[..scratch_len];
            for i in 0..ctx.num_frames {
                let mixin_slice = &mixin_out[i * ch..(i + 1) * ch];
                let out_slice = &mut conv_slice[i * ch..(i + 1) * ch];

                if M::IS_BF16 {
                    self.conv1d.process_single_frame_bf16::<M>(
                        ctx.layer_buffer_bf16,
                        out_slice,
                        ctx.buffer_start + i,
                        Some(mixin_slice),
                    );
                } else {
                    self.conv1d.process_single_frame::<M>(
                        ctx.layer_buffer,
                        out_slice,
                        ctx.buffer_start + i,
                        Some(mixin_slice),
                    );
                }
            }

            if ctx.is_first_layer {
                M::tanh_and_overwrite_block(ctx.head_input, conv_slice);
            } else {
                M::tanh_and_accumulate_block(ctx.head_input, conv_slice);
            }

            let lb_offset = ctx.buffer_start * ch;
            let residual_slice = ctx
                .layer_buffer
                .get_unchecked(lb_offset..lb_offset + scratch_len);

            self.one_by_one.process_residual_batch::<M>(
                conv_slice,
                residual_slice,
                ctx.output,
                ctx.num_frames,
            );

            if let (true, Some(bf16_out)) = (M::IS_BF16, ctx.output_bf16.as_mut()) {
                M::f32_to_bf16(ctx.output, bf16_out);
            }
        }
    }
}
