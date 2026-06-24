// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use core::mem::MaybeUninit;

use super::common::{WAVENET_MAX_NUM_FRAMES, WavenetProcessContext};
use super::conv1d::Conv1d;
use super::dense::DenseLayer;
use crate::math::common::SimdMath;

/// Complete Convolutional Cell (WaveNet Layer).
#[derive(Clone)]
pub struct WaveNetLayer<const COND: usize, const CH: usize, const K: usize> {
    /// This layer's parametric dilated Causal 1D convolution mesh.
    pub conv1d: Conv1d<CH, CH, K>,
    /// Conditional injection mixing network.
    pub input_mixin: DenseLayer<COND, CH>,
    /// 1x1 decompression linear affine transform of the layer.
    pub one_by_one: DenseLayer<CH, CH>,
}

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processes a full WaveNet layer, iterating `FastMath` in AVX2.
    ///
    /// Conv1D uses full-precision f32 weights; DenseLayer uses standard quantized paths.
    ///
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    #[inline]
    pub unsafe fn process_block_internal<M: SimdMath>(&self, ctx: WavenetProcessContext<'_>) {
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
            const {
                assert!(
                    CH * WAVENET_MAX_NUM_FRAMES <= 1024,
                    "topology CH exceeds stack buffer (1024)"
                );
            }
            debug_assert!(
                num_frames * CH <= 1024,
                "process_block_internal: num_frames*CH ({}) exceeds stack buffer (1024)",
                num_frames * CH,
            );

            // SAFETY: mixin_out is fully written by input_mixin.process_block before
            // any read. MaybeUninit avoids the 4 KB memset the compiler emits for
            // `[0.0f32; 1024]`.
            let mut mixin_out = MaybeUninit::<[f32; 1024]>::uninit();
            let mixin_out_ptr = mixin_out.as_mut_ptr() as *mut f32;
            let mixin_out_slice = core::slice::from_raw_parts_mut(mixin_out_ptr, num_frames * CH);
            self.input_mixin
                .process_block::<M>(condition, mixin_out_slice, num_frames);

            // SAFETY: conv_plus_mixin is fully written by the Conv1D path before
            // any read (tanh activation). MaybeUninit avoids a second 4 KB memset.
            let mut conv_plus_mixin = MaybeUninit::<[f32; 1024]>::uninit();
            let conv_plus_mixin_ptr = conv_plus_mixin.as_mut_ptr() as *mut f32;
            let conv_slice = core::slice::from_raw_parts_mut(conv_plus_mixin_ptr, num_frames * CH);

            // Dual-Frame Tiling with f32-native Conv1D path
            let mut i = 0;
            let mut chunks = conv_slice.chunks_exact_mut(2 * CH);
            for chunk in chunks.by_ref() {
                let (out_frame_f0, out_frame_f1) = chunk.split_at_mut(CH);

                let mix_idx_f0 = i * CH;
                let mix_idx_f1 = (i + 1) * CH;
                // SAFETY: mixin_out was fully written by input_mixin.process_block above.
                let mixin_f0 = core::slice::from_raw_parts(mixin_out_ptr.add(mix_idx_f0), CH);
                let mixin_f1 = core::slice::from_raw_parts(mixin_out_ptr.add(mix_idx_f1), CH);

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
                // SAFETY: mixin_out was fully written by input_mixin.process_block above.
                let mixin_slice = core::slice::from_raw_parts(mixin_out_ptr.add(mix_idx), CH);

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
