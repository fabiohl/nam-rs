// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    #[inline(always)]
    #[cfg(not(feature = "high-fidelity"))]
    pub unsafe fn process_block_internal<M: SimdMath>(&self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            condition_bf16,
            head_input,
            output,
            mut output_bf16,
            layer_buffer,
            layer_buffer_bf16,
            buffer_start,
            num_frames,
            ..
        } = ctx;

        unsafe {
            // [STEP 2: Conditioning (Input Mixin)]
            // Stack buffer of 1024 f32 = WAVENET_MAX_NUM_FRAMES(64) × max_CH(16).
            // Note: generic_const_exprs (nightly) would be needed for `[f32; CH * WAVENET_MAX_NUM_FRAMES]`
            // in stable Rust. The assert below catches any future topology that exceeds this size.
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
            let mut mixin_out = [0.0f32; 1024];
            let mixin_out_slice = &mut mixin_out[..num_frames * CH];
            if M::IS_BF16 {
                self.input_mixin
                    .process_bf16::<M>(condition_bf16, mixin_out_slice);
            } else {
                self.input_mixin
                    .process_block::<M>(condition, mixin_out_slice, num_frames);
            }

            // "Ahead-of-Time Conditioning": Temporal Buffer Fusion Optimization
            // On-stack temporary buffer to store intermediate results (Conv1D + Mixin)
            // before activation. CH=16 * MAX_FRAMES=64 = 1024 elements (4KB).
            let mut conv_plus_mixin = [0.0f32; 1024];
            let conv_slice = &mut conv_plus_mixin[..num_frames * CH];

            // [PHASE 1: Linear - Conv1D + Mixin]
            // Dual-Frame Tiling: Process 2 frames per iteration to amortize
            // Conv1D weight loading into registers.
            let mut i = 0;
            let mut chunks = conv_slice.chunks_exact_mut(2 * CH);
            for chunk in chunks.by_ref() {
                let (out_frame_f0, out_frame_f1) = chunk.split_at_mut(CH);

                let mix_idx_f0 = i * CH;
                let mix_idx_f1 = (i + 1) * CH;
                let mixin_f0 = mixin_out.get_unchecked(mix_idx_f0..mix_idx_f0 + CH);
                let mixin_f1 = mixin_out.get_unchecked(mix_idx_f1..mix_idx_f1 + CH);

                if M::IS_BF16 {
                    self.conv1d.process_dual_frame_bf16_with_mixin::<M>(
                        layer_buffer_bf16,
                        out_frame_f0,
                        out_frame_f1,
                        buffer_start + i,
                        buffer_start + i + 1,
                        mixin_f0,
                        mixin_f1,
                    );
                } else {
                    self.conv1d.process_dual_frame_with_mixin::<M>(
                        layer_buffer,
                        out_frame_f0,
                        out_frame_f1,
                        buffer_start + i,
                        buffer_start + i + 1,
                        mixin_f0,
                        mixin_f1,
                    );
                }
                i += 2;
            }

            // Handle the residual frame (odd)
            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                let mix_idx = i * CH;
                let mixin_slice = mixin_out.get_unchecked(mix_idx..mix_idx + CH);

                if M::IS_BF16 {
                    self.conv1d.process_single_frame_bf16_with_mixin::<M>(
                        layer_buffer_bf16,
                        rem,
                        buffer_start + i,
                        mixin_slice,
                    );
                } else {
                    self.conv1d.process_single_frame_with_mixin::<M>(
                        layer_buffer,
                        rem,
                        buffer_start + i,
                        mixin_slice,
                    );
                }
            }

            // [PHASE 2 & 3: Fused Activation and Head Update]
            // Apply Tanh and accumulate/overwrite in Head in a single memory pass.
            // This reduces bandwidth pressure (avoids 1 extra read and 1 extra write).
            if ctx.is_first_layer {
                M::tanh_and_overwrite_block(head_input, conv_slice);
            } else {
                M::tanh_and_accumulate_block(head_input, conv_slice);
            }

            // [PHASE 3: Output - 1x1 Residual]
            // [TF3] Optimization: 1x1 projection fused with residual sum in batch.
            // Eliminates the prior copy of the original state (layer_buffer) to output.
            let lb_offset = buffer_start * CH;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * CH);

            self.one_by_one.process_residual_batch::<M>(
                conv_slice,
                residual_slice,
                output,
                num_frames,
            );

            // 4. [T25] BF16 Fusion: Batch conversion if needed
            if let (true, Some(bf16_out)) = (M::IS_BF16, output_bf16.as_mut()) {
                M::f32_to_bf16(output, bf16_out);
            }
        }
    }

    /// Processes a full WaveNet layer in high-fidelity mode.
    ///
    /// Uses full-precision f32 weights (no quantization) and exact `f32::tanh`.
    /// Trade-off: higher latency/memory bandwidth for superior numerical fidelity.
    ///
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    #[cfg(feature = "high-fidelity")]
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(&self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            head_input,
            output,
            layer_buffer,
            buffer_start,
            num_frames,
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
                "process_block_internal hf: num_frames*CH ({}) exceeds stack buffer (1024)",
                num_frames * CH,
            );
            let mut mixin_out = [0.0f32; 1024];
            let mixin_out_slice = &mut mixin_out[..num_frames * CH];
            self.input_mixin
                .process_block_f32_native::<M>(condition, mixin_out_slice, num_frames);

            let mut conv_plus_mixin = [0.0f32; 1024];
            let conv_slice = &mut conv_plus_mixin[..num_frames * CH];

            // Dual-Frame Tiling with f32-native path
            let mut i = 0;
            let mut chunks = conv_slice.chunks_exact_mut(2 * CH);
            for chunk in chunks.by_ref() {
                let (out_frame_f0, out_frame_f1) = chunk.split_at_mut(CH);

                let mix_idx_f0 = i * CH;
                let mix_idx_f1 = (i + 1) * CH;
                let mixin_f0 = mixin_out.get_unchecked(mix_idx_f0..mix_idx_f0 + CH);
                let mixin_f1 = mixin_out.get_unchecked(mix_idx_f1..mix_idx_f1 + CH);

                self.conv1d.process_dual_frame_f32_native_with_mixin(
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
                let mixin_slice = mixin_out.get_unchecked(mix_idx..mix_idx + CH);

                self.conv1d.process_single_frame_f32_native_with_mixin(
                    layer_buffer,
                    rem,
                    buffer_start + i,
                    mixin_slice,
                );
            }

            if ctx.is_first_layer {
                M::tanh_and_overwrite_block(head_input, conv_slice);
            } else {
                M::tanh_and_accumulate_block(head_input, conv_slice);
            }

            let lb_offset = buffer_start * CH;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * CH);

            self.one_by_one.process_residual_batch_f32::<M>(
                conv_slice,
                residual_slice,
                output,
                num_frames,
            );
        }
    }
}
