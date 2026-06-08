// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic WaveNet layer (runtime dimensions).

use super::common::WavenetProcessContext;
use super::conv1d_dyn::Conv1dDyn;
use super::dense_dyn::DenseLayerDyn;
use crate::math::common::SimdMath;

/// WaveNet Layer with dynamic dimensions.
#[derive(Clone)]
pub struct WaveNetLayerDyn {
    /// Causal convolution core.
    pub conv1d: Conv1dDyn,
    /// Local input mixer (residuum).
    pub input_mixin: DenseLayerDyn,
    /// 1x1 transformer associated with the final output.
    pub one_by_one: DenseLayerDyn,
    /// Number of base channels.
    pub ch: usize,
    /// Enables the Gated Activation mechanism.
    pub gated: bool,
}

impl WaveNetLayerDyn {
    /// Executes the internal processing of a WaveNet layer with Dual-Frame Tiling.
    /// # Safety
    /// `ctx.block` must be large enough to hold `num_frames * out_ch` samples.
    /// Internal Orchestrator of the WaveNet Layer.
    /// This function is the 'maestro' that coordinates all the mathematical steps
    /// needed to process a single layer of the neural network.
    pub unsafe fn process_block_internal<M: SimdMath>(&self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            condition_bf16,
            head_input,
            output,
            layer_buffer,
            layer_buffer_bf16,
            buffer_start,
            block,
            num_frames,
            mut output_bf16,
            is_first_layer,
        } = ctx;
        let ch = self.ch;
        let out_ch = self.conv1d.out_ch;

        // --- On-Stack Temporary Buffer ---
        // We use a buffer aligned directly in execution memory (stack).
        // This avoids slow allocations and ensures that processing is
        // deterministic and ultra-fast for real-time audio.
        const MAX_STACK: usize = 8192;

        #[repr(align(64))]
        struct AlignedMixinBuffer([f32; MAX_STACK]);
        let mut mixin_out = AlignedMixinBuffer([0.0f32; MAX_STACK]);

        let mixin_len = num_frames * ch;
        assert!(
            mixin_len <= MAX_STACK,
            "mixin_len overflow: {} (max {})",
            mixin_len,
            MAX_STACK,
        );
        let mixin_out_slice = &mut mixin_out.0[..mixin_len];

        unsafe {
            // Decide between the BF16 (faster) or F32 (standard) path
            if M::IS_BF16 {
                // 1. Mixin (Preparation):
                // Process external conditions (like gain/tone) in batch.
                self.input_mixin.process_block_bf16::<M>(
                    condition_bf16,
                    mixin_out_slice,
                    num_frames,
                );

                // 2. Conv1D (The Core):
                // Apply the dilated convolution that 'listens' to the past.
                let mut i = 0;
                let active_block = &mut block[..num_frames * out_ch];
                let mut chunks = active_block.chunks_exact_mut(2 * out_ch);
                for chunk in chunks.by_ref() {
                    let (out_f0, out_f1) = chunk.split_at_mut(out_ch);
                    let mix_idx = i * ch;
                    let m_f0 = &mixin_out_slice[mix_idx..mix_idx + ch];
                    let m_f1 = &mixin_out_slice[mix_idx + ch..mix_idx + 2 * ch];

                    self.conv1d.process_dual_frame_bf16::<M>(
                        layer_buffer_bf16,
                        out_f0,
                        out_f1,
                        buffer_start + i,
                        buffer_start + i + 1,
                        Some(m_f0),
                        Some(m_f1),
                    );
                    i += 2;
                }
                let rem = chunks.into_remainder();
                if !rem.is_empty() {
                    let mix_idx = i * ch;
                    let m = &mixin_out_slice[mix_idx..mix_idx + ch];
                    self.conv1d.process_single_frame_bf16::<M>(
                        layer_buffer_bf16,
                        rem,
                        buffer_start + i,
                        Some(m),
                    );
                }
            } else {
                // Standard F32 path (Identical to the above, but with full precision).
                // 1. Mixin
                self.input_mixin
                    .process_block::<M>(condition, mixin_out_slice, num_frames);

                // 2. Conv1D
                let mut i = 0;
                let active_block = &mut block[..num_frames * out_ch];
                let mut chunks = active_block.chunks_exact_mut(2 * out_ch);
                for chunk in chunks.by_ref() {
                    let (out_f0, out_f1) = chunk.split_at_mut(out_ch);
                    let mix_idx = i * ch;
                    let m_f0 = &mixin_out_slice[mix_idx..mix_idx + ch];
                    let m_f1 = &mixin_out_slice[mix_idx + ch..mix_idx + 2 * ch];

                    self.conv1d.process_dual_frame::<M>(
                        layer_buffer,
                        out_f0,
                        out_f1,
                        buffer_start + i,
                        buffer_start + i + 1,
                        Some(m_f0),
                        Some(m_f1),
                    );
                    i += 2;
                }
                let rem = chunks.into_remainder();
                if !rem.is_empty() {
                    let mix_idx = i * ch;
                    let m = &mixin_out_slice[mix_idx..mix_idx + ch];
                    self.conv1d.process_single_frame::<M>(
                        layer_buffer,
                        rem,
                        buffer_start + i,
                        Some(m),
                    );
                }
            }

            // 3. Activation (Non-linearity):
            // Apply functions like Tanh or Gated to give the 'character' of the sound.
            if self.gated {
                // Gated Activation: Works like a gate that selectively opens and closes.
                if is_first_layer {
                    M::gated_activation_and_overwrite_block(
                        head_input,
                        &mut block[..num_frames * 2 * ch],
                        ch,
                    );
                } else {
                    M::gated_activation_and_accumulate_block(
                        head_input,
                        &mut block[..num_frames * 2 * ch],
                        ch,
                    );
                }

                // Realign data so the next step (GEMM)
                // is processed as a single contiguous memory block.
                for i in 1..num_frames {
                    block.copy_within(i * 2 * ch..i * 2 * ch + ch, i * ch);
                }
            } else {
                // Tanh: Classic activation that 'flattens' the signal to maintain stability.
                if is_first_layer {
                    M::tanh_and_overwrite_block(head_input, &mut block[..num_frames * ch]);
                } else {
                    M::tanh_and_accumulate_block(head_input, &mut block[..num_frames * ch]);
                }
            }

            // 4. Residual + 1x1 (The Final Mix):
            // Sum the original sound (residual) with what we just processed.
            // This allows the network to learn complex transformations without losing the foundation.
            let lb_offset = buffer_start * ch;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * ch);

            self.one_by_one.process_residual_batch::<M>(
                &block[..num_frames * ch],
                residual_slice,
                output,
                num_frames,
            );

            // 5. Final BF16 Conversion:
            // If we're using fast mode, clean the data for the next layer.
            if let (true, Some(bf16_out)) = (M::IS_BF16, output_bf16.as_mut()) {
                M::f32_to_bf16(output, bf16_out);
            }
        }
    }
}
