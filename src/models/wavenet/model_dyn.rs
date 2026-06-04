// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic WaveNet model (fallback for topologies not covered by Const Generics).

use super::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState, WavenetProcessContext};
use super::conv1d_dyn::Conv1dDyn;
use crate::math::common::{AlignedVec, SimdMath};
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

/// 1x1 Dense Layer with dynamic dimensions.
#[derive(Clone)]
pub struct DenseLayerDyn {
    /// Matrix weights [OUT][IN].
    pub weights: AlignedVec<u16>,
    /// Bias [OUT].
    pub bias: AlignedVec<f32>,
    /// Bias application flag.
    pub do_bias: bool,
    /// Input dimension.
    pub in_size: usize,
    /// Output dimension.
    pub out_size: usize,
}

impl DenseLayerDyn {
    /// Processes the layer by fusing with the output accumulator.
    ///
    /// # Safety
    ///
    /// Depends on the validity of input/output pointers and SIMD alignment.
    #[inline(always)]
    /// # Safety
    /// `output` must have size at least `num_frames * self.out_size`.
    pub unsafe fn process_acc_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::fused_add_gemm_batch(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Processes the layer by fusing with the residual sum.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the residual and output buffers have compatible sizes.
    #[inline(always)]
    pub unsafe fn process_residual_batch<M: SimdMath>(
        &self,
        input: &[f32],
        residual: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::fused_gemm_residual_batch(
                input,
                &self.weights,
                &self.bias,
                residual,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Processes the layer by replacing the output.
    ///
    /// # Safety
    /// `output` must have size at least `num_frames * self.out_size`.
    /// Depends on the validity of input and output buffers for num_frames.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::gemv_overwrite_batch(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Processes the layer using BF16.
    ///
    /// # Safety
    /// `output` must have size at least `num_frames * self.out_size`.
    /// Requires `M::IS_BF16` to be true and that the input/output buffers are valid.
    #[inline(always)]
    pub unsafe fn process_block_bf16<M: SimdMath>(
        &self,
        input: &[u16],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::gemv_overwrite_batch_bf16(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Fused projection for a single frame.
    ///
    /// # Safety
    ///
    /// Depends on the validity of single-frame buffers.
    #[inline(always)]
    pub unsafe fn process_fused<M: SimdMath>(&self, in_frame: &[f32], out_frame: &mut [f32]) {
        unsafe {
            M::fused_add_gemv(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }
}

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

                // [T1.2] Optimization: Realign data so the next step (GEMM)
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

/// Represents the entire vertical topology of a dynamic WaveNet branch, supporting multiple sequential dilations.
pub struct WaveNetLayerArrayDyn {
    /// Stacked list of layers with their respective dilations fixed in RAM at loading.
    pub layers: Vec<WaveNetLayerDyn>,
    /// Local mirror register of delay tape. Maintained for lock-free circular passes.
    pub states: Vec<WaveNetLayerState>,
    /// Dense resizer for the input channel.
    pub rechannel: DenseLayerDyn,
    /// Dense resizer for the parametric head sum mesh.
    pub head_rechannel: DenseLayerDyn,
    /// Temporary accumulator of sequential chains.
    pub array_outputs: AlignedVec<f32>,
    /// Accumulator of activations projected by the Head mesh.
    pub head_accum: AlignedVec<f32>,
    /// Ongoing final head projections (sum of multi-layer projections).
    pub head_outputs: AlignedVec<f32>,
    /// Auxiliary state buffer reused to inhibit heap allocation in RT threads.
    pub block_buffer: AlignedVec<f32>,
    /// Effective size of `block_buffer`. Equals `ch` or `2*ch` depending on gated.
    pub block_size: usize,
    /// Global analytical causal latency size of this cascade.
    pub receptive_field_size: usize,
    /// Transverse axis of base Channels (`C`).
    pub ch: usize,
    /// Summed projected reduction.
    pub head: usize,
    /// Cache of the last f32 conditioning.
    pub last_condition: AlignedVec<f32>,
    /// Cache of the last BF16 conditioning.
    pub last_condition_bf16: AlignedVec<u16>,
    /// Cache initialization flag.
    pub condition_init: bool,
}

impl WaveNetLayerArrayDyn {
    /// INFERENCE ORCHESTRATOR (Cascade Process):
    /// Performs synchronous inference of all layers of the Array in cascade.
    ///
    /// # Safety
    /// Depends on the integrity of the loaded matrices and the circular buffer states.
    pub unsafe fn process<M: crate::math::common::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
    ) {
        unsafe {
            self.process_internal_generic::<M>(layer_inputs, condition, num_frames, false);
        }
    }

    /// STATE PREWARM (Pre-warm).
    pub fn prewarm<M: crate::math::common::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
    ) {
        unsafe {
            self.process_internal_generic::<M>(layer_inputs, condition, 1, true);
        }
    }

    /// Generic implementation that unifies normal processing and pre-warm.
    /// [TA5.5] Logical duplication reduction of ~70%.
    #[inline(always)]
    unsafe fn process_internal_generic<M: crate::math::common::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
        prewarm_mode: bool,
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let ch = self.ch;
        let head = self.head;
        let states_ptr = self.states.as_mut_ptr();

        // 1) HEAD ACCUMULATOR RESET
        // (Eliminated: first layer overwrites head_accum directly)

        // 2) Lazy BF16 Conversion
        if M::IS_BF16 {
            let changed =
                prewarm_mode || !self.condition_init || condition != &self.last_condition[..];
            if changed {
                unsafe {
                    M::f32_to_bf16(condition, &mut self.last_condition_bf16);
                }
                self.last_condition.copy_from_slice(condition);
                self.condition_init = true;
            }
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;

            // 3) RECHANNEL (Input -> Residual)
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * ch],
                num_frames,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;
            let block_size = self.block_size;

            // 4) LAYER CASCADING
            for i in 0..num_layers {
                let layer = &self.layers[i];
                let current_state = &mut *states_ptr.add(i);

                // [T2.2] Software Prefetch of the next state in the cascade (L1).
                if i + 1 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 1) as *const i8);
                }
                if i + 2 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 2) as *const i8);
                }

                // [STEP 4.1: Pre-fill Ring Buffer (Backwards)]
                // If in prewarm mode, replicate the current input to the entire past.
                if prewarm_mode {
                    let start_idx = current_state.buffer_start * ch;
                    for offset in 1..=current_state.receptive_field_size {
                        debug_assert!(
                            current_state.buffer_start >= offset,
                            "backfill underflow: bs={}, off={}",
                            current_state.buffer_start,
                            offset
                        );
                        // SAFETY: garantido pelo construtor WaveNetLayerState::new que valida buffer_start >= receptive_field_size
                        let dst_start = current_state.buffer_start - offset;
                        let dst_idx = dst_start * ch;
                        debug_assert!(start_idx + ch <= current_state.layer_buffer.len());
                        debug_assert!(dst_idx + ch <= current_state.layer_buffer.len());
                        current_state
                            .layer_buffer
                            .copy_within(start_idx..start_idx + ch, dst_idx);
                        current_state
                            .layer_buffer_bf16
                            .copy_within(start_idx..start_idx + ch, dst_idx);
                    }
                }

                let ctx = WavenetProcessContext {
                    condition,
                    condition_bf16: &self.last_condition_bf16,
                    head_input: &mut self.head_accum[0..num_frames * ch],
                    output: if i == last_layer {
                        &mut self.array_outputs[0..num_frames * ch]
                    } else {
                        let next_state = &mut *states_ptr.add(i + 1);
                        let next_start = next_state.buffer_start * ch;
                        &mut next_state.layer_buffer[next_start..next_start + num_frames * ch]
                    },
                    output_bf16: None,
                    layer_buffer: &current_state.layer_buffer,
                    layer_buffer_bf16: &current_state.layer_buffer_bf16,
                    buffer_start: current_state.buffer_start,
                    block: &mut self.block_buffer[0..num_frames * block_size],
                    num_frames,
                    is_first_layer: i == 0,
                };

                layer.process_block_internal::<M>(ctx);

                // In prewarm mode we don't advance the circular pointer (static stabilization).
                if !prewarm_mode {
                    current_state.advance_frames(num_frames, ch);
                }
            }

            // 5) HEAD RECHANNEL (Skip Sum -> Output)
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * ch],
                &mut self.head_outputs[0..num_frames * head],
                num_frames,
            );
        }
    }
}

/// Final Dynamic Wrapper. Holds interconnected Arrays.
pub struct WaveNetDynModel {
    /// The Primary branch with the majority of the WaveNet causal field.
    pub array1: WaveNetLayerArrayDyn,
    /// Secondary Branch, mono causal reducer.
    pub array2: WaveNetLayerArrayDyn,
    /// Pre-linearization master volume adjustment.
    pub head_scale: f32,
    /// Total frame load this model assimilates before reliable output.
    pub receptive_field_size: usize,
    /// Dimensions of the head's final internal convergence.
    pub head: usize,
}

impl WaveNetDynModel {
    /// Processes the audio block in the causal matrix.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::common::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Cloned and purely SIMD-optimized `M` variant of full network processing.
    ///
    /// # Safety
    /// Must only be invoked via macro `dispatch_simd!`.
    unsafe fn process_internal<M: crate::math::common::SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) {
        let total_frames = input.len();
        let mut pos = 0;
        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                self.array1.process::<M>(in_slice, in_slice, num_frames);
                let array1_outputs = &self.array1.array_outputs[0..num_frames * self.array1.ch];
                self.array2
                    .process::<M>(array1_outputs, in_slice, num_frames);
            }

            unsafe {
                M::batch_wavenet_head_sum_dyn(
                    &self.array1.head_outputs[0..num_frames * self.head],
                    &self.array2.head_outputs[0..num_frames],
                    &mut output[pos..pos + num_frames],
                    self.head,
                    self.head_scale,
                );
            }
            pos += num_frames;
        }
    }

    /// Performs the initial `Prewarm` to stabilize the buffers.
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// # Safety
    /// Must only be invoked via macro `dispatch_simd!`.
    unsafe fn prewarm_internal<M: crate::math::common::SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];
        self.array1.prewarm::<M>(&layer_inputs_1, &condition);
        let array1_outputs = &self.array1.array_outputs[0..self.array1.ch];
        self.array2.prewarm::<M>(array1_outputs, &condition);
    }

    /// Reallocates `block_buffer` and `head_accum` in both arrays to support
    /// the given maximum buffer size, if larger than current capacity.
    pub fn set_max_buffer_size(&mut self, max_buf: usize) {
        let min_buf = max_buf.max(WAVENET_MAX_NUM_FRAMES);
        let ch1 = self.array1.ch;
        let block_size1 = self.array1.block_size;
        self.array1.head_accum.resize(min_buf * ch1, 0.0f32);
        self.array1
            .block_buffer
            .resize(min_buf * block_size1, 0.0f32);
        let ch2 = self.array2.ch;
        let block_size2 = self.array2.block_size;
        self.array2.head_accum.resize(min_buf * ch2, 0.0f32);
        self.array2
            .block_buffer
            .resize(min_buf * block_size2, 0.0f32);
    }
}
