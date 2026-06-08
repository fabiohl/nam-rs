// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic WaveNet model (fallback for topologies not covered by Const Generics).

use super::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState, WavenetProcessContext};
use super::dense_dyn::DenseLayerDyn;
use super::layer_dyn::WaveNetLayerDyn;
use crate::math::common::AlignedVec;
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

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
    /// Active number of layers for soft-degrade. Set to `layers.len()` by default.
    pub effective_layers: usize,
}

impl WaveNetLayerArrayDyn {
    /// Sets the effective number of layers for soft-degrade.
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        self.effective_layers = n.min(self.layers.len()).max(1);
    }

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
    /// Logical duplication reduction of ~70%.
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

            let num_layers = self.effective_layers;
            let last_layer = num_layers - 1;
            let block_size = self.block_size;

            // 4) LAYER CASCADING
            for i in 0..num_layers {
                let layer = &self.layers[i];
                let current_state = &mut *states_ptr.add(i);

                // Software Prefetch of the next state in the cascade (L1).
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
    /// Sets the effective number of layers on both arrays for soft-degrade.
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        self.array1.set_effective_layers(n);
        self.array2.set_effective_layers(n);
    }

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
