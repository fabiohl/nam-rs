// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState, WavenetProcessContext};
use super::conv1d::Conv1d;
use super::dense::DenseLayer;
use crate::math::common::{AlignedVec, SimdMath};
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

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

// (Processing context moved to wavenet_common.rs)

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processes a full WaveNet layer, iterating `FastMath` in AVX2.
    ///
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    #[inline(always)]
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
            assert!(
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
}

// (Layer state moved to wavenet_common.rs)

/// Grouped Multi-Layer WaveNet Unit.
pub struct WaveNetLayerArray<
    const IN: usize,
    const COND: usize,
    const CH: usize,
    const K: usize,
    const HEAD: usize,
> {
    /// Vec with structural topology (length defines blocks).
    pub layers: Vec<WaveNetLayer<COND, CH, K>>,
    /// RingBuffer states, one for each Layer in the system.
    pub states: Vec<WaveNetLayerState>,
    /// Initial `Dense` tensor opening.
    pub rechannel: DenseLayer<IN, CH>,
    /// Final tensor closure generating Head projection.
    pub head_rechannel: DenseLayer<CH, HEAD>,

    /// Pre-allocated temporary output array.
    /// Array output temporary accumulator.
    pub array_outputs: AlignedVec<f32>,
    /// CH-sized intermediate accumulator for layer contributions before the Head projection.
    pub head_accum: AlignedVec<f32>,
    /// Allocated global Linear projection memory (HEAD-sized).
    pub head_outputs: AlignedVec<f32>,
    /// Dimensional field size (global receptive field) for routing.
    pub receptive_field_size: usize,
    /// Shared activation buffer size.
    pub block_size: usize,
    /// Temporary accumulator for blocks (pre-allocated).
    pub block_buffer: AlignedVec<f32>,

    /// Conditioning buffer cached in BF16.
    pub last_condition_bf16: [u16; COND],
    /// Copy of the last f32 conditioning for comparison.
    pub last_condition: [f32; COND],
    /// Flag for first cache initialization.
    pub condition_init: bool,
}

impl<const IN: usize, const COND: usize, const CH: usize, const K: usize, const HEAD: usize>
    WaveNetLayerArray<IN, COND, CH, K, HEAD>
{
    /// Array's central processing. Fully shielded against allocations.
    ///
    /// # Safety
    /// State pointers iterate internally without bounds checks.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath, const PREWARM: bool>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let states_ptr = self.states.as_mut_ptr();

        // [STEP 1: Zero-Accumulator]
        // Zeros the "Skip Connections" output accumulator (Head) for this frame block.
        // This is essential because each layer of the array will add its contribution here.
        // (Eliminated: first layer overwrites head_accum directly)

        // [STEP 2: Lazy BF16 Conversion]
        if M::IS_BF16 {
            let changed = PREWARM || !self.condition_init || condition != &self.last_condition[..];

            if changed {
                unsafe {
                    M::f32_to_bf16(condition, &mut self.last_condition_bf16);
                }
                self.last_condition.copy_from_slice(condition);
                self.condition_init = true;
            }
        }

        unsafe {
            // [STEP 3: Dimensional Opening (Rechannel)]
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * CH],
                num_frames,
            );

            if M::IS_BF16 {
                M::f32_to_bf16(
                    &state_0.layer_buffer[start..start + num_frames * CH],
                    &mut state_0.layer_buffer_bf16[start..start + num_frames * CH],
                );
            }

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            // [STEP 4: Layer Inference Cascade]
            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if PREWARM {
                    // [STEP 4.1: Static State Propagation (Backfill)]
                    // In prewarm mode, we replicate the current sample to the entire history (Receptive Field).
                    // This ensures the network stabilizes instantly to the stationary value.
                    let start_idx = current_state.buffer_start * CH;
                    let src_range = start_idx..start_idx + CH;

                    for offset in 1..=current_state.receptive_field_size {
                        debug_assert!(
                            current_state.buffer_start >= offset,
                            "backfill underflow: bs={}, off={}",
                            current_state.buffer_start,
                            offset
                        );
                        // SAFETY: garantido pelo construtor WaveNetLayerState::new que valida buffer_start >= receptive_field_size
                        let dst_start = current_state.buffer_start - offset;
                        let dst_idx = dst_start * CH;
                        current_state
                            .layer_buffer
                            .copy_within(src_range.clone(), dst_idx);

                        if M::IS_BF16 {
                            current_state
                                .layer_buffer_bf16
                                .copy_within(src_range.clone(), dst_idx);
                        }
                    }
                }

                // Software Prefetch the next state in the cascade.
                // Bring the cache line of state i+1 (and i+2 if possible) into L1
                // while the processor resolves the arithmetic pipeline of the current layer.
                if i + 1 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 1) as *const i8);
                }
                if i + 2 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 2) as *const i8);
                }

                if i == last_layer {
                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        condition_bf16: &self.last_condition_bf16,
                        head_input: &mut self.head_accum[0..num_frames * CH],
                        output: &mut self.array_outputs[0..num_frames * CH],
                        output_bf16: None,
                        layer_buffer: &current_state.layer_buffer[..],
                        layer_buffer_bf16: &current_state.layer_buffer_bf16[..],
                        buffer_start: current_state.buffer_start,
                        num_frames,
                        block: &mut self.block_buffer[0..num_frames * self.block_size],
                        is_first_layer: i == 0,
                    });
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let n_start = next_state.buffer_start * CH;
                    let next_layer_buffer =
                        &mut next_state.layer_buffer[n_start..n_start + num_frames * CH];
                    let next_layer_buffer_bf16 = if M::IS_BF16 {
                        Some(&mut next_state.layer_buffer_bf16[n_start..n_start + num_frames * CH])
                    } else {
                        None
                    };

                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        condition_bf16: &self.last_condition_bf16,
                        head_input: &mut self.head_accum[0..num_frames * CH],
                        output: next_layer_buffer,
                        output_bf16: next_layer_buffer_bf16,
                        layer_buffer: &current_state.layer_buffer[..],
                        layer_buffer_bf16: &current_state.layer_buffer_bf16[..],
                        buffer_start: current_state.buffer_start,
                        num_frames,
                        block: &mut self.block_buffer[0..num_frames * self.block_size],
                        is_first_layer: i == 0,
                    });
                }

                if !PREWARM {
                    current_state.advance_frames(num_frames, CH);
                }
            }

            // [STEP 5: Dimensional Closure (Head Rechannel)]
            // The dense matrix funnels the accumulator (sum of skip-connections from all layers,
            // of size `CH`) into a smaller `HEAD` dimension (e.g., 16 -> 8 or 16 -> 1).
            // Mixed-Precision Selective: when `f32_weights` is available, use full FP32
            // precision for this critical final projection (tonal fidelity),
            // while the backbone runs quantized (BF16/F16) for performance.
            if self.head_rechannel.f32_weights.is_some() {
                self.head_rechannel.process_block_f32_native::<M>(
                    &self.head_accum[0..num_frames * CH],
                    &mut self.head_outputs[0..num_frames * HEAD],
                    num_frames,
                );
            } else {
                self.head_rechannel.process_block::<M>(
                    &self.head_accum[0..num_frames * CH],
                    &mut self.head_outputs[0..num_frames * HEAD],
                    num_frames,
                );
            }
        }
    }

    /// Processes data in Pre-warm mode to initialize and stabilize temporal memory.
    ///
    /// [SCIENTIFIC EXPLANATION]
    /// Causal convolution neural networks like WaveNet have an internal state that actively
    /// depends on N past steps (Receptive Field). When loading a fresh model, the allocated
    /// network memory (Ring Buffers) contains "pure zeros" or computational garbage.
    /// Pre-warm feeds a continuous inert signal (Absolute Silence) into the network so as to
    /// fill the entire past window. The resulting cold-start transients "drain"
    /// silently into oblivion, ensuring the first audio sample when turning on the device
    /// sounds organic and stable, without pops or clicks.
    #[inline(always)]
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    pub unsafe fn prewarm_internal<M: SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
    ) {
        unsafe {
            // Unified via shared code: we process 1 frame with the prewarm flag active.
            // [STEP 4.1] inside `process_block_internal` handles the backfill.
            self.process_block_internal::<M, true>(layer_inputs, condition, 1);
        }
    }
}

/// Complete WaveNet Model containing Two Heterogeneous Layer Array Blocks.
///
/// **Scientific Reference:** van den Oord, A., et al. (2016). *"WaveNet: A Generative Model for Raw Audio."* DeepMind.
///
/// `CH` = Array1 channels (layer 0 of the JSON, e.g., 16 for Standard)
/// `K`  = kernel size (always 3)
/// `HEAD` = Array1 head_size = Array2 channels (e.g., 8 for Standard)
///
/// Array2 uses `HEAD` channels and projects to 1 output (`HEAD2=1`),
/// following the C++ pattern: `WaveNetLayerArrayT<CH, 1, 1, HEAD, K, Dilations, true>`.
pub struct WaveNetModel<const CH: usize, const K: usize, const HEAD: usize> {
    /// Inner array 01: IN=1, COND=1, CH channels, HEAD outputs, no HeadBias.
    pub array1: WaveNetLayerArray<1, 1, CH, K, HEAD>,
    /// Inner array 02: IN=CH, COND=1, HEAD channels, 1 output, with HeadBias.
    pub array2: WaveNetLayerArray<CH, 1, HEAD, K, 1>,
    /// Final voltage compensation scale (Target Output Scale).
    pub head_scale: f32,
    /// Largest circular buffer required at the Kernel's temporal root.
    pub receptive_field_size: usize,
}

impl<const CH: usize, const K: usize, const HEAD: usize> WaveNetModel<CH, K, HEAD> {
    /// Resolves the full forward pass and produces waveform samples in zero allocation (DSP).
    ///
    /// Combines the outputs of both arrays: `sum(head1) + sum(head2)` × `head_scale`.
    ///
    /// **For Scientists and Devs:** This is where the performance "magic trick" happens (SIMD Dispatch).
    /// Instead of using slow `if/else` per frame to check the CPU (AVX2 vs AVX-512),
    /// the `dispatch_simd!` macro evaluates the hardware once and "teleports" execution
    /// to a cloned (monomorphized) version of this function strictly optimized for your processor.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { crate::math::common::dispatch_simd!(self, process_internal, input, output) };
    }

    #[inline(always)]
    /// Fast, generic routine that implements the neural network (WaveNet).
    /// The `<M: SimdMath>` constraint forces the compiler to generate assembly focused on
    /// large registers (256-bit or 512-bit) without branches (branchless).
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 {
            return;
        }

        let mut pos = 0;
        // [PROCESSING IN CHUNKS (BLOCKS)]
        // To maintain zero-allocation invariants (no temporary RAM vector allocations)
        // and respect the restricted L1/L2 Cache hierarchy, we limit processing
        // to `WAVENET_MAX_NUM_FRAMES` (typically 64 samples) at a time.
        // This loop iterates until it consumes the entire buffer (e.g., 256, 512, 1024 frames).
        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                // [STEP 1: Array1 Forward]
                // Conditioning and Input (1D: 1 channel) -> formatted as blocks of IN frames.
                // In the standard NAM topology, this Array performs convolutions using huge dilations
                // (e.g., from 1 to 512, 1 to 512 successively) to capture amplifier sub-bass.
                // Its output enters `array1.array_outputs` and the skips enter `array1.head_outputs`.
                self.array1
                    .process_block_internal::<M, false>(in_slice, in_slice, num_frames);

                // [STEP 2: Array2 Forward]
                // The second array typically acts as a closure perceptron layer
                // (smaller dimensions, dilations of only 1, processing the "mix" coming from Array1).
                let array1_outputs = &self.array1.array_outputs[0..num_frames * CH];
                self.array2.process_block_internal::<M, false>(
                    array1_outputs,
                    in_slice,
                    num_frames,
                );
            }

            // [STEP 3: Skip Sum + SIMD Final Scale]
            // SIMD summation of the Head projections of both arrays and scaling by `head_scale`.
            unsafe {
                M::batch_wavenet_head_sum::<HEAD>(
                    &self.array1.head_outputs[0..num_frames * HEAD],
                    &self.array2.head_outputs[0..num_frames],
                    &mut output[pos..pos + num_frames],
                    self.head_scale,
                );
            }
            pos += num_frames;
        }
    }

    /// Stabilizes the model by processing silence (Zero Input) for pre-warm.
    ///
    /// AVX-512 vs AVX2 dispatch is done via `SimdMathConfig::get().is_avx512` —
    /// Relaxed atomic read of a `LazyLock` initialized at startup, without calling
    /// `is_x86_feature_detected!` per invocation (cold-path, but consistent with
    /// the dispatch pattern of the rest of the codebase).
    #[cold]
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Prewarm strictly optimized for AVX-512 architecture.
    ///
    /// # Safety
    /// Requires a supported processor (AVX-512).
    #[target_feature(enable = "avx512f,avx512vl")]
    #[cold]
    pub unsafe fn prewarm_avx512(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::common::Avx512Math>() };
    }

    /// Prewarm strictly optimized for AVX2 architecture.
    ///
    /// # Safety
    /// Requires an x86-64-v3 (AVX2) processor.
    #[cold]
    pub unsafe fn prewarm_avx2(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::common::Avx2Math>() };
    }

    #[inline(always)]
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    #[cold]
    unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        unsafe {
            self.array1
                .prewarm_internal::<M>(&layer_inputs_1, &condition);
        }
        let array1_outputs = &self.array1.array_outputs[0..CH];
        unsafe {
            self.array2
                .prewarm_internal::<M>(array1_outputs, &condition);
        }
    }
}
