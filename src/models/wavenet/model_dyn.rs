// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::WAVENET_MAX_NUM_FRAMES;
use super::layer_array_dyn::WaveNetLayerArrayDyn;
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::{NamModel, StaticModel};

/// Complete WaveNet Model with runtime dimensions.
///
/// Mirrors the const-generic `WaveNetModel<CH, K, HEAD>` but accepts dimensions
/// at runtime, enabling arbitrary geometry loading without compile-time
/// specialization.
///
/// ## C++ Parity
///
/// Head scale is the **last** f32 read from the weights file
/// (`model.cpp:623-644`). The f32-native head_rechannel path is preserved
/// (`layer_array.rs:220`) for mixed-precision head projection fidelity.
///
/// ## Array Topology
///
/// - Array1: IN=1, COND=condition_size, CH channels, HEAD outputs
/// - Array2: IN=CH, COND=condition_size, HEAD channels, 1 output (with HeadBias)
///
/// ## condition_dsp (C++ `_process_condition`)
///
/// When `condition_dsp` is `Some`, the raw audio input is first processed by
/// the nested DSP sub-model before reaching the main layer arrays. The sub-model's
/// output channels replace the raw input as the `condition` parameter for both
/// arrays (layer_inputs remain the raw audio). This mirrors C++ `model.cpp:692-722`.
/// The sub-model is built eagerly during model construction and its `set_weights`
/// and `prewarm` are consumed independently from the main weight stream.
pub struct WaveNetModelDyn {
    /// Internal channel count (e.g., 16 for Standard).
    pub ch: usize,
    /// Kernel size (always 3 for WaveNet).
    pub k: usize,
    /// Head projection size (e.g., 8 for Standard).
    pub head: usize,
    /// Inner array 01: IN=1, COND=condition_size, CH channels, HEAD outputs, no HeadBias.
    pub array1: WaveNetLayerArrayDyn,
    /// Inner array 02: IN=CH, COND=condition_size, HEAD channels, 1 output, with HeadBias.
    pub array2: WaveNetLayerArrayDyn,
    /// Final voltage compensation scale (Target Output Scale).
    pub head_scale: f32,
    /// Largest circular buffer required at the Kernel's temporal root.
    pub receptive_field_size: usize,
    /// Optional nested condition DSP sub-model (C++ `_condition_dsp`).
    ///
    /// Built eagerly during model construction from the `condition_dsp` JSON
    /// object. Its `process()` is called with mono audio input; its multi-channel
    /// output replaces the raw input as the `condition` parameter passed to the
    /// layer arrays. When `None`, the raw input is used as both `layer_inputs`
    /// and `condition` (passthrough, cond≤1).
    pub condition_dsp: Option<Box<StaticModel>>,
    /// Pre-allocated output buffer for condition_dsp processing.
    ///
    /// Size: `cond × WAVENET_MAX_NUM_FRAMES`, where `cond` is the main model's
    /// `condition_size`. This matches the sub-model's `NumOutputChannels()`.
    pub condition_dsp_output: AlignedVec<f32>,
}

impl WaveNetModelDyn {
    /// Sets the effective number of layers on both arrays for soft-degrade.
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        self.array1.set_effective_layers(n);
        self.array2.set_effective_layers(n);
    }

    /// Resolves the full forward pass and produces waveform samples in zero allocation (DSP).
    ///
    /// Combines the outputs of both arrays: sum(head1) + sum(head2) × `head_scale`.
    ///
    /// **For Scientists and Devs:** This is where the performance "magic trick" happens (SIMD Dispatch).
    /// Instead of using slow `if/else` per frame to check the CPU (AVX2 vs AVX-512),
    /// the `dispatch_simd!` macro evaluates the hardware once and "teleports" execution
    /// to a cloned (monomorphized) version of this function strictly optimized for your processor.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { crate::math::common::dispatch_simd!(self, process_internal, input, output) };
    }

    /// Fast, generic routine that implements the neural network (WaveNet).
    /// The `<M: SimdMath>` constraint forces the compiler to generate assembly focused on
    /// large registers (256-bit or 512-bit) without branches (branchless).
    ///
    /// When `condition_dsp` is present, the raw input is first processed by the sub-DSP
    /// to produce multi-channel conditioning. The sub-model's output is used as the
    /// `condition` parameter for both layer arrays, while the raw input remains as
    /// `layer_inputs`. This mirrors C++ `model.cpp:737-825`.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 {
            return;
        }

        let ch = self.ch;
        let head = self.head;
        let cond = self.array1.cond;
        let mut pos = 0;

        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            let condition_slice: &[f32] = if let Some(ref mut cond_dsp) = self.condition_dsp {
                let cond_out = &mut self.condition_dsp_output[0..num_frames * cond];
                cond_dsp.process(in_slice, cond_out);
                cond_out
            } else {
                in_slice
            };

            unsafe {
                self.array1.process_block_internal::<M, false>(
                    in_slice,
                    condition_slice,
                    num_frames,
                    None,
                );

                let array1_head_out = &self.array1.head_outputs[0..num_frames * head];
                let array1_outputs = &self.array1.array_outputs[0..num_frames * ch];
                self.array2.process_block_internal::<M, false>(
                    array1_outputs,
                    condition_slice,
                    num_frames,
                    Some(array1_head_out),
                );
            }

            let head_dim = self.array2.head;
            let array2_head = &self.array2.head_outputs[0..num_frames * head_dim];
            let out_start = pos * head_dim;
            let out_slice = &mut output[out_start..out_start + num_frames * head_dim];
            out_slice.copy_from_slice(array2_head);
            unsafe {
                M::apply_gain(out_slice, self.head_scale);
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

    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    #[inline(always)]
    #[cold]
    unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        if let Some(ref mut cond_dsp) = self.condition_dsp {
            cond_dsp.prewarm(0);
        }

        let zero_input = [0.0f32];
        let cond = self.array1.cond;

        let condition: &[f32] = if let Some(ref mut cond_dsp) = self.condition_dsp {
            cond_dsp.process(&zero_input, &mut self.condition_dsp_output[0..cond]);
            &self.condition_dsp_output[0..cond]
        } else {
            &zero_input
        };

        unsafe {
            self.array1
                .prewarm_internal::<M>(&zero_input, condition, None);
        }
        let ch = self.ch;
        let head = self.head;
        let array1_outputs = &self.array1.array_outputs[0..ch];
        let array1_head_out = &self.array1.head_outputs[0..head];
        unsafe {
            self.array2
                .prewarm_internal::<M>(array1_outputs, condition, Some(array1_head_out));
        }
    }
}
