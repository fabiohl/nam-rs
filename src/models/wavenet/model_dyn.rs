// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::WAVENET_MAX_NUM_FRAMES;
use super::layer_array_dyn::WaveNetLayerArrayDyn;
use crate::math::common::SimdMath;

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
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 {
            return;
        }

        let ch = self.ch;
        let head = self.head;
        let mut pos = 0;

        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                self.array1
                    .process_block_internal::<M, false>(in_slice, in_slice, num_frames, None);

                let array1_head_out = &self.array1.head_outputs[0..num_frames * head];
                let array1_outputs = &self.array1.array_outputs[0..num_frames * ch];
                self.array2.process_block_internal::<M, false>(
                    array1_outputs,
                    in_slice,
                    num_frames,
                    Some(array1_head_out),
                );
            }

            let array2_head = &self.array2.head_outputs[0..num_frames];
            let out_slice = &mut output[pos..pos + num_frames];
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
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        unsafe {
            self.array1
                .prewarm_internal::<M>(&layer_inputs_1, &condition, None);
        }
        let ch = self.ch;
        let head = self.head;
        let array1_outputs = &self.array1.array_outputs[0..ch];
        let array1_head_out = &self.array1.head_outputs[0..head];
        unsafe {
            self.array2
                .prewarm_internal::<M>(array1_outputs, &condition, Some(array1_head_out));
        }
    }
}
