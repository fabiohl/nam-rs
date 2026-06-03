// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::{AlignedVec, SimdMath};

/// 1x1 Dense Layer (The Channel Mixer):
/// Think of this layer as a 'digital mixing console'. It blends the various
/// audio channels coming from the previous stage to create the final timbre combination.
#[derive(Clone)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    /// Weight matrix: Defines 'how much' of each channel goes into the mix.
    pub weights: AlignedVec<u16>,
    /// Bias: A basic 'volume' adjustment for each output channel.
    pub bias: AlignedVec<f32>,
    /// Flag indicating whether bias should be applied.
    pub do_bias: bool,
    /// Optional full-precision f32 weights for mixed-precision head projection.
    /// When present, `process_block_f32_native` can be used instead of the
    /// quantized SIMD path, preserving tonal fidelity in the critical final stage.
    pub f32_weights: Option<AlignedVec<f32>>,
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    /// Fused Processing:
    /// Multiplies, applies bias, and sums to the result, all in a single mathematical step.
    /// This is the most efficient way to process a single audio frame.
    ///
    /// # Safety
    /// The caller must guarantee that `in_frame` and `out_frame` have sizes compatible with `IN` and `OUT`.
    #[inline(always)]
    pub unsafe fn process_fused<M: SimdMath>(&self, in_frame: &[f32], out_frame: &mut [f32]) {
        unsafe {
            M::fused_add_gemv(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// Alias for `process_fused`.
    ///
    /// # Safety
    /// Depends on buffer validity and the `SimdMath` trait.
    #[inline(always)]
    pub unsafe fn process_acc_single_frame<M: SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        unsafe { self.process_fused::<M>(in_frame, out_frame) }
    }

    /// 'Clean' Processing (Overwrite):
    /// Similar to fused, but replaces what's in the output buffer
    /// instead of adding to the existing value.
    ///
    /// # Safety
    /// The caller must guarantee that `in_frame` and `out_frame` have sizes compatible with `IN` and `OUT`.
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        unsafe {
            M::gemv_overwrite(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// Fused Block Processing:
    /// Optimized version to process multiple samples (batches) at once,
    /// gaining significant speed on modern processors.
    ///
    /// # Safety
    /// The caller must guarantee that `input` and `output` have sizes compatible with `IN`, `OUT`, and `num_frames`.
    #[inline(always)]
    pub unsafe fn process_fused_block<M: SimdMath>(
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

    /// Alias for `process_fused_block`.
    ///
    /// # Safety
    /// Depends on buffer validity and the `SimdMath` trait.
    #[inline(always)]
    pub unsafe fn process_acc_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe { self.process_fused_block::<M>(input, output, num_frames) }
    }

    /// Residual Sum (The Final 'Shortcut'):
    /// This function does something amazing: it mixes channels AND adds the original sound
    /// (residual) to the result, all without needing to copy extra data in memory.
    ///
    /// # Safety
    /// The caller must guarantee compatible sizes and buffer validity.
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

    #[inline(always)]
    /// Processes iterative block by replacing (OVERWRITE) the given values instead of accumulating.
    ///
    /// # Safety
    /// The caller must guarantee that `input` and `output` have sizes compatible with `IN`, `OUT`, and `num_frames`.
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

    ///
    /// # Safety
    /// The caller must guarantee that `input` and `output` have sizes
    /// compatible with the layer's `IN` and `OUT` dimensions, and that the
    /// SIMD instructions requested by the dispatcher `M` are available.
    pub unsafe fn process_bf16<M: SimdMath>(&self, input: &[u16], output: &mut [f32]) {
        let num_frames = output.len() / OUT;
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

    /// Full-precision f32 head projection for mixed-precision inference.
    ///
    /// Dispatches to the appropriate SIMD kernel via the `SimdMath` trait,
    /// replacing the previous scalar triple-nested loop with shape-dependent
    /// vectorization (frame-batching for OUT≤4, channel-batching for OUT≥8).
    ///
    /// # Safety
    /// The caller must ensure that `in_frame` and `out_frame` have sizes
    /// compatible with `IN`, `OUT`, and `num_frames`, and that the SIMD
    /// instructions for `M` are available on the host CPU.
    #[inline(always)]
    pub unsafe fn process_block_f32_native<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        let f32_w = self
            .f32_weights
            .as_ref()
            .expect("process_block_f32_native requires f32_weights");
        unsafe {
            M::gemv_overwrite_batch_f32(
                input,
                f32_w,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }
}
