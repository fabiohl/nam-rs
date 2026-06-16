// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::{AlignedVec, SimdMath};

/// 1x1 Dense Layer (The Channel Mixer) with runtime dimensions.
///
/// Think of this layer as a 'digital mixing console'. It blends the various
/// audio channels coming from the previous stage to create the final timbre combination.
#[derive(Clone)]
pub struct DenseLayerDyn {
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of output channels.
    pub out_ch: usize,
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

impl DenseLayerDyn {
    /// Residual Sum (The Final 'Shortcut'):
    /// This function mixes channels AND adds the original sound
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

    /// Processes iterative block by replacing (OVERWRITE) the given values instead of accumulating.
    ///
    /// # Safety
    /// The caller must guarantee that `input` and `output` have sizes compatible with `in_ch`, `out_ch`, and `num_frames`.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        debug_assert_eq!(input.len(), self.in_ch * num_frames);
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

    /// Processes a BF16-conditioned block.
    ///
    /// # Safety
    /// The caller must guarantee that `input` and `output` have sizes
    /// compatible with the layer's `in_ch` and `out_ch` dimensions, and that the
    /// SIMD instructions requested by the dispatcher `M` are available.
    #[inline(always)]
    pub unsafe fn process_bf16<M: SimdMath>(&self, input: &[u16], output: &mut [f32]) {
        let num_frames = output.len() / self.out_ch;
        debug_assert_eq!(input.len(), self.in_ch * num_frames);
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
    /// The caller must ensure that `input` and `output` have sizes
    /// compatible with `in_ch`, `out_ch`, and `num_frames`, and that the SIMD
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
            if self.do_bias {
                M::gemv_with_bias_f32(input, f32_w, &self.bias, output, num_frames);
            } else {
                M::gemv_no_bias_f32(input, f32_w, output, num_frames);
            }
        }
    }
}
