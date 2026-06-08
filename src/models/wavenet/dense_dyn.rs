// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic dense layer (runtime dimensions).

use crate::math::common::{AlignedVec, SimdMath};

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
}
