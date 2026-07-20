// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::{WavenetProcessContext, effective_stride};
use super::conv1d::Conv1d;
use super::dense::DenseLayer;
use crate::math::common::{AlignedVec, SimdMath};

#[cfg(test)]
pub(crate) use telemetry_vars::*;

#[cfg(test)]
mod telemetry_vars {
    use std::cell::RefCell;
    use std::time::Duration;

    thread_local! {
        pub static ACC_MIXIN: RefCell<Duration> = const { RefCell::new(Duration::ZERO) };
        pub static ACC_CONV: RefCell<Duration> = const { RefCell::new(Duration::ZERO) };
        pub static ACC_TANH: RefCell<Duration> = const { RefCell::new(Duration::ZERO) };
        pub static ACC_ONE_BY_ONE: RefCell<Duration> = const { RefCell::new(Duration::ZERO) };
        pub static TELEMETRY_ACTIVE: RefCell<bool> = const { RefCell::new(false) };
    }
}

/// Complete Convolutional Cell (WaveNet Layer).
#[derive(Clone)]
pub struct WaveNetLayer<const COND: usize, const CH: usize, const K: usize> {
    /// This layer's parametric dilated Causal 1D convolution mesh.
    pub conv1d: Conv1d<CH, CH, K>,
    /// Conditional injection mixing network.
    pub input_mixin: DenseLayer<COND, CH>,
    /// 1x1 decompression linear affine transform of the layer.
    pub one_by_one: DenseLayer<CH, CH>,
    /// Pre-allocated scratch buffer for conditioning mixin output
    /// (size: `effective_stride(CH) * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_mixin: AlignedVec<f32>,
    /// Pre-allocated scratch buffer for Conv1D + mixin intermediate results
    /// (size: `effective_stride(CH) * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_conv: AlignedVec<f32>,
}

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processes a full WaveNet layer, iterating `FastMath` in AVX2.
    ///
    /// Conv1D uses full-precision f32 weights; DenseLayer uses standard quantized paths.
    ///
    /// For `CH=12` (WaveNet Lite), internal scratch buffers use stride 16 (64 B = 1 cache line)
    /// via `effective_stride`. Padding lanes 12-15 are zero-filled and never read.
    ///
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    #[inline]
    pub unsafe fn process_block_internal<M: SimdMath>(&mut self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            head_input,
            output,
            layer_buffer,
            buffer_start,
            num_frames,
            seed,
            is_first_layer,
            ..
        } = ctx;

        let stride = effective_stride::<{ CH }>();

        unsafe {
            debug_assert!(
                num_frames * stride <= self.scratch_mixin.len(),
                "process_block_internal: num_frames*stride ({}) exceeds scratch_mixin capacity ({})",
                num_frames * stride,
                self.scratch_mixin.len(),
            );
            assert!(
                num_frames * stride <= self.scratch_mixin.len(),
                "process_block_internal: num_frames*stride ({}) exceeds scratch_mixin capacity ({})",
                num_frames * stride,
                self.scratch_mixin.len(),
            );
            assert!(
                num_frames * stride <= self.scratch_conv.len(),
                "process_block_internal: num_frames*stride ({}) exceeds scratch_conv capacity ({})",
                num_frames * stride,
                self.scratch_conv.len(),
            );

            #[cfg(test)]
            let t_start = if crate::models::wavenet::layer::TELEMETRY_ACTIVE.with(|a| *a.borrow()) {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let mixin_out = self.scratch_mixin.get_unchecked_mut(..num_frames * stride);

            if CH == 12 {
                crate::math::gemm::gemv::stride16::broadcast_scale_with_bias_f32_avx2_padded(
                    condition,
                    &self.input_mixin.weights,
                    &self.input_mixin.bias,
                    mixin_out,
                    num_frames,
                );
            } else {
                self.input_mixin
                    .process_block::<M>(condition, mixin_out, num_frames);
            }

            #[cfg(test)]
            let t_mixin = if let Some(ts) = t_start {
                let now = std::time::Instant::now();
                crate::models::wavenet::layer::ACC_MIXIN
                    .with(|a| *a.borrow_mut() += now.duration_since(ts));
                Some(now)
            } else {
                None
            };

            let conv_slice = self.scratch_conv.get_unchecked_mut(..num_frames * stride);

            let mut i = 0;
            let mut chunks = conv_slice.chunks_exact_mut(2 * stride);
            for chunk in chunks.by_ref() {
                let (out_frame_f0, out_frame_f1) = chunk.split_at_mut(stride);

                let mix_idx_f0 = i * stride;
                let mix_idx_f1 = (i + 1) * stride;
                let mixin_f0 = &mixin_out[mix_idx_f0..mix_idx_f0 + CH];
                let mixin_f1 = &mixin_out[mix_idx_f1..mix_idx_f1 + CH];

                self.conv1d.process_dual_frame_with_mixin::<M>(
                    layer_buffer,
                    out_frame_f0,
                    out_frame_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    mixin_f0,
                    mixin_f1,
                    stride,
                );
                i += 2;
            }

            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                let mix_idx = i * stride;
                let mixin_slice = &mixin_out[mix_idx..mix_idx + CH];

                self.conv1d.process_single_frame_with_mixin::<M>(
                    layer_buffer,
                    rem,
                    buffer_start + i,
                    mixin_slice,
                    stride,
                );
            }

            #[cfg(test)]
            let t_conv = if let Some(ts) = t_mixin {
                let now = std::time::Instant::now();
                crate::models::wavenet::layer::ACC_CONV
                    .with(|a| *a.borrow_mut() += now.duration_since(ts));
                Some(now)
            } else {
                None
            };

            if let Some(s) = seed {
                M::tanh_and_accumulate_with_seed(head_input, conv_slice, s);
            } else if is_first_layer {
                M::tanh_and_overwrite_block(head_input, conv_slice);
            } else {
                M::tanh_and_accumulate_block(head_input, conv_slice);
            }

            #[cfg(test)]
            let t_tanh = if let Some(ts) = t_conv {
                let now = std::time::Instant::now();
                crate::models::wavenet::layer::ACC_TANH
                    .with(|a| *a.borrow_mut() += now.duration_since(ts));
                Some(now)
            } else {
                None
            };

            let lb_offset = buffer_start * stride;
            let residual_slice =
                layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * stride);

            if CH == 12 {
                crate::math::gemm::gemm_batch::stride16::fused_gemm_residual_batch_f32_12x12_padded(
                    conv_slice,
                    &self.one_by_one.weights,
                    &self.one_by_one.bias,
                    residual_slice,
                    output,
                    num_frames,
                    self.one_by_one.do_bias,
                );
            } else {
                self.one_by_one.process_residual_batch::<M>(
                    conv_slice,
                    residual_slice,
                    output,
                    num_frames,
                );
            }

            #[cfg(test)]
            if let Some(ts) = t_tanh {
                let now = std::time::Instant::now();
                crate::models::wavenet::layer::ACC_ONE_BY_ONE
                    .with(|a| *a.borrow_mut() += now.duration_since(ts));
            }
        }
    }
}
