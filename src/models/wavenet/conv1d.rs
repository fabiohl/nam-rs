// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Static Causal CNN Mesh for WaveNet Inference (Data-Oriented Design, SoA).
//!
//! **Cohesion Justification:** Single static 1D convolution unit: `ConvInput` trait +
//! `Conv1d` struct + single-frame kernel + mixin wrappers form a cohesive algorithmic unit.
//! `ConvInput` was extracted to `conv_input.rs` (S2.T06). Further splitting the
//! single-frame kernel would break the locality of `unsafe` aliasing contracts and
//! plain accumulators.

pub(crate) use super::conv_input::ConvInput;
#[cfg(test)]
use super::conv_input::init_accum_with_bias_mixin;
use super::conv_input::{dot_product_4x, load_4_accums, store_4_accums};
#[cfg(test)]
use crate::math::common::SimdMath;
use crate::math::common::{AlignedVec, PrefetchFn};

/// Dilated Causal Convolution (WaveNet Conv1D).
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Flattened weight matrix of size OUT * K * IN in full-precision f32.
    pub weights: AlignedVec<f32>,
    /// Causal bias, applied if do_bias is true. Total: OUT.
    pub bias: AlignedVec<f32>,
    /// Determines if the bias array should be added.
    pub do_bias: bool,
    /// Dilation factor on the causal temporal axis (e.g.: 1, 2, 4.. 512).
    pub dilation: usize,
    /// Pre-computed prefetch strategy (Branch Elimination).
    pub prefetch_fn: PrefetchFn,
}

impl<const IN: usize, const OUT: usize, const K: usize> Conv1d<IN, OUT, K> {
    /// Processes a single frame applying convolution to the ring buffer,
    /// fusing a Mixin vector (conditioning) directly into the accumulator.
    ///
    /// Uses full-precision f32 weights via `dot_product_4x` (AVX2/FMA or AVX-512 kernel).
    ///
    /// # Safety
    /// The caller must guarantee that `frame_idx`, `mixin`, `layer_buffer`,
    /// and `out_frame` have sizes compatible with the layer dimensions.
    #[inline(always)]
    pub unsafe fn process_single_frame_with_mixin(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: &[f32],
    ) {
        // [STEP 1: Accumulator Initialization]
        let full_blocks = OUT & !3;
        for i in (0..full_blocks).step_by(4) {
            let acc: &mut [f32; 4] =
                unsafe { &mut *(out_frame.as_mut_ptr().add(i) as *mut [f32; 4]) };
            if self.do_bias {
                acc.copy_from_slice(&self.bias[i..i + 4]);
                for j in 0..4 {
                    acc[j] += mixin[i + j];
                }
            } else {
                acc.copy_from_slice(&mixin[i..i + 4]);
            }
        }
        let rem = OUT & 3;
        if rem > 0 {
            let i = full_blocks;
            let rem_slice = &mut out_frame[i..OUT];
            if self.do_bias {
                rem_slice.copy_from_slice(&self.bias[i..OUT]);
                for j in 0..rem {
                    rem_slice[j] += mixin[i + j];
                }
            } else {
                rem_slice.copy_from_slice(&mixin[i..OUT]);
            }
        }

        // [STEP 2: Kernel Iteration with f32 weights]
        let mut in_taps = [[0.0f32; IN]; K];
        for (k, in_tap) in in_taps.iter_mut().enumerate() {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let in_slice_start = ((frame_idx as isize) + offset) as usize * IN;
            unsafe {
                in_tap.copy_from_slice(
                    layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN),
                );
            }
            unsafe {
                (self.prefetch_fn)(
                    layer_buffer.as_ptr().add(in_slice_start),
                    self.dilation * IN,
                    k,
                    K,
                    self.dilation,
                );
            }
        }

        let num_blocks = OUT.div_ceil(4);
        let mut out_c = 0;

        for _b in 0..num_blocks {
            let [mut r0, mut r1, mut r2, mut r3] = unsafe { load_4_accums(out_frame, out_c, OUT) };

            for (k, in_slice) in in_taps.iter().enumerate() {
                let w_start = (out_c / 4 * K + k) * IN * 4;
                let w_slice: &[[f32; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                let [t0, t1, t2, t3] = dot_product_4x(w_slice, in_slice);
                r0 += t0;
                r1 += t1;
                r2 += t2;
                r3 += t3;
            }

            unsafe { store_4_accums(out_frame, out_c, [r0, r1, r2, r3], OUT) };
            out_c += 4;
        }
    }

    /// Executes causal convolution over a flat bidirectional array (`layer_buffer`).
    ///
    /// # Safety
    /// Dynamically depends on the `SimdMath` trait provided.
    #[cfg(test)]
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
    ) {
        // [STEP 1: Accumulator Initialization — bias only, no mixin]
        let full_blocks = OUT & !3;
        for i in (0..full_blocks).step_by(4) {
            let acc: &mut [f32; 4] =
                unsafe { &mut *(out_frame.as_mut_ptr().add(i) as *mut [f32; 4]) };
            unsafe {
                init_accum_with_bias_mixin::<M>(acc, &self.bias, None, i, self.do_bias);
            }
        }
        let rem = OUT & 3;
        if rem > 0 {
            let i = full_blocks;
            let rem_slice = &mut out_frame[i..OUT];
            if self.do_bias {
                rem_slice.copy_from_slice(&self.bias[i..OUT]);
            } else {
                rem_slice.fill(0.0);
            }
        }

        // [STEP 2: Kernel Iteration with f32 weights]
        let mut in_taps = [[0.0f32; IN]; K];
        for (k, in_tap) in in_taps.iter_mut().enumerate() {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let in_slice_start = ((frame_idx as isize) + offset) as usize * IN;
            unsafe {
                in_tap.copy_from_slice(
                    layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN),
                );
            }
            unsafe {
                (self.prefetch_fn)(
                    layer_buffer.as_ptr().add(in_slice_start),
                    self.dilation * IN,
                    k,
                    K,
                    self.dilation,
                );
            }
        }

        let num_blocks = OUT.div_ceil(4);
        let mut out_c = 0;

        for _b in 0..num_blocks {
            let [mut r0, mut r1, mut r2, mut r3] = unsafe { load_4_accums(out_frame, out_c, OUT) };

            for (k, in_slice) in in_taps.iter().enumerate() {
                let w_start = (out_c / 4 * K + k) * IN * 4;
                let w_slice: &[[f32; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                let [t0, t1, t2, t3] = dot_product_4x(w_slice, in_slice);
                r0 += t0;
                r1 += t1;
                r2 += t2;
                r3 += t3;
            }

            unsafe { store_4_accums(out_frame, out_c, [r0, r1, r2, r3], OUT) };
            out_c += 4;
        }
    }

    /// Processes a sequential iterative block.
    /// For cache efficiency, instead of processing the entire layer by multiple blocks,
    /// we limit calls to consecutive frame-by-frame calls (`process_single_frame`).
    ///
    /// # Safety
    /// Pointer must be valid and num_frames must fit within the layer_buffer bounds.
    #[cfg(test)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            let out_frame = unsafe { block.get_unchecked_mut(i * OUT..i * OUT + OUT) };
            unsafe {
                self.process_single_frame::<M>(layer_buffer, out_frame, buffer_start + i);
            }
        }
    }
}
