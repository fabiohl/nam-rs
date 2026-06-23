// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dual-Frame Processing of Causal WaveNet Convolution.
//!
//! Extension of `Conv1d` with methods that process two frames simultaneously
//! (Temporal Tiling), maximizing weight reuse in registers.
//!
//! ## Coesion justification (S2.T07 — no split)
//!
//! This file is a cohesive unit: a single `impl Conv1d` block extending the
//! static convolution with dual-frame (Temporal Tiling) processing.

use super::conv_input::{load_4_accums, store_4_accums};
use super::conv1d::Conv1d;
use crate::loader::dispatcher::wavenet::layout::select_interleave_width;
use crate::math::common::SimdMath;

impl<const IN: usize, const OUT: usize, const K: usize> Conv1d<IN, OUT, K> {
    /// Fused variant that processes two frames simultaneously, adding Mixin vectors
    /// (conditioning) directly to the accumulators.
    /// This approach maximizes the utilization of weights loaded into registers (Temporal Tiling).
    ///
    /// Uses full-precision f32 weights via `M::dot_product_4x_f32_dual` (AVX2/FMA or AVX-512 kernel).
    ///
    /// # Safety
    /// `layer_buffer` and `mixin` must have appropriate sizes.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_dual_frame_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: &[f32],
        mixin_f1: &[f32],
    ) {
        let interleave_width = select_interleave_width(OUT);

        if interleave_width != 4 {
            unsafe {
                self.process_single_frame_with_mixin::<M>(
                    layer_buffer,
                    out_frame_f0,
                    frame_idx_f0,
                    mixin_f0,
                );
                self.process_single_frame_with_mixin::<M>(
                    layer_buffer,
                    out_frame_f1,
                    frame_idx_f1,
                    mixin_f1,
                );
            }
            return;
        }
        // --- 1. Setup (Bias and Mixin) ---
        let full_blocks = OUT & !3;
        for i in (0..full_blocks).step_by(4) {
            let acc_f0: &mut [f32; 4] =
                unsafe { &mut *(out_frame_f0.as_mut_ptr().add(i) as *mut [f32; 4]) };
            let acc_f1: &mut [f32; 4] =
                unsafe { &mut *(out_frame_f1.as_mut_ptr().add(i) as *mut [f32; 4]) };
            if self.do_bias {
                acc_f0.copy_from_slice(&self.bias[i..i + 4]);
                acc_f1.copy_from_slice(&self.bias[i..i + 4]);
                for j in 0..4 {
                    acc_f0[j] += mixin_f0[i + j];
                    acc_f1[j] += mixin_f1[i + j];
                }
            } else {
                acc_f0.copy_from_slice(&mixin_f0[i..i + 4]);
                acc_f1.copy_from_slice(&mixin_f1[i..i + 4]);
            }
        }
        let rem = OUT & 3;
        if rem > 0 {
            let i = full_blocks;
            let rem_f0 = &mut out_frame_f0[i..OUT];
            let rem_f1 = &mut out_frame_f1[i..OUT];
            if self.do_bias {
                rem_f0.copy_from_slice(&self.bias[i..OUT]);
                rem_f1.copy_from_slice(&self.bias[i..OUT]);
                for j in 0..rem {
                    rem_f0[j] += mixin_f0[i + j];
                    rem_f1[j] += mixin_f1[i + j];
                }
            } else {
                rem_f0.copy_from_slice(&mixin_f0[i..OUT]);
                rem_f1.copy_from_slice(&mixin_f1[i..OUT]);
            }
        }

        // --- 2. Preload taps ---
        let mut in_taps_f0 = [[0.0f32; IN]; K];
        let mut in_taps_f1 = [[0.0f32; IN]; K];
        for k in 0..K {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let in_start_f0 = ((frame_idx_f0 as isize) + offset) as usize * IN;
            let in_start_f1 = ((frame_idx_f1 as isize) + offset) as usize * IN;
            unsafe {
                in_taps_f0[k]
                    .copy_from_slice(layer_buffer.get_unchecked(in_start_f0..in_start_f0 + IN));
                in_taps_f1[k]
                    .copy_from_slice(layer_buffer.get_unchecked(in_start_f1..in_start_f1 + IN));
                (self.prefetch_fn)(
                    layer_buffer.as_ptr().add(in_start_f0),
                    self.dilation * IN,
                    k,
                    K,
                    self.dilation,
                );
            }
        }

        // --- 3. Central loop with f32 weights ---
        let num_blocks = OUT.div_ceil(4);
        let mut out_c = 0;

        for _b in 0..num_blocks {
            let [mut r0_f0, mut r1_f0, mut r2_f0, mut r3_f0] =
                unsafe { load_4_accums(out_frame_f0, out_c, OUT) };
            let [mut r0_f1, mut r1_f1, mut r2_f1, mut r3_f1] =
                unsafe { load_4_accums(out_frame_f1, out_c, OUT) };

            for k in 0..K {
                let w_start = (out_c / 4 * K + k) * IN * 4;
                let w_slice: &[[f32; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                let (t_f0, t_f1) =
                    unsafe { M::dot_product_4x_f32_dual(w_slice, &in_taps_f0[k], &in_taps_f1[k]) };

                r0_f0 += t_f0[0];
                r1_f0 += t_f0[1];
                r2_f0 += t_f0[2];
                r3_f0 += t_f0[3];
                r0_f1 += t_f1[0];
                r1_f1 += t_f1[1];
                r2_f1 += t_f1[2];
                r3_f1 += t_f1[3];
            }

            unsafe { store_4_accums(out_frame_f0, out_c, [r0_f0, r1_f0, r2_f0, r3_f0], OUT) };
            unsafe { store_4_accums(out_frame_f1, out_c, [r0_f1, r1_f1, r2_f1, r3_f1], OUT) };
            out_c += 4;
        }
    }
}
