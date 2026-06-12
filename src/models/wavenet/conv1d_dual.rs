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
//! static convolution with dual-frame (Temporal Tiling) processing. The
//! structure is 3 methods — two thin public wrappers (`process_dual_frame_with_mixin`,
//! `process_dual_frame_bf16_with_mixin`) plus one dominant generic kernel
//! (`process_dual_frame_generic`) that is the central computation engine shared
//! by all paths. The internal dispatchers are trivial routing methods.
//! Splitting would fragment this tightly coupled unit without meaningful
//! structural benefit, as no sub-component is independently reusable or
//! testable in isolation.

use super::conv_input::{init_accum_with_bias_mixin, load_4_accums, store_kahan_4_accums};
use super::conv1d::{Conv1d, ConvInput};
use crate::math::common::{SimdMath, kahan_add};

impl<const IN: usize, const OUT: usize, const K: usize> Conv1d<IN, OUT, K> {
    /// Fused variant that processes two frames simultaneously, adding Mixin vectors (conditioning) directly to the accumulators.
    /// This approach maximizes the utilization of weights loaded into registers (Temporal Tiling).
    ///
    /// # Safety
    /// `layer_buffer` and `mixin` must have appropriate sizes.
    /// Dual Frame Processing with Mixin:
    /// This function computes two audio moments at once, already integrating
    /// external settings (mixin) to save processing time.
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
        unsafe {
            self.process_dual_frame_internal::<M>(
                layer_buffer,
                out_frame_f0,
                out_frame_f1,
                frame_idx_f0,
                frame_idx_f1,
                Some(mixin_f0),
                Some(mixin_f1),
            );
        }
    }

    /// Internal Organizer:
    /// Prepares data for the 'Universal Engine' (Generic), deciding how
    /// information will be sent for computation.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn process_dual_frame_internal<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        unsafe {
            self.process_dual_frame_generic::<M, f32>(
                layer_buffer,
                out_frame_f0,
                out_frame_f1,
                frame_idx_f0,
                frame_idx_f1,
                mixin_f0,
                mixin_f1,
            );
        }
    }

    /// Universal Engine (Generic Engine):
    /// This is the central intelligence that performs the heavy math. Thanks to the use
    /// of generic types (T: ConvInput), this same code works in both
    /// full precision mode and ultra-fast mode (BF16).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn process_dual_frame_generic<M: SimdMath, T: ConvInput>(
        &self,
        layer_buffer: &[T],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        // --- 1. Setup (Bias and Mixin) ---
        let full_blocks = OUT & !3;
        for i in (0..full_blocks).step_by(4) {
            // SAFETY: `i + 4 <= OUT` since `i` ranges up to `full_blocks` in steps of 4, where `full_blocks = OUT & !3`.
            // The slice pointers are well-aligned and point to valid sequences of 4 floats, which aligns with `[f32; 4]`.
            // The lifetimes are correctly constrained by the borrows of `out_frame_f0` and `out_frame_f1`.
            let acc_f0: &mut [f32; 4] =
                unsafe { &mut *(out_frame_f0.as_mut_ptr().add(i) as *mut [f32; 4]) };
            let acc_f1: &mut [f32; 4] =
                unsafe { &mut *(out_frame_f1.as_mut_ptr().add(i) as *mut [f32; 4]) };
            unsafe {
                init_accum_with_bias_mixin::<M>(acc_f0, &self.bias, mixin_f0, i, self.do_bias);
                init_accum_with_bias_mixin::<M>(acc_f1, &self.bias, mixin_f1, i, self.do_bias);
            }
        }
        let rem = OUT & 3;
        if rem > 0 {
            let i = full_blocks;
            // f0 remainder
            let rem_slice_f0 = &mut out_frame_f0[i..OUT];
            let rem_slice_f1 = &mut out_frame_f1[i..OUT];
            if let (Some(m0), Some(m1)) = (mixin_f0, mixin_f1) {
                if self.do_bias {
                    rem_slice_f0.copy_from_slice(&self.bias[i..OUT]);
                    rem_slice_f1.copy_from_slice(&self.bias[i..OUT]);
                    unsafe {
                        M::accumulate_head(rem_slice_f0, &m0[i..OUT]);
                        M::accumulate_head(rem_slice_f1, &m1[i..OUT]);
                    }
                } else {
                    rem_slice_f0.copy_from_slice(&m0[i..OUT]);
                    rem_slice_f1.copy_from_slice(&m1[i..OUT]);
                }
            } else if self.do_bias {
                rem_slice_f0.copy_from_slice(&self.bias[i..OUT]);
                rem_slice_f1.copy_from_slice(&self.bias[i..OUT]);
            } else {
                rem_slice_f0.fill(0.0);
                rem_slice_f1.fill(0.0);
            }
        }

        // --- 2. Look into the Past (Dilation) ---
        let mut in_taps_f0 = [[T::default(); IN]; K];
        let mut in_taps_f1 = [[T::default(); IN]; K];
        for k in 0..K {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let in_slice_start_f0 = ((frame_idx_f0 as isize) + offset) as usize * IN;
            let in_slice_start_f1 = ((frame_idx_f1 as isize) + offset) as usize * IN;
            unsafe {
                in_taps_f0.get_unchecked_mut(k).copy_from_slice(
                    layer_buffer.get_unchecked(in_slice_start_f0..in_slice_start_f0 + IN),
                );
                in_taps_f1.get_unchecked_mut(k).copy_from_slice(
                    layer_buffer.get_unchecked(in_slice_start_f1..in_slice_start_f1 + IN),
                );
                (self.prefetch_fn)(
                    T::cast_ptr(layer_buffer.as_ptr().add(in_slice_start_f0)),
                    self.dilation * IN,
                    k,
                    K,
                    self.dilation,
                );
            }
        }

        let num_blocks = OUT.div_ceil(4);
        let mut out_c = 0;

        // --- 3. Central Computation Loop (4-Blocks) ---
        for b in 0..num_blocks {
            let [mut r0_f0, mut r1_f0, mut r2_f0, mut r3_f0] =
                unsafe { load_4_accums(out_frame_f0, out_c, OUT) };
            let [mut r0_f1, mut r1_f1, mut r2_f1, mut r3_f1] =
                unsafe { load_4_accums(out_frame_f1, out_c, OUT) };

            let mut c0_f0 = 0.0f32;
            let mut c1_f0 = 0.0f32;
            let mut c2_f0 = 0.0f32;
            let mut c3_f0 = 0.0f32;
            let mut c0_f1 = 0.0f32;
            let mut c1_f1 = 0.0f32;
            let mut c2_f1 = 0.0f32;
            let mut c3_f1 = 0.0f32;

            for k in 0..K {
                let w_start = (b * K + k) * IN * 4;
                let w_slice: &[[u16; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                let in_slice_f0 = &in_taps_f0[k];
                let in_slice_f1 = &in_taps_f1[k];

                let (t_f0, t_f1) = unsafe {
                    T::dot_product_4x_interleaved_dual_frame::<M>(w_slice, in_slice_f0, in_slice_f1)
                };

                let (s, c) = kahan_add(r0_f0, c0_f0, t_f0[0]);
                r0_f0 = s;
                c0_f0 = c;
                let (s, c) = kahan_add(r1_f0, c1_f0, t_f0[1]);
                r1_f0 = s;
                c1_f0 = c;
                let (s, c) = kahan_add(r2_f0, c2_f0, t_f0[2]);
                r2_f0 = s;
                c2_f0 = c;
                let (s, c) = kahan_add(r3_f0, c3_f0, t_f0[3]);
                r3_f0 = s;
                c3_f0 = c;
                let (s, c) = kahan_add(r0_f1, c0_f1, t_f1[0]);
                r0_f1 = s;
                c0_f1 = c;
                let (s, c) = kahan_add(r1_f1, c1_f1, t_f1[1]);
                r1_f1 = s;
                c1_f1 = c;
                let (s, c) = kahan_add(r2_f1, c2_f1, t_f1[2]);
                r2_f1 = s;
                c2_f1 = c;
                let (s, c) = kahan_add(r3_f1, c3_f1, t_f1[3]);
                r3_f1 = s;
                c3_f1 = c;
            }

            unsafe { store_kahan_4_accums(out_frame_f0, out_c, [r0_f0, r1_f0, r2_f0, r3_f0], OUT) };
            unsafe { store_kahan_4_accums(out_frame_f1, out_c, [r0_f1, r1_f1, r2_f1, r3_f1], OUT) };
            out_c += 4;
        }
    }

    /// Fused BF16 variant that processes two frames simultaneously, adding Mixin vectors directly to the accumulators.
    /// This approach maximizes the utilization of weights (VNNI) loaded into registers (Temporal Tiling).
    ///
    /// # Safety
    /// The caller must guarantee that `layer_buffer` and `mixin` have appropriate sizes.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_dual_frame_bf16_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: &[f32],
        mixin_f1: &[f32],
    ) {
        unsafe {
            self.process_dual_frame_bf16_internal::<M>(
                layer_buffer,
                out_frame_f0,
                out_frame_f1,
                frame_idx_f0,
                frame_idx_f1,
                Some(mixin_f0),
                Some(mixin_f1),
            );
        }
    }

    /// Internal BF16 Organizer:
    /// Routes 16-bit data to the Universal Engine for computation.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn process_dual_frame_bf16_internal<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        unsafe {
            self.process_dual_frame_generic::<M, u16>(
                layer_buffer,
                out_frame_f0,
                out_frame_f1,
                frame_idx_f0,
                frame_idx_f1,
                mixin_f0,
                mixin_f1,
            );
        }
    }
}
