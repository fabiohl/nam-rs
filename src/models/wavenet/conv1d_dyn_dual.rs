// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::MAX_KERNEL;
use super::conv_input::{
    load_4_accums, load_8_accums, load_16_accums, store_4_accums, store_8_accums, store_16_accums,
};
use super::conv1d_dyn::Conv1dDyn;
use crate::math::common::SimdMath;

impl Conv1dDyn {
    /// F32-native dual-frame convolution (full-precision f32 weights).
    ///
    /// Processes two audio frames simultaneously using Temporal Tiling for
    /// maximum L1 cache reuse of convolution weights.
    ///
    /// # Safety
    /// `out_f0` and `out_f1` must have lengths of at least `self.out_ch`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn process_dual_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_f0: &mut [f32],
        out_f1: &mut [f32],
        idx_f0: usize,
        idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        let iw = self.interleave_width;
        let num_blocks = self.out_ch.div_ceil(iw);
        let mut tap_ptrs_f0 = [core::ptr::null::<f32>(); MAX_KERNEL];
        let mut tap_ptrs_f1 = [core::ptr::null::<f32>(); MAX_KERNEL];
        let k_limit = self.kernel.min(MAX_KERNEL);

        for (k, (tap_f0, tap_f1)) in tap_ptrs_f0
            .iter_mut()
            .zip(tap_ptrs_f1.iter_mut())
            .enumerate()
            .take(k_limit)
        {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
            let in_start_f0 = ((idx_f0 as isize) + offset) as usize * self.in_ch;
            let in_start_f1 = ((idx_f1 as isize) + offset) as usize * self.in_ch;

            unsafe {
                *tap_f0 = layer_buffer.as_ptr().add(in_start_f0);
                *tap_f1 = layer_buffer.as_ptr().add(in_start_f1);

                (self.prefetch_fn)(
                    *tap_f0,
                    self.dilation * self.in_ch,
                    k,
                    self.kernel,
                    self.dilation,
                );
            }
        }

        let in_ch = self.in_ch;
        let kernel = self.kernel;
        let do_bias = self.do_bias;

        // --- 1. Setup (Bias and Mixin) ---
        for b in 0..num_blocks {
            let off = b * iw;
            let w = iw.min(self.out_ch - off);
            for j in 0..w {
                let m0 = if let Some(m) = mixin_f0 {
                    if off + j < m.len() { m[off + j] } else { 0.0 }
                } else {
                    0.0
                };
                let m1 = if let Some(m) = mixin_f1 {
                    if off + j < m.len() { m[off + j] } else { 0.0 }
                } else {
                    0.0
                };
                if do_bias {
                    out_f0[off + j] = self.bias[off + j] + m0;
                    out_f1[off + j] = self.bias[off + j] + m1;
                } else {
                    out_f0[off + j] = m0;
                    out_f1[off + j] = m1;
                }
            }
        }

        // --- 2. Central loop with f32 weights ---
        match iw {
            16 => {
                for b in 0..num_blocks {
                    let out_c = b * 16;
                    let mut r_f0 = unsafe { load_16_accums(out_f0, out_c, self.out_ch) };
                    let mut r_f1 = unsafe { load_16_accums(out_f1, out_c, self.out_ch) };
                    let w = 16.min(self.out_ch - out_c);
                    for k in 0..kernel {
                        let tap_f0 = unsafe { *tap_ptrs_f0.get_unchecked(k) };
                        let tap_f1 = unsafe { *tap_ptrs_f1.get_unchecked(k) };
                        let w_start = (b * kernel + k) * in_ch * 16;
                        let w_slice: &[[f32; 16]] = unsafe {
                            let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 16];
                            core::slice::from_raw_parts(ptr, in_ch)
                        };
                        let in_f0 = unsafe { core::slice::from_raw_parts(tap_f0, in_ch) };
                        let in_f1 = unsafe { core::slice::from_raw_parts(tap_f1, in_ch) };
                        let (t_f0, t_f1) =
                            unsafe { M::dot_product_16x_f32_dual(w_slice, in_f0, in_f1) };
                        for i in 0..w {
                            r_f0[i] += t_f0[i];
                            r_f1[i] += t_f1[i];
                        }
                    }
                    unsafe { store_16_accums(out_f0, out_c, r_f0, self.out_ch) };
                    unsafe { store_16_accums(out_f1, out_c, r_f1, self.out_ch) };
                }
            }
            8 => {
                for b in 0..num_blocks {
                    let out_c = b * 8;
                    let mut r_f0 = unsafe { load_8_accums(out_f0, out_c, self.out_ch) };
                    let mut r_f1 = unsafe { load_8_accums(out_f1, out_c, self.out_ch) };
                    let w = 8.min(self.out_ch - out_c);
                    for k in 0..kernel {
                        let tap_f0 = unsafe { *tap_ptrs_f0.get_unchecked(k) };
                        let tap_f1 = unsafe { *tap_ptrs_f1.get_unchecked(k) };
                        let w_start = (b * kernel + k) * in_ch * 8;
                        let w_slice: &[[f32; 8]] = unsafe {
                            let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 8];
                            core::slice::from_raw_parts(ptr, in_ch)
                        };
                        let in_f0 = unsafe { core::slice::from_raw_parts(tap_f0, in_ch) };
                        let in_f1 = unsafe { core::slice::from_raw_parts(tap_f1, in_ch) };
                        let (t_f0, t_f1) =
                            unsafe { M::dot_product_8x_f32_dual(w_slice, in_f0, in_f1) };
                        for i in 0..w {
                            r_f0[i] += t_f0[i];
                            r_f1[i] += t_f1[i];
                        }
                    }
                    unsafe { store_8_accums(out_f0, out_c, r_f0, self.out_ch) };
                    unsafe { store_8_accums(out_f1, out_c, r_f1, self.out_ch) };
                }
            }
            _ => {
                for b in 0..num_blocks {
                    let out_c = b * 4;
                    let [mut r0_f0, mut r1_f0, mut r2_f0, mut r3_f0] =
                        unsafe { load_4_accums(out_f0, out_c, self.out_ch) };
                    let [mut r0_f1, mut r1_f1, mut r2_f1, mut r3_f1] =
                        unsafe { load_4_accums(out_f1, out_c, self.out_ch) };
                    for k in 0..kernel {
                        let tap_f0 = unsafe { *tap_ptrs_f0.get_unchecked(k) };
                        let tap_f1 = unsafe { *tap_ptrs_f1.get_unchecked(k) };
                        let w_start = (b * kernel + k) * in_ch * 4;
                        let w_slice: &[[f32; 4]] = unsafe {
                            let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                            core::slice::from_raw_parts(ptr, in_ch)
                        };
                        let in_f0 = unsafe { core::slice::from_raw_parts(tap_f0, in_ch) };
                        let in_f1 = unsafe { core::slice::from_raw_parts(tap_f1, in_ch) };
                        let (t_f0, t_f1) =
                            unsafe { M::dot_product_4x_f32_dual(w_slice, in_f0, in_f1) };
                        r0_f0 += t_f0[0];
                        r1_f0 += t_f0[1];
                        r2_f0 += t_f0[2];
                        r3_f0 += t_f0[3];
                        r0_f1 += t_f1[0];
                        r1_f1 += t_f1[1];
                        r2_f1 += t_f1[2];
                        r3_f1 += t_f1[3];
                    }
                    unsafe {
                        store_4_accums(out_f0, out_c, [r0_f0, r1_f0, r2_f0, r3_f0], self.out_ch)
                    };
                    unsafe {
                        store_4_accums(out_f1, out_c, [r0_f1, r1_f1, r2_f1, r3_f1], self.out_ch)
                    };
                }
            }
        }
    }
}
