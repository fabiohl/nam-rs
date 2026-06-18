// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::MAX_KERNEL;
use super::conv1d_dyn::Conv1dDyn;

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
    pub(crate) unsafe fn process_dual_frame(
        &self,
        layer_buffer: &[f32],
        out_f0: &mut [f32],
        out_f1: &mut [f32],
        idx_f0: usize,
        idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        use crate::models::wavenet::conv_input::dot_product_4x_dual;

        let num_blocks = self.num_blocks;
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

        for b in 0..num_blocks {
            let out_c = b * 4;
            let (mut r0_f0, mut r1_f0, mut r2_f0, mut r3_f0);
            let (mut r0_f1, mut r1_f1, mut r2_f1, mut r3_f1);

            unsafe {
                let (mv0_f0, mv1_f0, mv2_f0, mv3_f0) = Self::load_mixin_4(mixin_f0, out_c);
                if do_bias {
                    r0_f0 = *self.bias.get_unchecked(out_c) + mv0_f0;
                    r1_f0 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1_f0;
                    r2_f0 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2_f0;
                    r3_f0 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3_f0;
                } else {
                    r0_f0 = mv0_f0;
                    r1_f0 = mv1_f0;
                    r2_f0 = mv2_f0;
                    r3_f0 = mv3_f0;
                }

                let (mv0_f1, mv1_f1, mv2_f1, mv3_f1) = Self::load_mixin_4(mixin_f1, out_c);
                if do_bias {
                    r0_f1 = *self.bias.get_unchecked(out_c) + mv0_f1;
                    r1_f1 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1_f1;
                    r2_f1 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2_f1;
                    r3_f1 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3_f1;
                } else {
                    r0_f1 = mv0_f1;
                    r1_f1 = mv1_f1;
                    r2_f1 = mv2_f1;
                    r3_f1 = mv3_f1;
                }

                for k in 0..kernel {
                    let tap_f0 = *tap_ptrs_f0.get_unchecked(k);
                    let tap_f1 = *tap_ptrs_f1.get_unchecked(k);

                    let w_start = (b * kernel + k) * in_ch * 4;
                    let w_slice: &[[f32; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                        core::slice::from_raw_parts(ptr, in_ch)
                    };

                    let in_f0 = core::slice::from_raw_parts(tap_f0, in_ch);
                    let in_f1 = core::slice::from_raw_parts(tap_f1, in_ch);

                    let (t_f0, t_f1) = dot_product_4x_dual(w_slice, in_f0, in_f1);

                    r0_f0 += t_f0[0];
                    r1_f0 += t_f0[1];
                    r2_f0 += t_f0[2];
                    r3_f0 += t_f0[3];
                    r0_f1 += t_f1[0];
                    r1_f1 += t_f1[1];
                    r2_f1 += t_f1[2];
                    r3_f1 += t_f1[3];
                }

                if out_c + 3 < self.out_ch {
                    *out_f0.get_unchecked_mut(out_c) = r0_f0;
                    *out_f0.get_unchecked_mut(out_c + 1) = r1_f0;
                    *out_f0.get_unchecked_mut(out_c + 2) = r2_f0;
                    *out_f0.get_unchecked_mut(out_c + 3) = r3_f0;

                    *out_f1.get_unchecked_mut(out_c) = r0_f1;
                    *out_f1.get_unchecked_mut(out_c + 1) = r1_f1;
                    *out_f1.get_unchecked_mut(out_c + 2) = r2_f1;
                    *out_f1.get_unchecked_mut(out_c + 3) = r3_f1;
                } else {
                    let r_f0 = [r0_f0, r1_f0, r2_f0, r3_f0];
                    let r_f1 = [r0_f1, r1_f1, r2_f1, r3_f1];
                    for lane in 0..4 {
                        if out_c + lane < self.out_ch {
                            *out_f0.get_unchecked_mut(out_c + lane) = r_f0[lane];
                            *out_f1.get_unchecked_mut(out_c + lane) = r_f1[lane];
                        }
                    }
                }
            }
        }
    }
}
