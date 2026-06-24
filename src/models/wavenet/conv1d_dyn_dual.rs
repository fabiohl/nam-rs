// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::MAX_KERNEL;
use super::conv1d_dyn::Conv1dDyn;
use crate::math::common::{SimdMath, prefetch_strategy_2stage, prefetch_strategy_simple};

const TAP_BUF_FLOATS: usize = MAX_KERNEL * 64;

impl Conv1dDyn {
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

                if self.dilation >= 128 {
                    prefetch_strategy_2stage(
                        *tap_f0,
                        self.dilation * self.in_ch,
                        k,
                        self.kernel,
                        self.dilation,
                    );
                } else {
                    prefetch_strategy_simple(
                        *tap_f0,
                        self.dilation * self.in_ch,
                        k,
                        self.kernel,
                        self.dilation,
                    );
                }
            }
        }

        let in_ch = self.in_ch;
        let kernel = self.kernel;
        let do_bias = self.do_bias;
        let tap_total = kernel * in_ch;
        debug_assert!(tap_total <= TAP_BUF_FLOATS);

        let mut tap_buf_f0 = [0.0f32; TAP_BUF_FLOATS];
        let mut tap_buf_f1 = [0.0f32; TAP_BUF_FLOATS];
        for k in 0..kernel {
            let src_f0 = unsafe { core::slice::from_raw_parts(tap_ptrs_f0[k], in_ch) };
            let src_f1 = unsafe { core::slice::from_raw_parts(tap_ptrs_f1[k], in_ch) };
            tap_buf_f0[k * in_ch..(k + 1) * in_ch].copy_from_slice(src_f0);
            tap_buf_f1[k * in_ch..(k + 1) * in_ch].copy_from_slice(src_f1);
        }
        let flat_taps_f0: &[f32] = &tap_buf_f0[..tap_total];
        let flat_taps_f1: &[f32] = &tap_buf_f1[..tap_total];

        match iw {
            16 => {
                for b in 0..num_blocks {
                    let out_c = b * 16;
                    let w = 16.min(self.out_ch - out_c);
                    let mut init_f0 = [0.0f32; 16];
                    let mut init_f1 = [0.0f32; 16];
                    for (j, (item_f0, item_f1)) in init_f0
                        .iter_mut()
                        .zip(init_f1.iter_mut())
                        .enumerate()
                        .take(w)
                    {
                        let m0 = if let Some(m) = mixin_f0 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        let m1 = if let Some(m) = mixin_f1 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        if do_bias {
                            *item_f0 = self.bias[out_c + j] + m0;
                            *item_f1 = self.bias[out_c + j] + m1;
                        } else {
                            *item_f0 = m0;
                            *item_f1 = m1;
                        }
                    }
                    let w_start = b * kernel * in_ch * 16;
                    let w_slice: &[[f32; 16]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 16];
                        core::slice::from_raw_parts(ptr, kernel * in_ch)
                    };
                    let (r_f0, r_f1) = unsafe {
                        M::dot_product_16x_f32_dual_accumulate(
                            w_slice,
                            flat_taps_f0,
                            flat_taps_f1,
                            &init_f0,
                            &init_f1,
                        )
                    };
                    unsafe {
                        for (j, (&v_f0, &v_f1)) in r_f0.iter().zip(r_f1.iter()).enumerate().take(w)
                        {
                            *out_f0.get_unchecked_mut(out_c + j) = v_f0;
                            *out_f1.get_unchecked_mut(out_c + j) = v_f1;
                        }
                    }
                }
            }
            8 => {
                for b in 0..num_blocks {
                    let out_c = b * 8;
                    let w = 8.min(self.out_ch - out_c);
                    let mut init_f0 = [0.0f32; 8];
                    let mut init_f1 = [0.0f32; 8];
                    for (j, (item_f0, item_f1)) in init_f0
                        .iter_mut()
                        .zip(init_f1.iter_mut())
                        .enumerate()
                        .take(w)
                    {
                        let m0 = if let Some(m) = mixin_f0 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        let m1 = if let Some(m) = mixin_f1 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        if do_bias {
                            *item_f0 = self.bias[out_c + j] + m0;
                            *item_f1 = self.bias[out_c + j] + m1;
                        } else {
                            *item_f0 = m0;
                            *item_f1 = m1;
                        }
                    }
                    let w_start = b * kernel * in_ch * 8;
                    let w_slice: &[[f32; 8]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 8];
                        core::slice::from_raw_parts(ptr, kernel * in_ch)
                    };
                    let (r_f0, r_f1) = unsafe {
                        M::dot_product_8x_f32_dual_accumulate(
                            w_slice,
                            flat_taps_f0,
                            flat_taps_f1,
                            &init_f0,
                            &init_f1,
                        )
                    };
                    unsafe {
                        for (j, (&v_f0, &v_f1)) in r_f0.iter().zip(r_f1.iter()).enumerate().take(w)
                        {
                            *out_f0.get_unchecked_mut(out_c + j) = v_f0;
                            *out_f1.get_unchecked_mut(out_c + j) = v_f1;
                        }
                    }
                }
            }
            _ => {
                for b in 0..num_blocks {
                    let out_c = b * 4;
                    let w = 4.min(self.out_ch - out_c);
                    let mut init_f0 = [0.0f32; 4];
                    let mut init_f1 = [0.0f32; 4];
                    for (j, (item_f0, item_f1)) in init_f0
                        .iter_mut()
                        .zip(init_f1.iter_mut())
                        .enumerate()
                        .take(w)
                    {
                        let m0 = if let Some(m) = mixin_f0 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        let m1 = if let Some(m) = mixin_f1 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        if do_bias {
                            *item_f0 = self.bias[out_c + j] + m0;
                            *item_f1 = self.bias[out_c + j] + m1;
                        } else {
                            *item_f0 = m0;
                            *item_f1 = m1;
                        }
                    }
                    let w_start = b * kernel * in_ch * 4;
                    let w_slice: &[[f32; 4]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                        core::slice::from_raw_parts(ptr, kernel * in_ch)
                    };
                    let (r_f0, r_f1) = unsafe {
                        M::dot_product_4x_f32_dual_accumulate(
                            w_slice,
                            flat_taps_f0,
                            flat_taps_f1,
                            &init_f0,
                            &init_f1,
                        )
                    };
                    unsafe {
                        for (j, (&v_f0, &v_f1)) in r_f0.iter().zip(r_f1.iter()).enumerate().take(w)
                        {
                            *out_f0.get_unchecked_mut(out_c + j) = v_f0;
                            *out_f1.get_unchecked_mut(out_c + j) = v_f1;
                        }
                    }
                }
            }
        }
    }
}
