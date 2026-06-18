// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::MAX_KERNEL;
use super::conv1d::ConvInput;
use super::conv1d_dyn::Conv1dDyn;
use crate::math::common::SimdMath;

impl Conv1dDyn {
    #[inline(always)]
    pub(crate) unsafe fn process_single_frame_generic<M: SimdMath, T: ConvInput>(
        &self,
        layer_buffer: &[T],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.num_blocks;
        debug_assert!(
            self.kernel <= MAX_KERNEL,
            "kernel {} exceeds MAX_KERNEL",
            self.kernel
        );
        debug_assert!(
            self.weights.len() >= num_blocks * 4 * self.in_ch * self.kernel,
            "weights length {} is less than expected {}",
            self.weights.len(),
            num_blocks * 4 * self.in_ch * self.kernel
        );
        let mut tap_ptrs = [core::ptr::null::<T>(); MAX_KERNEL];
        let k_limit = self.kernel.min(MAX_KERNEL);

        for (k, tap_ptr) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
            let in_slice_start = ((frame_idx as isize) + offset) as usize * self.in_ch;
            unsafe {
                *tap_ptr = layer_buffer.as_ptr().add(in_slice_start);
                (self.prefetch_fn)(
                    T::cast_ptr(*tap_ptr),
                    self.dilation * self.in_ch,
                    k,
                    self.kernel,
                    self.dilation,
                );
            }
        }

        for b in 0..num_blocks {
            let out_c = b * 4;
            let mut r0;
            let mut r1;
            let mut r2;
            let mut r3;

            unsafe {
                let (mv0, mv1, mv2, mv3) = Self::load_mixin_4(mixin, out_c);
                if self.do_bias {
                    r0 = *self.bias.get_unchecked(out_c) + mv0;
                    r1 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1;
                    r2 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2;
                    r3 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3;
                } else {
                    r0 = mv0;
                    r1 = mv1;
                    r2 = mv2;
                    r3 = mv3;
                }

                for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                    let w_start = (b * self.kernel + k) * self.in_ch * 4;
                    let w_slice: &[[u16; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                        core::slice::from_raw_parts(ptr, self.in_ch)
                    };

                    let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);

                    let [t0, t1, t2, t3] = T::dot_product_4x_interleaved::<M>(w_slice, in_slice);
                    r0 += t0;
                    r1 += t1;
                    r2 += t2;
                    r3 += t3;
                }

                if out_c + 3 < self.out_ch {
                    *out_frame.get_unchecked_mut(out_c) = r0;
                    *out_frame.get_unchecked_mut(out_c + 1) = r1;
                    *out_frame.get_unchecked_mut(out_c + 2) = r2;
                    *out_frame.get_unchecked_mut(out_c + 3) = r3;
                } else {
                    let r = [r0, r1, r2, r3];
                    for (lane, &val) in r.iter().enumerate() {
                        if out_c + lane < self.out_ch {
                            *out_frame.get_unchecked_mut(out_c + lane) = val;
                        }
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn process_block_generic<M: SimdMath, T: ConvInput>(
        &self,
        layer_buffer: &[T],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert_eq!(num_frames * self.out_ch, block.len());
        let mut i = 0;

        let mut chunks = block.chunks_exact_mut(2 * self.out_ch);
        for chunk in chunks.by_ref() {
            let (out_f0, out_f1) = chunk.split_at_mut(self.out_ch);

            let (m_f0, m_f1) = if let Some(m) = mixin {
                let start0 = i * self.out_ch;
                let end0 = (start0 + self.out_ch).min(m.len());
                let start1 = (i + 1) * self.out_ch;
                let end1 = (start1 + self.out_ch).min(m.len());
                (
                    if start0 < m.len() {
                        Some(&m[start0..end0])
                    } else {
                        None
                    },
                    if start1 < m.len() {
                        Some(&m[start1..end1])
                    } else {
                        None
                    },
                )
            } else {
                (None, None)
            };

            unsafe {
                self.process_dual_frame_generic::<M, T>(
                    layer_buffer,
                    out_f0,
                    out_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    m_f0,
                    m_f1,
                );
            }
            i += 2;
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let m = mixin.map(|m| &m[i * self.out_ch..(i + 1) * self.out_ch]);
            unsafe {
                self.process_single_frame_generic::<M, T>(layer_buffer, rem, buffer_start + i, m);
            }
        }
    }

    /// F32-native block processing (full-precision f32 weights).
    ///
    /// Same dual-frame tiling as `process_block_generic`, but uses full-precision
    /// f32 weights and scalar dot products, bypassing quantization drift.
    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) unsafe fn process_block_f32_native(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert_eq!(num_frames * self.out_ch, block.len());
        let mut i = 0;

        let mut chunks = block.chunks_exact_mut(2 * self.out_ch);
        for chunk in chunks.by_ref() {
            let (out_f0, out_f1) = chunk.split_at_mut(self.out_ch);

            let (m_f0, m_f1) = if let Some(m) = mixin {
                let start0 = i * self.out_ch;
                let end0 = (start0 + self.out_ch).min(m.len());
                let start1 = (i + 1) * self.out_ch;
                let end1 = (start1 + self.out_ch).min(m.len());
                (
                    if start0 < m.len() {
                        Some(&m[start0..end0])
                    } else {
                        None
                    },
                    if start1 < m.len() {
                        Some(&m[start1..end1])
                    } else {
                        None
                    },
                )
            } else {
                (None, None)
            };

            unsafe {
                self.process_dual_frame_f32_native_generic(
                    layer_buffer,
                    out_f0,
                    out_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    m_f0,
                    m_f1,
                );
            }
            i += 2;
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let m = mixin.map(|m| &m[i * self.out_ch..(i + 1) * self.out_ch]);
            unsafe {
                self.process_single_frame_f32_native(layer_buffer, rem, buffer_start + i, m);
            }
        }
    }
}
