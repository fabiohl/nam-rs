// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Runtime-dimensional convolution components for WaveNet architectures.
//!
//! Contains the fundamental convolution structures that operate with
//! runtime-defined dimensions, serving as a foundation for A2 architecture
//! stages and static WaveNet test/stress kernels.

use crate::math::common::{AlignedVec, PrefetchFn};

use super::common::MAX_KERNEL;

/// Structure for causal 1D convolution with dynamic dimensions.
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1dDyn {
    /// Full-precision f32 convolution weights [OUT][KERNEL][IN] (interleaved 4-wide).
    pub weights: AlignedVec<f32>,
    /// Bias vector [OUT].
    pub bias: AlignedVec<f32>,
    /// Flag indicating whether bias should be applied.
    pub do_bias: bool,
    /// Temporal dilation factor.
    pub dilation: usize,
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of output channels.
    pub out_ch: usize,
    /// Number of 4-channel blocks.
    pub num_blocks: usize,
    /// Physical kernel size.
    pub kernel: usize,
    /// Pre-computed prefetch strategy.
    pub prefetch_fn: PrefetchFn,
}

impl Conv1dDyn {
    /// F32-native single-frame convolution (full-precision f32 weights).
    ///
    /// # Safety
    /// The caller must guarantee that `layer_buffer` and `out_frame` have sizes
    /// compatible with the layer dimensions.
    #[inline(always)]
    pub unsafe fn process_single_frame(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        use crate::models::wavenet::conv_input::dot_product_4x;

        let num_blocks = self.num_blocks;
        let in_ch = self.in_ch;
        let out_ch = self.out_ch;
        let kernel = self.kernel;
        let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
        let k_limit = kernel.min(MAX_KERNEL);

        for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (kernel as isize));
            let in_start = ((frame_idx as isize) + offset) as usize * in_ch;
            unsafe {
                *tap = layer_buffer.as_ptr().add(in_start);
                (self.prefetch_fn)(*tap, self.dilation * in_ch, k, kernel, self.dilation);
            }
        }

        for b in 0..num_blocks {
            let out_c = b * 4;
            let (mu0, mu1, mu2, mu3) = unsafe { Self::load_mixin_4(mixin, out_c) };
            let (mut r0, mut r1, mut r2, mut r3);
            unsafe {
                if self.do_bias {
                    r0 = *self.bias.get_unchecked(out_c) + mu0;
                    r1 = if out_c + 1 < out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mu1;
                    r2 = if out_c + 2 < out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mu2;
                    r3 = if out_c + 3 < out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mu3;
                } else {
                    r0 = mu0;
                    r1 = mu1;
                    r2 = mu2;
                    r3 = mu3;
                }

                for k in 0..kernel {
                    let tap = *tap_ptrs.get_unchecked(k);
                    let w_start = (b * kernel + k) * in_ch * 4;
                    let w_slice: &[[f32; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                        core::slice::from_raw_parts(ptr, in_ch)
                    };
                    let in_slice = core::slice::from_raw_parts(tap, in_ch);
                    let [t0, t1, t2, t3] = dot_product_4x(w_slice, in_slice);
                    r0 += t0;
                    r1 += t1;
                    r2 += t2;
                    r3 += t3;
                }

                if out_c + 3 < out_ch {
                    *out_frame.get_unchecked_mut(out_c) = r0;
                    *out_frame.get_unchecked_mut(out_c + 1) = r1;
                    *out_frame.get_unchecked_mut(out_c + 2) = r2;
                    *out_frame.get_unchecked_mut(out_c + 3) = r3;
                } else {
                    let r = [r0, r1, r2, r3];
                    for (lane, &val) in r.iter().enumerate() {
                        if out_c + lane < out_ch {
                            *out_frame.get_unchecked_mut(out_c + lane) = val;
                        }
                    }
                }
            }
        }
    }

    /// F32-native block processing (full-precision f32 weights).
    ///
    /// Same dual-frame tiling as the generic block processing, but uses full-precision
    /// f32 weights and scalar dot products.
    #[inline(always)]
    pub(crate) unsafe fn process_block(
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
                self.process_dual_frame(
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
                self.process_single_frame(layer_buffer, rem, buffer_start + i, m);
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn load_mixin_4(mixin: Option<&[f32]>, out_c: usize) -> (f32, f32, f32, f32) {
        if let Some(m) = mixin {
            if out_c + 3 < m.len() {
                unsafe {
                    (
                        *m.get_unchecked(out_c),
                        *m.get_unchecked(out_c + 1),
                        *m.get_unchecked(out_c + 2),
                        *m.get_unchecked(out_c + 3),
                    )
                }
            } else {
                let mut v = [0.0f32; 4];
                for (i, val) in v.iter_mut().enumerate() {
                    if out_c + i < m.len() {
                        unsafe {
                            *val = *m.get_unchecked(out_c + i);
                        }
                    }
                }
                (v[0], v[1], v[2], v[3])
            }
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    }
}
