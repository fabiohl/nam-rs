// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Common and dynamic components for WaveNet architectures.
//!
//! Contains the fundamental structures (Conv1D, Dense, Layer) that operate with
//! runtime-defined dimensions, serving as a foundation for the dynamic model
//! and future A2 architecture stages.
//!
//! IMPORTANT: A2 architecture support is in "placeholder" stage
//! pending stabilization of the reference implementation.

use super::conv1d::ConvInput;
use crate::math::common::{AlignedVec, PrefetchFn, SimdMath};

/// Maximum frames to process in one callback pulse.
pub const WAVENET_MAX_NUM_FRAMES: usize = 64;
/// Circular temporal padding of memories in the Ring Buffers framework.
pub const LAYER_ARRAY_BUFFER_PADDING: usize = 24;
/// Maximum supported kernel size.
pub const MAX_KERNEL: usize = 16;

/// Structure for causal 1D convolution with dynamic dimensions.
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1dDyn {
    /// Convolution weights [OUT][KERNEL][IN] (quantized u16).
    pub weights: AlignedVec<u16>,
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
    /// Processes two audio frames simultaneously using single-precision (f32) inputs.
    ///
    /// # Algorithmic Details & Optimizations
    /// - **Temporal Tiling & Weight Reuse**: The critical performance bottleneck of WaveNet inference
    ///   is the memory bandwidth/latency of loading convolution weights. By processing two independent
    ///   temporal frames (`idx_f0` and `idx_f1`) simultaneously, loaded weights are reused across both
    ///   calculations. This reduces weight loads from L1 cache/registers by 50% per calculated frame.
    /// - **Instruction-Level Parallelism (ILP)**: Interleaving instructions for two independent frames
    ///   allows the CPU's execution pipelines to hide instruction latencies (e.g., FMA latencies). Instead
    ///   of waiting on dependency chains of a single frame's accumulator, we alternate operations between
    ///   the two frames, saturating execution units.
    /// - **Interleaved Layout**: Weights are stored in a `[OUT/4][KERNEL][IN][4]` layout. This allows the
    ///   underlying SIMD engine to perform 4-lane wide vector dot products on output channels in parallel
    ///   using a single weight vector load.
    /// - **Fallback / Tail Trade-off**: When the total block size `num_frames` is odd, the even pairs
    ///   are processed in dual-frame batches, and the final remaining frame falls back to the single-frame
    ///   variant ([`Self::process_single_frame`]) which does not benefit from the weight reuse optimization.
    ///
    /// # Safety
    /// - `out_f0` and `out_f1` must have lengths of at least `self.out_ch`.
    /// - `layer_buffer` must contain valid elements for the dilated tap indexes calculated from
    ///   `idx_f0` and `idx_f1`.
    /// - Pointers derived from `layer_buffer` must not violate Rust's aliasing rules (they should be
    ///   disjoint or read-only).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_dual_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_f0: &mut [f32],
        out_f1: &mut [f32],
        idx_f0: usize,
        idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        unsafe {
            self.process_dual_frame_generic::<M, f32>(
                layer_buffer,
                out_f0,
                out_f1,
                idx_f0,
                idx_f1,
                mixin_f0,
                mixin_f1,
            );
        }
    }

    /// Processes two audio frames simultaneously using half-precision (BF16) inputs.
    ///
    /// # Algorithmic Details & Optimizations
    /// - **Temporal Tiling & Weight Reuse**: The critical performance bottleneck of WaveNet inference
    ///   is the memory bandwidth/latency of loading convolution weights. By processing two independent
    ///   temporal frames (`idx_f0` and `idx_f1`) simultaneously, loaded weights are reused across both
    ///   calculations. This reduces weight loads from L1 cache/registers by 50% per calculated frame.
    /// - **Instruction-Level Parallelism (ILP)**: Interleaving instructions for two independent frames
    ///   allows the CPU's execution pipelines to hide instruction latencies (e.g., FMA latencies). Instead
    ///   of waiting on dependency chains of a single frame's accumulator, we alternate operations between
    ///   the two frames, saturating execution units.
    /// - **Interleaved Layout**: Weights are stored in a `[OUT/4][KERNEL][IN][4]` layout. This allows the
    ///   underlying SIMD engine to perform 4-lane wide vector dot products on output channels in parallel
    ///   using a single weight vector load.
    /// - **BF16 / VNNI Acceleration**: By utilizing BF16 inputs, memory bandwidth is further halved, and
    ///   vector hardware instructions (like VNNI or AVX-512 BF16) can perform dot products at double the
    ///   throughput of f32.
    /// - **Fallback / Tail Trade-off**: When the total block size `num_frames` is odd, the even pairs
    ///   are processed in dual-frame batches, and the final remaining frame falls back to the single-frame
    ///   variant ([`Self::process_single_frame_bf16`]) which does not benefit from the weight reuse optimization.
    ///
    /// # Safety
    /// - `out_f0` and `out_f1` must have lengths of at least `self.out_ch`.
    /// - `layer_buffer` must contain valid elements for the dilated tap indexes calculated from
    ///   `idx_f0` and `idx_f1`.
    /// - Pointers derived from `layer_buffer` must not violate Rust's aliasing rules (they should be
    ///   disjoint or read-only).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_dual_frame_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_f0: &mut [f32],
        out_f1: &mut [f32],
        idx_f0: usize,
        idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        unsafe {
            self.process_dual_frame_generic::<M, u16>(
                layer_buffer,
                out_f0,
                out_f1,
                idx_f0,
                idx_f1,
                mixin_f0,
                mixin_f1,
            );
        }
    }

    /// Processes a sample block with optional mixin (f32).
    ///
    /// # Safety
    /// `block` must have size at least `num_frames * self.out_ch`.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        unsafe {
            self.process_block_generic::<M, f32>(
                layer_buffer,
                block,
                buffer_start,
                num_frames,
                mixin,
            );
        }
    }

    /// Processes a sample block using BF16.
    ///
    /// # Safety
    /// `block` must have size at least `num_frames * self.out_ch`.
    #[inline(always)]
    pub unsafe fn process_block_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        unsafe {
            self.process_block_generic::<M, u16>(
                layer_buffer,
                block,
                buffer_start,
                num_frames,
                mixin,
            );
        }
    }

    /// Processes a single frame (f32).
    ///
    /// # Safety
    /// `out_frame` must have size compatible with `self.out_ch`.
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        unsafe {
            self.process_single_frame_generic::<M, f32>(layer_buffer, out_frame, frame_idx, mixin);
        }
    }

    /// Processes a single BF16 frame.
    ///
    /// # Safety
    /// `out_frame` must have size compatible with `self.out_ch`.
    #[inline(always)]
    pub unsafe fn process_single_frame_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        unsafe {
            self.process_single_frame_generic::<M, u16>(layer_buffer, out_frame, frame_idx, mixin);
        }
    }

    #[inline(always)]
    unsafe fn load_mixin_4(mixin: Option<&[f32]>, out_c: usize) -> (f32, f32, f32, f32) {
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

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn process_dual_frame_generic<M: SimdMath, T: ConvInput>(
        &self,
        layer_buffer: &[T],
        out_f0: &mut [f32],
        out_f1: &mut [f32],
        idx_f0: usize,
        idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        let num_blocks = self.num_blocks;
        debug_assert!(
            self.weights.len() >= num_blocks * 4 * self.in_ch * self.kernel,
            "weights length {} is less than expected {}",
            self.weights.len(),
            num_blocks * 4 * self.in_ch * self.kernel
        );
        debug_assert!(
            self.kernel <= MAX_KERNEL,
            "kernel {} excede MAX_KERNEL",
            self.kernel
        );
        let mut tap_ptrs_f0 = [core::ptr::null::<T>(); MAX_KERNEL];
        let mut tap_ptrs_f1 = [core::ptr::null::<T>(); MAX_KERNEL];
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
                    T::cast_ptr(*tap_f0),
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
                    let w_slice: &[[u16; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                        core::slice::from_raw_parts(ptr, in_ch)
                    };

                    let in_f0 = core::slice::from_raw_parts(tap_f0, in_ch);
                    let in_f1 = core::slice::from_raw_parts(tap_f1, in_ch);

                    let (t_f0, t_f1) =
                        T::dot_product_4x_interleaved_dual_frame::<M>(w_slice, in_f0, in_f1);

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

    #[inline(always)]
    unsafe fn process_single_frame_generic<M: SimdMath, T: ConvInput>(
        &self,
        layer_buffer: &[T],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.num_blocks;
        debug_assert!(
            self.kernel <= MAX_KERNEL,
            "kernel {} excede MAX_KERNEL",
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
    unsafe fn process_block_generic<M: SimdMath, T: ConvInput>(
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
}
