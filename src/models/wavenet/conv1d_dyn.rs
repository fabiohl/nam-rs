// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Common and dynamic components for WaveNet architectures.
//!
//! Contains the fundamental structures (Conv1D, Dense, Layer) that operate with
//! runtime-defined dimensions, serving as a foundation for the dynamic model
//! and future A2 architecture stages.

use crate::math::common::{AlignedVec, PrefetchFn, SimdMath};

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
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
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
