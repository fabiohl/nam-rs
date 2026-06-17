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
use super::conv_input::{init_accum_with_bias_mixin, load_4_accums, store_4_accums};
use crate::math::common::{AlignedVec, PrefetchFn, SimdMath};

/// Dilated Causal Convolution (WaveNet Conv1D).
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Flattened weight matrix of size OUT * K * IN.
    pub weights: AlignedVec<u16>,
    /// Optional full-precision f32 weights for high-fidelity mode.
    /// When present, `process_*_f32_native` methods use these directly,
    /// bypassing quantization drift entirely at the cost of 2× memory bandwidth.
    #[cfg(feature = "high-fidelity")]
    pub f32_weights: AlignedVec<f32>,
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
    /// Executes causal convolution over a flat bidirectional array (`layer_buffer`).
    ///
    /// ## Optimization: Proactive Software Prefetch
    ///
    /// For large dilations (256, 512), accesses to `layer_buffer` skip
    /// thousands of floats between consecutive kernel taps, causing predictable
    /// L1 cache misses. The `_mm_prefetch` issued for the **next tap**
    /// while the current tap is processed via FMA allows the memory subsystem
    /// to proactively bring in the cache line — cost of 1 cycle (masked by the
    /// FMA pipeline), benefit of ~5–10% latency in layers with high dilation.
    ///
    /// # Safety
    /// Dynamically depends on the `SimdMath` trait provided.
    ///
    /// Processes a single frame applying convolution to the ring buffer (optimized via FMA 4x).
    #[cfg(test)]
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
    ) {
        unsafe {
            self.process_single_frame_internal::<M>(layer_buffer, out_frame, frame_idx, None);
        }
    }

    #[inline(always)]
    /// Fused variant that adds a Mixin vector (conditioning) directly to the accumulator.
    /// Sums the mixin and processes Conv1D for a single frame.
    ///
    /// # Safety
    /// The caller must guarantee that `frame_idx` and `mixin` are valid.
    pub unsafe fn process_single_frame_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: &[f32],
    ) {
        unsafe {
            self.process_single_frame_internal::<M>(
                layer_buffer,
                out_frame,
                frame_idx,
                Some(mixin),
            );
        }
    }

    #[inline(always)]
    unsafe fn process_single_frame_internal<M: SimdMath>(
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

    #[inline(always)]
    unsafe fn process_single_frame_generic<M: SimdMath, T: ConvInput>(
        &self,
        layer_buffer: &[T],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        // [STEP 1: Accumulator Initialization]
        let full_blocks = OUT & !3;
        for i in (0..full_blocks).step_by(4) {
            // SAFETY: `i + 4 <= OUT` since `i` ranges up to `full_blocks` in steps of 4, where `full_blocks = OUT & !3`.
            // The slice pointer is well-aligned and points to a valid sequence of 4 floats, which aligns with `[f32; 4]`.
            // The lifetime is correctly constrained by the borrow of `out_frame`.
            let acc: &mut [f32; 4] =
                unsafe { &mut *(out_frame.as_mut_ptr().add(i) as *mut [f32; 4]) };
            unsafe {
                init_accum_with_bias_mixin::<M>(acc, &self.bias, mixin, i, self.do_bias);
            }
        }
        let rem = OUT & 3;
        if rem > 0 {
            let i = full_blocks;
            let rem_slice = &mut out_frame[i..OUT];
            if let Some(m) = mixin {
                if self.do_bias {
                    rem_slice.copy_from_slice(&self.bias[i..OUT]);
                    unsafe {
                        M::accumulate_head(rem_slice, &m[i..OUT]);
                    }
                } else {
                    rem_slice.copy_from_slice(&m[i..OUT]);
                }
            } else if self.do_bias {
                rem_slice.copy_from_slice(&self.bias[i..OUT]);
            } else {
                rem_slice.fill(0.0);
            }
        }

        // [STEP 2: Kernel Iteration (Receptive Field)]
        // Loop Inversion: Channel-First Tiling.
        // Process all taps (K) for one output channel block before moving to the next.
        // This keeps accumulators in SIMD registers, reducing L1 cache traffic.

        // Pre-loading taps (Input data) for the current block.
        // Since K and IN are small (e.g., 3 and 16), the stack copy cost is compensated
        // by the elimination of address recomputation and better locality in the b-first loop.
        let mut in_taps = [[T::default(); IN]; K];
        for (k, in_tap) in in_taps.iter_mut().enumerate() {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let in_slice_start = ((frame_idx as isize) + offset) as usize * IN;
            unsafe {
                in_tap.copy_from_slice(
                    layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN),
                );
            }

            // Prefetch via pre-computed strategy (Branchless)
            unsafe {
                (self.prefetch_fn)(
                    T::cast_ptr(layer_buffer.as_ptr().add(in_slice_start)),
                    self.dilation * IN,
                    k,
                    K,
                    self.dilation,
                );
            }
        }

        // Interleaved 1D Convolution Processing by Blocks:
        // To optimize computation throughput and cache usage, we process output channels
        // grouped in blocks of 4 elements. This enables computing 4 outputs in parallel using
        // SIMD instructions that read weights and inputs in a highly combined fashion.
        let num_blocks = OUT.div_ceil(4);
        let mut out_c = 0;

        for b in 0..num_blocks {
            // Load the 4 temporary accumulators from the current output frame.
            let [mut r0, mut r1, mut r2, mut r3] = unsafe { load_4_accums(out_frame, out_c, OUT) };

            for (k, in_slice) in in_taps.iter().enumerate() {
                let w_start = (b * K + k) * IN * 4;
                let w_slice: &[[u16; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                // Performs the 4-channel interleaved dot product at once.
                let [t0, t1, t2, t3] =
                    unsafe { T::dot_product_4x_interleaved::<M>(w_slice, in_slice) };
                r0 += t0;
                r1 += t1;
                r2 += t2;
                r3 += t3;
            }

            unsafe { store_4_accums(out_frame, out_c, [r0, r1, r2, r3], OUT) };
            out_c += 4;
        }
    }

    /// Processes a single frame using BF16 buffers (VNNI).
    ///
    /// # Safety
    /// The caller must guarantee that `layer_buffer` and `out_frame` have sizes
    /// compatible with the layer's `IN` and `OUT` dimensions, and that the
    /// SIMD instructions requested by the dispatcher `M` are available.
    #[inline(always)]
    pub unsafe fn process_single_frame_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame: &mut [f32],
        frame_idx: usize,
    ) {
        unsafe {
            self.process_single_frame_bf16_internal::<M>(layer_buffer, out_frame, frame_idx, None);
        }
    }

    #[inline(always)]
    /// Fused BF16 variant that adds a Mixin vector directly to the accumulator.
    /// Sums the mixin and processes Conv1D (BF16) for a single frame.
    ///
    /// # Safety
    /// The caller must guarantee that `frame_idx` and `mixin` are valid.
    pub unsafe fn process_single_frame_bf16_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: &[f32],
    ) {
        unsafe {
            self.process_single_frame_bf16_internal::<M>(
                layer_buffer,
                out_frame,
                frame_idx,
                Some(mixin),
            );
        }
    }

    #[inline(always)]
    unsafe fn process_single_frame_bf16_internal<M: SimdMath>(
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
            // [STEP: Per-Frame Delegation]
            // Slice the output buffer (multi-channel output of size `OUT`) and dispatch to computation.
            let out_frame = unsafe { block.get_unchecked_mut(i * OUT..i * OUT + OUT) };
            unsafe {
                self.process_single_frame::<M>(layer_buffer, out_frame, buffer_start + i);
            }
        }
    }

    /// F32-native single-frame convolution with mixin (high-fidelity mode).
    ///
    /// Uses `self.f32_weights` directly, bypassing F16/BF16 quantization drift.
    /// Same layout and algorithm as the quantized path, but with full-precision weights.
    ///
    /// # Safety
    /// The caller must guarantee that `frame_idx`, `mixin`, `layer_buffer`,
    /// and `out_frame` have sizes compatible with the layer dimensions.
    #[cfg(feature = "high-fidelity")]
    #[inline(always)]
    pub unsafe fn process_single_frame_f32_native_with_mixin(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: &[f32],
    ) {
        use super::conv_input::dot_product_4x_f32;

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

        for b in 0..num_blocks {
            let [mut r0, mut r1, mut r2, mut r3] =
                unsafe { super::conv_input::load_4_accums(out_frame, out_c, OUT) };

            for (k, in_slice) in in_taps.iter().enumerate() {
                let w_start = (b * K + k) * IN * 4;
                let w_slice: &[[f32; 4]] = unsafe {
                    let ptr = self.f32_weights.as_ptr().add(w_start) as *const [f32; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                let [t0, t1, t2, t3] = dot_product_4x_f32(w_slice, in_slice);
                r0 += t0;
                r1 += t1;
                r2 += t2;
                r3 += t3;
            }

            unsafe {
                super::conv_input::store_4_accums(out_frame, out_c, [r0, r1, r2, r3], OUT);
            }
            out_c += 4;
        }
    }
}
