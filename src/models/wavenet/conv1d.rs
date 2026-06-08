// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Static Causal CNN Mesh for WaveNet Inference (Data-Oriented Design, SoA).
//!
//! All structures use `Const Generics` for mathematical dimensions and pre-allocated vectors
//! ensuring a strict instantiation policy (Zero-Allocation during processing).
//! Dynamic loops resolve computations in deterministic FMA sequences via AVX2.

pub(crate) use super::conv_input::ConvInput;
use crate::math::common::{AlignedVec, PrefetchFn, SimdMath, kahan_add};

/// Dilated Causal Convolution (WaveNet Conv1D).
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Flattened weight matrix of size OUT * K * IN.
    pub weights: AlignedVec<u16>,
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
        if let Some(m) = mixin {
            if self.do_bias {
                out_frame.copy_from_slice(&self.bias[0..OUT]);
                unsafe {
                    M::accumulate_head(out_frame, m);
                }
            } else {
                out_frame.copy_from_slice(m);
            }
        } else if self.do_bias {
            out_frame.copy_from_slice(&self.bias[0..OUT]);
        } else {
            out_frame.fill(0.0);
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
            let mut r0;
            let mut r1;
            let mut r2;
            let mut r3;

            // Load the 4 temporary accumulators from the current output frame.
            unsafe {
                r0 = *out_frame.get_unchecked(out_c);
                if OUT.is_multiple_of(4) || out_c + 3 < OUT {
                    r1 = *out_frame.get_unchecked(out_c + 1);
                    r2 = *out_frame.get_unchecked(out_c + 2);
                    r3 = *out_frame.get_unchecked(out_c + 3);
                } else {
                    r1 = if out_c + 1 < OUT {
                        *out_frame.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    };
                    r2 = if out_c + 2 < OUT {
                        *out_frame.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    };
                    r3 = if out_c + 3 < OUT {
                        *out_frame.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    };
                }
            }

            // Kahan compensation variables: track lost low-order bits per channel
            // to bound the per-tap accumulation error to O(eps) instead of O(K·eps).
            let mut c0 = 0.0f32;
            let mut c1 = 0.0f32;
            let mut c2 = 0.0f32;
            let mut c3 = 0.0f32;

            // For each tap (delay/offset in the circular audio buffer) of the convolution
            for (k, in_slice) in in_taps.iter().enumerate() {
                let w_start = (b * K + k) * IN * 4;
                let w_slice: &[[u16; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                // Performs the 4-channel interleaved dot product at once.
                let [t0, t1, t2, t3] =
                    unsafe { T::dot_product_4x_interleaved::<M>(w_slice, in_slice) };
                // Kahan compensated accumulation per channel
                let (s, c) = kahan_add(r0, c0, t0);
                r0 = s;
                c0 = c;
                let (s, c) = kahan_add(r1, c1, t1);
                r1 = s;
                c1 = c;
                let (s, c) = kahan_add(r2, c2, t2);
                r2 = s;
                c2 = c;
                let (s, c) = kahan_add(r3, c3, t3);
                r3 = s;
                c3 = c;
            }

            // Write back the 4 processed accumulators to the output buffer in-place.
            unsafe {
                *out_frame.get_unchecked_mut(out_c) = r0;
                if OUT.is_multiple_of(4) || out_c + 3 < OUT {
                    *out_frame.get_unchecked_mut(out_c + 1) = r1;
                    *out_frame.get_unchecked_mut(out_c + 2) = r2;
                    *out_frame.get_unchecked_mut(out_c + 3) = r3;
                } else {
                    if out_c + 1 < OUT {
                        *out_frame.get_unchecked_mut(out_c + 1) = r1;
                    }
                    if out_c + 2 < OUT {
                        *out_frame.get_unchecked_mut(out_c + 2) = r2;
                    }
                    if out_c + 3 < OUT {
                        *out_frame.get_unchecked_mut(out_c + 3) = r3;
                    }
                }
            }
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
}
