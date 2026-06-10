// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Fully unrolled CH=3 GEMV kernels for the A2 fast-path (T2.1).
//!
//! When `out_ch == 3`, the generic 4-wide interleaved scheme wastes 25% of SIMD
//! lanes (lane 3 accumulates zero weights). These kernels completely unroll the
//! convolution loops — no `for k in 0..kernel`, no `for i in 0..in_ch` — using
//! compile-time constant indexing for optimal instruction scheduling.
//!
//! Supports kernel sizes 6 and 15 (the only A2 kernel sizes).
//!
//! ## Source of truth
//! - `a2_fast.cpp` (strategy `Channels=3`, GEMV unrolled).

use crate::models::wavenet::conv1d_dyn::Conv1dDyn;
use core::arch::x86_64::*;

impl Conv1dDyn {
    /// Fully unrolled CH=3 GEMV — replaces `process_single_frame_generic` when `out_ch == 3`.
    ///
    /// Each (tap, input_channel) pair is a single `_mm_fmadd_ps` instruction,
    /// loading 4 interleaved weights (lanes 0-2 valid, lane 3 = zero) and
    /// broadcasting one state element.
    #[inline(always)]
    pub(crate) unsafe fn process_single_ch3_unrolled(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert_eq!(self.out_ch, 3);
        debug_assert!(self.kernel == 6 || self.kernel == 15);

        match self.kernel {
            6 => self.process_single_ch3_k6(layer_buffer, out_frame, frame_idx, mixin),
            15 => self.process_single_ch3_k15(layer_buffer, out_frame, frame_idx, mixin),
            _ => unreachable!(),
        }
    }

    /// Unrolled K=6 GEMV for 3-channel input/output.
    ///
    /// 6 taps × 3 input channels = 18 `_mm_fmadd_ps` instructions.
    /// Uses `#[target_feature(enable = "f16c")]` for `_mm_cvtph_ps`.
    #[target_feature(enable = "f16c")]
    unsafe fn process_single_ch3_k6(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        let in_ch = self.in_ch;
        let d = self.dilation as isize;
        let k_limit = self.kernel as isize; // 6

        unsafe {
            let buf = layer_buffer.as_ptr();

            // Pre-compute tap base indices (column * in_ch).
            // Generic kernel formula: offset = dilation * (k + 1 - kernel)
            let t0 = ((frame_idx as isize) + d * (1_isize - k_limit)) as usize * in_ch;
            let t1 = ((frame_idx as isize) + d * (2_isize - k_limit)) as usize * in_ch;
            let t2 = ((frame_idx as isize) + d * (3_isize - k_limit)) as usize * in_ch;
            let t3 = ((frame_idx as isize) + d * (4_isize - k_limit)) as usize * in_ch;
            let t4 = ((frame_idx as isize) + d * (5_isize - k_limit)) as usize * in_ch;
            let t5 = ((frame_idx as isize) + d * (6_isize - k_limit)) as usize * in_ch;

            // Initialize accumulators with bias + mixin.
            let (b0, b1, b2, b3) = Self::load_mixin_4(mixin, 0);
            let mut acc = if self.do_bias {
                _mm_setr_ps(
                    *self.bias.get_unchecked(0) + b0,
                    *self.bias.get_unchecked(1) + b1,
                    *self.bias.get_unchecked(2) + b2,
                    b3, // lane 3: no bias (out_ch == 3), just mixin or 0
                )
            } else {
                _mm_setr_ps(b0, b1, b2, b3)
            };

            let w_ptr = self.weights.as_ptr();

            // ── K=6 unrolled: 6 taps × 3 input channels = 18 FMAs ────────────
            // Uses a macro to generate identical instruction sequences,
            // eliminating loop overhead completely.
            macro_rules! fma3_k6 {
                ($tap_base:ident, $k:expr) => {
                    // i=0
                    {
                        let wp = w_ptr.add(($k * in_ch + 0) * 4);
                        let wv = _mm_cvtph_ps(_mm_loadl_epi64(wp as *const __m128i));
                        let sv = _mm_set1_ps(*buf.add($tap_base + 0));
                        acc = _mm_fmadd_ps(wv, sv, acc);
                    }
                    // i=1
                    {
                        let wp = w_ptr.add(($k * in_ch + 1) * 4);
                        let wv = _mm_cvtph_ps(_mm_loadl_epi64(wp as *const __m128i));
                        let sv = _mm_set1_ps(*buf.add($tap_base + 1));
                        acc = _mm_fmadd_ps(wv, sv, acc);
                    }
                    // i=2
                    {
                        let wp = w_ptr.add(($k * in_ch + 2) * 4);
                        let wv = _mm_cvtph_ps(_mm_loadl_epi64(wp as *const __m128i));
                        let sv = _mm_set1_ps(*buf.add($tap_base + 2));
                        acc = _mm_fmadd_ps(wv, sv, acc);
                    }
                };
            }

            fma3_k6!(t0, 0);
            fma3_k6!(t1, 1);
            fma3_k6!(t2, 2);
            fma3_k6!(t3, 3);
            fma3_k6!(t4, 4);
            fma3_k6!(t5, 5);

            _mm_storeu_ps(out_frame.as_mut_ptr(), acc);
        }
    }

    /// Unrolled K=15 GEMV for 3-channel input/output.
    ///
    /// 15 taps × 3 input channels = 45 `_mm_fmadd_ps` instructions.
    #[target_feature(enable = "f16c")]
    unsafe fn process_single_ch3_k15(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        let in_ch = self.in_ch;
        let d = self.dilation as isize;
        let k_limit = self.kernel as isize; // 15

        unsafe {
            let buf = layer_buffer.as_ptr();

            // Pre-compute all 15 tap base indices.
            // Generic kernel formula: offset = dilation * (k + 1 - kernel)
            macro_rules! tap {
                ($idx:expr) => {
                    ((frame_idx as isize) + d * (($idx as isize) + 1 - k_limit)) as usize * in_ch
                };
            }

            let t0 = tap!(0);
            let t1 = tap!(1);
            let t2 = tap!(2);
            let t3 = tap!(3);
            let t4 = tap!(4);
            let t5 = tap!(5);
            let t6 = tap!(6);
            let t7 = tap!(7);
            let t8 = tap!(8);
            let t9 = tap!(9);
            let t10 = tap!(10);
            let t11 = tap!(11);
            let t12 = tap!(12);
            let t13 = tap!(13);
            let t14 = tap!(14);

            // Initialize accumulators with bias + mixin.
            let (b0, b1, b2, b3) = Self::load_mixin_4(mixin, 0);
            let mut acc = if self.do_bias {
                _mm_setr_ps(
                    *self.bias.get_unchecked(0) + b0,
                    *self.bias.get_unchecked(1) + b1,
                    *self.bias.get_unchecked(2) + b2,
                    b3,
                )
            } else {
                _mm_setr_ps(b0, b1, b2, b3)
            };

            let w_ptr = self.weights.as_ptr();

            // ── K=15 unrolled: 15 taps × 3 input channels = 45 FMAs ───────────
            macro_rules! fma3_k15 {
                ($tap_base:ident, $k:expr) => {
                    // i=0
                    {
                        let wp = w_ptr.add(($k * in_ch + 0) * 4);
                        let wv = _mm_cvtph_ps(_mm_loadl_epi64(wp as *const __m128i));
                        let sv = _mm_set1_ps(*buf.add($tap_base + 0));
                        acc = _mm_fmadd_ps(wv, sv, acc);
                    }
                    // i=1
                    {
                        let wp = w_ptr.add(($k * in_ch + 1) * 4);
                        let wv = _mm_cvtph_ps(_mm_loadl_epi64(wp as *const __m128i));
                        let sv = _mm_set1_ps(*buf.add($tap_base + 1));
                        acc = _mm_fmadd_ps(wv, sv, acc);
                    }
                    // i=2
                    {
                        let wp = w_ptr.add(($k * in_ch + 2) * 4);
                        let wv = _mm_cvtph_ps(_mm_loadl_epi64(wp as *const __m128i));
                        let sv = _mm_set1_ps(*buf.add($tap_base + 2));
                        acc = _mm_fmadd_ps(wv, sv, acc);
                    }
                };
            }

            fma3_k15!(t0, 0);
            fma3_k15!(t1, 1);
            fma3_k15!(t2, 2);
            fma3_k15!(t3, 3);
            fma3_k15!(t4, 4);
            fma3_k15!(t5, 5);
            fma3_k15!(t6, 6);
            fma3_k15!(t7, 7);
            fma3_k15!(t8, 8);
            fma3_k15!(t9, 9);
            fma3_k15!(t10, 10);
            fma3_k15!(t11, 11);
            fma3_k15!(t12, 12);
            fma3_k15!(t13, 13);
            fma3_k15!(t14, 14);

            _mm_storeu_ps(out_frame.as_mut_ptr(), acc);
        }
    }
}

#[cfg(test)]
#[path = "conv1d_ch3_test.rs"]
mod tests;
