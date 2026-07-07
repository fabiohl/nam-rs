// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Native Complex FFT — Radix-2 Decimation-in-Time (DIT).
//!
//! Generic over `f32` and `f64`. All twiddle factors and bit-reversal
//! tables are pre-computed at construction time, so the [`FftPlanner::process`]
//! routine operates on SoA (Struct-of-Arrays) buffers with zero heap
//! allocations — safe for real-time audio threads.
//!
//! # Algorithm
//!
//! Iterative Cooley-Tukey DIT Radix-2. Butterflies use `mul_add` (FMA)
//! for improved numerical accuracy.
//!
//! For `f32` processing, stages where `half >= 8` are accelerated via
//! AVX2/AVX-512 SIMD butterfly kernels, dispatched at runtime through
//! the [`dispatch_simd`] macro.

use crate::dispatch_simd;
use std::f64::consts::TAU;
use std::ops::{Add, Mul, Neg, Sub};

/// Minimal float abstraction for `f32` and `f64` FFT operations.
pub trait FftFloat:
    Copy
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
{
    /// Returns the mathematical constant τ = 2π.
    fn tau() -> Self;

    /// Converts a `usize` to this float type.
    fn from_usize(n: usize) -> Self;

    /// Computes the sine of `self` (radians).
    fn sin(self) -> Self;

    /// Computes the cosine of `self` (radians).
    fn cos(self) -> Self;

    /// Returns the reciprocal `1 / self`.
    fn recip(self) -> Self;

    /// Fused multiply-add: `self * a + b`.
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl FftFloat for f32 {
    #[inline]
    fn tau() -> Self {
        core::f32::consts::TAU
    }

    #[inline]
    fn from_usize(n: usize) -> Self {
        n as f32
    }

    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        f32::cos(self)
    }

    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }
}

impl FftFloat for f64 {
    #[inline]
    fn tau() -> Self {
        TAU
    }

    #[inline]
    fn from_usize(n: usize) -> Self {
        n as f64
    }

    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }

    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
}

/// Pre-computed complex FFT plan (Radix-2 DIT).
///
/// Construction (`new`) allocates bit-reversal and twiddle-factor tables.
/// The [`process`](Self::process) method performs the in-place transform
/// without any further heap allocations.
pub struct FftPlanner<T: FftFloat> {
    n: usize,
    bit_reverse: Vec<usize>,
    twiddle_re: Vec<T>,
    twiddle_im: Vec<T>,
    stage_twiddle_re: Vec<T>,
    stage_twiddle_im: Vec<T>,
    stage_offsets: Vec<usize>,
}

impl<T: FftFloat> FftPlanner<T> {
    /// Creates a new FFT plan for size `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not a power of two or is zero.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "FFT size must be positive");
        assert!(
            n.is_power_of_two(),
            "FFT size must be a power of two, got {n}"
        );

        let n_half = n / 2;

        // --- Bit-reversal lookup table ---
        let mut bit_reverse = vec![0usize; n];
        let mut j = 0usize;
        for entry in bit_reverse.iter_mut().skip(1) {
            let mut bit = n_half;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            *entry = j;
        }

        // --- Twiddle factors: W_N^k = e^{-2πi k / N} for k = 0 .. n/2 ---
        let tau = T::tau();
        let n_t = T::from_usize(n);
        let mut twiddle_re = Vec::with_capacity(n_half);
        let mut twiddle_im = Vec::with_capacity(n_half);
        for k in 0..n_half {
            let angle = tau * T::from_usize(k) * n_t.recip();
            twiddle_re.push(angle.cos());
            twiddle_im.push(-angle.sin());
        }

        // --- Stage-reordered twiddle tables for SIMD ---
        let num_stages = n.ilog2() as usize;
        let total = n.saturating_sub(1);
        let mut stage_twiddle_re = Vec::with_capacity(total);
        let mut stage_twiddle_im = Vec::with_capacity(total);
        let mut stage_offsets = Vec::with_capacity(num_stages);

        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            stage_offsets.push(stage_twiddle_re.len());
            for j in 0..half {
                let w_idx = j * step;
                stage_twiddle_re.push(twiddle_re[w_idx]);
                stage_twiddle_im.push(twiddle_im[w_idx]);
            }
            len <<= 1;
        }

        Self {
            n,
            bit_reverse,
            twiddle_re,
            twiddle_im,
            stage_twiddle_re,
            stage_twiddle_im,
            stage_offsets,
        }
    }

    /// Returns the FFT size.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the size is zero (never, guarded at construction).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    const AVX2_SIMD_WIDTH: usize = 8;

    /// Performs the forward complex FFT **in-place** on SoA buffers.
    ///
    /// Buffers `re` and `im` must each have length `n`.
    /// No heap allocations occur inside this method.
    pub fn process(&self, re: &mut [T], im: &mut [T]) {
        debug_assert_eq!(re.len(), self.n, "re length mismatch");
        debug_assert_eq!(im.len(), self.n, "im length mismatch");

        let n = self.n;

        // 1. Bit-reversal permutation
        // SAFETY: bit_reverse has length n, and i ∈ 0..n.
        for i in 0..n {
            let j = unsafe { *self.bit_reverse.get_unchecked(i) };
            if i < j {
                // SAFETY: i < j < n by bit-reversal property; both indices are in bounds.
                unsafe {
                    std::ptr::swap(re.get_unchecked_mut(i), re.get_unchecked_mut(j));
                    std::ptr::swap(im.get_unchecked_mut(i), im.get_unchecked_mut(j));
                }
            }
        }

        // 2. Iterative DIT Radix-2 butterflies
        self.run_butterflies(re, im, false);
    }

    /// Performs the inverse complex FFT **in-place** on SoA buffers.
    ///
    /// Uses conjugated twiddle factors and applies `1/n` scaling at the
    /// end so that `process_inverse(process_forward(x)) == x`.
    ///
    /// Buffers `re` and `im` must each have length `n`.
    /// No heap allocations occur inside this method.
    pub fn process_inverse(&self, re: &mut [T], im: &mut [T]) {
        debug_assert_eq!(re.len(), self.n, "re length mismatch");
        debug_assert_eq!(im.len(), self.n, "im length mismatch");

        let n = self.n;

        // 1. Bit-reversal permutation
        // SAFETY: bit_reverse has length n, and i ∈ 0..n.
        for i in 0..n {
            let j = unsafe { *self.bit_reverse.get_unchecked(i) };
            if i < j {
                // SAFETY: i < j < n by bit-reversal property; both indices are in bounds.
                unsafe {
                    std::ptr::swap(re.get_unchecked_mut(i), re.get_unchecked_mut(j));
                    std::ptr::swap(im.get_unchecked_mut(i), im.get_unchecked_mut(j));
                }
            }
        }

        // 2. Iterative DIT Radix-2 butterflies with conjugated twiddle factors
        self.run_butterflies(re, im, true);

        // 3. Scale by 1/n
        let scale = T::from_usize(n).recip();
        for sample in re.iter_mut() {
            *sample = *sample * scale;
        }
        for sample in im.iter_mut() {
            *sample = *sample * scale;
        }
    }

    /// Runs the butterfly stages, dispatching to SIMD when T=f32 and
    /// `half >= SIMD_WIDTH`.
    #[inline]
    fn run_butterflies(&self, re: &mut [T], im: &mut [T], inverse: bool) {
        let n = self.n;
        let mut len = 2;
        let mut stage_idx = 0usize;
        while len <= n {
            let half = len / 2;
            let step = n / len;

            // SIMD path for f32 when half is large enough
            if core::mem::size_of::<T>() == core::mem::size_of::<f32>()
                && half >= Self::AVX2_SIMD_WIDTH
            {
                // SAFETY: T is f32 (guarded by size_of check above).
                // stage_idx < num_stages, twiddle offsets are precomputed.
                let tw_offset = *unsafe { self.stage_offsets.get_unchecked(stage_idx) };
                let tw_re_ptr =
                    unsafe { self.stage_twiddle_re.as_ptr().add(tw_offset) as *const f32 };
                let tw_im_ptr =
                    unsafe { self.stage_twiddle_im.as_ptr().add(tw_offset) as *const f32 };
                let re_ptr = re.as_mut_ptr() as *mut f32;
                let im_ptr = im.as_mut_ptr() as *mut f32;

                for k in (0..n).step_by(len) {
                    // SAFETY: re_ptr, im_ptr, tw_re_ptr, tw_im_ptr are
                    // valid; k is a group start; half butterflies fit.
                    // CPU ISA is verified by dispatch_simd! runtime check.
                    dispatch_simd!(fft_butterfly_stage(
                        re_ptr, im_ptr, half, tw_re_ptr, tw_im_ptr, k, inverse
                    ));
                }
            } else {
                // Scalar path for f64 or small half
                for k in (0..n).step_by(len) {
                    for j in 0..half {
                        let w_idx = j * step;
                        // SAFETY: w_idx < n/2; twiddle tables have capacity n/2.
                        let w_re = unsafe { *self.twiddle_re.get_unchecked(w_idx) };
                        let w_im = if inverse {
                            -unsafe { *self.twiddle_im.get_unchecked(w_idx) }
                        } else {
                            unsafe { *self.twiddle_im.get_unchecked(w_idx) }
                        };

                        let idx1 = k + j;
                        let idx2 = k + j + half;
                        // SAFETY: k < n, j < half, len ≤ n → idx1, idx2 < n.
                        let (re_idx2, im_idx2, re_idx1, im_idx1) = unsafe {
                            let r2 = *re.get_unchecked(idx2);
                            let i2 = *im.get_unchecked(idx2);
                            let r1 = *re.get_unchecked(idx1);
                            let i1 = *im.get_unchecked(idx1);
                            (r2, i2, r1, i1)
                        };

                        let t_re = w_re.mul_add(re_idx2, -w_im * im_idx2);
                        let t_im = w_re.mul_add(im_idx2, w_im * re_idx2);

                        unsafe {
                            *re.get_unchecked_mut(idx2) = re_idx1 - t_re;
                            *im.get_unchecked_mut(idx2) = im_idx1 - t_im;
                            *re.get_unchecked_mut(idx1) = re_idx1 + t_re;
                            *im.get_unchecked_mut(idx1) = im_idx1 + t_im;
                        }
                    }
                }
            }

            len <<= 1;
            stage_idx += 1;
        }
    }
}

pub use crate::math::dsp::rfft::RfftPlanner;

#[cfg(test)]
#[path = "fft_test.rs"]
mod tests;
