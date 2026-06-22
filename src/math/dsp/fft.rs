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

        Self {
            n,
            bit_reverse,
            twiddle_re,
            twiddle_im,
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

    /// Performs the forward complex FFT **in-place** on SoA buffers.
    ///
    /// Buffers `re` and `im` must each have length `n`.
    /// No heap allocations occur inside this method.
    pub fn process(&self, re: &mut [T], im: &mut [T]) {
        assert_eq!(re.len(), self.n, "re length mismatch");
        assert_eq!(im.len(), self.n, "im length mismatch");

        let n = self.n;

        // 1. Bit-reversal permutation
        for i in 0..n {
            let j = self.bit_reverse[i];
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        // 2. Iterative DIT Radix-2 butterflies
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for k in (0..n).step_by(len) {
                for j in 0..half {
                    let w_idx = j * step;
                    // SAFETY: w_idx < n/2 ≤ twiddle capacity (guaranteed by construction)
                    let w_re = self.twiddle_re[w_idx];
                    let w_im = self.twiddle_im[w_idx];

                    let idx1 = k + j;
                    let idx2 = k + j + half;

                    let t_re = w_re.mul_add(re[idx2], -w_im * im[idx2]);
                    let t_im = w_re.mul_add(im[idx2], w_im * re[idx2]);

                    re[idx2] = re[idx1] - t_re;
                    im[idx2] = im[idx1] - t_im;
                    re[idx1] = re[idx1] + t_re;
                    im[idx1] = im[idx1] + t_im;
                }
            }
            len <<= 1;
        }
    }

    /// Performs the inverse complex FFT **in-place** on SoA buffers.
    ///
    /// Uses conjugated twiddle factors and applies `1/n` scaling at the
    /// end so that `process_inverse(process_forward(x)) == x`.
    ///
    /// Buffers `re` and `im` must each have length `n`.
    /// No heap allocations occur inside this method.
    pub fn process_inverse(&self, re: &mut [T], im: &mut [T]) {
        assert_eq!(re.len(), self.n, "re length mismatch");
        assert_eq!(im.len(), self.n, "im length mismatch");

        let n = self.n;

        // 1. Bit-reversal permutation
        for i in 0..n {
            let j = self.bit_reverse[i];
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        // 2. Iterative DIT Radix-2 butterflies with conjugated twiddle factors
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for k in (0..n).step_by(len) {
                for j in 0..half {
                    let w_idx = j * step;
                    let w_re = self.twiddle_re[w_idx];
                    let w_im = -self.twiddle_im[w_idx]; // conjugate

                    let idx1 = k + j;
                    let idx2 = k + j + half;

                    let t_re = w_re.mul_add(re[idx2], -w_im * im[idx2]);
                    let t_im = w_re.mul_add(im[idx2], w_im * re[idx2]);

                    re[idx2] = re[idx1] - t_re;
                    im[idx2] = im[idx1] - t_im;
                    re[idx1] = re[idx1] + t_re;
                    im[idx1] = im[idx1] + t_im;
                }
            }
            len <<= 1;
        }

        // 3. Scale by 1/n
        let scale = T::from_usize(n).recip();
        for sample in re.iter_mut() {
            *sample = *sample * scale;
        }
        for sample in im.iter_mut() {
            *sample = *sample * scale;
        }
    }
}

/// Pre-computed real-to-complex FFT plan (Radix-2 DIT).
///
/// Converts a purely real buffer of size `N` into a compact complex
/// spectrum of size `N/2 + 1` using an `N/2`-point complex FFT followed
/// by an O(N) Hermitian-symmetry unpacking step.
///
/// Construction (`new`) allocates the inner half-size complex FFT plan,
/// post-processing twiddle factors, and scratch buffers.
/// [`process_forward`](Self::process_forward) performs the transform
/// reusing the pre-allocated scratch space — safe for real-time audio
/// threads.
pub struct RfftPlanner<T: FftFloat> {
    n: usize,
    fft_n2: FftPlanner<T>,
    post_twiddle_re: Vec<T>,
    post_twiddle_im: Vec<T>,
    scratch_re: Vec<T>,
    scratch_im: Vec<T>,
}

impl<T: FftFloat> RfftPlanner<T> {
    /// Creates a new RFFT plan for a real input of size `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not a power of two, is zero, or is not even
    /// (guaranteed by the power-of-two check).
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "RFFT size must be positive");
        assert!(
            n.is_power_of_two(),
            "RFFT size must be a power of two, got {n}"
        );

        let n_half = n / 2;
        let fft_n2 = FftPlanner::new(n_half);

        let tau = T::tau();
        let n_t = T::from_usize(n);
        let mut post_twiddle_re = Vec::with_capacity(n_half + 1);
        let mut post_twiddle_im = Vec::with_capacity(n_half + 1);
        for k in 0..=n_half {
            let angle = tau * T::from_usize(k) * n_t.recip();
            post_twiddle_re.push(angle.cos());
            post_twiddle_im.push(-angle.sin());
        }

        let scratch_re = vec![T::from_usize(0); n_half];
        let scratch_im = vec![T::from_usize(0); n_half];

        Self {
            n,
            fft_n2,
            post_twiddle_re,
            post_twiddle_im,
            scratch_re,
            scratch_im,
        }
    }

    /// Returns the original real input size `N`.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the size is zero (never, guarded at construction).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Real-to-complex forward FFT.
    ///
    /// Given a purely-real input `input` of length `N`, writes the
    /// non-redundant half of the complex spectrum (size `N/2 + 1`) into
    /// `out_re` and `out_im`.
    ///
    /// Uses pre-allocated scratch buffers internally — no heap
    /// allocations occur inside this method.
    ///
    /// # Panics
    ///
    /// Panics if `input` length ≠ `N`, or if `out_re` / `out_im` length ≠ `N/2 + 1`.
    pub fn process_forward(&mut self, input: &[T], out_re: &mut [T], out_im: &mut [T]) {
        let n = self.n;
        let n_half = n / 2;
        let expected = n_half + 1;
        assert_eq!(input.len(), n, "input length mismatch");
        assert_eq!(out_re.len(), expected, "out_re length mismatch");
        assert_eq!(out_im.len(), expected, "out_im length mismatch");

        // 1. Pack even/odd samples into scratch complex array of size N/2
        for i in 0..n_half {
            self.scratch_re[i] = input[2 * i];
            self.scratch_im[i] = input[2 * i + 1];
        }

        // 2. Complex FFT of size N/2
        self.fft_n2
            .process(&mut self.scratch_re, &mut self.scratch_im);

        // 3. Post-processing via Hermitian symmetry
        // DC: X[0] = H[0].re + H[0].im
        out_re[0] = self.scratch_re[0] + self.scratch_im[0];
        out_im[0] = T::from_usize(0);

        if n_half > 1 {
            let half = T::from_usize(2).recip();
            for k in 1..n_half {
                let nk = n_half - k;

                let re_k = self.scratch_re[k];
                let im_k = self.scratch_im[k];
                let re_nk = self.scratch_re[nk];
                let im_nk = self.scratch_im[nk];

                let even_re = half * (re_k + re_nk);
                let even_im = half * (im_k - im_nk);
                let odd_re = half * (re_k - re_nk);
                let odd_im = half * (im_k + im_nk);

                let w_re = self.post_twiddle_re[k];
                let w_im = self.post_twiddle_im[k];

                out_re[k] = even_re + odd_re.mul_add(w_im, odd_im * w_re);
                out_im[k] = even_im - odd_re.mul_add(w_re, -odd_im * w_im);
            }
        }

        // Nyquist: X[N/2] = H[0].re - H[0].im
        out_re[n_half] = self.scratch_re[0] - self.scratch_im[0];
        out_im[n_half] = T::from_usize(0);
    }

    /// Complex-to-real inverse FFT.
    ///
    /// Given a compact complex spectrum `in_re`/`in_im` of length `N/2 + 1`
    /// (as produced by [`process_forward`](Self::process_forward)), computes
    /// the inverse real FFT into `out` of length `N`.
    ///
    /// The input buffers are mutated in-place (reused as scratch space).
    /// No heap allocations occur inside this method.
    ///
    /// # Panics
    ///
    /// Panics if `in_re`/`in_im` length ≠ `N/2 + 1`, or `out` length ≠ `N`.
    pub fn process_inverse(&self, in_re: &mut [T], in_im: &mut [T], out: &mut [T]) {
        let n = self.n;
        let n_half = n / 2;
        let expected = n_half + 1;
        assert_eq!(in_re.len(), expected, "in_re length mismatch");
        assert_eq!(in_im.len(), expected, "in_im length mismatch");
        assert_eq!(out.len(), n, "output length mismatch");

        let two = T::from_usize(2);
        let half = two.recip();

        // 1. Pre-processing: recover packed N/2 complex array from compact spectrum.
        //    DC + Nyquist → H[0]
        {
            let x0_re = in_re[0];
            let xn2_re = in_re[n_half];
            in_re[0] = (x0_re + xn2_re) * half;
            in_im[0] = (x0_re - xn2_re) * half;
        }

        // 2. Pre-processing: k = 1 .. N/4 (paired with nk = N/2 - k)
        let quarter_n = n / 4;
        for k in 1..quarter_n {
            let nk = n_half - k;

            let xk_re = in_re[k];
            let xk_im = in_im[k];
            let xnk_re = in_re[nk];
            let xnk_im = in_im[nk];

            let even_re = (xk_re + xnk_re) * half;
            let even_im = (xk_im - xnk_im) * half;

            let diff_re = (xk_re - xnk_re) * half;
            let sum_im = (xk_im + xnk_im) * half;

            let w_re = self.post_twiddle_re[k];
            let w_im = self.post_twiddle_im[k];

            let odd_re = w_im.mul_add(diff_re, -w_re * sum_im);
            let odd_im = w_re.mul_add(diff_re, w_im * sum_im);

            in_re[k] = even_re + odd_re;
            in_im[k] = even_im + odd_im;
            in_re[nk] = even_re - odd_re;
            in_im[nk] = -even_im + odd_im;
        }

        // 3. Pre-processing: middle bin k = N/4 (when N divisible by 4)
        if quarter_n > 0 && n_half.is_multiple_of(2) {
            let k_mid = quarter_n;
            let x_re = in_re[k_mid];
            let x_im = in_im[k_mid];
            in_re[k_mid] = x_re;
            in_im[k_mid] = -x_im;
        }

        // 4. Inverse complex FFT of size N/2 (in-place)
        self.fft_n2
            .process_inverse(&mut in_re[..n_half], &mut in_im[..n_half]);

        // 5. Unpack even/odd samples into real output
        for k in 0..n_half {
            out[2 * k] = in_re[k];
            out[2 * k + 1] = in_im[k];
        }
    }
}

#[cfg(test)]
#[path = "fft_test.rs"]
mod tests;
