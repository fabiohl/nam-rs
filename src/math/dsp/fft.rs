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

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for floating-point comparisons.
    const EPS_F32: f32 = 1e-5;
    const EPS_F64: f64 = 1e-9;

    // -----------------------------------------------------------------
    // helpers
    // -----------------------------------------------------------------

    fn assert_slice_approx_eq_f32(a: &[f32], b: &[f32], eps: f32) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < eps,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    fn assert_slice_approx_eq_f64(a: &[f64], b: &[f64], eps: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < eps,
                "mismatch at index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    // -----------------------------------------------------------------
    // construction
    // -----------------------------------------------------------------

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two() {
        FftPlanner::<f32>::new(3);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two_f64() {
        FftPlanner::<f64>::new(6);
    }

    #[test]
    #[should_panic(expected = "positive")]
    fn rejects_zero() {
        FftPlanner::<f32>::new(0);
    }

    #[test]
    fn accepts_powers_of_two() {
        for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let fft = FftPlanner::<f64>::new(n);
            assert_eq!(fft.len(), n);
            assert_eq!(fft.bit_reverse.len(), n);
            assert_eq!(fft.twiddle_re.len(), n / 2);
            assert_eq!(fft.twiddle_im.len(), n / 2);
        }
    }

    // -----------------------------------------------------------------
    // impulse → DC
    // -----------------------------------------------------------------

    #[test]
    fn impulse_dc_f32() {
        for n in [1, 2, 4, 8, 16, 32, 64] {
            let fft = FftPlanner::<f32>::new(n);
            let mut re = vec![0.0f32; n];
            let mut im = vec![0.0f32; n];
            re[0] = 1.0;
            fft.process(&mut re, &mut im);
            let expected = vec![1.0f32; n];
            assert_slice_approx_eq_f32(&re, &expected, EPS_F32);
            let expected_im = vec![0.0f32; n];
            assert_slice_approx_eq_f32(&im, &expected_im, EPS_F32);
        }
    }

    #[test]
    fn impulse_dc_f64() {
        let fft = FftPlanner::<f64>::new(8);
        let mut re = vec![0.0f64; 8];
        let mut im = vec![0.0f64; 8];
        re[0] = 1.0;
        fft.process(&mut re, &mut im);
        for (i, v) in re.iter().enumerate() {
            assert!((v - 1.0).abs() < EPS_F64, "re[{i}] = {v}");
        }
        for (i, v) in im.iter().enumerate() {
            assert!(v.abs() < EPS_F64, "im[{i}] = {v}");
        }
    }

    // -----------------------------------------------------------------
    // round-trip: forward → inverse ≡ identity
    // -----------------------------------------------------------------

    #[test]
    fn roundtrip_f32() {
        for n in [1, 2, 4, 8, 16, 32, 64, 128] {
            let fft = FftPlanner::<f32>::new(n);
            let original_re: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let original_im: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();

            let mut re = original_re.clone();
            let mut im = original_im.clone();

            fft.process(&mut re, &mut im);
            fft.process_inverse(&mut re, &mut im);

            assert_slice_approx_eq_f32(&re, &original_re, 1e-3f32 * n as f32);
            assert_slice_approx_eq_f32(&im, &original_im, 1e-3f32 * n as f32);
        }
    }

    #[test]
    fn roundtrip_f64() {
        for n in [1, 2, 4, 8, 16, 32, 64, 128] {
            let fft = FftPlanner::<f64>::new(n);
            let original_re: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let original_im: Vec<f64> = (0..n).map(|i| -(i as f64)).collect();

            let mut re = original_re.clone();
            let mut im = original_im.clone();

            fft.process(&mut re, &mut im);
            fft.process_inverse(&mut re, &mut im);

            assert_slice_approx_eq_f64(&re, &original_re, 1e-6f64 * n as f64);
            assert_slice_approx_eq_f64(&im, &original_im, 1e-6f64 * n as f64);
        }
    }

    // -----------------------------------------------------------------
    // linearity: FFT(a + b) = FFT(a) + FFT(b)
    // -----------------------------------------------------------------

    #[test]
    fn linearity_f64() {
        let n = 16;
        let fft = FftPlanner::<f64>::new(n);

        let a_re: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let a_im: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let b_re: Vec<f64> = (0..n).map(|i| (2.0 * i as f64).cos()).collect();
        let b_im: Vec<f64> = (0..n).map(|i| (3.0 * i as f64).sin()).collect();

        // FFT(a)
        let mut fa_re = a_re.clone();
        let mut fa_im = a_im.clone();
        fft.process(&mut fa_re, &mut fa_im);

        // FFT(b)
        let mut fb_re = b_re.clone();
        let mut fb_im = b_im.clone();
        fft.process(&mut fb_re, &mut fb_im);

        // FFT(a+b)
        let mut sum_re: Vec<f64> = a_re.iter().zip(&b_re).map(|(x, y)| x + y).collect();
        let mut sum_im: Vec<f64> = a_im.iter().zip(&b_im).map(|(x, y)| x + y).collect();
        fft.process(&mut sum_re, &mut sum_im);

        // Sum of individual FFTs
        for i in 0..n {
            let expected_re = fa_re[i] + fb_re[i];
            let expected_im = fa_im[i] + fb_im[i];
            assert!(
                (sum_re[i] - expected_re).abs() < EPS_F64,
                "linearity re[{i}]: {} vs {}",
                sum_re[i],
                expected_re
            );
            assert!(
                (sum_im[i] - expected_im).abs() < EPS_F64,
                "linearity im[{i}]: {} vs {}",
                sum_im[i],
                expected_im
            );
        }
    }

    // -----------------------------------------------------------------
    // known values: N=4
    // -----------------------------------------------------------------

    #[test]
    fn known_n4_f64() {
        let fft = FftPlanner::<f64>::new(4);
        let mut re = vec![1.0, 2.0, 3.0, 4.0];
        let mut im = vec![0.0; 4];
        fft.process(&mut re, &mut im);

        // FFT of [1,2,3,4]:
        // DC   = 1+2+3+4 = 10
        // bin1 = (1+2w+3w^2+4w^3) where w=e^{-2πi/4}=-i
        //       = 1 - 2i - 3 + 4i = -2 + 2i
        // bin2 = 1 - 2 + 3 - 4 = -2
        // bin3 = conjugate of bin1 = -2 - 2i
        assert!((re[0] - 10.0).abs() < EPS_F64);
        assert!((im[0] - 0.0).abs() < EPS_F64);
        assert!((re[1] - (-2.0)).abs() < EPS_F64);
        assert!((im[1] - 2.0).abs() < EPS_F64);
        assert!((re[2] - (-2.0)).abs() < EPS_F64);
        assert!((im[2] - 0.0).abs() < EPS_F64);
        assert!((re[3] - (-2.0)).abs() < EPS_F64);
        assert!((im[3] - (-2.0)).abs() < EPS_F64);
    }

    // -----------------------------------------------------------------
    // known values: N=8 — single cosine
    // -----------------------------------------------------------------

    #[test]
    fn cosine_n8_f64() {
        let n = 8;
        let fft = FftPlanner::<f64>::new(n);
        let freq = 2.0; // two cycles in N=8 → bin 2 and bin 6
        let mut re: Vec<f64> = (0..n)
            .map(|i| (TAU * freq * i as f64 / n as f64).cos())
            .collect();
        let mut im = vec![0.0f64; n];
        fft.process(&mut re, &mut im);

        // Cosine at bin 2: peaks at indices 2 and 6 with amplitude n/2 = 4
        assert!((re[2] - 4.0).abs() < EPS_F64, "re[2] = {}", re[2]);
        assert!((re[6] - 4.0).abs() < EPS_F64, "re[6] = {}", re[6]);
        // Other bins should be ~0
        for &i in &[0, 1, 3, 4, 5, 7] {
            assert!(re[i].abs() < 1e-9, "re[{i}] = {}", re[i]);
            assert!(im[i].abs() < 1e-9, "im[{i}] = {}", im[i]);
        }
    }

    // -----------------------------------------------------------------
    // process panics on wrong buffer length
    // -----------------------------------------------------------------

    #[test]
    #[should_panic(expected = "re length mismatch")]
    fn process_wrong_len() {
        let fft = FftPlanner::<f32>::new(8);
        let mut re = vec![0.0f32; 7];
        let mut im = vec![0.0f32; 8];
        fft.process(&mut re, &mut im);
    }

    // -----------------------------------------------------------------
    // N=1 edge case
    // -----------------------------------------------------------------

    #[test]
    fn n1_f64() {
        let fft = FftPlanner::<f64>::new(1);
        let mut re = vec![42.0];
        let mut im = vec![7.0];
        fft.process(&mut re, &mut im);
        assert!((re[0] - 42.0).abs() < EPS_F64);
        assert!((im[0] - 7.0).abs() < EPS_F64);
        fft.process_inverse(&mut re, &mut im);
        assert!((re[0] - 42.0).abs() < EPS_F64);
        assert!((im[0] - 7.0).abs() < EPS_F64);
    }
}
