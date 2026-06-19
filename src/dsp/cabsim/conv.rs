// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Uniform-Partitioned Overlap-Save (UPOLS) convolution engine.
//!
//! Implements real-time convolution of an audio stream with an impulse response
//! using the Uniform-Partitioned Overlap-Save method in the frequency domain.
//!
//! ## Design
//!
//! - **Partition size** equals the audio block size (typically 64–2048 samples).
//!   Latency is exactly `partition_size` samples.
//! - **FFT size** is `2 × partition_size` (rounded up to next power of two).
//! - **Kernel pre-FFT**: all IR partitions are transformed to the frequency domain
//!   at construction time, outside the audio thread.
//! - **FDL (Frequency Delay Line)** is pre-allocated as a contiguous buffer of
//!   complex spectra, one per partition.
//! - **Zero-alloc hot-path**: `process()` only mutates pre-allocated buffers.
//!   It never allocates, never blocks, and never panics.
//!
//! ## Reference
//!
//! Gardner, W. G. "Efficient Convolution without Input-Output Delay"
//! JAES Vol. 43, No. 3, 1995 March.

use crate::math::common::AlignedVec;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

/// Uniform-Partitioned Overlap-Save convolution engine.
///
/// All memory is allocated at construction time (`ConvEngine::new()`).
/// The [`process`](ConvEngine::process) method is zero-alloc and safe for real-time
/// audio threads.
///
/// ## Latency
///
/// UPOLS introduces exactly `partition_size` samples of latency
/// (one full audio block).
pub struct ConvEngine {
    /// FFT size (2 × partition_size rounded up to next power of two).
    fft_size: usize,
    /// Number of samples per input/output block.
    partition_size: usize,
    /// Number of IR partitions.
    num_partitions: usize,
    /// Pre-FFT'd kernel partitions (real part).
    /// Flat storage: `num_partitions × fft_size` f32 values.
    h_fdl_re: AlignedVec<f32>,
    /// Pre-FFT'd kernel partitions (imaginary part).
    h_fdl_im: AlignedVec<f32>,
    /// Frequency Delay Line (FDL): circular buffer of input spectra (real part).
    /// Flat storage: `num_partitions × fft_size` f32 values.
    fdl_re: AlignedVec<f32>,
    /// FDL imaginary part.
    fdl_im: AlignedVec<f32>,
    /// Write index into the FDL circular buffer.
    fdl_idx: usize,
    /// Input overlap buffer for overlap-save (length = `fft_size`).
    /// Layout: the most recent `partition_size` samples are loaded at
    /// offset `fft_size - partition_size`. After FFT and IFFT,
    /// the valid output starts at offset `fft_size - partition_size`.
    input_buf: AlignedVec<f32>,
    /// Forward FFT scratch buffer.
    fft_scratch: Vec<Complex<f32>>,
    /// Inverse FFT scratch buffer.
    ifft_scratch: Vec<Complex<f32>>,
    /// Work buffer for forward FFT (length = fft_size).
    fft_buf: Vec<Complex<f32>>,
    /// Accumulation buffer in frequency domain, real part (length = fft_size).
    acc_re: AlignedVec<f32>,
    /// Accumulation buffer imaginary part.
    acc_im: AlignedVec<f32>,
    /// Forward FFT plan.
    fft: Arc<dyn Fft<f32>>,
    /// Inverse FFT plan.
    ifft: Arc<dyn Fft<f32>>,
    /// IFFT scale factor = 1.0 / fft_size.
    ifft_scale: f32,
    /// Cached output start index (= fft_size - partition_size).
    output_start: usize,
}

impl ConvEngine {
    /// Builds a UPOLS convolution engine for the given impulse response.
    ///
    /// The IR is partitioned into blocks of `partition_size` samples.
    /// All FFTs of the kernel partitions are computed here — outside the
    /// audio thread — so that [`process`](ConvEngine::process) is zero-alloc.
    ///
    /// # Parameters
    /// - `ir`: impulse response samples (mono, f32).
    /// - `partition_size`: size of each partition / audio block size.
    ///
    /// # Returns
    /// A fully initialized `ConvEngine`. If `ir` is empty, the engine
    /// acts as a passthrough (output = input).
    pub fn new(ir: &[f32], partition_size: usize) -> Self {
        assert!(partition_size > 0, "partition_size must be positive");

        let fft_size = (2 * partition_size).next_power_of_two();
        let output_start = fft_size - partition_size;

        // Partition IR: P = ceil(N / B)
        let num_partitions = if ir.is_empty() {
            0
        } else {
            ir.len().div_ceil(partition_size)
        };

        // Build FFT plans
        let mut planner = FftPlanner::new();
        let fft: Arc<dyn Fft<f32>> = planner.plan_fft_forward(fft_size);
        let ifft: Arc<dyn Fft<f32>> = planner.plan_fft_inverse(fft_size);

        let ifft_scale = 1.0 / fft_size as f32;

        // Pre-allocate scratch buffers
        let fft_scratch_len = fft.get_inplace_scratch_len();
        let ifft_scratch_len = ifft.get_inplace_scratch_len();
        let mut fft_scratch = vec![Complex::new(0.0_f32, 0.0_f32); fft_scratch_len];
        let ifft_scratch = vec![Complex::new(0.0_f32, 0.0_f32); ifft_scratch_len];

        // Pre-FFT each kernel partition
        let h_fdl_part_len = num_partitions * fft_size;
        let mut h_fdl_re = AlignedVec::new(h_fdl_part_len, 0.0_f32);
        let mut h_fdl_im = AlignedVec::new(h_fdl_part_len, 0.0_f32);
        let mut fft_buf = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];

        for p in 0..num_partitions {
            let ir_start = p * partition_size;
            let ir_end = (ir_start + partition_size).min(ir.len());
            // Zero out the FFT buffer
            fft_buf.fill(Complex::new(0.0, 0.0));

            // Place partition samples at the beginning (causal convolution)
            for (i, &sample) in ir[ir_start..ir_end].iter().enumerate() {
                fft_buf[i] = Complex::new(sample, 0.0);
            }

            fft.process_with_scratch(&mut fft_buf, &mut fft_scratch);

            // Store in h_fdl (separate re, im)
            let base = p * fft_size;
            for (k, c) in fft_buf.iter().enumerate() {
                h_fdl_re[base + k] = c.re;
                h_fdl_im[base + k] = c.im;
            }
        }

        // Pre-allocate FDL (all zeros initially)
        let fdl_part_len = num_partitions * fft_size;
        let fdl_re = AlignedVec::new(fdl_part_len, 0.0_f32);
        let fdl_im = AlignedVec::new(fdl_part_len, 0.0_f32);

        // Pre-allocate other buffers
        let input_buf = AlignedVec::new(fft_size, 0.0_f32);
        let fft_buf_final = vec![Complex::new(0.0_f32, 0.0_f32); fft_size];
        let acc_re = AlignedVec::new(fft_size, 0.0_f32);
        let acc_im = AlignedVec::new(fft_size, 0.0_f32);

        Self {
            fft_size,
            partition_size,
            num_partitions,
            h_fdl_re,
            h_fdl_im,
            fdl_re,
            fdl_im,
            fdl_idx: 0,
            input_buf,
            fft_scratch,
            ifft_scratch,
            fft_buf: fft_buf_final,
            acc_re,
            acc_im,
            fft,
            ifft,
            ifft_scale,
            output_start,
        }
    }

    /// Returns the partition size (== audio block size) in samples.
    #[inline(always)]
    pub fn partition_size(&self) -> usize {
        self.partition_size
    }

    /// Returns the FFT size used for frequency-domain processing.
    #[inline(always)]
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }

    /// Returns the number of IR partitions.
    #[inline(always)]
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Returns the algorithmic latency in samples (= `partition_size`).
    #[inline(always)]
    pub fn latency_samples(&self) -> usize {
        self.partition_size
    }

    /// Returns `true` if no IR is loaded (passthrough mode).
    #[inline(always)]
    pub fn is_passthrough(&self) -> bool {
        self.num_partitions == 0
    }

    /// Processes one block of mono audio through the convolution engine.
    ///
    /// ## RT-Safety
    ///
    /// This function is **zero-alloc**, **lock-free**, and never panics.
    /// It only mutates pre-allocated internal buffers.
    ///
    /// ## Parameters
    /// - `input`: slice of exactly `partition_size` samples.
    /// - `output`: slice of exactly `partition_size` samples where the
    ///   convolved result is written.
    ///
    /// ## Panic Safety
    ///
    /// This function uses unchecked indexing internally for performance,
    /// but all bounds are guaranteed by construction (pre-allocated sizes).
    #[inline]
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.partition_size);
        debug_assert_eq!(output.len(), self.partition_size);

        if self.num_partitions == 0 {
            // Passthrough: no IR loaded
            output.copy_from_slice(input);
            return;
        }

        // ── Step 1: Shift input buffer (overlap-save) ──
        // Discard the oldest `partition_size` samples and shift the tail forward.
        // Then load `partition_size` new samples at the end.
        let in_len = self.fft_size;
        let out_start = self.output_start;
        self.input_buf.copy_within(self.partition_size..in_len, 0);

        // Load new samples at the end
        self.input_buf[out_start..in_len].copy_from_slice(input);

        // ── Step 2: Forward FFT of input segment ──
        for (i, &sample) in self.input_buf.iter().enumerate() {
            self.fft_buf[i] = Complex::new(sample, 0.0);
        }
        self.fft
            .process_with_scratch(&mut self.fft_buf, &mut self.fft_scratch);

        // ── Step 3: Store in FDL (circular buffer) ──
        let fdl_base = self.fdl_idx * self.fft_size;
        for (k, c) in self.fft_buf.iter().enumerate() {
            self.fdl_re[fdl_base + k] = c.re;
            self.fdl_im[fdl_base + k] = c.im;
        }

        // ── Step 4: Frequency-domain MAC over all partitions ──
        let p_count = self.num_partitions;
        let n_bins = self.fft_size;

        // Zero the accumulator
        self.acc_re.fill(0.0);
        self.acc_im.fill(0.0);

        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::{
                _mm256_add_ps, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_load_ps, _mm256_mul_ps,
                _mm256_store_ps,
            };
            // SAFETY: AlignedVec guarantees 64-byte alignment, sufficient for
            // _mm256_load_ps/_mm256_store_ps (32-byte). All indices are
            // bounded by pre-allocated buffer sizes.
            unsafe {
                if p_count == 1 {
                    let fdl_start = self.fdl_idx * self.fft_size;
                    let mut k = 0usize;
                    while k + 8 <= n_bins {
                        let h_re = _mm256_load_ps(self.h_fdl_re.as_ptr().add(k));
                        let h_im = _mm256_load_ps(self.h_fdl_im.as_ptr().add(k));
                        let x_re = _mm256_load_ps(self.fdl_re.as_ptr().add(fdl_start + k));
                        let x_im = _mm256_load_ps(self.fdl_im.as_ptr().add(fdl_start + k));

                        let re_prod = _mm256_mul_ps(h_re, x_re);
                        let re_res = _mm256_fnmadd_ps(h_im, x_im, re_prod);

                        let im_prod = _mm256_mul_ps(h_re, x_im);
                        let im_res = _mm256_fmadd_ps(h_im, x_re, im_prod);

                        _mm256_store_ps(self.acc_re.as_mut_ptr().add(k), re_res);
                        _mm256_store_ps(self.acc_im.as_mut_ptr().add(k), im_res);

                        k += 8;
                    }
                    for k in k..n_bins {
                        let h_re = self.h_fdl_re[k];
                        let h_im = self.h_fdl_im[k];
                        let x_re = self.fdl_re[fdl_start + k];
                        let x_im = self.fdl_im[fdl_start + k];
                        self.acc_re[k] = h_re * x_re - h_im * x_im;
                        self.acc_im[k] = h_re * x_im + h_im * x_re;
                    }
                } else {
                    for p in 0..p_count {
                        let fdl_p = (self.fdl_idx + p_count - p) % p_count;
                        let fdl_start = fdl_p * self.fft_size;
                        let h_start = p * self.fft_size;

                        let mut k = 0usize;
                        while k + 8 <= n_bins {
                            let h_re = _mm256_load_ps(self.h_fdl_re.as_ptr().add(h_start + k));
                            let h_im = _mm256_load_ps(self.h_fdl_im.as_ptr().add(h_start + k));
                            let x_re = _mm256_load_ps(self.fdl_re.as_ptr().add(fdl_start + k));
                            let x_im = _mm256_load_ps(self.fdl_im.as_ptr().add(fdl_start + k));
                            let acc_re_curr = _mm256_load_ps(self.acc_re.as_ptr().add(k));
                            let acc_im_curr = _mm256_load_ps(self.acc_im.as_ptr().add(k));

                            let re_prod = _mm256_mul_ps(h_re, x_re);
                            let re_res = _mm256_fnmadd_ps(h_im, x_im, re_prod);
                            let re_sum = _mm256_add_ps(acc_re_curr, re_res);

                            let im_prod = _mm256_mul_ps(h_re, x_im);
                            let im_res = _mm256_fmadd_ps(h_im, x_re, im_prod);
                            let im_sum = _mm256_add_ps(acc_im_curr, im_res);

                            _mm256_store_ps(self.acc_re.as_mut_ptr().add(k), re_sum);
                            _mm256_store_ps(self.acc_im.as_mut_ptr().add(k), im_sum);

                            k += 8;
                        }
                        for k in k..n_bins {
                            let h_re = self.h_fdl_re[h_start + k];
                            let h_im = self.h_fdl_im[h_start + k];
                            let x_re = self.fdl_re[fdl_start + k];
                            let x_im = self.fdl_im[fdl_start + k];
                            self.acc_re[k] += h_re * x_re - h_im * x_im;
                            self.acc_im[k] += h_re * x_im + h_im * x_re;
                        }
                    }
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            if p_count == 1 {
                // Fast path: single partition — no loop over p
                let fdl_start = self.fdl_idx * self.fft_size;
                for k in 0..n_bins {
                    let h_re = self.h_fdl_re[k];
                    let h_im = self.h_fdl_im[k];
                    let x_re = self.fdl_re[fdl_start + k];
                    let x_im = self.fdl_im[fdl_start + k];
                    self.acc_re[k] = h_re * x_re - h_im * x_im;
                    self.acc_im[k] = h_re * x_im + h_im * x_re;
                }
            } else {
                for p in 0..p_count {
                    let fdl_p = (self.fdl_idx + p_count - p) % p_count;
                    let fdl_start = fdl_p * self.fft_size;
                    let h_start = p * self.fft_size;

                    for k in 0..n_bins {
                        let h_re = self.h_fdl_re[h_start + k];
                        let h_im = self.h_fdl_im[h_start + k];
                        let x_re = self.fdl_re[fdl_start + k];
                        let x_im = self.fdl_im[fdl_start + k];
                        self.acc_re[k] += h_re * x_re - h_im * x_im;
                        self.acc_im[k] += h_re * x_im + h_im * x_re;
                    }
                }
            }
        }

        // ── Step 5: Merge acc_re/acc_im into fft_buf for IFFT (rustfft requires interleaved) ──
        for k in 0..n_bins {
            self.fft_buf[k] = Complex::new(self.acc_re[k], self.acc_im[k]);
        }
        self.ifft
            .process_with_scratch(&mut self.fft_buf, &mut self.ifft_scratch);

        // ── Step 6: Extract valid output (overlap-save discard) ──
        for (i, c) in self.fft_buf[out_start..out_start + self.partition_size]
            .iter()
            .enumerate()
        {
            output[i] = c.re * self.ifft_scale;
        }

        // ── Step 7: Advance FDL write index ──
        self.fdl_idx += 1;
        if self.fdl_idx >= p_count {
            self.fdl_idx = 0;
        }
    }
}

#[cfg(test)]
#[path = "conv_test.rs"]
mod conv_test;
