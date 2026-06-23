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
//!   at construction time via native `RfftPlanner`, outside the audio thread.
//! - **FDL (Frequency Delay Line)** is pre-allocated as contiguous SoA buffers
//!   of real/imaginary spectra (size `fft_size/2 + 1` bins per partition).
//! - **Zero-alloc hot-path**: `process()` only mutates pre-allocated buffers.
//!   It never allocates, never blocks, and never panics.
//!
//! ## Reference
//!
//! Gardner, W. G. "Efficient Convolution without Input-Output Delay"
//! JAES Vol. 43, No. 3, 1995 March.

use crate::math::common::AlignedVec;
use crate::math::dsp::fft::RfftPlanner;

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
    /// Number of frequency bins per partition (= fft_size / 2 + 1).
    n_bins: usize,
    /// Number of samples per input/output block.
    partition_size: usize,
    /// Number of IR partitions.
    num_partitions: usize,
    /// Pre-FFT'd kernel partitions (real part).
    /// Flat storage: `num_partitions × n_bins` f32 values.
    h_fdl_re: AlignedVec<f32>,
    /// Pre-FFT'd kernel partitions (imaginary part).
    h_fdl_im: AlignedVec<f32>,
    /// Frequency Delay Line (FDL): circular buffer of input spectra (real part).
    /// Flat storage: `num_partitions × n_bins` f32 values.
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
    /// Native RFFT planner (handles both forward RFFT and inverse IRFFT).
    rfft: RfftPlanner<f32>,
    /// Forward RFFT output: real part (length = `n_bins`).
    fft_buf_re: AlignedVec<f32>,
    /// Forward RFFT output: imaginary part (length = `n_bins`).
    fft_buf_im: AlignedVec<f32>,
    /// Accumulation buffer in frequency domain, real part (length = `n_bins`).
    acc_re: AlignedVec<f32>,
    /// Accumulation buffer imaginary part (length = `n_bins`).
    acc_im: AlignedVec<f32>,
    /// Time-domain output buffer for IRFFT (length = `fft_size`).
    output_buf: AlignedVec<f32>,
    /// Cached output start index (= fft_size - partition_size).
    output_start: usize,
}

impl ConvEngine {
    /// Builds a UPOLS convolution engine for the given impulse response.
    ///
    /// The IR is partitioned into blocks of `partition_size` samples.
    /// All RFFTs of the kernel partitions are computed here — outside the
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
        let n_bins = fft_size / 2 + 1;
        let output_start = fft_size - partition_size;

        // Partition IR: P = ceil(N / B)
        let num_partitions = if ir.is_empty() {
            0
        } else {
            ir.len().div_ceil(partition_size)
        };

        // Build native RFFT plan (handles both forward RFFT and inverse IRFFT)
        let mut rfft = RfftPlanner::<f32>::new(fft_size);

        // Pre-FFT each kernel partition
        let h_fdl_part_len = num_partitions * n_bins;
        let mut h_fdl_re = AlignedVec::new(h_fdl_part_len, 0.0_f32);
        let mut h_fdl_im = AlignedVec::new(h_fdl_part_len, 0.0_f32);

        let mut ir_buf = vec![0.0f32; fft_size];
        let mut tmp_re = vec![0.0f32; n_bins];
        let mut tmp_im = vec![0.0f32; n_bins];

        for p in 0..num_partitions {
            let ir_start = p * partition_size;
            let ir_end = (ir_start + partition_size).min(ir.len());

            ir_buf.fill(0.0);
            for (i, &sample) in ir[ir_start..ir_end].iter().enumerate() {
                ir_buf[i] = sample;
            }

            rfft.process_forward(&ir_buf, &mut tmp_re, &mut tmp_im);

            let base = p * n_bins;
            for k in 0..n_bins {
                h_fdl_re[base + k] = tmp_re[k];
                h_fdl_im[base + k] = tmp_im[k];
            }
        }

        // Pre-allocate FDL (all zeros initially)
        let fdl_part_len = num_partitions * n_bins;
        let fdl_re = AlignedVec::new(fdl_part_len, 0.0_f32);
        let fdl_im = AlignedVec::new(fdl_part_len, 0.0_f32);

        // Pre-allocate runtime buffers
        let input_buf = AlignedVec::new(fft_size, 0.0_f32);
        let fft_buf_re = AlignedVec::new(n_bins, 0.0_f32);
        let fft_buf_im = AlignedVec::new(n_bins, 0.0_f32);
        let acc_re = AlignedVec::new(n_bins, 0.0_f32);
        let acc_im = AlignedVec::new(n_bins, 0.0_f32);
        let output_buf = AlignedVec::new(fft_size, 0.0_f32);

        Self {
            fft_size,
            n_bins,
            partition_size,
            num_partitions,
            h_fdl_re,
            h_fdl_im,
            fdl_re,
            fdl_im,
            fdl_idx: 0,
            input_buf,
            rfft,
            fft_buf_re,
            fft_buf_im,
            acc_re,
            acc_im,
            output_buf,
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
        let in_len = self.fft_size;
        let out_start = self.output_start;
        self.input_buf.copy_within(self.partition_size..in_len, 0);

        // Load new samples at the end
        self.input_buf[out_start..in_len].copy_from_slice(input);

        // ── Step 2: Forward RFFT of input segment ──
        self.rfft
            .process_forward(&self.input_buf, &mut self.fft_buf_re, &mut self.fft_buf_im);

        // ── Step 3: Store in FDL (circular buffer) ──
        let fdl_base = self.fdl_idx * self.n_bins;
        for k in 0..self.n_bins {
            self.fdl_re[fdl_base + k] = self.fft_buf_re[k];
            self.fdl_im[fdl_base + k] = self.fft_buf_im[k];
        }

        // ── Step 4: Frequency-domain MAC over all partitions ──
        let p_count = self.num_partitions;
        let n_bins = self.n_bins;

        // Zero the accumulator
        self.acc_re.fill(0.0);
        self.acc_im.fill(0.0);

        use core::arch::x86_64::{
            _mm256_add_ps, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
            _mm256_storeu_ps,
        };
        // SAFETY: AlignedVec guarantees 64-byte alignment. _mm256_loadu_ps
        // has no alignment requirement. All indices are bounded by
        // pre-allocated buffer sizes.
        // Note: we use _mm256_loadu_ps/_mm256_storeu_ps (unaligned)
        // because n_bins = fft_size/2 + 1 may not be a multiple of 8,
        // so p * n_bins + k can be unaligned for odd p.
        unsafe {
            if p_count == 1 {
                let fdl_start = self.fdl_idx * self.n_bins;
                let mut k = 0usize;
                while k + 8 <= n_bins {
                    let h_re = _mm256_loadu_ps(self.h_fdl_re.as_ptr().add(k));
                    let h_im = _mm256_loadu_ps(self.h_fdl_im.as_ptr().add(k));
                    let x_re = _mm256_loadu_ps(self.fdl_re.as_ptr().add(fdl_start + k));
                    let x_im = _mm256_loadu_ps(self.fdl_im.as_ptr().add(fdl_start + k));

                    let re_prod = _mm256_mul_ps(h_re, x_re);
                    let re_res = _mm256_fnmadd_ps(h_im, x_im, re_prod);

                    let im_prod = _mm256_mul_ps(h_re, x_im);
                    let im_res = _mm256_fmadd_ps(h_im, x_re, im_prod);

                    _mm256_storeu_ps(self.acc_re.as_mut_ptr().add(k), re_res);
                    _mm256_storeu_ps(self.acc_im.as_mut_ptr().add(k), im_res);

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
                    let fdl_start = fdl_p * self.n_bins;
                    let h_start = p * self.n_bins;

                    let mut k = 0usize;
                    while k + 8 <= n_bins {
                        let h_re = _mm256_loadu_ps(self.h_fdl_re.as_ptr().add(h_start + k));
                        let h_im = _mm256_loadu_ps(self.h_fdl_im.as_ptr().add(h_start + k));
                        let x_re = _mm256_loadu_ps(self.fdl_re.as_ptr().add(fdl_start + k));
                        let x_im = _mm256_loadu_ps(self.fdl_im.as_ptr().add(fdl_start + k));
                        let acc_re_curr = _mm256_loadu_ps(self.acc_re.as_ptr().add(k));
                        let acc_im_curr = _mm256_loadu_ps(self.acc_im.as_ptr().add(k));

                        let re_prod = _mm256_mul_ps(h_re, x_re);
                        let re_res = _mm256_fnmadd_ps(h_im, x_im, re_prod);
                        let re_sum = _mm256_add_ps(acc_re_curr, re_res);

                        let im_prod = _mm256_mul_ps(h_re, x_im);
                        let im_res = _mm256_fmadd_ps(h_im, x_re, im_prod);
                        let im_sum = _mm256_add_ps(acc_im_curr, im_res);

                        _mm256_storeu_ps(self.acc_re.as_mut_ptr().add(k), re_sum);
                        _mm256_storeu_ps(self.acc_im.as_mut_ptr().add(k), im_sum);

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

        // ── Step 5: Inverse RFFT (complex → real) ──
        // process_inverse takes in_re/in_im of length N/2+1 (n_bins) and
        // produces real output of length N (fft_size). The inverse scaling
        // is handled internally by the IRFFT algorithm.
        self.rfft
            .process_inverse(&mut self.acc_re, &mut self.acc_im, &mut self.output_buf);

        // ── Step 6: Extract valid output (overlap-save discard) ──
        output.copy_from_slice(&self.output_buf[out_start..out_start + self.partition_size]);

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
