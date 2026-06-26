// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Zero-latency partitioned FFT convolution state for the Linear model.
//!
//! Implements hybrid convolution: the direct head (first `P` samples) is
//! computed in the time domain, while the tail (remaining `N-P` samples) is
//! computed in the frequency domain via overlap-save FFT blocks.
//!
//! # Architecture
//!
//! The IR `h[n]` of length `N` is partitioned into:
//! - **Head** (`h[0..P]`): convolved directly in the time domain with zero
//!   latency (no block delay).
//! - **Tail** (`h[P..N]`): partitioned into `K = ceil((N-P)/P)` blocks of
//!   at most `P` samples each, zero-padded to `2P`, and pre-transformed
//!   into the frequency domain via RFFT.
//!
//! At runtime, every `P` samples the input window of size `2P` is FFT'd
//! and stored in a circular frequency delay line (FDL). The accumulated
//! product of the FDL entries with the corresponding tail spectra is
//! inverse-transformed, yielding `P` valid output samples that are buffered
//! for per-sample consumption.
//!
//! # RT-Safety
//! - All buffers are pre-allocated at construction (`AlignedVec<f32>`,
//!   64-byte aligned for SIMD).
//! - The `RfftPlanner` pre-computes twiddle/bit-reversal tables once.
//! - Hot-path operations use only stack-allocated temporaries and
//!   pre-existing slice references — zero heap allocation, zero locks.

use crate::math::common::AlignedVec;
use crate::math::dsp::fft::RfftPlanner;
use core::fmt;

/// State for zero-latency partitioned FFT (overlap-save) convolution.
///
/// Handles the tail portion of the impulse response (samples from partition
/// size `P` to the end `N-1`). The head (samples 0..P) is computed directly
/// in the time domain by `LinearModel::process_sample`.
///
/// # Buffer Layout
///
/// | Buffer            | Size              | Purpose                                   |
/// |-------------------|-------------------|-------------------------------------------|
/// | `h_fdl_re/im`     | `K × (P+1)`       | Pre-computed tail IR spectra (flat)       |
/// | `fdl_re/im`       | `K × (P+1)`       | Circular frequency delay line (flat)      |
/// | `input_buf`       | `2P`              | Input window for forward RFFT             |
/// | `fft_re/im`       | `P+1`             | Forward RFFT output (compact spectrum)    |
/// | `acc_re/im`       | `P+1`             | Complex MAC accumulation                  |
/// | `output_buf`      | `2P`              | IFFT output (time domain)                 |
/// | `tail_output_buf` | `P`               | Valid tail samples ready for consumption  |
///
/// where `K = ceil((N-P)/P)` is the number of tail partitions.
///
/// The spectrum buffers (`h_fdl_*` and `fdl_*`) are stored as flat
/// `AlignedVec<f32>` with stride `P+1` (number of bins). Partition `k`
/// occupies indices `[k * num_bins .. (k+1) * num_bins]`. This flat layout
/// avoids pointer indirection in the hot-path MAC loop and keeps all FDL
/// data in a single contiguous region for cache locality.
pub struct LinearFftState {
    /// Partition size `P` (head length = tail block size). Must be a power
    /// of two ≤ `N`.
    pub p: usize,
    /// Total receptive field `N` (= IR length).
    pub n: usize,
    /// Number of tail partitions `K = ceil((N-P)/P)`.
    pub num_partitions: usize,
    /// Number of complex bins per partition = `P + 1`.
    pub num_bins: usize,
    /// Real-to-complex FFT planner for block size `2P`.
    pub rfft: RfftPlanner<f32>,
    /// Pre-computed real spectra of the tail IR partitions. Flat buffer of
    /// length `K × num_bins`. Partition `k` starts at index `k * num_bins`.
    pub h_fdl_re: AlignedVec<f32>,
    /// Pre-computed imaginary spectra of the tail IR partitions.
    pub h_fdl_im: AlignedVec<f32>,
    /// Frequency delay line — real part. Flat circular buffer of past input
    /// spectra, length `K × num_bins`. Partition `k` starts at `k * num_bins`.
    pub fdl_re: AlignedVec<f32>,
    /// Frequency delay line — imaginary part.
    pub fdl_im: AlignedVec<f32>,
    /// Circular write index into `fdl_re` / `fdl_im` (0..K-1).
    /// Advanced before storing a new input spectrum.
    pub fdl_write_idx: usize,
    /// Input window buffer of size `2P` for the forward RFFT.
    /// Filled from the `MirroredBuffer` history in `LinearModel`.
    pub input_buf: AlignedVec<f32>,
    /// Forward RFFT output — real bins (size `P+1`).
    pub fft_re: AlignedVec<f32>,
    /// Forward RFFT output — imaginary bins (size `P+1`).
    pub fft_im: AlignedVec<f32>,
    /// Complex MAC accumulation buffer — real part (size `P+1`).
    pub acc_re: AlignedVec<f32>,
    /// Complex MAC accumulation buffer — imaginary part (size `P+1`).
    pub acc_im: AlignedVec<f32>,
    /// IFFT output buffer (size `2P`). Valid tail samples reside in
    /// indices `P..2P-1`.
    pub output_buf: AlignedVec<f32>,
    /// Circular buffer holding the `P` valid tail output samples from the
    /// most recent `process_tail_block` call. Read sequentially by
    /// `LinearModel::process_sample` in the FFT path.
    pub tail_output_buf: AlignedVec<f32>,
    /// Current read position within `tail_output_buf` (0 ≤ `sample_counter` < `P`).
    /// Incremented by `LinearModel::process_sample`; triggers a new tail
    /// block computation when it reaches `P`.
    pub sample_counter: usize,
}

impl fmt::Debug for LinearFftState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinearFftState")
            .field("p", &self.p)
            .field("n", &self.n)
            .field("num_partitions", &self.num_partitions)
            .field("num_bins", &self.num_bins)
            .field("fdl_write_idx", &self.fdl_write_idx)
            .field("sample_counter", &self.sample_counter)
            .finish_non_exhaustive()
    }
}

impl LinearFftState {
    /// Creates a new `LinearFftState` for the given impulse response.
    ///
    /// `p` is the partition size (head length), which must be a power of
    /// two and ≤ `weights.len()`. `weights` are the IR samples in
    /// **forward-time order** (i.e., `weights[0]` is the response at the
    /// current sample). They are read-only: only the tail portion
    /// (`weights[P..N]`) is used by this state; the head (`weights[0..P]`)
    /// is convolved directly by `LinearModel`.
    ///
    /// All internal buffers are pre-allocated with 64-byte alignment
    /// (`AlignedVec<f32>`) at construction time. No further heap
    /// allocations occur during processing.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a power of two, or if `p > weights.len()`.
    pub fn new(p: usize, weights: &[f32]) -> Self {
        let n = weights.len();
        assert!(p.is_power_of_two(), "P must be a power of two, got {p}");
        assert!(p <= n, "P ({p}) must be ≤ N ({n})");

        let block_size = 2 * p;
        let num_bins = p + 1;
        // ceil((N-P)/P), saturated at 0 when P==N
        let num_partitions = if n > p { (n - p).div_ceil(p) } else { 0 };

        let mut rfft = RfftPlanner::<f32>::new(block_size);

        let fdl_total = num_partitions * num_bins;
        let mut h_fdl_re = AlignedVec::<f32>::new(fdl_total, 0.0f32);
        let mut h_fdl_im = AlignedVec::<f32>::new(fdl_total, 0.0f32);

        // Reusable padded buffer for the forward RFFT (zero-padded IR segment)
        let mut padded = AlignedVec::<f32>::new(block_size, 0.0f32);
        let mut h_re = AlignedVec::<f32>::new(num_bins, 0.0f32);
        let mut h_im = AlignedVec::<f32>::new(num_bins, 0.0f32);

        for k in 0..num_partitions {
            let start = p + k * p;
            let end = (start + p).min(n);
            let seg_len = end - start;

            // Copy IR segment and zero-fill the tail of padded
            for i in 0..seg_len {
                padded[i] = weights[start + i];
            }
            padded[seg_len..].fill(0.0f32);

            rfft.process_forward(&padded, &mut h_re, &mut h_im);

            // Store spectrum at stride position k * num_bins
            let offset = k * num_bins;
            h_fdl_re[offset..offset + num_bins].copy_from_slice(&h_re);
            h_fdl_im[offset..offset + num_bins].copy_from_slice(&h_im);
        }

        // Initialize FDL with zeros (no past input yet)
        let fdl_re = AlignedVec::<f32>::new(fdl_total, 0.0f32);
        let fdl_im = AlignedVec::<f32>::new(fdl_total, 0.0f32);

        // fdl_write_idx starts at K-1 so that the first advance wraps to 0
        let fdl_write_idx = num_partitions.saturating_sub(1);

        Self {
            p,
            n,
            num_partitions,
            num_bins,
            rfft,
            h_fdl_re,
            h_fdl_im,
            fdl_re,
            fdl_im,
            fdl_write_idx,
            input_buf: AlignedVec::<f32>::new(block_size, 0.0f32),
            fft_re: AlignedVec::<f32>::new(num_bins, 0.0f32),
            fft_im: AlignedVec::<f32>::new(num_bins, 0.0f32),
            acc_re: AlignedVec::<f32>::new(num_bins, 0.0f32),
            acc_im: AlignedVec::<f32>::new(num_bins, 0.0f32),
            output_buf: AlignedVec::<f32>::new(block_size, 0.0f32),
            tail_output_buf: AlignedVec::<f32>::new(p, 0.0f32),
            sample_counter: 0,
        }
    }

    /// Returns a slice over the real spectrum of tail partition `k`.
    ///
    /// # Panics
    ///
    /// Panics if `k >= num_partitions`.
    #[inline]
    pub fn h_fdl_re_partition(&self, k: usize) -> &[f32] {
        let offset = k * self.num_bins;
        &self.h_fdl_re[offset..offset + self.num_bins]
    }

    /// Resets all runtime buffers to zero and re-initializes counters.
    ///
    /// This operation is allocation-free: it only zero-fills the existing
    /// pre-allocated buffers. The pre-computed `h_fdl_*` spectra are
    /// **not** modified (they depend only on the IR, which is static).
    #[cold]
    pub fn reset(&mut self) {
        self.fdl_re.fill(0.0f32);
        self.fdl_im.fill(0.0f32);
        self.input_buf.fill(0.0f32);
        self.fft_re.fill(0.0f32);
        self.fft_im.fill(0.0f32);
        self.acc_re.fill(0.0f32);
        self.acc_im.fill(0.0f32);
        self.output_buf.fill(0.0f32);
        self.tail_output_buf.fill(0.0f32);
        self.fdl_write_idx = self.num_partitions.saturating_sub(1);
        self.sample_counter = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zero_tail_partitions_when_p_equals_n() {
        let weights = vec![1.0f32; 256];
        let state = LinearFftState::new(256, &weights);
        assert_eq!(state.p, 256);
        assert_eq!(state.n, 256);
        assert_eq!(state.num_partitions, 0);
        assert_eq!(state.num_bins, 257);
        assert_eq!(state.h_fdl_re.len(), 0);
        assert_eq!(state.fdl_re.len(), 0);
        assert_eq!(state.tail_output_buf.len(), 256);
    }

    #[test]
    fn new_one_tail_partition_when_n_equals_2p() {
        let weights = vec![1.0f32; 512];
        let state = LinearFftState::new(256, &weights);
        assert_eq!(state.p, 256);
        assert_eq!(state.n, 512);
        assert_eq!(state.num_partitions, 1);
        assert_eq!(state.num_bins, 257);
        // Flat length = K × num_bins = 1 × 257
        assert_eq!(state.h_fdl_re.len(), 257);
        assert_eq!(state.fdl_re.len(), 257);
    }

    #[test]
    fn new_partitions_for_ir_8192_p_512() {
        let weights = vec![0.0f32; 8192];
        let state = LinearFftState::new(512, &weights);
        // ceil((8192-512)/512) = ceil(7680/512) = 15
        assert_eq!(state.num_partitions, 15);
        assert_eq!(state.num_bins, 513);
        // Flat length = 15 × 513 = 7695
        assert_eq!(state.h_fdl_re.len(), 7695);
        assert_eq!(state.fdl_re.len(), 7695);
        assert_eq!(state.tail_output_buf.len(), 512);
        assert_eq!(state.input_buf.len(), 1024);
    }

    #[test]
    fn new_partitions_uneven_last_block() {
        // N=1000, P=256: ceil((1000-256)/256) = ceil(744/256) = 3
        let weights = vec![0.0f32; 1000];
        let state = LinearFftState::new(256, &weights);
        assert_eq!(state.num_partitions, 3);
        // Last partition covers IR[768..1000] = 232 samples, zero-padded to 512
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn new_panics_on_non_power_of_two_p() {
        let weights = vec![0.0f32; 1024];
        LinearFftState::new(300, &weights);
    }

    #[test]
    #[should_panic(expected = "P (1024) must be ≤ N (512)")]
    fn new_panics_when_p_greater_than_n() {
        let weights = vec![0.0f32; 512];
        LinearFftState::new(1024, &weights);
    }

    #[test]
    fn reset_zeros_runtime_buffers() {
        let mut weights = vec![0.0f32; 256];
        weights[0] = 1.0;
        let mut state = LinearFftState::new(128, &weights);
        assert_eq!(state.num_partitions, 1);

        // Dirty the runtime buffers
        state.tail_output_buf[0] = 99.0;
        state.fdl_re[0] = 99.0;
        state.fdl_im[0] = 99.0;
        state.sample_counter = 42;
        state.fdl_write_idx = 0;

        state.reset();

        assert_eq!(state.sample_counter, 0);
        assert_eq!(state.fdl_write_idx, state.num_partitions.saturating_sub(1));
        for v in state.tail_output_buf.iter() {
            assert!((*v).abs() < f32::EPSILON);
        }
        // FDL must be zeroed
        for v in state.fdl_re.iter() {
            assert!((*v).abs() < f32::EPSILON);
        }
        // h_fdl should NOT be zeroed (pre-computed IR spectra)
        assert!(!state.h_fdl_re.is_empty());
    }

    #[test]
    fn debug_format_does_not_leak_internal_buffers() {
        let weights = vec![1.0f32; 512];
        let state = LinearFftState::new(256, &weights);
        let dbg = format!("{state:?}");
        assert!(dbg.contains("LinearFftState"));
        assert!(dbg.contains("p"));
        assert!(dbg.contains("n"));
        assert!(dbg.contains("..")); // finish_non_exhaustive marker
    }

    #[test]
    fn h_spectra_nonzero_for_nonzero_ir_tail() {
        let mut weights = vec![0.0f32; 512];
        // Put an impulse in the tail at index P (256)
        weights[256] = 1.0;
        let state = LinearFftState::new(256, &weights);
        assert_eq!(state.num_partitions, 1);
        // The spectrum of an impulse delayed by 0 within its block is all ones (real)
        // But it's zero-padded to 512 then FFT'd of 512 real → 257 complex bins
        // DC should be 1.0 (sum of all samples = 1.0)
        // Partition 0, bin 0 = state.h_fdl_re[0]
        assert!((state.h_fdl_re[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn weights_head_portion_not_used_for_spectra() {
        // IR: head [1,2,3,4], tail [5,6,7,8] with P=4
        let weights: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let state = LinearFftState::new(4, &weights);
        assert_eq!(state.num_partitions, 1);
        // The head portion [1,2,3,4] should NOT affect h_fdl.
        // Only tail [5,6,7,8] is transformed (zero-padded to 8).
        // DC of [5,6,7,8,0,0,0,0] = 5+6+7+8 = 26
        // Partition 0, bin 0 = state.h_fdl_re[0]
        assert!((state.h_fdl_re[0] - 26.0).abs() < 1e-4);
    }

    #[test]
    fn h_fdl_re_partition_accessor() {
        let weights = vec![1.0f32; 512];
        let state = LinearFftState::new(256, &weights);
        assert_eq!(state.num_partitions, 1);
        let part = state.h_fdl_re_partition(0);
        assert_eq!(part.len(), 257);
        // DC bin should be 256.0 (sum of 256 ones, zero-padded)
        assert!((part[0] - 256.0).abs() < 1e-4);
    }
}
