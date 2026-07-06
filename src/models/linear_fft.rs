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

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;
use crate::math::common::Avx2Math;
use crate::math::common::Avx512Math;
use crate::math::common::Avx512VnniBf16Math;
use crate::math::common::dispatch::InstructionSet;
use crate::math::common::dispatch::SimdMathConfig;
use crate::math::common::traits::SimdMath;
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
    /// Points to the next position that will be written. In `process_tail_block`,
    /// the FDL is read before writing: old spectra are consumed for the tail
    /// convolution, then the new input spectrum replaces the oldest entry.
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
    /// Instruction set captured at construction time to avoid runtime CPU
    /// feature checks in the audio hot path.
    pub isa: InstructionSet,
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
    pub fn new(p: usize, weights: &[f32]) -> Result<Self, NamErrorCode> {
        let n = weights.len();
        assert!(p.is_power_of_two(), "P must be a power of two, got {p}");
        assert!(p <= n, "P ({p}) must be ≤ N ({n})");

        let block_size = 2 * p;
        let num_bins = p + 1;
        // ceil((N-P)/P), saturated at 0 when P==N
        let num_partitions = if n > p { (n - p).div_ceil(p) } else { 0 };

        let mut rfft = RfftPlanner::<f32>::new(block_size);

        let fdl_total = num_partitions * num_bins;
        let mut h_fdl_re = AlignedVec::<f32>::new(fdl_total, 0.0f32)?;
        let mut h_fdl_im = AlignedVec::<f32>::new(fdl_total, 0.0f32)?;

        // Reusable padded buffer for the forward RFFT (zero-padded IR segment)
        let mut padded = AlignedVec::<f32>::new(block_size, 0.0f32)?;
        let mut h_re = AlignedVec::<f32>::new(num_bins, 0.0f32)?;
        let mut h_im = AlignedVec::<f32>::new(num_bins, 0.0f32)?;

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
        let fdl_re = AlignedVec::<f32>::new(fdl_total, 0.0f32)?;
        let fdl_im = AlignedVec::<f32>::new(fdl_total, 0.0f32)?;

        // fdl_write_idx starts at K-1 so that the first read uses all-zero
        // FDL entries (correct for silence before the first block).
        let fdl_write_idx = num_partitions.saturating_sub(1);
        let isa = SimdMathConfig::current().instruction_set;

        Ok(Self {
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
            input_buf: AlignedVec::<f32>::new(block_size, 0.0f32)?,
            fft_re: AlignedVec::<f32>::new(num_bins, 0.0f32)?,
            fft_im: AlignedVec::<f32>::new(num_bins, 0.0f32)?,
            acc_re: AlignedVec::<f32>::new(num_bins, 0.0f32)?,
            acc_im: AlignedVec::<f32>::new(num_bins, 0.0f32)?,
            output_buf: AlignedVec::<f32>::new(block_size, 0.0f32)?,
            tail_output_buf: AlignedVec::<f32>::new(p, 0.0f32)?,
            sample_counter: 0,
            isa,
        })
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

    /// Processes one block of tail convolution using overlap-save FFT.
    ///
    /// `input_window` must be a contiguous slice of the last `2P` input
    /// samples (oldest to newest), typically obtained from the
    /// `MirroredBuffer` via `history[write_pos - 2*P .. write_pos]`.
    ///
    /// After this call completes, `tail_output_buf` contains `P` valid
    /// tail output samples ready for sequential per-sample consumption
    /// via `sample_counter`.
    ///
    /// # Algorithm (overlap-save, zero-latency hybrid)
    ///
    /// 1. Compute forward RFFT of the `2P`-sample input window.
    /// 2. Read past input spectra from the circular FDL (delays `P`,
    ///    `2P`, …, `K×P`), multiply by the corresponding pre-computed
    ///    tail IR spectra in the frequency domain using SIMD complex MAC.
    /// 3. Store the new input spectrum in the FDL and advance the write
    ///    index (overwrites the oldest entry, now `K+1` blocks ago).
    /// 4. Inverse RFFT the accumulated spectrum back to time domain.
    /// 5. Extract the valid `P` output samples (overlap-save: indices
    ///    `P..2P-1`) into `tail_output_buf`.
    ///
    /// # RT-Safety
    ///
    /// Zero heap allocation, zero locks, zero panics in production
    /// (debug assertions only). All buffers were pre-allocated at
    /// construction time. The ISA for SIMD dispatch was captured once
    /// at construction time — no runtime CPU feature checks on the hot
    /// path.
    pub fn process_tail_block(&mut self, input_window: &[f32]) {
        let p = self.p;
        let block_size = 2 * p;
        let num_bins = self.num_bins;
        let num_partitions = self.num_partitions;

        debug_assert_eq!(input_window.len(), block_size);

        if num_partitions == 0 {
            return;
        }

        // ── Step 1: Copy input window and compute forward RFFT ──
        self.input_buf[..block_size].copy_from_slice(input_window);
        self.rfft
            .process_forward(&self.input_buf, &mut self.fft_re, &mut self.fft_im);

        // ── Step 2: Frequency-domain MAC ──
        // The tail output for the NEXT block needs the current input spectrum
        // (block B) for partition 0 (delay P) and FDL entries (blocks B-1,
        // B-2, ...) for partitions 1..K-1 (delays 2P, 3P, ..., K×P).
        // SAFETY: all slices have length num_bins, guaranteed by construction.
        // ISA was captured at construction time.
        self.acc_re[..num_bins].fill(0.0);
        self.acc_im[..num_bins].fill(0.0);

        // Partition 0 (delays P..2P−1): uses the current block's input spectrum.
        unsafe {
            match self.isa {
                InstructionSet::Avx512VnniBf16 => Avx512VnniBf16Math::complex_mac_accumulate(
                    &self.h_fdl_re[..num_bins],
                    &self.h_fdl_im[..num_bins],
                    &self.fft_re[..num_bins],
                    &self.fft_im[..num_bins],
                    &mut self.acc_re[..num_bins],
                    &mut self.acc_im[..num_bins],
                ),
                InstructionSet::Avx512 => Avx512Math::complex_mac_accumulate(
                    &self.h_fdl_re[..num_bins],
                    &self.h_fdl_im[..num_bins],
                    &self.fft_re[..num_bins],
                    &self.fft_im[..num_bins],
                    &mut self.acc_re[..num_bins],
                    &mut self.acc_im[..num_bins],
                ),
                InstructionSet::Avx2 => Avx2Math::complex_mac_accumulate(
                    &self.h_fdl_re[..num_bins],
                    &self.h_fdl_im[..num_bins],
                    &self.fft_re[..num_bins],
                    &self.fft_im[..num_bins],
                    &mut self.acc_re[..num_bins],
                    &mut self.acc_im[..num_bins],
                ),
            }
        }

        // Partitions 1..K−1 (delays 2P..K×P): use past input spectra from FDL.
        for k in 1..num_partitions {
            let input_idx = (self.fdl_write_idx + num_partitions - k) % num_partitions;
            let fdl_start = input_idx * num_bins;
            let h_start = k * num_bins;

            unsafe {
                match self.isa {
                    InstructionSet::Avx512VnniBf16 => Avx512VnniBf16Math::complex_mac_accumulate(
                        &self.h_fdl_re[h_start..h_start + num_bins],
                        &self.h_fdl_im[h_start..h_start + num_bins],
                        &self.fdl_re[fdl_start..fdl_start + num_bins],
                        &self.fdl_im[fdl_start..fdl_start + num_bins],
                        &mut self.acc_re[..num_bins],
                        &mut self.acc_im[..num_bins],
                    ),
                    InstructionSet::Avx512 => Avx512Math::complex_mac_accumulate(
                        &self.h_fdl_re[h_start..h_start + num_bins],
                        &self.h_fdl_im[h_start..h_start + num_bins],
                        &self.fdl_re[fdl_start..fdl_start + num_bins],
                        &self.fdl_im[fdl_start..fdl_start + num_bins],
                        &mut self.acc_re[..num_bins],
                        &mut self.acc_im[..num_bins],
                    ),
                    InstructionSet::Avx2 => Avx2Math::complex_mac_accumulate(
                        &self.h_fdl_re[h_start..h_start + num_bins],
                        &self.h_fdl_im[h_start..h_start + num_bins],
                        &self.fdl_re[fdl_start..fdl_start + num_bins],
                        &self.fdl_im[fdl_start..fdl_start + num_bins],
                        &mut self.acc_re[..num_bins],
                        &mut self.acc_im[..num_bins],
                    ),
                }
            }
        }

        // ── Step 3: Store new input spectrum in FDL ──
        {
            let fdl_base = self.fdl_write_idx * num_bins;
            self.fdl_re[fdl_base..fdl_base + num_bins].copy_from_slice(&self.fft_re);
            self.fdl_im[fdl_base..fdl_base + num_bins].copy_from_slice(&self.fft_im);
        }

        // ── Step 4: Advance FDL write index ──
        self.fdl_write_idx += 1;
        if self.fdl_write_idx >= num_partitions {
            self.fdl_write_idx = 0;
        }

        // ── Step 5: Inverse RFFT (complex → real, 2P samples) ──
        self.rfft
            .process_inverse(&mut self.acc_re, &mut self.acc_im, &mut self.output_buf);

        // ── Step 6: Extract valid output (overlap-save: samples P..2P-1) ──
        self.tail_output_buf[..p].copy_from_slice(&self.output_buf[p..block_size]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zero_tail_partitions_when_p_equals_n() {
        let weights = vec![1.0f32; 256];
        let state = LinearFftState::new(256, &weights)
            .expect("construction should succeed for test-sized buffers");
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
        let state = LinearFftState::new(256, &weights)
            .expect("construction should succeed for test-sized buffers");
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
        let state = LinearFftState::new(512, &weights)
            .expect("construction should succeed for test-sized buffers");
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
        let state = LinearFftState::new(256, &weights)
            .expect("construction should succeed for test-sized buffers");
        assert_eq!(state.num_partitions, 3);
        // Last partition covers IR[768..1000] = 232 samples, zero-padded to 512
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn new_panics_on_non_power_of_two_p() {
        let weights = vec![0.0f32; 1024];
        LinearFftState::new(300, &weights)
            .expect("construction should succeed for test-sized buffers");
    }

    #[test]
    #[should_panic(expected = "P (1024) must be ≤ N (512)")]
    fn new_panics_when_p_greater_than_n() {
        let weights = vec![0.0f32; 512];
        LinearFftState::new(1024, &weights)
            .expect("construction should succeed for test-sized buffers");
    }

    #[test]
    fn reset_zeros_runtime_buffers() {
        let mut weights = vec![0.0f32; 256];
        weights[0] = 1.0;
        let mut state = LinearFftState::new(128, &weights)
            .expect("construction should succeed for test-sized buffers");
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
        let state = LinearFftState::new(256, &weights)
            .expect("construction should succeed for test-sized buffers");
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
        let state = LinearFftState::new(256, &weights)
            .expect("construction should succeed for test-sized buffers");
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
        let state = LinearFftState::new(4, &weights)
            .expect("construction should succeed for test-sized buffers");
        assert_eq!(state.num_partitions, 1);
        // The head portion [1,2,3,4] should NOT affect h_fdl.
        // Only tail [5,6,7,8] is transformed (zero-padded to 8).
        // DC of [5,6,7,8,0,0,0,0] = 5+6+7+8 = 26
        // Partition 0, bin 0 = state.h_fdl_re[0]
        assert!((state.h_fdl_re[0] - 26.0).abs() < 1e-4);
    }

    #[test]
    fn process_tail_block_noop_when_zero_partitions() {
        let weights = vec![1.0f32; 256];
        let mut state = LinearFftState::new(256, &weights)
            .expect("construction should succeed for test-sized buffers");
        assert_eq!(state.num_partitions, 0);
        // Should not panic when called with no tail partitions
        state.process_tail_block(&[0.0f32; 512]);
        // tail_output_buf should remain all zeros
        for &v in state.tail_output_buf.iter() {
            assert!((v - 0.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn process_tail_block_first_call_uses_current_spectrum() {
        // P=2, N=4, IR=[1.0, 2.0, 3.0, 4.0], tail=[3.0, 4.0]
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let mut state = LinearFftState::new(2, &weights)
            .expect("construction should succeed for test-sized buffers");
        assert_eq!(state.num_partitions, 1);
        // Fix: first call uses current input spectrum.
        // conv([3,4], [1,2,3,4]) at positions [2,3] = [3*3+4*2=17, 3*4+4*3=24]
        state.process_tail_block(&[1.0, 2.0, 3.0, 4.0]);
        assert!(
            (state.tail_output_buf[0] - 17.0).abs() < 1e-4,
            "expected 17, got {}",
            state.tail_output_buf[0]
        );
        assert!(
            (state.tail_output_buf[1] - 24.0).abs() < 1e-4,
            "expected 24, got {}",
            state.tail_output_buf[1]
        );
    }

    #[test]
    fn process_tail_block_numerical_correctness() {
        // P=2, N=4, IR = [0.5, 1.5, 3.0, 4.0]
        // head = [0.5, 1.5], tail = [3.0, 4.0]
        let ir = vec![0.5, 1.5, 3.0, 4.0];
        let p = 2;
        let mut state = LinearFftState::new(p, &ir)
            .expect("construction should succeed for test-sized buffers");

        // Block 1: feed [1, 2, 3, 4] — uses current input spectrum.
        // conv([3,4], [1,2,3,4]) valid region [2,3] = [17, 24]
        let block1 = [1.0, 2.0, 3.0, 4.0];
        state.process_tail_block(&block1);
        assert!(
            (state.tail_output_buf[0] - 17.0).abs() < 1e-4,
            "block 1[0]: expected 17, got {}",
            state.tail_output_buf[0]
        );
        assert!(
            (state.tail_output_buf[1] - 24.0).abs() < 1e-4,
            "block 1[1]: expected 24, got {}",
            state.tail_output_buf[1]
        );

        // Block 2: feed [5, 6, 7, 8] — uses current input spectrum.
        // conv([3,4], [5,6,7,8]) valid region [2,3] = [45, 52]
        let block2 = [5.0, 6.0, 7.0, 8.0];
        state.process_tail_block(&block2);
        assert!(
            (state.tail_output_buf[0] - 45.0).abs() < 1e-4,
            "block 2[0]: expected 45, got {}",
            state.tail_output_buf[0]
        );
        assert!(
            (state.tail_output_buf[1] - 52.0).abs() < 1e-4,
            "block 2[1]: expected 52, got {}",
            state.tail_output_buf[1]
        );
    }

    #[test]
    fn process_tail_block_multiple_partitions() {
        // P=2, N=8, K=ceil((8-2)/2)=3
        // IR = [h0, h1, t0, t1, t2, t3, t4, t5]
        let ir: Vec<f32> = vec![
            0.0, 0.0, // head (unused by FFT tail)
            1.0, 0.0, // tail partition 0 (delays 2,3)
            0.0, 1.0, // tail partition 1 (delays 4,5)
            0.5, 0.5, // tail partition 2 (delays 6,7)
        ];
        let p = 2;
        let mut state = LinearFftState::new(p, &ir)
            .expect("construction should succeed for test-sized buffers");
        assert_eq!(state.num_partitions, 3);

        // Block 1: feed [0, 0, 1, 0]
        // k=0: conv([1,0], [0,0,1,0]) at [2,3] = [1, 0]
        // k=1: FDL zeros → [0, 0]
        // k=2: FDL zeros → [0, 0]
        state.process_tail_block(&[0.0, 0.0, 1.0, 0.0]);
        assert!(
            (state.tail_output_buf[0] - 1.0).abs() < 1e-4,
            "block 1[0]: expected 1, got {}",
            state.tail_output_buf[0]
        );
        assert!(
            (state.tail_output_buf[1] - 0.0).abs() < 1e-4,
            "block 1[1]: expected 0, got {}",
            state.tail_output_buf[1]
        );

        // Block 2: feed [0, 1, 0, 0]
        // k=0: conv([1,0], [0,1,0,0]) at [2,3] = [0, 0]
        // k=1: FDL[2]=block 1's [0,0,1,0]. conv([0,1], [0,0,1,0]) at [2,3] = [0, 1]
        // k=2: FDL zeros → [0, 0]
        state.process_tail_block(&[0.0, 1.0, 0.0, 0.0]);
        assert!(
            (state.tail_output_buf[0] - 0.0).abs() < 1e-4,
            "block 2[0]: expected 0, got {}",
            state.tail_output_buf[0]
        );
        assert!(
            (state.tail_output_buf[1] - 1.0).abs() < 1e-4,
            "block 2[1]: expected 1, got {}",
            state.tail_output_buf[1]
        );

        // Block 3: feed [1, 0, 0, 0]
        // k=0: conv([1,0], [1,0,0,0]) at [2,3] = [0, 0]
        // k=1: FDL[0]=block 2's [0,1,0,0]. conv([0,1], [0,1,0,0]) at [2,3] = [1, 0]
        // k=2: FDL[2]=block 1's [0,0,1,0]. conv([0.5,0.5], [0,0,1,0]) at [2,3] = [0.5, 0.5]
        // Total: [1.5, 0.5]
        state.process_tail_block(&[1.0, 0.0, 0.0, 0.0]);
        assert!(
            (state.tail_output_buf[0] - 1.5).abs() < 1e-4,
            "block 3[0]: expected 1.5, got {}",
            state.tail_output_buf[0]
        );
        assert!(
            (state.tail_output_buf[1] - 0.5).abs() < 1e-4,
            "block 3[1]: expected 0.5, got {}",
            state.tail_output_buf[1]
        );
    }

    #[test]
    fn process_tail_block_fdl_circular_advance() {
        // P=2, N=8, K=3
        let ir = vec![0.0f32; 8];
        let p = 2;
        let mut state = LinearFftState::new(p, &ir)
            .expect("construction should succeed for test-sized buffers");
        assert_eq!(state.num_partitions, 3);
        assert_eq!(state.fdl_write_idx, 2); // K-1

        // Call 1: write at 2, advance to 0
        state.process_tail_block(&[1.0; 4]);
        assert_eq!(state.fdl_write_idx, 0);

        // Call 2: write at 0, advance to 1
        state.process_tail_block(&[2.0; 4]);
        assert_eq!(state.fdl_write_idx, 1);

        // Call 3: write at 1, advance to 2
        state.process_tail_block(&[3.0; 4]);
        assert_eq!(state.fdl_write_idx, 2);

        // Call 4: wraps back to 0
        state.process_tail_block(&[4.0; 4]);
        assert_eq!(state.fdl_write_idx, 0);
    }

    #[test]
    fn process_tail_block_reset_then_process() {
        let ir = vec![1.0, 2.0, 3.0, 4.0];
        let mut state = LinearFftState::new(2, &ir)
            .expect("construction should succeed for test-sized buffers");

        // Prime FDL with one block
        state.process_tail_block(&[1.0; 4]);
        assert_eq!(state.fdl_write_idx, 0);

        // Reset should clear FDL and counters
        state.reset();
        assert_eq!(state.fdl_write_idx, 0); // K-1 for K=1
        assert_eq!(state.sample_counter, 0);

        // After reset, first call uses current spectrum.
        // conv([3,4], [5,6,7,8]) at [2,3] = [3*7+4*6=45, 3*8+4*7=52]
        state.process_tail_block(&[5.0, 6.0, 7.0, 8.0]);
        assert!(
            (state.tail_output_buf[0] - 45.0).abs() < 1e-4,
            "expected 45, got {}",
            state.tail_output_buf[0]
        );
        assert!(
            (state.tail_output_buf[1] - 52.0).abs() < 1e-4,
            "expected 52, got {}",
            state.tail_output_buf[1]
        );
    }
}
