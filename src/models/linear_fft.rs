// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Zero-latency partitioned FFT convolution state for the Linear model.
//!
//! Implements hybrid convolution: the direct head (first `P` samples) is
//! computed in the time domain, while the tail (remaining `N-P` samples) is
//! computed in the frequency domain via overlap-save FFT blocks.
//!
//! Detailed buffer layout and processing logic is specified in Sprint 4 tasks
//! 4 and 5. This file currently holds the structural definition only.

/// State for zero-latency partitioned FFT (overlap-save) convolution.
///
/// Handles the tail portion of the impulse response (samples from partition
/// size `P` to the end `N-1`). The head (samples 0..P) is computed directly
/// in the time domain by `LinearModel::process_sample`.
///
/// # Buffers (to be implemented in Sprint 4)
/// - `rfft`: forward/inverse FFT planner.
/// - `h_fdl_re`, `h_fdl_im`: pre-computed spectra of the IR tail partitions.
/// - `fdl_re`, `fdl_im`: frequency delay line (circular) for overlap-save.
/// - `input_buf`: input window of size `2P`.
/// - `output_buf`: complex output buffer post-IFFT.
/// - `tail_output_buf`: circular buffer with ready-to-read tail samples.
/// - `sample_counter`: internal read index for the tail block (0..P-1).
#[derive(Debug)]
pub struct LinearFftState {
    /// Partition size `P` (head length = tail block size).
    pub p: usize,
    /// Total receptive field `N` (= IR length).
    pub n: usize,
}
