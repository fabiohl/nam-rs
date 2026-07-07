// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Linear Model — Finite Impulse Response (FIR) network architecture for NAM.
//!
//! The Linear architecture implements a simple linear filter: the output at each
//! time step is obtained by the dot product of the model weights with a window of
//! input history (receptive field), plus a scalar bias:
//!
//! `output = bias + dot(weights, history_window)`
//!
//! Weights are stored in **reversed** order (matching C++ `nam::Linear` internal
//! layout) so that a dot product with the oldest-to-newest history window yields
//! the FIR convolution directly. The input history is stored in a
//! `MirroredBuffer<f32>`, which provides branch-free, contiguous access via
//! mirrored memory mapping — eliminating ring-buffer wrap-around logic in the
//! audio hot-path.
//!
//! # C++ Parity
//! This implementation matches `NeuralAmpModelerCore/NAM/dsp.cpp:255-301`
//! exactly: JSON weights are reversed on construction, and the dot product is
//! computed with the oldest-to-newest history window plus the scalar bias,
//! without tanh or head_scale (those are exclusive to WaveNet).

use super::NamModel;
use super::linear_fft::LinearFftState;
use super::sealed;
use crate::common::diagnostics::NamErrorCode;
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::loader::nam_json::LinearImplementation;
use crate::math::common::AlignedVec;
use log::warn;

/// Runtime convolution mode for the Linear model.
///
/// Controls whether the model uses direct time-domain convolution or
/// zero-latency partitioned FFT (hybrid: direct head + FFT tail).
#[derive(Debug)]
pub enum LinearMode {
    /// Direct time-domain convolution — dot product over the full receptive field.
    Direct,
    /// FFT partitioned convolution with `LinearFftState` for the tail.
    Fft(Box<LinearFftState>),
}

/// Linear Model — lightweight FIR-based neural model.
///
/// This is the simplest NAM architecture: a single linear layer (dot product)
/// applied over the recent sample history with an optional scalar bias.
///
/// # RT-Safety
/// - Zero allocation on the hot-path (`process`).
/// - Uses `MirroredBuffer` for branch-free ring buffer access.
/// - No locks, no `unwrap()`, no I/O.
pub struct LinearModel {
    /// FIR filter weights stored in **reversed** order (matching C++ internal
    /// layout). JSON weights are reversed on construction, so that
    /// `dot(weights, oldest_to_newest_window)` produces the FIR convolution.
    /// 64-byte aligned for AVX2/AVX-512 SIMD loads.
    pub weights: AlignedVec<f32>,
    /// Scalar bias added after the dot product.
    pub bias: f32,
    /// Circular buffer of past input samples, backed by mirrored memory mapping
    /// for branch-free contiguous access across the wrap boundary.
    pub history: MirroredBuffer<f32>,
    /// Current write position in the `history` ring buffer (0..receptive_field-1).
    pub write_pos: usize,
    /// Number of input samples in the receptive field (= `weights.len()`).
    pub receptive_field: usize,
    /// Precalculated limit * 2 to avoid runtime multiplication overflow checks.
    double_limit: usize,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Convolution implementation mode as configured in the JSON.
    pub implementation: LinearImplementation,
    /// Runtime convolution mode — `Direct` or `Fft` with partitioned FFT state.
    pub mode: LinearMode,
}

/// Minimum receptive field (taps) for auto-selecting FFT partitioned convolution.
///
/// Below this threshold, time-domain direct convolution is more efficient
/// due to FFT overhead.
const FFT_AUTO_THRESHOLD: usize = 256;

/// Largest power of two ≤ `n`.
const fn largest_power_of_two_le(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut v = n;
    let mut r = 1;
    while v > 1 {
        r <<= 1;
        v >>= 1;
    }
    r
}

/// Selects the partition size `P` for FFT hybrid convolution.
///
/// Returns the largest power of two ≤ `receptive_field / 2`, guaranteeing
/// that `2 * P ≤ receptive_field` — which ensures the `block_start`
/// subtraction never underflows in the hot-path.
fn select_partition_size(receptive_field: usize) -> usize {
    let max_p = receptive_field / 2;
    largest_power_of_two_le(max_p.max(1))
}

impl LinearModel {
    /// Creates a new LinearModel with the given weights, bias, and implementation.
    ///
    /// Weights are expected in **forward-time order** as stored in the `.nam`
    /// JSON (w[0] is the response at the current sample). They are reversed
    /// internally to match the C++ `nam::Linear` layout.
    ///
    /// `implementation` controls the convolution strategy (`Auto`, `Direct`, `Fft`)
    /// as configured in the model's JSON:
    /// - `Direct`: always uses time-domain dot product.
    /// - `Auto`: uses FFT when `receptive_field >= 256`, otherwise Direct.
    /// - `Fft`: uses FFT partitioned convolution; falls back to Direct with a
    ///   warning if the receptive field is too small (< 256).
    ///
    /// Allocates the `MirroredBuffer` for the input history. The buffer is
    /// initialized to zero (silence) by the operating system via `mmap`.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the `MirroredBuffer` allocation fails
    /// (e.g., out of memory or virtual address space).
    pub fn new(
        weights: Vec<f32>,
        bias: f32,
        implementation: LinearImplementation,
    ) -> std::io::Result<Self> {
        let receptive_field = weights.len();
        let mode = Self::resolve_mode(implementation, receptive_field, &weights)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;
        let mut aligned = AlignedVec::from_vec(weights)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;
        aligned.reverse();
        let history = MirroredBuffer::<f32>::new(receptive_field)?;
        let limit = history.size();
        let double_limit = limit.checked_mul(2).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Limit overflow")
        })?;
        Ok(Self {
            weights: aligned,
            bias,
            history,
            write_pos: limit,
            receptive_field,
            double_limit,
            prewarm_on_reset: true,
            implementation,
            mode,
        })
    }

    /// Resolves which convolution mode to use based on the requested
    /// implementation and the receptive field size.
    fn resolve_mode(
        implementation: LinearImplementation,
        receptive_field: usize,
        weights: &[f32],
    ) -> Result<LinearMode, NamErrorCode> {
        match implementation {
            LinearImplementation::Direct => Ok(LinearMode::Direct),
            LinearImplementation::Auto => {
                if receptive_field >= FFT_AUTO_THRESHOLD {
                    let p = select_partition_size(receptive_field);
                    if p < receptive_field {
                        return Ok(LinearMode::Fft(Box::new(LinearFftState::new(p, weights)?)));
                    }
                }
                Ok(LinearMode::Direct)
            }
            LinearImplementation::Fft => {
                if receptive_field < FFT_AUTO_THRESHOLD {
                    warn!(
                        "[Linear] Fft requested but receptive_field={receptive_field} < {FFT_AUTO_THRESHOLD} \
                         — falling back to Direct"
                    );
                    return Ok(LinearMode::Direct);
                }
                let p = select_partition_size(receptive_field);
                Ok(LinearMode::Fft(Box::new(LinearFftState::new(p, weights)?)))
            }
        }
    }

    /// Processes a single audio sample using the Linear model.
    ///
    /// 1. Writes the sample into the ring buffer (`history`).
    /// 2. Advances the write pointer in the mirrored area.
    /// 3. Dispatches according to the active `mode`:
    ///    - **Direct**: dot product over the full receptive field + bias.
    ///    - **FFT**: dot product over the head (`P` taps) + bias + tail sample
    ///      from the pre-computed `tail_output_buf`. Every `P` samples, a new
    ///      tail block is computed via `LinearFftState::process_tail_block`.
    ///
    /// # Safety
    /// `self.weights` must be 64-byte aligned (guaranteed by `AlignedVec`).
    #[inline(always)]
    unsafe fn process_sample(&mut self, input: f32) -> f32 {
        self.history[self.write_pos] = input;

        self.write_pos += 1;
        if self.write_pos >= self.double_limit {
            self.write_pos -= self.history.size();
        }

        match &mut self.mode {
            LinearMode::Direct => {
                let start = self.write_pos - self.receptive_field;
                let window = &self.history[start..self.write_pos];
                let dot = unsafe {
                    crate::math::dsp::stereo::convolve_mono(
                        self.weights.as_ptr(),
                        window.as_ptr(),
                        self.receptive_field,
                    )
                };
                self.bias + dot
            }
            LinearMode::Fft(state) => {
                let p = state.p;

                // Head convolution: last P weights × last P window samples
                let head_weights_ptr =
                    unsafe { self.weights.as_ptr().add(self.receptive_field - p) };
                let head_start = self.write_pos - p;
                let head_window = &self.history[head_start..self.write_pos];
                let head_dot = unsafe {
                    crate::math::dsp::stereo::convolve_mono(
                        head_weights_ptr,
                        head_window.as_ptr(),
                        p,
                    )
                };

                let y_tail = state.tail_output_buf[state.sample_counter];
                state.sample_counter += 1;

                if state.sample_counter >= p {
                    let block_start = self.write_pos - 2 * p;
                    let block_window = &self.history[block_start..self.write_pos];
                    state.process_tail_block(block_window);
                    state.sample_counter = 0;
                }

                self.bias + head_dot + y_tail
            }
        }
    }

    /// Processes a block of audio samples.
    ///
    /// # Safety
    /// `self.weights` must be 64-byte aligned.
    #[inline(always)]
    pub unsafe fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let n = core::cmp::min(input.len(), output.len());
        for i in 0..n {
            // SAFETY: process_sample requires self.weights to be 64-byte aligned (guaranteed by AlignedVec).
            unsafe {
                output[i] = self.process_sample(input[i]);
            }
        }
    }

    /// Fills the history buffer with zeros, resets the write pointer, and
    /// reinitializes the FFT state (if active).
    #[cold]
    pub fn prewarm(&mut self, _num_samples: usize) {
        let size = self.history.size();
        for i in 0..(size * 2) {
            self.history[i] = 0.0;
        }
        self.write_pos = size;
        if let LinearMode::Fft(ref mut state) = self.mode {
            state.reset();
        }
    }

    /// Resets internal state: zeroes the history buffer, write pointer,
    /// and FFT state (if active).
    #[cold]
    pub fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        let size = self.history.size();
        for i in 0..(size * 2) {
            self.history[i] = 0.0;
        }
        self.write_pos = size;
        if let LinearMode::Fft(ref mut state) = self.mode {
            state.reset();
        }
    }
}

impl sealed::Sealed for LinearModel {}

impl NamModel for LinearModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // SAFETY: weights are 64-byte aligned (AlignedVec).
        unsafe { self.process(input, output) };
    }

    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        self.prewarm(num_samples);
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        if self.prewarm_on_reset {
            self.reset(sample_rate, max_buffer_size);
        }
        Ok(())
    }

    fn prewarm_samples(&self) -> usize {
        0
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
    }
}

#[cfg(test)]
#[path = "linear_test.rs"]
mod tests;
