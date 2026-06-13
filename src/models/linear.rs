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
use super::sealed;
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::math::common::AlignedVec;

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
}

impl LinearModel {
    /// Creates a new LinearModel with the given weights, bias.
    ///
    /// Weights are expected in **forward-time order** as stored in the `.nam`
    /// JSON (w[0] is the response at the current sample). They are reversed
    /// internally to match the C++ `nam::Linear` layout.
    ///
    /// Allocates the `MirroredBuffer` for the input history. The buffer is
    /// initialized to zero (silence) by the operating system via `mmap`.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the `MirroredBuffer` allocation fails
    /// (e.g., out of memory or virtual address space).
    pub fn new(weights: Vec<f32>, bias: f32) -> std::io::Result<Self> {
        let receptive_field = weights.len();
        let mut aligned = AlignedVec::from_vec(weights);
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
        })
    }

    /// Processes a single audio sample using the Linear model.
    ///
    /// 1. Writes the sample into the ring buffer (`history`).
    /// 2. Advances the write pointer in the mirrored area.
    /// 3. Obtains a contiguous slice representing the receptive field window.
    /// 4. Computes the dot product plus the scalar bias via AVX2/AVX-512 SIMD.
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

        let start = self.write_pos - self.receptive_field;
        let window = &self.history[start..self.write_pos];
        // SAFETY: weights are 64-byte aligned (AlignedVec), window is contiguous
        // from MirroredBuffer (page-aligned), taps matches window/receptive_field.
        let dot = unsafe {
            crate::math::dsp::stereo::convolve_mono(
                self.weights.as_ptr(),
                window.as_ptr(),
                self.receptive_field,
            )
        };
        self.bias + dot
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
            unsafe { output[i] = self.process_sample(input[i]); }
        }
    }

    /// Fills the history buffer with zeros and resets the write pointer.
    #[cold]
    pub fn prewarm(&mut self, _num_samples: usize) {
        let size = self.history.size();
        for i in 0..(size * 2) {
            self.history[i] = 0.0;
        }
        self.write_pos = size;
    }

    /// Resets internal state: zeroes the history buffer and write pointer.
    #[cold]
    pub fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        let size = self.history.size();
        for i in 0..(size * 2) {
            self.history[i] = 0.0;
        }
        self.write_pos = size;
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

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) {
        self.reset(sample_rate, max_buffer_size);
    }

    fn prewarm_samples(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_unit_weight() {
        let model = LinearModel::new(vec![1.0], 0.0).unwrap();
        assert_eq!(model.receptive_field, 1);
        assert_eq!(model.weights.len(), 1);
    }

    #[test]
    fn test_linear_known_output() {
        let weights = vec![0.2, 0.3, 0.5];
        let bias = 0.1;
        let mut model = LinearModel::new(weights, bias).unwrap();

        model.prewarm(0);

        // After prewarm, history is all zeros. Stored weights are reversed: [0.5, 0.3, 0.2]
        // Feed [1.0]: window (oldest→newest) = [0, 0, 1.0]
        //   dot = 0.5*0 + 0.3*0 + 0.2*1.0 = 0.2 + bias=0.1 = 0.3
        let out0 = unsafe { model.process_sample(1.0) };
        let expected0 = 0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 1.0 + 0.1;
        assert!(
            (out0 - expected0).abs() < 1e-6,
            "out0={out0}, expected={expected0}"
        );

        // Feed [2.0]: window (oldest→newest) = [0, 1.0, 2.0]
        //   dot = 0.5*0 + 0.3*1.0 + 0.2*2.0 = 0.7 + bias=0.1 = 0.8
        let out1 = unsafe { model.process_sample(2.0) };
        let expected1 = 0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 2.0 + 0.1;
        assert!(
            (out1 - expected1).abs() < 1e-6,
            "out1={out1}, expected={expected1}"
        );

        // Feed [3.0]: window (oldest→newest) = [1.0, 2.0, 3.0]
        //   dot = 0.5*1.0 + 0.3*2.0 + 0.2*3.0 = 0.5+0.6+0.6 = 1.7 + bias=0.1 = 1.8
        let out2 = unsafe { model.process_sample(3.0) };
        let expected2 = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0 + 0.1;
        assert!(
            (out2 - expected2).abs() < 1e-6,
            "out2={out2}, expected={expected2}"
        );
    }

    #[test]
    fn test_linear_zero_output() {
        let mut model = LinearModel::new(vec![0.0, 0.0, 0.0], 0.0).unwrap();
        model.prewarm(0);
        let out = unsafe { model.process_sample(5.0) };
        assert!((out - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_process_block() {
        let mut model = LinearModel::new(vec![1.0], 0.0).unwrap();
        model.prewarm(0);

        let input = [0.1, 0.2, 0.3];
        let mut output = [0.0f32; 3];
        unsafe { model.process(&input, &mut output) };

        // With weight=1 (reversed), bias=0: output = input
        for i in 0..3 {
            assert!(
                (output[i] - input[i]).abs() < 1e-6,
                "output[{i}]={}, expected={}",
                output[i],
                input[i]
            );
        }
    }

    #[test]
    fn test_linear_reset() {
        let mut model = LinearModel::new(vec![0.5, 0.5], 0.0).unwrap();
        model.prewarm(0);

        let out1 = unsafe { model.process_sample(1.0) };
        model.reset(0, 0);

        let out2 = unsafe { model.process_sample(1.0) };
        assert!(
            (out1 - out2).abs() < 1e-6,
            "reset should reproduce the same output: {out1} != {out2}"
        );
    }

    #[test]
    fn test_linear_prewarm_samples_zero() {
        let model = LinearModel::new(vec![1.0; 16], 0.0).unwrap();
        assert_eq!(model.prewarm_samples(), 0);
    }
}
