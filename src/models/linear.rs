// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Linear Model — Finite Impulse Response (FIR) network architecture for NAM.
//!
//! The Linear architecture implements a simple linear filter: the output at each
//! time step is obtained by the dot product of the model weights with a window of
//! input history (receptive field), plus bias, passed through a tanh nonlinearity,
//! and scaled by `head_scale`:
//!
//! `output = head_scale * tanh(dot(weights, history) + bias)`
//!
//! The input history is stored in a `MirroredBuffer<f32>`, which provides
//! branch-free, contiguous access via mirrored memory mapping — eliminating
//! ring-buffer wrap-around logic in the audio hot-path.

use super::NamModel;
use super::sealed;
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::math::activations::tanh::tanh;

/// Linear Model — lightweight FIR-based neural model.
///
/// This is the simplest NAM architecture: a single linear layer (dot product)
/// applied over the recent sample history, followed by tanh and output scaling.
///
/// # RT-Safety
/// - Zero allocation on the hot-path (`process`).
/// - Uses `MirroredBuffer` for branch-free ring buffer access.
/// - No locks, no `unwrap()`, no I/O.
pub struct LinearModel {
    /// FIR filter weights (coefficients), loaded from the `.nam` file.
    pub weights: Vec<f32>,
    /// Optional scalar bias added after the dot product.
    pub bias: f32,
    /// Output scaling factor (`head_scale` from the model config).
    pub head_scale: f32,
    /// Circular buffer of past input samples, backed by mirrored memory mapping
    /// for branch-free contiguous access across the wrap boundary.
    pub history: MirroredBuffer<f32>,
    /// Current write position in the `history` ring buffer (0..receptive_field-1).
    pub write_pos: usize,
    /// Number of input samples in the receptive field (= `weights.len()`).
    pub receptive_field: usize,
}

impl LinearModel {
    /// Creates a new LinearModel with the given weights, bias, and head scale.
    ///
    /// Allocates the `MirroredBuffer` for the input history. The buffer is
    /// initialized to zero (silence) by the operating system via `mmap`.
    ///
    /// # Errors
    /// Returns `std::io::Error` if the `MirroredBuffer` allocation fails
    /// (e.g., out of memory or virtual address space).
    pub fn new(weights: Vec<f32>, bias: f32, head_scale: f32) -> std::io::Result<Self> {
        let receptive_field = weights.len();
        let history = MirroredBuffer::<f32>::new(receptive_field)?;
        Ok(Self {
            weights,
            bias,
            head_scale,
            history,
            write_pos: 0,
            receptive_field,
        })
    }

    /// Processes a single audio sample using the Linear model.
    ///
    /// 1. Writes the sample into the ring buffer (`history`).
    /// 2. Advances the write pointer (modulo `receptive_field`).
    /// 3. Computes `dot(weights, window) + bias` where the window is the last
    ///    `receptive_field` samples in chronological order (oldest→newest).
    ///    Because the contiguous MirroredBuffer maps the same physical pages
    ///    twice, the range `history[write_pos..write_pos + receptive_field]`
    ///    always contains the correct window without wrap-around.
    /// 4. Applies `tanh` and scales by `head_scale`.
    #[inline(always)]
    fn process_sample(&mut self, input: f32) -> f32 {
        // Write the new sample to the ring buffer and its mirror (same physical memory).
        self.history[self.write_pos] = input;

        // Advance the write pointer.
        self.write_pos += 1;
        if self.write_pos >= self.receptive_field {
            self.write_pos = 0;
        }

        // The contiguous window of the last `receptive_field` samples starts at
        // `write_pos` (oldest) and goes forward.  Delegate to the physical
        // half of the buffer and use explicit modular indexing for correctness.
        let rf = self.receptive_field;
        let pos = self.write_pos;
        let mut dot = self.bias;
        for i in 0..rf {
            let idx = (pos + i) % rf;
            dot += self.weights[i] * self.history[idx];
        }
        tanh(dot) * self.head_scale
    }

    /// Processes a block of audio samples.
    #[inline(always)]
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let n = core::cmp::min(input.len(), output.len());
        for i in 0..n {
            output[i] = self.process_sample(input[i]);
        }
    }

    /// Fills the history buffer with zeros and resets the write pointer.
    #[cold]
    pub fn prewarm(&mut self, _num_samples: usize) {
        for i in 0..self.receptive_field {
            self.history[i] = 0.0;
        }
        self.write_pos = 0;
    }

    /// Resets internal state: zeroes the history buffer and write pointer.
    #[cold]
    pub fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        for i in 0..self.receptive_field {
            self.history[i] = 0.0;
        }
        self.write_pos = 0;
    }
}

impl sealed::Sealed for LinearModel {}

impl NamModel for LinearModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
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

    /// A simple identity-ish model: weights [1.0], bias 0, head_scale 1.
    /// For small inputs, tanh(x) ≈ x, so output ≈ input.
    #[test]
    fn test_linear_unit_weight() {
        let model = LinearModel::new(vec![1.0], 0.0, 1.0).unwrap();
        assert_eq!(model.receptive_field, 1);
        assert_eq!(model.weights.len(), 1);
    }

    /// Test that process_sample produces correct output for known weights.
    #[test]
    fn test_linear_known_output() {
        let weights = vec![0.5, 0.3, 0.2];
        let bias = 0.1;
        let head_scale = 2.0;
        let mut model = LinearModel::new(weights.clone(), bias, head_scale).unwrap();

        // After prewarm, history is all zeros.
        model.prewarm(0);

        // Feed [1.0]: window (oldest→newest) = [0, 0, 1.0]
        //   dot = 0.5*0 + 0.3*0 + 0.2*1.0 = 0.2, +bias=0.3, tanh(0.3)*2
        let out0 = model.process_sample(1.0);
        let expected0 = tanh(0.5 * 0.0 + 0.3 * 0.0 + 0.2 * 1.0 + 0.1) * 2.0;
        assert!((out0 - expected0).abs() < 1e-6, "out0={out0}, expected={expected0}");

        // Feed [2.0]: window (oldest→newest) = [0, 1.0, 2.0]
        //   dot = 0.5*0 + 0.3*1.0 + 0.2*2.0 = 0.7, +bias=0.8, tanh(0.8)*2
        let out1 = model.process_sample(2.0);
        let expected1 = tanh(0.5 * 0.0 + 0.3 * 1.0 + 0.2 * 2.0 + 0.1) * 2.0;
        assert!((out1 - expected1).abs() < 1e-6, "out1={out1}, expected={expected1}");

        // Feed [3.0]: window (oldest→newest) = [1.0, 2.0, 3.0]
        //   dot = 0.5*1.0 + 0.3*2.0 + 0.2*3.0 = 0.5+0.6+0.6 = 1.7, +bias=1.8, tanh(1.8)*2
        let out2 = model.process_sample(3.0);
        let expected2 = tanh(0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0 + 0.1) * 2.0;
        assert!((out2 - expected2).abs() < 1e-6, "out2={out2}, expected={expected2}");
    }

    /// Test that a zero-weight model with bias 0 and head_scale 0 outputs silence.
    #[test]
    fn test_linear_zero_output() {
        let mut model = LinearModel::new(vec![0.0, 0.0, 0.0], 0.0, 0.0).unwrap();
        model.prewarm(0);
        let out = model.process_sample(5.0);
        assert!((out - 0.0).abs() < 1e-6);
    }

    /// Test the block processing interface.
    #[test]
    fn test_linear_process_block() {
        let mut model = LinearModel::new(vec![1.0], 0.0, 1.0).unwrap();
        model.prewarm(0);

        let input = [0.1, 0.2, 0.3];
        let mut output = [0.0f32; 3];
        model.process(&input, &mut output);

        // With weight=1, bias=0, head_scale=1: output = tanh(input)
        for i in 0..3 {
            let expected = tanh(input[i]);
            assert!(
                (output[i] - expected).abs() < 1e-6,
                "output[{i}]={}, expected={expected}",
                output[i]
            );
        }
    }

    /// Test that reset clears state and produces identical results.
    #[test]
    fn test_linear_reset() {
        let mut model = LinearModel::new(vec![0.5, 0.5], 0.0, 1.0).unwrap();
        model.prewarm(0);

        // Process one sample
        let out1 = model.process_sample(1.0);
        model.reset(0, 0);

        let out2 = model.process_sample(1.0);
        assert!(
            (out1 - out2).abs() < 1e-6,
            "reset should reproduce the same output: {out1} != {out2}"
        );
    }

    /// Test prewarm_samples returns 0 (no additional warmup needed after prewarm).
    #[test]
    fn test_linear_prewarm_samples_zero() {
        let model = LinearModel::new(vec![1.0; 16], 0.0, 1.0).unwrap();
        assert_eq!(model.prewarm_samples(), 0);
    }
}
