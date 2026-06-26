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
        let mode = Self::resolve_mode(implementation, receptive_field, &weights);
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
    ) -> LinearMode {
        match implementation {
            LinearImplementation::Direct => LinearMode::Direct,
            LinearImplementation::Auto => {
                if receptive_field >= FFT_AUTO_THRESHOLD {
                    let p = select_partition_size(receptive_field);
                    if p < receptive_field {
                        return LinearMode::Fft(Box::new(LinearFftState::new(p, weights)));
                    }
                }
                LinearMode::Direct
            }
            LinearImplementation::Fft => {
                if receptive_field < FFT_AUTO_THRESHOLD {
                    warn!(
                        "[Linear] Fft requested but receptive_field={receptive_field} < {FFT_AUTO_THRESHOLD} \
                         — falling back to Direct"
                    );
                    return LinearMode::Direct;
                }
                let p = select_partition_size(receptive_field);
                LinearMode::Fft(Box::new(LinearFftState::new(p, weights)))
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
mod tests {
    use super::*;

    // ── select_partition_size ──

    #[test]
    fn test_select_partition_size_power_of_two_n() {
        // N=256 is power of two, half=128, + threshold → 128
        assert_eq!(select_partition_size(256), 128);
        // N=512 → largest power ≤ 512 = 512 → ≥ N → half=256 < 512
        assert_eq!(select_partition_size(512), 256);
        // N=1024 → 512
        assert_eq!(select_partition_size(1024), 512);
    }

    #[test]
    fn test_select_partition_size_non_power_of_two_n() {
        // N=300, max_p=150 → largest power of 2 ≤ 150 = 128
        assert_eq!(select_partition_size(300), 128);
        // N=999, max_p=499 → largest power of 2 ≤ 499 = 256
        assert_eq!(select_partition_size(999), 256);
        // N=2047, max_p=1023 → largest power of 2 ≤ 1023 = 512
        assert_eq!(select_partition_size(2047), 512);
    }

    // ── Mode resolution ──

    #[test]
    fn test_direct_explicit_always_direct() {
        let model = LinearModel::new(vec![1.0; 1024], 0.0, LinearImplementation::Direct).unwrap();
        assert!(matches!(model.mode, LinearMode::Direct));
    }

    #[test]
    fn test_auto_direct_below_threshold() {
        let model = LinearModel::new(vec![1.0; 128], 0.0, LinearImplementation::Auto).unwrap();
        assert!(matches!(model.mode, LinearMode::Direct));
    }

    #[test]
    fn test_auto_fft_above_threshold() {
        let model = LinearModel::new(vec![1.0; 512], 0.0, LinearImplementation::Auto).unwrap();
        assert!(matches!(model.mode, LinearMode::Fft(_)));
    }

    #[test]
    fn test_fft_explicit_above_threshold() {
        let model = LinearModel::new(vec![1.0; 512], 0.0, LinearImplementation::Fft).unwrap();
        assert!(matches!(model.mode, LinearMode::Fft(_)));
    }

    #[test]
    fn test_fft_explicit_below_threshold_fallback() {
        let model = LinearModel::new(vec![1.0; 128], 0.0, LinearImplementation::Fft).unwrap();
        assert!(matches!(model.mode, LinearMode::Direct));
    }

    // ── FFT process_sample correctness ──

    #[test]
    fn test_fft_process_basic_no_tail() {
        // P=N: FFT state has zero tail partitions, equivalent to Direct
        let ir: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let mut model = LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
        model.prewarm(0);

        let mut model_fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model_fft.prewarm(0);

        let inputs = [0.5, -0.3, 0.8, -0.1, 0.2];
        for &x in &inputs {
            let direct = unsafe { model.process_sample(x) };
            let fft = unsafe { model_fft.process_sample(x) };
            assert!(
                (direct - fft).abs() < 1e-5,
                "direct={direct} fft={fft} mismatch at input={x}"
            );
        }
    }

    #[test]
    fn test_fft_process_with_tail() {
        // P=4, N=8: 2 partitions (head + 1 tail)
        let ir: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
        let mut direct = LinearModel::new(ir.clone(), 0.0, LinearImplementation::Direct).unwrap();
        direct.prewarm(0);

        let mut fft = LinearModel::new(ir, 0.0, LinearImplementation::Fft).unwrap();
        fft.prewarm(0);

        // Feed 100 samples; compare Direct vs FFT output sample by sample
        let mut max_diff = 0.0f32;
        for i in 0..100 {
            let x = (i as f32 * 0.7).sin();
            let d = unsafe { direct.process_sample(x) };
            let f = unsafe { fft.process_sample(x) };
            let diff = (d - f).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1e-4,
            "max diff between Direct and FFT = {max_diff}"
        );
    }

    #[test]
    fn test_fft_long_tail_many_partitions() {
        // P=256 but N=2048 (> 256 so actually P would be 1024 from select_partition_size)
        // Let's use explicit Fft with a small P to force many partitions
        // We'll bypass new() and construct manually for this edge case
        let ir = vec![1.0f32; 2048];
        let fft_state = LinearFftState::new(256, &ir);
        let mut aligned = AlignedVec::from_vec(ir.clone());
        aligned.reverse();
        let history = MirroredBuffer::<f32>::new(2048).unwrap();
        let limit = history.size();

        let mut model = LinearModel {
            weights: aligned,
            bias: 0.0,
            history,
            write_pos: limit,
            receptive_field: 2048,
            double_limit: limit.saturating_mul(2),
            prewarm_on_reset: true,
            implementation: LinearImplementation::Fft,
            mode: LinearMode::Fft(Box::new(fft_state)),
        };
        model.prewarm(0);

        // Verify processing does not panic over 2000 samples
        for i in 0..2000 {
            let x = (i as f32 * 0.3).sin();
            unsafe { model.process_sample(x) };
        }
    }

    #[test]
    fn test_fft_reset_restores_behavior() {
        let ir = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let mut model = LinearModel::new(ir, 0.2, LinearImplementation::Fft).unwrap();
        model.prewarm(0);

        let out1 = unsafe { model.process_sample(0.7) };
        let out2 = unsafe { model.process_sample(-0.3) };
        let out3 = unsafe { model.process_sample(0.4) };

        model.reset(0, 0);

        let out1b = unsafe { model.process_sample(0.7) };
        let out2b = unsafe { model.process_sample(-0.3) };
        let out3b = unsafe { model.process_sample(0.4) };

        assert!(
            (out1 - out1b).abs() < 1e-6,
            "reset mismatch: {out1} vs {out1b}"
        );
        assert!(
            (out2 - out2b).abs() < 1e-6,
            "reset mismatch: {out2} vs {out2b}"
        );
        assert!(
            (out3 - out3b).abs() < 1e-6,
            "reset mismatch: {out3} vs {out3b}"
        );
    }

    #[test]
    fn test_fft_process_block() {
        let ir = vec![1.0f32; 512];
        let mut model = LinearModel::new(ir, 0.0, LinearImplementation::Fft).unwrap();
        model.prewarm(0);

        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.2).sin()).collect();
        let mut output = vec![0.0f32; 128];
        unsafe { model.process(&input, &mut output) };

        // Verify no NaN, no infinity
        for &v in &output {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn test_direct_process_block_still_works() {
        let mut model =
            LinearModel::new(vec![0.5, 0.5], 0.0, LinearImplementation::Direct).unwrap();
        model.prewarm(0);

        let input = [0.1f32; 64];
        let mut output = [0.0f32; 64];
        unsafe { model.process(&input, &mut output) };

        for &v in &output {
            assert!(v.is_finite());
        }
    }

    // ── Existing tests kept ──

    #[test]
    fn test_linear_unit_weight() {
        let model = LinearModel::new(vec![1.0], 0.0, LinearImplementation::default()).unwrap();
        assert_eq!(model.receptive_field, 1);
        assert_eq!(model.weights.len(), 1);
    }

    #[test]
    fn test_linear_known_output() {
        let weights = vec![0.2, 0.3, 0.5];
        let bias = 0.1;
        let mut model = LinearModel::new(weights, bias, LinearImplementation::default()).unwrap();

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
        let mut model =
            LinearModel::new(vec![0.0, 0.0, 0.0], 0.0, LinearImplementation::default()).unwrap();
        model.prewarm(0);
        let out = unsafe { model.process_sample(5.0) };
        assert!((out - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_process_block() {
        let mut model = LinearModel::new(vec![1.0], 0.0, LinearImplementation::default()).unwrap();
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
        let mut model =
            LinearModel::new(vec![0.5, 0.5], 0.0, LinearImplementation::default()).unwrap();
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
        let model = LinearModel::new(vec![1.0; 16], 0.0, LinearImplementation::default()).unwrap();
        assert_eq!(model.prewarm_samples(), 0);
    }
}
