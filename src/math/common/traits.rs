// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! [Definition of the abstract interface for SIMD operations.

/// Abstraction trait for static dispatch of SIMD mathematical operations.
///
/// # Safety
/// All implementations of this trait use x86-64 SIMD intrinsics that require
/// specific CPU features (AVX2/FMA minimum). The caller must ensure that the CPU
/// supports the declared features via `#[target_feature]` in the concrete implementation.
/// Slices passed must be valid and accessible for reading/writing as indicated.
///
/// # Operation Groups
///
/// The operations of this trait are organized into the following groups:
/// - **(A) Dot Products**: Scalar/4x/dual-frame dot products (e.g., `dot_product`).
/// - **(B) GEMV/GEMM Fused**: Fused matrix-vector/matrix-matrix kernels (e.g., `fused_add_gemv`).
/// - **(C) Activations**: Tanh/sigmoid activation functions and gated fusions (e.g., `tanh_slice`).
/// - **(D) Conversions**: f32 ↔ bf16 conversion utilities (e.g., `f32_to_bf16`).
/// - **(E) LSTM Gates**: Specific kernels for LSTM cell gates (e.g., `fused_lstm_gates_dyn`).
pub trait SimdMath {
    /// SIMD register type used (e.g.: __m256 or __m512).
    type V: Copy;

    /// Indicates whether this implementation uses weights and signals in BF16 format.
    const IS_BF16: bool = false;

    // --- (A) Dot Products ---

    /// Computes the dot product between two f32 vectors.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32;

    /// Computes the dot product between two BF16 vectors.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32;

    /// Computes 4 simultaneous BF16 dot products (interleaved) with f32 input.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4];

    /// Computes 4 simultaneous BF16 dot products (interleaved) with BF16 input.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4];

    /// Computes 4 simultaneous BF16 dot products (interleaved) for 2 parallel frames.
    /// Returns a tuple with the 4 results of frame 0 and the 4 results of frame 1.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]);

    /// Computes 4 simultaneous BF16 dot products (interleaved) for 2 parallel frames (BF16 input).
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
        weights: &[[u16; 4]],
        state_f0: &[u16],
        state_f1: &[u16],
    ) -> ([f32; 4], [f32; 4]);

    /// Computes 4 simultaneous BF16 dot products.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4];

    /// Horizontal sum of a buffer.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32;

    // --- (B) GEMV/GEMM Fused ---

    /// Fused add + GEMV kernel.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// Fused add + batch GEMM kernel.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// Fused residual batch GEMM kernel.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// GEMV kernel with overwrite.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// GEMV kernel with overwrite in batch.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn gemv_overwrite_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// GEMV kernel with overwrite (BF16 input).
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// GEMV kernel with overwrite in batch (BF16 input).
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn gemv_overwrite_batch_bf16(
        in_frames: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// GEMV kernel with overwrite and bias in batch using native f32 weights.
    ///
    /// Always adds bias. Used for mixed-precision head projection where
    /// the final stage requires full FP32 precision while the backbone
    /// runs quantized.
    ///
    /// # Safety
    /// Buffers must be valid. The caller must ensure that `in_frames`,
    /// `weights`, and `out_frames` have compatible dimensions.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    unsafe fn gemv_with_bias_f32(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
    );

    /// GEMV kernel with overwrite (no bias) in batch using native f32 weights.
    ///
    /// Overwrites without adding bias. Used for mixed-precision head
    /// projection where the final stage requires full FP32 precision
    /// while the backbone runs quantized.
    ///
    /// # Safety
    /// Buffers must be valid. The caller must ensure that `in_frames`,
    /// `weights`, and `out_frames` have compatible dimensions.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    unsafe fn gemv_no_bias_f32(
        in_frames: &[f32],
        weights: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
    );

    /// GEMV kernel with overwrite for 4 simultaneous gates.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn gemv_overwrite_4gate(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    );

    /// GEMV kernel with overwrite for 4 simultaneous gates (BF16 input).
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn gemv_overwrite_bf16_4gate(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    );

    // --- (C) Activations ---

    /// Accumulates the contents of one vector into another.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]);

    /// Fused Tanh + Head Accumulate.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]);

    /// Fused Gated Activation + Head Accumulate.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    );

    /// Fused Tanh + Head Overwrite.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]);

    /// Fused Gated Activation + Head Overwrite.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn gated_activation_and_overwrite_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    );

    /// Computes the maximum energy between two channels (Stereo).
    /// Returns `max(energy_l, energy_r)`.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32;

    /// Computes the energy (Mean Square) of a block.
    /// $E = \frac{1}{N} \sum x_i^2$
    ///
    /// # Safety
    /// The buffer must be valid.
    unsafe fn compute_energy(data: &[f32]) -> f32;

    /// Computes the maximum absolute difference between two blocks.
    /// $\max(|a_i - b_i|)$
    ///
    /// # Safety
    /// The buffers must be valid and have the same length.
    unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32;

    /// Computes the peak absolute value of both channels.
    /// Returns `(max(|left_i|), max(|right_i|))`
    ///
    /// # Safety
    /// The buffers must be valid and have the same length.
    unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32);

    /// Computes the peak absolute value of a single channel.
    /// Returns `max(|x_i|)`
    ///
    /// # Safety
    /// The buffer must be valid.
    unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32;

    /// Applies Tanh to a slice.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn tanh_slice(slice: &mut [f32]);

    /// Applies Sigmoid to a slice.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn sigmoid_slice(slice: &mut [f32]);

    /// Tanh activation on a block.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn activation_tanh_block(buf: &mut [f32]);

    // --- (D) Conversions ---

    /// Conversion from F32 to BF16.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]);

    /// Stores the contents of a SIMD register as BF16 (truncated).
    ///
    /// # Safety
    /// The pointer must be valid and have enough space.
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V);

    // --- (E) LSTM Gates ---

    /// Fused kernel for dynamic LSTM gate processing.
    /// Performs activations (sigmoid/tanh) and state update (cell/hidden) in a single step.
    ///
    /// # Safety
    /// Buffers must have sizes compatible with `hidden_size`.
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    );

    /// Stereo convolution (used in the resampler).
    /// Performs the dot product between a coefficient bank and two input buffers (L/R).
    ///
    /// # Safety
    /// `coeffs`, `input_l`, and `input_r` must be valid pointers to at least `taps` elements.
    /// `coeffs` must be aligned according to the SIMD register.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32);

    /// Dual stereo convolution (reuses input loads).
    /// Performs the dot product between two coefficient banks and two input buffers (L/R).
    ///
    /// # Safety
    /// `coeffs0`, `coeffs1`, `input_l`, and `input_r` must be valid pointers to at least `taps` elements.
    /// `coeffs0` and `coeffs1` must be aligned according to the SIMD register.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    unsafe fn convolve_stereo_dual(
        coeffs0: *const f32,
        coeffs1: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> ((f32, f32), (f32, f32));

    /// Mono convolution (used in the resampler).
    /// Performs the dot product between a coefficient bank and an input buffer.
    ///
    /// # Safety
    /// `coeffs` and `input` must be valid pointers to at least `taps` elements.
    /// `coeffs` must be aligned according to the SIMD register.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32;

    /// Dual mono convolution (reuses input loads).
    /// Performs the dot product between two coefficient banks and one input buffer.
    ///
    /// # Safety
    /// `coeffs0`, `coeffs1`, and `input` must be valid pointers to at least `taps` elements.
    /// `coeffs0` and `coeffs1` must be aligned according to the SIMD register.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    unsafe fn convolve_mono_dual(
        coeffs0: *const f32,
        coeffs1: *const f32,
        input: *const f32,
        taps: usize,
    ) -> (f32, f32);

    /// Applies gain and detects clipping in mono in a single pass.
    /// Returns `true` if any resulting sample has `|x| > 1.0`.
    ///
    /// # Safety
    /// The buffer must be valid.
    unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool;

    /// Applies gain and detects clipping in stereo in a single pass.
    /// Returns `true` if any resulting sample has `|x| > 1.0`.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool;

    /// Applies constant gain in stereo (without clipping detection).
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32);

    /// Applies constant gain to a mono buffer.
    ///
    /// # Safety
    /// The buffer must be valid.
    unsafe fn apply_gain(data: &mut [f32], gain: f32);

    /// Applies a linear gain ramp to a mono buffer.
    ///
    /// # Safety
    /// The buffer must be valid.
    unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32);

    /// Crossfade blend: `out[i] = out[i] * (1-t) + pending[i] * t`.
    /// Computed as `fma(pending[i] - out[i], t, out[i])` for single-rounding precision.
    ///
    /// # Safety
    /// Buffers must be valid and have at least `min(out.len(), pending.len())` elements.
    unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32);

    /// Applies a linear gain ramp in stereo.
    ///
    /// # Safety
    /// Buffers must be valid.
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32);

    /// Adds a broadcast constant (dither offset) to every element of a mono buffer.
    ///
    /// # Safety
    /// The buffer must be valid.
    unsafe fn apply_dither_add(data: &mut [f32], offset: f32);
}
