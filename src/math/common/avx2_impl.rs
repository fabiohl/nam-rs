// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! AVX2 implementations of the `SimdMath` trait.
//!
//! This module contains the `Avx2Math` struct.
//! that implement the `SimdMath` trait using AVX2/FMA instructions.
//! Methods delegate to kernel functions in `math::gemm`, `math::wavenet`,
//! `math::lstm`, `math::dsp`, and `math::common::utility`.

use crate::math::common::traits::SimdMath;
use core::arch::x86_64::*;

/// Concrete implementation of the SimdMath trait for processors with AVX2 and FMA support.
///
/// This is where we "connect the wires": we connect the abstract mathematical operations of the system
/// to the ultra-fast functions documented above. This struct ensures that NAM-rs
/// takes full advantage of modern hardware to process audio in real time.
pub struct Avx2Math;

impl SimdMath for Avx2Math {
    type V = __m256;

    // Dot Product: Multiplies weights by signal and sums the result (the "DNA" of neural networks).
    // In AVX2, we use 256-bit registers that process 8 numbers at once.
    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::gemm::dot::dot_product_avx2(a, b) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product_bf16(_a: &[u16], _b: &[u16]) -> f32 {
        unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::gemm::dot_4x::dot_product_4x_interleaved_avx2(weights, state) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::gemm::dot_4x::dot_product_4x_interleaved_dual_frame_avx2(
                weights, state_f0, state_f1,
            )
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product_4x_f32(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::gemm::dot_4x::dot_product_4x_f32_avx2(weights, state) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product_4x_f32_dual(
        weights: &[[f32; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::gemm::dot_4x::dot_product_4x_f32_dual_avx2(weights, state_f0, state_f1)
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product_8x_f32(weights: &[[f32; 8]], state: &[f32]) -> [f32; 8] {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::gemm::dot_8x::dot_product_8x_f32_avx2(weights, state) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product_16x_f32(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::gemm::dot_16x::dot_product_16x_f32_avx2(weights, state) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn dot_product_bf16_4x(
        _w0: &[u16],
        _w1: &[u16],
        _w2: &[u16],
        _w3: &[u16],
        _in_frame: &[u16],
    ) -> [f32; 4] {
        unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
    }

    // GEMV Operations: Matrix-Vector multiplication, used in almost all model layers.
    // The "fused" prefix indicates that the Bias vector addition is combined (fused) with the multiplication
    // to save memory accesses and processor instructions.
    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            // Delegates the computation to the optimized AVX2 matrix-vector multiplication kernel.
            super::super::gemm::gemv::fused_add_gemv_avx2(
                in_frame, weights, bias, out_frame, do_bias,
            )
        }
    }

    /// Performs matrix multiplication on a batch of vectors via AVX2.
    /// Useful when processing multiple audio frames concurrently to reduce overheads.
    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            // Delegates the batch matrix-matrix multiplication (GEMM) computation to the AVX2 kernel.
            super::super::gemm::gemm_batch::fused_add_gemm_batch_avx2(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
    }

    /// Performs matrix-vector multiplication also adding the residual connection (skip connection)
    /// from the previous layer. Widely used in the WaveNet residual block architecture.
    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            // Delegates the multiplication with integrated residual sum and bias to the AVX2 kernel.
            super::super::gemm::gemm_batch::fused_gemm_residual_batch_avx2(
                in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
            )
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn fused_gemm_residual_batch_f32(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::gemm::gemm_batch::fused_gemm_residual_batch_f32_avx2(
                in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
            )
        }
    }

    /// Version that overwrites the output buffer directly with the matrix-vector multiplication result,
    /// without accumulating with pre-existing values in the buffer.
    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::gemm::gemv::gemv_overwrite_avx2(
                in_frame, weights, bias, out_frame, do_bias,
            )
        }
    }

    /// Version that overwrites the output buffer accepting input data represented in BF16 (16-bit)
    /// and BF16 weights, performing accumulation in f32 to preserve fidelity.
    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gemv_overwrite_bf16(
        _in_frame: &[u16],
        _weights: &[u16],
        _bias: &[f32],
        _out_frame: &mut [f32],
        _do_bias: bool,
    ) {
        unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
    }

    // LSTM Gates (4-gate): Simultaneously computes the 4 memory controls of the LSTM network.
    // Gate computation (Input, Forget, Cell Candidate, and Output) shares the same input
    // states. Computing them in parallel drastically reduces cache jumps.
    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gemv_overwrite_4gate(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        let ih = in_frame.len();
        let stride = ih * hidden_size;
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            // We split the single contiguous weight matrix into 4 blocks (strides) corresponding to each gate:
            // - weights[0..stride]: Input Gate weights.
            // - weights[stride..2*stride]: Forget Gate weights.
            // - weights[2*stride..3*stride]: Cell Candidate weights.
            // - weights[3*stride..4*stride]: Output Gate weights.
            super::super::gemm::gemv_4gate::gemv_4gate_avx2(
                in_frame,
                &weights[0..stride],
                &weights[stride..2 * stride],
                &weights[2 * stride..3 * stride],
                &weights[3 * stride..4 * stride],
                bias,
                out_gates,
                do_bias,
            )
        }
    }

    /// Equivalent to `gemv_overwrite_4gate` but processing input data represented
    /// in the BF16 reduced precision format.
    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gemv_overwrite_bf16_4gate(
        _in_frame: &[u16],
        _weights: &[u16],
        _bias: &[f32],
        _out_gates: &mut [f32],
        _hidden_size: usize,
        _do_bias: bool,
    ) {
        unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::wavenet::accumulate::accumulate_head_avx2(dest, src) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::wavenet::accumulate::tanh_and_accumulate_block_avx2(head_input, block)
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::wavenet::accumulate::gated_activation_and_accumulate_block_avx2(
                head_input, block, ch,
            )
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::wavenet::accumulate::tanh_and_overwrite_block_avx2(head_input, block)
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gated_activation_and_overwrite_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::wavenet::accumulate::gated_activation_and_overwrite_block_avx2(
                head_input, block, ch,
            )
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn f32_to_bf16(_src: &[f32], _dest: &mut [u16]) {
        unreachable!("AVX2 IS_BF16=false; BF16 paths are never reached at runtime")
    }

    /// Converts a register containing 8 32-bit floats (Self::V) to the compact BF16 (16-bit)
    /// format and stores the results in memory.
    ///
    /// ## Bitwise SIMD Magic Details:
    /// To convert f32 to BF16 without spending many CPU cycles on slow mathematical conversions,
    /// the technique leverages the structural similarity between IEEE 754 single-precision float and BF16:
    /// Both share the same dynamic range (8 exponent bits), but BF16 discards the
    /// 16 least significant mantissa bits (truncation/rounding).
    #[inline(always)]
    // SAFETY: ptr is a valid mutable pointer to 8 u16 elements; v is a valid __m256 register;
    // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch). Intrinsic transformation
    // uses bitwise repacking; the _mm_storeu_si128 performs an unaligned 128-bit store.
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
        // SAFETY: ptr points to at least 8 writable u16 elements (128-bit store).
        unsafe {
            // 1. Reinterpret the f32 register (__m256) as 32-bit integers (__m256i). Cost: 0 cycles.
            let v_i = _mm256_castps_si256(v);
            // 2. Shift the integers 16 bits to the right. The upper half (BF16 mantissa)
            // moves to the lower half of each 32-bit element.
            let v_shifted = _mm256_srli_epi32(v_i, 16);
            // 3. Pack with unsigned saturation (Packus) 32-bit elements into 16-bit elements.
            // This consolidates the useful 16-bit data.
            let packed = _mm256_packus_epi32(v_shifted, v_shifted);
            // 4. Permute 64-bit lanes using pattern (8/0x08) to regroup the valid results
            // that got mixed up due to the inherent behavior of the _mm256_packus_epi32 instruction.
            let permuted = _mm256_permute4x64_epi64(packed, 8);
            // 5. Extract the lower half (128 bits of a 256-bit register), containing the 8 BF16 values.
            let v_low = _mm256_castsi256_si128(permuted);
            // 6. Write the 8 compacted BF16 values directly to the destination in RAM.
            _mm_storeu_si128(ptr as *mut __m128i, v_low);
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn tanh_slice(slice: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::tanh_slice_avx2(slice) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::sigmoid_slice_avx2(slice) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn relu_slice(slice: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::relu_slice_avx2(slice) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn prelu_slice(slice: &mut [f32], slopes: &[f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::prelu_slice_avx2(slice, slopes) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn softsign_slice(slice: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::softsign_slice_avx2(slice) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn silu_slice(slice: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::silu_slice_avx2(slice) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn hard_tanh_slice(slice: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::hard_tanh_slice_avx2(slice) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn hard_swish_slice(slice: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::hard_swish_slice_avx2(slice) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn fast_tanh_slice(slice: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::fast_tanh_slice_avx2(slice) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn leaky_hard_tanh_slice(
        slice: &mut [f32],
        min_val: f32,
        max_val: f32,
        min_slope: f32,
        max_slope: f32,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            crate::math::activations::leaky_hard_tanh_slice_avx2(
                slice, min_val, max_val, min_slope, max_slope,
            )
        }
    }

    #[inline(always)]
    // SAFETY: ptr is a valid raw pointer to N valid f32 elements;
    // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        // SAFETY: ptr and N satisfy the function's documented invariants.
        unsafe { super::utility::horizontal_sum_avx2(ptr, N) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { crate::math::activations::tanh_slice_avx2(buf) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::lstm::fused_lstm_gates_dyn_avx2(
                gates,
                cell_state,
                hidden_state,
                hidden_size,
            )
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::stereo::compute_energy_stereo_avx2(l, r) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn compute_energy(data: &[f32]) -> f32 {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::stereo::compute_energy_avx2(data) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::stereo::compute_max_diff_avx2(a, b) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::stereo::compute_peak_abs_stereo_avx2(left, right) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32 {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::stereo::compute_peak_abs_mono_avx2(data) }
    }

    // Convolve Stereo: Applies filtering (equalization) to both channels (left and right) simultaneously.
    //
    // ## SIMD Throughput Optimization:
    // Since the impulse response (FIR filter coefficients) is identical for both channels in standard
    // stereo processing, we load the coefficients once into AVX2 registers and
    // multiply them concurrently by the left (input_l) and right (input_r) audio samples.
    // This doubles the computation efficiency compared to filtering each channel sequentially.
    #[inline(always)]
    // SAFETY: coeffs, input_l, and input_r are valid raw pointers to taps valid f32 elements;
    // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch). Kernel may use aligned loads.
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        // SAFETY: raw pointers satisfy the function's documented invariants.
        unsafe { super::super::dsp::stereo::convolve_stereo_avx2(coeffs, input_l, input_r, taps) }
    }

    #[inline(always)]
    // SAFETY: coeffs0, coeffs1, input_l, and input_r are valid raw pointers to taps valid f32
    // elements; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn convolve_stereo_dual(
        coeffs0: *const f32,
        coeffs1: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> ((f32, f32), (f32, f32)) {
        // SAFETY: raw pointers satisfy the function's documented invariants.
        unsafe {
            super::super::dsp::stereo::convolve_stereo_dual_avx2(
                coeffs0, coeffs1, input_l, input_r, taps,
            )
        }
    }

    #[inline(always)]
    // SAFETY: coeffs and input are valid raw pointers to taps valid f32 elements;
    // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
        // SAFETY: raw pointers satisfy the function's documented invariants.
        unsafe { super::super::dsp::stereo::convolve_mono_avx2(coeffs, input, taps) }
    }

    #[inline(always)]
    // SAFETY: coeffs0, coeffs1, and input are valid raw pointers to taps valid f32 elements;
    // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn convolve_mono_dual(
        coeffs0: *const f32,
        coeffs1: *const f32,
        input: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        // SAFETY: raw pointers satisfy the function's documented invariants.
        unsafe { super::super::dsp::stereo::convolve_mono_dual_avx2(coeffs0, coeffs1, input, taps) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::gain::apply_gain_and_detect_clipping_mono_avx2(data, gain) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::dsp::gain::apply_gain_and_detect_clipping_stereo_avx2(left, right, gain)
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::gain::apply_gain_stereo_avx2(left, right, gain) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::gain::apply_gain_avx2(data, gain) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::gain::apply_ramp_avx2(data, start, step) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::gain::apply_ramp_stereo_avx2(left, right, start, step) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::gain::apply_dither_add_avx2(data, offset) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe { super::super::dsp::gain::crossfade_blend_mono_avx2(out, pending, t) }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gemv_overwrite_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        let in_len = in_frames.len() / num_frames;
        let out_len = out_frames.len() / num_frames;
        for i in 0..num_frames {
            let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
            let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
            // SAFETY: in_slice and out_slice are valid sub-slices of the batch arrays;
            // AVX2+FMA ISA verified by caller via dispatch.
            unsafe {
                super::super::gemm::gemv::gemv_overwrite_avx2(
                    in_slice, weights, bias, out_slice, do_bias,
                )
            };
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gemv_with_bias_f32(
        in_frames: &[f32],
        weights: &[f32],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::gemm::gemv::gemv_with_bias_f32_avx2(
                in_frames, weights, bias, out_frames, num_frames,
            )
        }
    }

    #[inline(always)]
    // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
    unsafe fn gemv_no_bias_f32(
        in_frames: &[f32],
        weights: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
    ) {
        // SAFETY: arguments satisfy the function's documented invariants.
        unsafe {
            super::super::gemm::gemv::gemv_no_bias_f32_avx2(
                in_frames, weights, out_frames, num_frames,
            )
        }
    }
}
