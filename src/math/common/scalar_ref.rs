// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Scalar Reference Implementations for DSP math kernels.
//!
//! # Purpose
//!
//! The role of this module is to be the **scalar reference implementation**: simple,
//! correct, unoptimized algorithms that serve as an oracle in parity tests
//! for AVX2 and AVX-512 kernels. Every vector kernel must produce the same
//! numerical result (within floating-point tolerance) as its scalar counterpart here.
//!
//! # Source of Truth
//!
//! The definitive mathematical specification of each operation is NeuralAmpModelerCore (C++).
//! The implementations here are faithful scalar translations of that reference code,
//! used exclusively for internal validation.

// Re-exports of Wavenet fallbacks (Task 3.4 — maintains path compatibility)
use crate::math::common::kahan_add;
pub use crate::math::wavenet::accumulate::{
    accumulate_head_fallback, gated_activation_and_accumulate_block_fallback,
    gated_activation_and_overwrite_block_fallback, tanh_and_accumulate_block_fallback,
    tanh_and_overwrite_block_fallback,
};


/// Computes the "Dot Product" between two sets of numbers.
/// Imagine multiplying each item from one list by the corresponding item of another list
/// and, in the end, summing everything up.
///
/// - `a`: List of decimal numbers (f32).
/// - `b`: List of "weights" stored in compact form (u16/f16).
pub unsafe fn dot_product_fallback(a: &[f32], b: &[u16]) -> f32 {
    // We pick the size of the smaller list to ensure we won't "run over" memory.
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32; // Start the sum at zero.

    for i in 0..len {
        unsafe {
            // The weight 'b' is "shrunken" (f16). Here we transform it back into
            // a normal decimal number (f32) so we can do the math.
            let fb = half::f16::from_bits(*b.get_unchecked(i)).to_f32();

            // We multiply the input value 'a' by the weight 'fb' and add to the total.
            // `get_unchecked` is like telling the computer: "Go straight to this address,
            // I guarantee it exists", which saves a bit of speed.
            sum += *a.get_unchecked(i) * fb;
        }
    }
    sum // Returns the final result of the sum.
}

/// Version of the Dot Product for the "BF16" (Brain Floating Point) format.
/// This is a decimal number format widely used in AI because
/// it takes half the space but preserves the "scale" of large numbers.
pub unsafe fn dot_product_bf16_fallback(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32;

    for i in 0..len {
        unsafe {
            // Here we do some bit "magic": we shift the number 16 places to the left.
            // This transforms the compact BF16 format back into standard f32 decimal.
            let fa = f32::from_bits((*a.get_unchecked(i) as u32) << 16);
            let fb = f32::from_bits((*b.get_unchecked(i) as u32) << 16);

            // Multiply and accumulate into the sum.
            sum += fa * fb;
        }
    }
    sum
}

/// Native f32 dot_product for mixed-precision head projection.
///
/// Used for the final head weights (WaveNet head_rechannel, LSTM head_weights)
/// when running in full FP32 precision, while the backbone uses quantized
/// (BF16/F16) weights for performance.
pub fn dot_product_f32_native(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Interleaved Dot Product (4x).
/// Instead of computing a single sum, this function computes 4 sums at the same time
/// using the same input data but different weights.
/// Useful when a sound (state) affects 4 different "channels" or "neurons".
pub unsafe fn dot_product_4x_interleaved_fallback(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];
    let mut comp = [0.0f32; 4];

    for i in 0..len {
        unsafe {
            let s = *state.get_unchecked(i);
            let w = weights.get_unchecked(i);

            let w0 = half::f16::from_bits(w[0]).to_f32();
            let w1 = half::f16::from_bits(w[1]).to_f32();
            let w2 = half::f16::from_bits(w[2]).to_f32();
            let w3 = half::f16::from_bits(w[3]).to_f32();

            let (s0, c0) = kahan_add(sum[0], comp[0], w0 * s);
            sum[0] = s0;
            comp[0] = c0;
            let (s1, c1) = kahan_add(sum[1], comp[1], w1 * s);
            sum[1] = s1;
            comp[1] = c1;
            let (s2, c2) = kahan_add(sum[2], comp[2], w2 * s);
            sum[2] = s2;
            comp[2] = c2;
            let (s3, c3) = kahan_add(sum[3], comp[3], w3 * s);
            sum[3] = s3;
            comp[3] = c3;
        }
    }
    sum
}

/// Same logic as above (4 sums at once), but using the compact BF16 format.
pub unsafe fn dot_product_4x_interleaved_bf16_fallback(
    weights: &[[u16; 4]],
    state: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];
    let mut comp = [0.0f32; 4];

    for i in 0..len {
        unsafe {
            let s = f32::from_bits((*state.get_unchecked(i) as u32) << 16);
            let w = weights.get_unchecked(i);

            let w0 = f32::from_bits((w[0] as u32) << 16);
            let w1 = f32::from_bits((w[1] as u32) << 16);
            let w2 = f32::from_bits((w[2] as u32) << 16);
            let w3 = f32::from_bits((w[3] as u32) << 16);

            let (s0, c0) = kahan_add(sum[0], comp[0], w0 * s);
            sum[0] = s0;
            comp[0] = c0;
            let (s1, c1) = kahan_add(sum[1], comp[1], w1 * s);
            sum[1] = s1;
            comp[1] = c1;
            let (s2, c2) = kahan_add(sum[2], comp[2], w2 * s);
            sum[2] = s2;
            comp[2] = c2;
            let (s3, c3) = kahan_add(sum[3], comp[3], w3 * s);
            sum[3] = s3;
            comp[3] = c3;
        }
    }
    sum
}

/// Interleaved Dot Product for "Dual Frame" (Two audio frames).
/// This function is even more hardworking: it computes 4 sums for the first
/// audio frame AND 4 sums for the second frame, all in a single loop.
/// This saves time because we read the weights from memory only once.
pub unsafe fn dot_product_4x_interleaved_dual_frame_fallback(
    weights: &[[u16; 4]],
    state_f0: &[f32], // First audio frame (e.g.: current sample)
    state_f1: &[f32], // Second frame (e.g.: previous or next sample)
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut sum_f0 = [0.0f32; 4];
    let mut sum_f1 = [0.0f32; 4];
    let mut comp_f0 = [0.0f32; 4];
    let mut comp_f1 = [0.0f32; 4];

    for i in 0..len {
        unsafe {
            let s0 = *state_f0.get_unchecked(i);
            let s1 = *state_f1.get_unchecked(i);
            let w = weights.get_unchecked(i);

            // We unpack the 4 weights just once to use in both frames.
            let w0 = half::f16::from_bits(w[0]).to_f32();
            let w1 = half::f16::from_bits(w[1]).to_f32();
            let w2 = half::f16::from_bits(w[2]).to_f32();
            let w3 = half::f16::from_bits(w[3]).to_f32();

            // Kahan-compensated accumulations for frame 0.
            let (s, c) = kahan_add(sum_f0[0], comp_f0[0], w0 * s0);
            sum_f0[0] = s;
            comp_f0[0] = c;
            let (s, c) = kahan_add(sum_f0[1], comp_f0[1], w1 * s0);
            sum_f0[1] = s;
            comp_f0[1] = c;
            let (s, c) = kahan_add(sum_f0[2], comp_f0[2], w2 * s0);
            sum_f0[2] = s;
            comp_f0[2] = c;
            let (s, c) = kahan_add(sum_f0[3], comp_f0[3], w3 * s0);
            sum_f0[3] = s;
            comp_f0[3] = c;

            // Kahan-compensated accumulations for frame 1.
            let (s, c) = kahan_add(sum_f1[0], comp_f1[0], w0 * s1);
            sum_f1[0] = s;
            comp_f1[0] = c;
            let (s, c) = kahan_add(sum_f1[1], comp_f1[1], w1 * s1);
            sum_f1[1] = s;
            comp_f1[1] = c;
            let (s, c) = kahan_add(sum_f1[2], comp_f1[2], w2 * s1);
            sum_f1[2] = s;
            comp_f1[2] = c;
            let (s, c) = kahan_add(sum_f1[3], comp_f1[3], w3 * s1);
            sum_f1[3] = s;
            comp_f1[3] = c;
        }
    }
    (sum_f0, sum_f1)
}

/// Same "Dual Frame" logic as above, but everything in BF16 format.
pub unsafe fn dot_product_4x_interleaved_dual_frame_bf16_fallback(
    weights: &[[u16; 4]],
    state_f0: &[u16],
    state_f1: &[u16],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut sum_f0 = [0.0f32; 4];
    let mut sum_f1 = [0.0f32; 4];
    let mut comp_f0 = [0.0f32; 4];
    let mut comp_f1 = [0.0f32; 4];

    for i in 0..len {
        unsafe {
            let s0 = f32::from_bits((*state_f0.get_unchecked(i) as u32) << 16);
            let s1 = f32::from_bits((*state_f1.get_unchecked(i) as u32) << 16);
            let w = weights.get_unchecked(i);

            let w0 = f32::from_bits((w[0] as u32) << 16);
            let w1 = f32::from_bits((w[1] as u32) << 16);
            let w2 = f32::from_bits((w[2] as u32) << 16);
            let w3 = f32::from_bits((w[3] as u32) << 16);

            let (s, c) = kahan_add(sum_f0[0], comp_f0[0], w0 * s0);
            sum_f0[0] = s;
            comp_f0[0] = c;
            let (s, c) = kahan_add(sum_f0[1], comp_f0[1], w1 * s0);
            sum_f0[1] = s;
            comp_f0[1] = c;
            let (s, c) = kahan_add(sum_f0[2], comp_f0[2], w2 * s0);
            sum_f0[2] = s;
            comp_f0[2] = c;
            let (s, c) = kahan_add(sum_f0[3], comp_f0[3], w3 * s0);
            sum_f0[3] = s;
            comp_f0[3] = c;

            let (s, c) = kahan_add(sum_f1[0], comp_f1[0], w0 * s1);
            sum_f1[0] = s;
            comp_f1[0] = c;
            let (s, c) = kahan_add(sum_f1[1], comp_f1[1], w1 * s1);
            sum_f1[1] = s;
            comp_f1[1] = c;
            let (s, c) = kahan_add(sum_f1[2], comp_f1[2], w2 * s1);
            sum_f1[2] = s;
            comp_f1[2] = c;
            let (s, c) = kahan_add(sum_f1[3], comp_f1[3], w3 * s1);
            sum_f1[3] = s;
            comp_f1[3] = c;
        }
    }
    (sum_f0, sum_f1)
}

/// Computes 4 dot products at once for BF16.
/// This is a shortcut to call the single-line function 4 times.
pub unsafe fn dot_product_bf16_4x_fallback(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    in_frame: &[u16],
) -> [f32; 4] {
    unsafe {
        [
            dot_product_bf16_fallback(in_frame, w0),
            dot_product_bf16_fallback(in_frame, w1),
            dot_product_bf16_fallback(in_frame, w2),
            dot_product_bf16_fallback(in_frame, w3),
        ]
    }
}

/// Batch Matrix Processing (GEMM).
/// GEMM stands for "General Matrix Multiplication". It's the heart of neural networks.
/// This function processes multiple audio "frames" at once.
pub unsafe fn fused_add_gemm_batch_fallback(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32], // "Bias" is a fixed offset added at the end (like the 'b' in y = ax + b).
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    // Figure out how much space each frame occupies in memory.
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    // For each frame in the batch...
    for f in 0..num_frames {
        unsafe {
            // ...call the function that processes a single vector (GEMV).
            fused_add_gemv_fallback(
                in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
                weights,
                bias,
                out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
                do_bias,
            );
        }
    }
}

/// Matrix-Vector Multiplication (GEMV) that ADDS to the existing result.
/// Think of it as injecting a new processing layer on top of what was already there.
pub unsafe fn fused_add_gemv_fallback(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    // For each output "neuron"...
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        // We start with the 'bias' (offset) or with zero.
        let mut sum = if do_bias { b } else { 0.0 };

        // We iterate through all inputs and corresponding weights.
        for in_c in 0..in_len {
            unsafe {
                // Gets the compressed weight and unpacks it.
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                // Multiplies the input by the weight and accumulates.
                sum += *in_frame.get_unchecked(in_c) * w;
            }
        }
        unsafe {
            // IMPORTANT: Here we use '+=' to ADD to what was already in the output buffer.
            *out_frame.get_unchecked_mut(out_c) += sum;
        }
    }
}

/// Matrix-Vector Multiplication (GEMV) that OVERWRITES the result.
/// Unlike the previous one, this erases what was in the output and places the new value.
pub unsafe fn gemv_overwrite_fallback(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            unsafe {
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
        }
        unsafe {
            // IMPORTANT: Here we use '=' to CLEAR and set the new value.
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Residual Matrix Multiplication in a batch.
/// In "Residual" neural networks, we add the processing result to the original signal.
/// It's like saying: "Change the sound just a little bit relative to what it was."
pub unsafe fn fused_gemm_residual_batch_fallback(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32],
    residual: &[f32], // This is the "clean" signal that will be added at the end.
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    for frame_idx in 0..num_frames {
        unsafe {
            for (out_c, &b) in bias.iter().enumerate().take(out_len) {
                let mut sum = if do_bias { b } else { 0.0 };
                for in_c in 0..in_len {
                    let w_bits = *weights.get_unchecked(in_c * out_len + out_c);
                    let w = half::f16::from_bits(w_bits).to_f32();
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                // Final result = (Matrix Processing) + (Original Residual Signal).
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) =
                    sum + *residual.get_unchecked(frame_idx * out_len + out_c);
            }
        }
    }
}

/// Matrix-Vector Multiplication (Overwrite) using BF16 input and weights.
pub unsafe fn gemv_overwrite_bf16_fallback(
    in_frame: &[u16],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            unsafe {
                // Unpack BF16 input -> f32.
                let s = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
                // Unpack BF16 weight -> f32 and multiply.
                sum += s * f32::from_bits(
                    (*weights.get_unchecked(in_c * out_len + out_c) as u32) << 16,
                );
            }
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Converts high-precision decimal numbers (f32) to the compact format (BF16).
/// This saves a lot of memory and is used to store neural network "weights".
pub unsafe fn f32_to_bf16_fallback(src: &[f32], dest: &mut [u16]) {
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        // Gets the 16 most important bits of the number and discards the rest.
        *d = (s.to_bits() >> 16) as u16;
    }
}

/// Applies the Tanh "squashing" function across an entire list of numbers.
pub unsafe fn tanh_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = v.tanh();
    }
}

/// Applies the "Sigmoid" function across an entire list of numbers.
/// Sigmoid squashes numbers to stay between 0.0 and 1.0.
/// It's great for creating "gates" or automatic volume controls.
pub unsafe fn sigmoid_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Sums all numbers in a list and returns a single final value.
pub unsafe fn horizontal_sum_fallback(ptr: *const f32, len: usize) -> f32 {
    let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
    slice.iter().sum()
}

/// Stereo Convolution (used in the Resampler).
/// Convolution is like applying a filter (like an equalizer).
/// Here we do it for the Left (L) and Right (R) channels simultaneously.
pub unsafe fn convolve_stereo_fallback(
    coeffs: *const f32,  // Filter coefficients.
    input_l: *const f32, // Left channel input.
    input_r: *const f32, // Right channel input.
    taps: usize,         // Filter "length".
) -> (f32, f32) {
    let mut sum_l = 0.0f32;
    let mut sum_r = 0.0f32;
    for i in 0..taps {
        let h = *coeffs.add(i);
        sum_l += h * *input_l.add(i);
        sum_r += h * *input_r.add(i);
    }
    (sum_l, sum_r)
}

/// Dual Stereo Convolution (used in the Resampler).
/// Performs two consecutive stereo convolutions reusing input loads.
pub unsafe fn convolve_stereo_dual_fallback(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> ((f32, f32), (f32, f32)) {
    let mut sum0_l = 0.0f32;
    let mut sum0_r = 0.0f32;
    let mut sum1_l = 0.0f32;
    let mut sum1_r = 0.0f32;
    for i in 0..taps {
        let h0 = *coeffs0.add(i);
        let h1 = *coeffs1.add(i);
        let xl = *input_l.add(i);
        let xr = *input_r.add(i);
        sum0_l += h0 * xl;
        sum0_r += h0 * xr;
        sum1_l += h1 * xl;
        sum1_r += h1 * xr;
    }
    ((sum0_l, sum0_r), (sum1_l, sum1_r))
}

/// Mono Dual Convolution (used in the Resampler).
/// Performs two mono convolutions on the same input buffer, reusing the loaded input samples.
pub unsafe fn convolve_mono_dual_fallback(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input: *const f32,
    taps: usize,
) -> (f32, f32) {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    for i in 0..taps {
        let x = *input.add(i);
        sum0 += *coeffs0.add(i) * x;
        sum1 += *coeffs1.add(i) * x;
    }
    (sum0, sum1)
}

/// Mono Convolution (used in the Resampler).
pub unsafe fn convolve_mono_fallback(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..taps {
        sum += *coeffs.add(i) * *input.add(i);
    }
    sum
}

/// Scalar fallback for the 4 LSTM gates.
/// Each gate controls a different aspect: input, forget, content, and output.
/// Used directly by `avx512.rs` and `avx2.rs` for non-vectorized operations.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_fallback(
    in_frame: &[f32],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    unsafe {
        // Processes each of the 4 gates separately.
        gemv_overwrite_fallback(
            in_frame,
            w0,
            &bias[0..out_len],
            &mut out[0..out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w1,
            &bias[out_len..2 * out_len],
            &mut out[out_len..2 * out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w2,
            &bias[2 * out_len..3 * out_len],
            &mut out[2 * out_len..3 * out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w3,
            &bias[3 * out_len..4 * out_len],
            &mut out[3 * out_len..4 * out_len],
            do_bias,
        );
    }
}

/// BF16 version for the 4 LSTM gates.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_bf16_fallback(
    in_frame: &[u16],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    unsafe {
        gemv_overwrite_bf16_fallback(
            in_frame,
            w0,
            &bias[0..out_len],
            &mut out[0..out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w1,
            &bias[out_len..2 * out_len],
            &mut out[out_len..2 * out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w2,
            &bias[2 * out_len..3 * out_len],
            &mut out[2 * out_len..3 * out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w3,
            &bias[3 * out_len..4 * out_len],
            &mut out[3 * out_len..4 * out_len],
            do_bias,
        );
    }
}

/// Applies a volume (gain) by multiplying each sample by the desired value.
pub unsafe fn apply_gain_fallback(data: &mut [f32], gain: f32) {
    for x in data.iter_mut() {
        *x *= gain;
    }
}

/// Computes the energy (Mean Square) of a block via scalar.
pub unsafe fn compute_energy_fallback(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum / (len as f32)
}

/// Computes the maximum energy between two channels via scalar.
pub unsafe fn compute_energy_stereo_fallback(l: &[f32], r: &[f32]) -> f32 {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return 0.0;
    }
    let mut sum_l = 0.0f32;
    let mut sum_r = 0.0f32;
    for i in 0..len {
        let xl = *l.get_unchecked(i);
        let xr = *r.get_unchecked(i);
        sum_l += xl * xl;
        sum_r += xr * xr;
    }
    let energy_l = sum_l / (len as f32);
    let energy_r = sum_r / (len as f32);
    energy_l.max(energy_r)
}

/// Computes the maximum absolute difference between two blocks via scalar.
pub unsafe fn compute_max_diff_fallback(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return 0.0;
    }
    let mut max_diff = 0.0f32;
    for i in 0..len {
        let d = (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    max_diff
}

/// Computes the peak absolute value of both channels (fallback).
pub unsafe fn compute_peak_abs_stereo_fallback(left: &[f32], right: &[f32]) -> (f32, f32) {
    let mut peak_l = 0.0f32;
    let mut peak_r = 0.0f32;
    let len = core::cmp::min(left.len(), right.len());
    for i in 0..len {
        let al = (*left.get_unchecked(i)).abs();
        let ar = (*right.get_unchecked(i)).abs();
        if al > peak_l {
            peak_l = al;
        }
        if ar > peak_r {
            peak_r = ar;
        }
    }
    (peak_l, peak_r)
}

/// Batch GEMV overwrite with native f32 weights and inputs.
///
/// Performs `num_frames` independent matrix-vector multiplications.
/// Layout: input is frame-major `[f0_in.., f1_in.., ...]`, output is
/// frame-major `[f0_out.., f1_out.., ...]`. Weights are column-major
/// `weights[in_c * OUT + out_c]`.
///
/// This is the scalar reference oracle for the SIMD kernels.
pub fn gemv_overwrite_batch_f32_fallback(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    for n in 0..num_frames {
        for out_c in 0..out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                sum += in_frames[n * in_len + in_c] * weights[in_c * out_len + out_c];
            }
            out_frames[n * out_len + out_c] = sum;
        }
    }
}
