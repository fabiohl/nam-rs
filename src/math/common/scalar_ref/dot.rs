// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::kahan_add;

/// Computes the "Dot Product" between two sets of numbers.
/// Imagine multiplying each item from one list by the corresponding item of another list
/// and, in the end, summing everything up.
///
/// - `a`: List of decimal numbers (f32).
/// - `b`: List of "weights" stored in compact form (u16/f16).
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_fallback(a: &[f32], b: &[u16]) -> f32 {
    // We pick the size of the smaller list to ensure we won't "run over" memory.
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32; // Start the sum at zero.

    for i in 0..len {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_bf16_fallback(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32;

    for i in 0..len {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
#[inline(always)]
pub fn dot_product_f32_native(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let a_sub = &a[..len];
    let b_sub = &b[..len];
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += a_sub[i] * b_sub[i];
    }
    sum
}

/// Interleaved Dot Product (4x).
/// Instead of computing a single sum, this function computes 4 sums at the same time
/// using the same input data but different weights.
/// Useful when a sound (state) affects 4 different "channels" or "neurons".
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_4x_interleaved_fallback(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];
    let mut comp = [0.0f32; 4];

    for i in 0..len {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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

/// F32-native 4-lane interleaved dot product (scalar reference for SIMD validation).
///
/// Uses `mul_add` (FMA3 fused multiply-add) to match the rounding of the AVX2/FMA
/// kernel (`dot_product_4x_f32_avx2`). Both paths produce bit‑identical results.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()`.
#[inline]
pub unsafe fn dot_product_4x_f32_scalar(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut r = [0.0f32; 4];
    for i in 0..len {
        let w = weights.get_unchecked(i);
        let s = state.get_unchecked(i);
        r[0] = (*w.get_unchecked(0)).mul_add(*s, r[0]);
        r[1] = (*w.get_unchecked(1)).mul_add(*s, r[1]);
        r[2] = (*w.get_unchecked(2)).mul_add(*s, r[2]);
        r[3] = (*w.get_unchecked(3)).mul_add(*s, r[3]);
    }
    r
}

/// F32-native 4-lane interleaved dual-frame dot product (scalar reference).
///
/// Processes two state vectors against the same weight slice using `mul_add`
/// (FMA3) to match the rounding of the AVX2/FMA kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[inline]
pub unsafe fn dot_product_4x_f32_dual_scalar(
    weights: &[[f32; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut r0 = [0.0f32; 4];
    let mut r1 = [0.0f32; 4];
    for i in 0..len {
        let w = weights.get_unchecked(i);
        let s0 = state_f0.get_unchecked(i);
        let s1 = state_f1.get_unchecked(i);
        r0[0] = (*w.get_unchecked(0)).mul_add(*s0, r0[0]);
        r0[1] = (*w.get_unchecked(1)).mul_add(*s0, r0[1]);
        r0[2] = (*w.get_unchecked(2)).mul_add(*s0, r0[2]);
        r0[3] = (*w.get_unchecked(3)).mul_add(*s0, r0[3]);
        r1[0] = (*w.get_unchecked(0)).mul_add(*s1, r1[0]);
        r1[1] = (*w.get_unchecked(1)).mul_add(*s1, r1[1]);
        r1[2] = (*w.get_unchecked(2)).mul_add(*s1, r1[2]);
        r1[3] = (*w.get_unchecked(3)).mul_add(*s1, r1[3]);
    }
    (r0, r1)
}

/// Same logic as above (4 sums at once), but using the compact BF16 format.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_4x_interleaved_bf16_fallback(
    weights: &[[u16; 4]],
    state: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];
    let mut comp = [0.0f32; 4];

    for i in 0..len {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
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
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
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
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_bf16_4x_fallback(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    in_frame: &[u16],
) -> [f32; 4] {
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    unsafe {
        [
            dot_product_bf16_fallback(in_frame, w0),
            dot_product_bf16_fallback(in_frame, w1),
            dot_product_bf16_fallback(in_frame, w2),
            dot_product_bf16_fallback(in_frame, w3),
        ]
    }
}
