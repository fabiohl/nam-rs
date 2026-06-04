// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// Converts high-precision decimal numbers (f32) to the compact format (BF16).
/// This saves a lot of memory and is used to store neural network "weights".
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
pub unsafe fn f32_to_bf16_fallback(src: &[f32], dest: &mut [u16]) {
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        // Gets the 16 most important bits of the number and discards the rest.
        *d = (s.to_bits() >> 16) as u16;
    }
}

/// Applies the Tanh "squashing" function across an entire list of numbers.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
pub unsafe fn tanh_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = v.tanh();
    }
}

/// Applies the "Sigmoid" function across an entire list of numbers.
/// Sigmoid squashes numbers to stay between 0.0 and 1.0.
/// It's great for creating "gates" or automatic volume controls.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
pub unsafe fn sigmoid_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Sums all numbers in a list and returns a single final value.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
pub unsafe fn horizontal_sum_fallback(ptr: *const f32, len: usize) -> f32 {
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
    slice.iter().sum()
}

/// Applies a volume (gain) by multiplying each sample by the desired value.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
pub unsafe fn apply_gain_fallback(data: &mut [f32], gain: f32) {
    for x in data.iter_mut() {
        *x *= gain;
    }
}

/// Computes the energy (Mean Square) of a block via scalar.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
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
