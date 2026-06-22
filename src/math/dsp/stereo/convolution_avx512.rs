// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use core::arch::x86_64::*;

/// Stereo Convolution: Applies a filter (coefficients) to two audio channels at the same time.
/// It's like passing sound through an equalizer or simulating a room (reverb).
#[target_feature(enable = "avx512f")]
pub unsafe fn convolve_stereo_avx512(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    debug_assert!(
        (coeffs as usize).is_multiple_of(64),
        "coeffs must be 64-byte aligned"
    );
    let mut sum_l0 = _mm512_setzero_ps();
    let mut sum_l1 = _mm512_setzero_ps();
    let mut sum_r0 = _mm512_setzero_ps();
    let mut sum_r1 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 32 <= taps {
        let h0 = _mm512_load_ps(coeffs.add(i));
        let x0_l = _mm512_loadu_ps(input_l.add(i));
        let x0_r = _mm512_loadu_ps(input_r.add(i));
        sum_l0 = _mm512_fmadd_ps(h0, x0_l, sum_l0);
        sum_r0 = _mm512_fmadd_ps(h0, x0_r, sum_r0);

        let h1 = _mm512_load_ps(coeffs.add(i + 16));
        let x1_l = _mm512_loadu_ps(input_l.add(i + 16));
        let x1_r = _mm512_loadu_ps(input_r.add(i + 16));
        sum_l1 = _mm512_fmadd_ps(h1, x1_l, sum_l1);
        sum_r1 = _mm512_fmadd_ps(h1, x1_r, sum_r1);

        i += 32;
    }

    let sum_l = _mm512_add_ps(sum_l0, sum_l1);
    let sum_r = _mm512_add_ps(sum_r0, sum_r1);
    let mut out_l = _mm512_reduce_add_ps(sum_l);
    let mut out_r = _mm512_reduce_add_ps(sum_r);

    while i < taps {
        let h = *coeffs.add(i);
        out_l += h * *input_l.add(i);
        out_r += h * *input_r.add(i);
        i += 1;
    }

    (out_l, out_r)
}

/// Stereo Dual Convolution AVX-512.
/// Performs two stereo convolutions (for two coefficient sets coeffs0 and coeffs1)
/// over the same input buffers input_l and input_r.
/// Loads input samples once and applies them to both coefficient sets.
#[target_feature(enable = "avx512f")]
pub unsafe fn convolve_stereo_dual_avx512(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> ((f32, f32), (f32, f32)) {
    debug_assert!(
        (coeffs0 as usize).is_multiple_of(64),
        "coeffs0 must be 64-byte aligned"
    );
    debug_assert!(
        (coeffs1 as usize).is_multiple_of(64),
        "coeffs1 must be 64-byte aligned"
    );
    let mut sum0_l = _mm512_setzero_ps();
    let mut sum0_r = _mm512_setzero_ps();
    let mut sum1_l = _mm512_setzero_ps();
    let mut sum1_r = _mm512_setzero_ps();
    let mut i = 0;

    while i + 16 <= taps {
        let x_l = _mm512_loadu_ps(input_l.add(i));
        let x_r = _mm512_loadu_ps(input_r.add(i));

        let h0 = _mm512_load_ps(coeffs0.add(i));
        sum0_l = _mm512_fmadd_ps(h0, x_l, sum0_l);
        sum0_r = _mm512_fmadd_ps(h0, x_r, sum0_r);

        let h1 = _mm512_load_ps(coeffs1.add(i));
        sum1_l = _mm512_fmadd_ps(h1, x_l, sum1_l);
        sum1_r = _mm512_fmadd_ps(h1, x_r, sum1_r);

        i += 16;
    }

    let mut out0_l = _mm512_reduce_add_ps(sum0_l);
    let mut out0_r = _mm512_reduce_add_ps(sum0_r);
    let mut out1_l = _mm512_reduce_add_ps(sum1_l);
    let mut out1_r = _mm512_reduce_add_ps(sum1_r);

    while i < taps {
        let h0 = *coeffs0.add(i);
        let h1 = *coeffs1.add(i);
        let xl = *input_l.add(i);
        let xr = *input_r.add(i);
        out0_l += h0 * xl;
        out0_r += h0 * xr;
        out1_l += h1 * xl;
        out1_r += h1 * xr;
        i += 1;
    }

    ((out0_l, out0_r), (out1_l, out1_r))
}

/// Mono Dual Convolution AVX-512.
/// Performs two mono convolutions on the same input buffer, reusing the loaded input samples.
#[target_feature(enable = "avx512f")]
pub unsafe fn convolve_mono_dual_avx512(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input: *const f32,
    taps: usize,
) -> (f32, f32) {
    debug_assert!(
        (coeffs0 as usize).is_multiple_of(64),
        "coeffs0 must be 64-byte aligned"
    );
    debug_assert!(
        (coeffs1 as usize).is_multiple_of(64),
        "coeffs1 must be 64-byte aligned"
    );
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 16 <= taps {
        let x = _mm512_loadu_ps(input.add(i));

        let h0 = _mm512_load_ps(coeffs0.add(i));
        sum0 = _mm512_fmadd_ps(h0, x, sum0);

        let h1 = _mm512_load_ps(coeffs1.add(i));
        sum1 = _mm512_fmadd_ps(h1, x, sum1);

        i += 16;
    }

    let mut out0 = _mm512_reduce_add_ps(sum0);
    let mut out1 = _mm512_reduce_add_ps(sum1);

    while i < taps {
        let xl = *input.add(i);
        out0 += *coeffs0.add(i) * xl;
        out1 += *coeffs1.add(i) * xl;
        i += 1;
    }

    (out0, out1)
}

/// Mono Convolution AVX-512.
/// Loads coefficients and applies them to a single channel.
#[target_feature(enable = "avx512f")]
pub unsafe fn convolve_mono_avx512(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    debug_assert!(
        (coeffs as usize).is_multiple_of(64),
        "coeffs must be 64-byte aligned"
    );
    let mut sum0 = _mm512_setzero_ps();
    let mut sum1 = _mm512_setzero_ps();
    let mut i = 0;

    while i + 32 <= taps {
        let h0 = _mm512_load_ps(coeffs.add(i));
        let x0 = _mm512_loadu_ps(input.add(i));
        sum0 = _mm512_fmadd_ps(h0, x0, sum0);

        let h1 = _mm512_load_ps(coeffs.add(i + 16));
        let x1 = _mm512_loadu_ps(input.add(i + 16));
        sum1 = _mm512_fmadd_ps(h1, x1, sum1);

        i += 32;
    }

    let sum = _mm512_add_ps(sum0, sum1);
    let mut out = _mm512_reduce_add_ps(sum);

    while i < taps {
        let h = *coeffs.add(i);
        out += h * *input.add(i);
        i += 1;
    }

    out
}
