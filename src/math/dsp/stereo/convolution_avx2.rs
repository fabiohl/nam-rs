// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use core::arch::x86_64::*;

/// Stereo Interleaved Convolution AVX2.
/// Loads coefficients once and applies them to both channels.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_stereo_avx2(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    debug_assert!(
        (coeffs as usize).is_multiple_of(32),
        "coeffs must be 32-byte aligned"
    );
    let mut sum_l0 = _mm256_setzero_ps();
    let mut sum_l1 = _mm256_setzero_ps();
    let mut sum_r0 = _mm256_setzero_ps();
    let mut sum_r1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= taps {
        let h0 = _mm256_load_ps(coeffs.add(i));
        let x0_l = _mm256_loadu_ps(input_l.add(i));
        let x0_r = _mm256_loadu_ps(input_r.add(i));
        sum_l0 = _mm256_fmadd_ps(h0, x0_l, sum_l0);
        sum_r0 = _mm256_fmadd_ps(h0, x0_r, sum_r0);

        let h1 = _mm256_load_ps(coeffs.add(i + 8));
        let x1_l = _mm256_loadu_ps(input_l.add(i + 8));
        let x1_r = _mm256_loadu_ps(input_r.add(i + 8));
        sum_l1 = _mm256_fmadd_ps(h1, x1_l, sum_l1);
        sum_r1 = _mm256_fmadd_ps(h1, x1_r, sum_r1);

        i += 16;
    }

    while i + 8 <= taps {
        let h = _mm256_load_ps(coeffs.add(i));
        let x_l = _mm256_loadu_ps(input_l.add(i));
        let x_r = _mm256_loadu_ps(input_r.add(i));
        sum_l0 = _mm256_fmadd_ps(h, x_l, sum_l0);
        sum_r0 = _mm256_fmadd_ps(h, x_r, sum_r0);
        i += 8;
    }

    // Horizontal reduction L
    let sum_l = _mm256_add_ps(sum_l0, sum_l1);
    let hi128_l = _mm256_extractf128_ps(sum_l, 1);
    let lo128_l = _mm256_castps256_ps128(sum_l);
    let s128_l = _mm_add_ps(lo128_l, hi128_l);
    let shuf_l = _mm_movehdup_ps(s128_l);
    let sums_l = _mm_add_ps(s128_l, shuf_l);
    let shuf2_l = _mm_movehl_ps(sums_l, sums_l);
    let r_l = _mm_add_ss(sums_l, shuf2_l);
    let mut out_l = 0.0f32;
    _mm_store_ss(&mut out_l, r_l);

    // Horizontal reduction R
    let sum_r = _mm256_add_ps(sum_r0, sum_r1);
    let hi128_r = _mm256_extractf128_ps(sum_r, 1);
    let lo128_r = _mm256_castps256_ps128(sum_r);
    let s128_r = _mm_add_ps(lo128_r, hi128_r);
    let shuf_r = _mm_movehdup_ps(s128_r);
    let sums_r = _mm_add_ps(s128_r, shuf_r);
    let shuf2_r = _mm_movehl_ps(sums_r, sums_r);
    let r_r = _mm_add_ss(sums_r, shuf2_r);
    let mut out_r = 0.0f32;
    _mm_store_ss(&mut out_r, r_r);

    while i < taps {
        let h = *coeffs.add(i);
        out_l += h * *input_l.add(i);
        out_r += h * *input_r.add(i);
        i += 1;
    }

    (out_l, out_r)
}

/// Stereo Dual Convolution AVX2.
/// Performs two stereo convolutions (for two coefficient sets coeffs0 and coeffs1)
/// over the same input buffers input_l and input_r.
/// Loads input samples once and applies them to both coefficient sets.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_stereo_dual_avx2(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> ((f32, f32), (f32, f32)) {
    debug_assert!(
        (coeffs0 as usize).is_multiple_of(32),
        "coeffs0 must be 32-byte aligned"
    );
    debug_assert!(
        (coeffs1 as usize).is_multiple_of(32),
        "coeffs1 must be 32-byte aligned"
    );
    let mut sum0_l0 = _mm256_setzero_ps();
    let mut sum0_r0 = _mm256_setzero_ps();
    let mut sum0_l1 = _mm256_setzero_ps();
    let mut sum0_r1 = _mm256_setzero_ps();

    let mut sum1_l0 = _mm256_setzero_ps();
    let mut sum1_r0 = _mm256_setzero_ps();
    let mut sum1_l1 = _mm256_setzero_ps();
    let mut sum1_r1 = _mm256_setzero_ps();

    let mut i = 0;

    while i + 16 <= taps {
        let x0_l = _mm256_loadu_ps(input_l.add(i));
        let x0_r = _mm256_loadu_ps(input_r.add(i));

        let h0_0 = _mm256_load_ps(coeffs0.add(i));
        sum0_l0 = _mm256_fmadd_ps(h0_0, x0_l, sum0_l0);
        sum0_r0 = _mm256_fmadd_ps(h0_0, x0_r, sum0_r0);

        let h1_0 = _mm256_load_ps(coeffs1.add(i));
        sum1_l0 = _mm256_fmadd_ps(h1_0, x0_l, sum1_l0);
        sum1_r0 = _mm256_fmadd_ps(h1_0, x0_r, sum1_r0);

        let x1_l = _mm256_loadu_ps(input_l.add(i + 8));
        let x1_r = _mm256_loadu_ps(input_r.add(i + 8));

        let h0_1 = _mm256_load_ps(coeffs0.add(i + 8));
        sum0_l1 = _mm256_fmadd_ps(h0_1, x1_l, sum0_l1);
        sum0_r1 = _mm256_fmadd_ps(h0_1, x1_r, sum0_r1);

        let h1_1 = _mm256_load_ps(coeffs1.add(i + 8));
        sum1_l1 = _mm256_fmadd_ps(h1_1, x1_l, sum1_l1);
        sum1_r1 = _mm256_fmadd_ps(h1_1, x1_r, sum1_r1);

        i += 16;
    }

    while i + 8 <= taps {
        let x_l = _mm256_loadu_ps(input_l.add(i));
        let x_r = _mm256_loadu_ps(input_r.add(i));

        let h0 = _mm256_load_ps(coeffs0.add(i));
        sum0_l0 = _mm256_fmadd_ps(h0, x_l, sum0_l0);
        sum0_r0 = _mm256_fmadd_ps(h0, x_r, sum0_r0);

        let h1 = _mm256_load_ps(coeffs1.add(i));
        sum1_l0 = _mm256_fmadd_ps(h1, x_l, sum1_l0);
        sum1_r0 = _mm256_fmadd_ps(h1, x_r, sum1_r0);

        i += 8;
    }

    // Combine accumulators
    let sum0_l = _mm256_add_ps(sum0_l0, sum0_l1);
    let sum0_r = _mm256_add_ps(sum0_r0, sum0_r1);
    let sum1_l = _mm256_add_ps(sum1_l0, sum1_l1);
    let sum1_r = _mm256_add_ps(sum1_r0, sum1_r1);

    // Horizontal reduction sum0_l
    let hi128_0l = _mm256_extractf128_ps(sum0_l, 1);
    let lo128_0l = _mm256_castps256_ps128(sum0_l);
    let s128_0l = _mm_add_ps(lo128_0l, hi128_0l);
    let shuf_0l = _mm_movehdup_ps(s128_0l);
    let sums_0l = _mm_add_ps(s128_0l, shuf_0l);
    let shuf2_0l = _mm_movehl_ps(sums_0l, sums_0l);
    let r_0l = _mm_add_ss(sums_0l, shuf2_0l);
    let mut out0_l = 0.0f32;
    _mm_store_ss(&mut out0_l, r_0l);

    // Horizontal reduction sum0_r
    let hi128_0r = _mm256_extractf128_ps(sum0_r, 1);
    let lo128_0r = _mm256_castps256_ps128(sum0_r);
    let s128_0r = _mm_add_ps(lo128_0r, hi128_0r);
    let shuf_0r = _mm_movehdup_ps(s128_0r);
    let sums_0r = _mm_add_ps(s128_0r, shuf_0r);
    let shuf2_0r = _mm_movehl_ps(sums_0r, sums_0r);
    let r_0r = _mm_add_ss(sums_0r, shuf2_0r);
    let mut out0_r = 0.0f32;
    _mm_store_ss(&mut out0_r, r_0r);

    // Horizontal reduction sum1_l
    let hi128_1l = _mm256_extractf128_ps(sum1_l, 1);
    let lo128_1l = _mm256_castps256_ps128(sum1_l);
    let s128_1l = _mm_add_ps(lo128_1l, hi128_1l);
    let shuf_1l = _mm_movehdup_ps(s128_1l);
    let sums_1l = _mm_add_ps(s128_1l, shuf_1l);
    let shuf2_1l = _mm_movehl_ps(sums_1l, sums_1l);
    let r_1l = _mm_add_ss(sums_1l, shuf2_1l);
    let mut out1_l = 0.0f32;
    _mm_store_ss(&mut out1_l, r_1l);

    // Horizontal reduction sum1_r
    let hi128_1r = _mm256_extractf128_ps(sum1_r, 1);
    let lo128_1r = _mm256_castps256_ps128(sum1_r);
    let s128_1r = _mm_add_ps(lo128_1r, hi128_1r);
    let shuf_1r = _mm_movehdup_ps(s128_1r);
    let sums_1r = _mm_add_ps(s128_1r, shuf_1r);
    let shuf2_1r = _mm_movehl_ps(sums_1r, sums_1r);
    let r_1r = _mm_add_ss(sums_1r, shuf2_1r);
    let mut out1_r = 0.0f32;
    _mm_store_ss(&mut out1_r, r_1r);

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

/// Mono Dual Convolution AVX2.
/// Performs two mono convolutions on the same input buffer, reusing the loaded input samples.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_mono_dual_avx2(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input: *const f32,
    taps: usize,
) -> (f32, f32) {
    debug_assert!(
        (coeffs0 as usize).is_multiple_of(32),
        "coeffs0 must be 32-byte aligned"
    );
    debug_assert!(
        (coeffs1 as usize).is_multiple_of(32),
        "coeffs1 must be 32-byte aligned"
    );
    let mut sum0_0 = _mm256_setzero_ps();
    let mut sum0_1 = _mm256_setzero_ps();
    let mut sum1_0 = _mm256_setzero_ps();
    let mut sum1_1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= taps {
        let x0 = _mm256_loadu_ps(input.add(i));

        let h0_0 = _mm256_load_ps(coeffs0.add(i));
        sum0_0 = _mm256_fmadd_ps(h0_0, x0, sum0_0);

        let h1_0 = _mm256_load_ps(coeffs1.add(i));
        sum1_0 = _mm256_fmadd_ps(h1_0, x0, sum1_0);

        let x1 = _mm256_loadu_ps(input.add(i + 8));

        let h0_1 = _mm256_load_ps(coeffs0.add(i + 8));
        sum0_1 = _mm256_fmadd_ps(h0_1, x1, sum0_1);

        let h1_1 = _mm256_load_ps(coeffs1.add(i + 8));
        sum1_1 = _mm256_fmadd_ps(h1_1, x1, sum1_1);

        i += 16;
    }

    while i + 8 <= taps {
        let x = _mm256_loadu_ps(input.add(i));

        let h0 = _mm256_load_ps(coeffs0.add(i));
        sum0_0 = _mm256_fmadd_ps(h0, x, sum0_0);

        let h1 = _mm256_load_ps(coeffs1.add(i));
        sum1_0 = _mm256_fmadd_ps(h1, x, sum1_0);

        i += 8;
    }

    let sum0 = _mm256_add_ps(sum0_0, sum0_1);
    let sum1 = _mm256_add_ps(sum1_0, sum1_1);

    // Horizontal reduction sum0
    let hi128_0 = _mm256_extractf128_ps(sum0, 1);
    let lo128_0 = _mm256_castps256_ps128(sum0);
    let s128_0 = _mm_add_ps(lo128_0, hi128_0);
    let shuf_0 = _mm_movehdup_ps(s128_0);
    let sums_0 = _mm_add_ps(s128_0, shuf_0);
    let shuf2_0 = _mm_movehl_ps(sums_0, sums_0);
    let r_0 = _mm_add_ss(sums_0, shuf2_0);
    let mut out0 = 0.0f32;
    _mm_store_ss(&mut out0, r_0);

    // Horizontal reduction sum1
    let hi128_1 = _mm256_extractf128_ps(sum1, 1);
    let lo128_1 = _mm256_castps256_ps128(sum1);
    let s128_1 = _mm_add_ps(lo128_1, hi128_1);
    let shuf_1 = _mm_movehdup_ps(s128_1);
    let sums_1 = _mm_add_ps(s128_1, shuf_1);
    let shuf2_1 = _mm_movehl_ps(sums_1, sums_1);
    let r_1 = _mm_add_ss(sums_1, shuf2_1);
    let mut out1 = 0.0f32;
    _mm_store_ss(&mut out1, r_1);

    while i < taps {
        let xl = *input.add(i);
        out0 += *coeffs0.add(i) * xl;
        out1 += *coeffs1.add(i) * xl;
        i += 1;
    }

    (out0, out1)
}

/// Mono Convolution AVX2.
/// Loads coefficients and applies them to a single channel.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_mono_avx2(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    debug_assert!(
        (coeffs as usize).is_multiple_of(32),
        "coeffs must be 32-byte aligned"
    );
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= taps {
        let h0 = _mm256_load_ps(coeffs.add(i));
        let x0 = _mm256_loadu_ps(input.add(i));
        sum0 = _mm256_fmadd_ps(h0, x0, sum0);

        let h1 = _mm256_load_ps(coeffs.add(i + 8));
        let x1 = _mm256_loadu_ps(input.add(i + 8));
        sum1 = _mm256_fmadd_ps(h1, x1, sum1);

        i += 16;
    }

    while i + 8 <= taps {
        let h = _mm256_load_ps(coeffs.add(i));
        let x = _mm256_loadu_ps(input.add(i));
        sum0 = _mm256_fmadd_ps(h, x, sum0);
        i += 8;
    }

    // Horizontal reduction
    let sum = _mm256_add_ps(sum0, sum1);
    let hi128 = _mm256_extractf128_ps(sum, 1);
    let lo128 = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let r = _mm_add_ss(sums, shuf2);
    let mut out = 0.0f32;
    _mm_store_ss(&mut out, r);

    while i < taps {
        let h = *coeffs.add(i);
        out += h * *input.add(i);
        i += 1;
    }

    out
}
