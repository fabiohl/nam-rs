// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use core::arch::x86_64::*;

/// Convolução Stereo Interleaved AVX2.
/// Carrega coeficientes uma única vez e aplica a ambos os canais.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_stereo_avx2(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
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

    // Redução horizontal L
    let sum_l = _mm256_add_ps(sum_l0, sum_l1);
    let hi128_l = _mm256_extractf128_ps(sum_l, 1);
    let lo128_l = _mm256_castps256_ps128(sum_l);
    let s128_l = _mm_add_ps(lo128_l, hi128_l);
    let shuf_l = _mm_movehdup_ps(s128_l);
    let sums_l = _mm_add_ps(s128_l, shuf_l);
    let shuf2_l = _mm_movehl_ps(sums_l, sums_l);
    let r_l = _mm_add_ss(sums_l, shuf2_l);
    let mut out_l = _mm_cvtss_f32(r_l);

    // Redução horizontal R
    let sum_r = _mm256_add_ps(sum_r0, sum_r1);
    let hi128_r = _mm256_extractf128_ps(sum_r, 1);
    let lo128_r = _mm256_castps256_ps128(sum_r);
    let s128_r = _mm_add_ps(lo128_r, hi128_r);
    let shuf_r = _mm_movehdup_ps(s128_r);
    let sums_r = _mm_add_ps(s128_r, shuf_r);
    let shuf2_r = _mm_movehl_ps(sums_r, sums_r);
    let r_r = _mm_add_ss(sums_r, shuf2_r);
    let mut out_r = _mm_cvtss_f32(r_r);

    while i < taps {
        let h = *coeffs.add(i);
        out_l += h * *input_l.add(i);
        out_r += h * *input_r.add(i);
        i += 1;
    }

    (out_l, out_r)
}

/// Convolução Stereo Dual AVX2.
/// Realiza duas convoluções estéreo (para dois conjuntos de coeficientes coeffs0 e coeffs1)
/// sobre os mesmos buffers de entrada input_l e input_r.
/// Carrega amostras de entrada uma única vez e aplica a ambos os conjuntos de coeficientes.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_stereo_dual_avx2(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> ((f32, f32), (f32, f32)) {
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

    // Redução horizontal sum0_l
    let hi128_0l = _mm256_extractf128_ps(sum0_l, 1);
    let lo128_0l = _mm256_castps256_ps128(sum0_l);
    let s128_0l = _mm_add_ps(lo128_0l, hi128_0l);
    let shuf_0l = _mm_movehdup_ps(s128_0l);
    let sums_0l = _mm_add_ps(s128_0l, shuf_0l);
    let shuf2_0l = _mm_movehl_ps(sums_0l, sums_0l);
    let r_0l = _mm_add_ss(sums_0l, shuf2_0l);
    let mut out0_l = _mm_cvtss_f32(r_0l);

    // Redução horizontal sum0_r
    let hi128_0r = _mm256_extractf128_ps(sum0_r, 1);
    let lo128_0r = _mm256_castps256_ps128(sum0_r);
    let s128_0r = _mm_add_ps(lo128_0r, hi128_0r);
    let shuf_0r = _mm_movehdup_ps(s128_0r);
    let sums_0r = _mm_add_ps(s128_0r, shuf_0r);
    let shuf2_0r = _mm_movehl_ps(sums_0r, sums_0r);
    let r_0r = _mm_add_ss(sums_0r, shuf2_0r);
    let mut out0_r = _mm_cvtss_f32(r_0r);

    // Redução horizontal sum1_l
    let hi128_1l = _mm256_extractf128_ps(sum1_l, 1);
    let lo128_1l = _mm256_castps256_ps128(sum1_l);
    let s128_1l = _mm_add_ps(lo128_1l, hi128_1l);
    let shuf_1l = _mm_movehdup_ps(s128_1l);
    let sums_1l = _mm_add_ps(s128_1l, shuf_1l);
    let shuf2_1l = _mm_movehl_ps(sums_1l, sums_1l);
    let r_1l = _mm_add_ss(sums_1l, shuf2_1l);
    let mut out1_l = _mm_cvtss_f32(r_1l);

    // Redução horizontal sum1_r
    let hi128_1r = _mm256_extractf128_ps(sum1_r, 1);
    let lo128_1r = _mm256_castps256_ps128(sum1_r);
    let s128_1r = _mm_add_ps(lo128_1r, hi128_1r);
    let shuf_1r = _mm_movehdup_ps(s128_1r);
    let sums_1r = _mm_add_ps(s128_1r, shuf_1r);
    let shuf2_1r = _mm_movehl_ps(sums_1r, sums_1r);
    let r_1r = _mm_add_ss(sums_1r, shuf2_1r);
    let mut out1_r = _mm_cvtss_f32(r_1r);

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

/// Convolução Mono AVX2.
/// Carrega coeficientes e aplica a um único canal.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_mono_avx2(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
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

    // Redução horizontal
    let sum = _mm256_add_ps(sum0, sum1);
    let hi128 = _mm256_extractf128_ps(sum, 1);
    let lo128 = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let r = _mm_add_ss(sums, shuf2);
    let mut out = _mm_cvtss_f32(r);

    while i < taps {
        let h = *coeffs.add(i);
        out += h * *input.add(i);
        i += 1;
    }

    out
}
