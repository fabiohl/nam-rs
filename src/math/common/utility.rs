// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Utilitários SIMD para reduções e operações horizontais.

use core::arch::x86_64::*;

/// Soma horizontal de um registrador AVX2 (256-bit) para escalar f32.
///
/// Executa a redução via extração de lanes de 128-bit e somas sucessivas.
/// Total de instruções: ~7 (incluindo store_ss).
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let r = _mm_add_ss(sums, shuf2);
    let mut out = 0.0f32;
    _mm_store_ss(&mut out, r);
    out
}

/// Soma horizontal de um registrador AVX-512 (512-bit) para escalar f32.
///
/// Utiliza o intrínseco nativo de redução do AVX-512 Foundation.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn hsum_avx512(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// Soma horizontal de um buffer f32 via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn horizontal_sum_avx2(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm256_setzero_ps();

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        sum_v = _mm256_add_ps(sum_v, v);
        i += 8;
    }

    let mut total = hsum_avx2(sum_v);

    while i < len {
        total += *ptr.add(i);
        i += 1;
    }

    total
}

/// Soma horizontal de um buffer f32 via AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn horizontal_sum_avx512(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();

    while i + 16 <= len {
        let v = _mm512_loadu_ps(ptr.add(i));
        sum_v = _mm512_add_ps(sum_v, v);
        i += 16;
    }

    let mut total = hsum_avx512(sum_v);

    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let v = _mm512_maskz_loadu_ps(mask, ptr.add(i));
        total += hsum_avx512(v);
    }

    total
}
