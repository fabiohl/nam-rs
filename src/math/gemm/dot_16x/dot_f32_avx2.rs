// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Dot Product 16x f32 — AVX2/FMA kernel (f32 native weights).
//!
//! Processes 16 interleaved weight rows using two `__m256` registers per row
//! (lo and hi halves of the 16-element array), paired with a single state
//! broadcast per iteration.
//!
//! # Precision
//! Both the scalar reference (`mul_add`) and this kernel (`_mm256_fmadd_ps`) use
//! the same FMA3 fused multiply‑add instruction, producing mathematically
//! equivalent results. The 4‑way accumulator unrolling splits the summation
//! across independent chains (`acc_lo0..3` and `acc_hi0..3`) with a final
//! horizontal reduction, which may yield slightly different rounding (< 2 ULP)
//! compared to the strictly‑serial FMA chain of the scalar reference. No
//! dequantization or precision conversion is involved.

use crate::dot4x_simd4;
use core::arch::x86_64::*;

/// 16‑lane interleaved dot product (`weights: &[[f32; 16]]`, `state: &[f32]`)
/// with AVX2/FMA.
///
/// # Strategy
/// - 16 weights per row split into two `__m256` loads (lo `[0..8)`, hi `[8..16)`).
/// - State scalar broadcast to all lanes of both halves via `_mm256_set1_ps`.
/// - Main loop processes 4 input samples per iteration using 4 independent
///   pairs of `__m256` accumulators (`acc_lo0..3`, `acc_hi0..3`), interleaved
///   by index `i`, to break the FMA latency chain and allow OoO execution to
///   overlap them.
/// - Tail (< 4 elements) falls back to a single‑accumulator‑pair loop.
/// - Final reduction: tree‑sum both halves independently, then store as
///   contiguous `[f32; 16]`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that the memory regions
/// are valid for unaligned load.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_16x_f32_avx2(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
    let len = state.len();
    let mut acc_lo0 = _mm256_setzero_ps();
    let mut acc_lo1 = _mm256_setzero_ps();
    let mut acc_lo2 = _mm256_setzero_ps();
    let mut acc_lo3 = _mm256_setzero_ps();
    let mut acc_hi0 = _mm256_setzero_ps();
    let mut acc_hi1 = _mm256_setzero_ps();
    let mut acc_hi2 = _mm256_setzero_ps();
    let mut acc_hi3 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        dot4x_simd4!(i, len, {
            let s0 = _mm256_set1_ps(*state.get_unchecked(i));
            let w0_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w0_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            acc_lo0 = _mm256_fmadd_ps(w0_lo, s0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(w0_hi, s0, acc_hi0);

            let s1 = _mm256_set1_ps(*state.get_unchecked(i + 1));
            let w1_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let w1_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 1) as *const f32).add(8));
            acc_lo1 = _mm256_fmadd_ps(w1_lo, s1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(w1_hi, s1, acc_hi1);

            let s2 = _mm256_set1_ps(*state.get_unchecked(i + 2));
            let w2_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let w2_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 2) as *const f32).add(8));
            acc_lo2 = _mm256_fmadd_ps(w2_lo, s2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(w2_hi, s2, acc_hi2);

            let s3 = _mm256_set1_ps(*state.get_unchecked(i + 3));
            let w3_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let w3_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 3) as *const f32).add(8));
            acc_lo3 = _mm256_fmadd_ps(w3_lo, s3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(w3_hi, s3, acc_hi3);
        });

        while i < len {
            let s = _mm256_set1_ps(*state.get_unchecked(i));
            let w_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            acc_lo0 = _mm256_fmadd_ps(w_lo, s, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(w_hi, s, acc_hi0);
            i += 1;
        }

        acc_lo0 = _mm256_add_ps(acc_lo0, acc_lo1);
        acc_lo2 = _mm256_add_ps(acc_lo2, acc_lo3);
        acc_lo0 = _mm256_add_ps(acc_lo0, acc_lo2);

        acc_hi0 = _mm256_add_ps(acc_hi0, acc_hi1);
        acc_hi2 = _mm256_add_ps(acc_hi2, acc_hi3);
        acc_hi0 = _mm256_add_ps(acc_hi0, acc_hi2);

        let mut out = [0.0f32; 16];
        _mm256_storeu_ps(out.as_mut_ptr(), acc_lo0);
        _mm256_storeu_ps(out.as_mut_ptr().add(8), acc_hi0);
        out
    }
}
