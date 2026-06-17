// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Dot Product 4x f32 — AVX2/FMA kernel (hi‑fi mode).
//!
//! Processes `state[i] · weights[i]` for 4 interleaved output channels.
//!
//! # Bit‑exactness guarantee
//! Both the scalar reference (`mul_add`) and this kernel (`_mm_fmadd_ps`) use
//! the same FMA3 fused multiply‑add, producing **identical rounding** on any
//! x86 CPU with FMA support (x86‑64‑v3). No dequantization or precision
//! conversion is involved — the result is bit‑identical by construction.

use core::arch::x86_64::*;

/// 4‑lane interleaved dot product (`weights: &[[f32; 4]]`, `state: &[f32]`) with
/// AVX2/FMA.
///
/// # Strategy
/// - One weight row (`[f32;4]`) is loaded into `__m128`.
/// - `state[i]` is broadcast into `__m128` via `_mm_set1_ps`.
/// - `_mm_fmadd_ps(w128, s_broadcast, acc128)` → one FMA per sample per channel,
///   matching the scalar `mul_add` rounding chain exactly.
/// - On x86‑64‑v3 the compiler may schedule multiple iterations ahead of time
///   (OoO execution / loop unrolling), keeping latency covered without manually
///   unrolling in registers.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that the memory regions
/// are valid for unaligned load.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_4x_f32_avx2(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len();
    let mut acc = _mm_setzero_ps();
    let mut i = 0;

    unsafe {
        while i < len {
            let s = _mm_set1_ps(*state.get_unchecked(i));
            let w = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            acc = _mm_fmadd_ps(w, s, acc);
            i += 1;
        }

        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), acc);
        out
    }
}

/// Dual-frame 4‑lane interleaved dot product (`weights: &[[f32; 4]]`,
/// `state_f0: &[f32]`, `state_f1: &[f32]`) with AVX2/FMA.
///
/// # Strategy
/// - One weight row (`[f32;4]`) is loaded into `__m128` and broadcast into
///   both halves of `__m256`.
/// - `state_f0[i]` and `state_f1[i]` are each broadcast across 4 lanes and
///   packed into a single `__m256` (`_mm256_set_m128`).
/// - Single `_mm256_fmadd_ps` accumulates both frames simultaneously — the
///   weight is reused, matching the dual-frame semantics of the scalar path.
/// - At the end, `_mm256_extractf128_ps` splits back into per-frame `__m128`
///   results.
///
/// # Bit‑exactness
/// Both the scalar reference (`mul_add`) and this kernel (`_mm_fmadd_ps`)
/// use FMA3 fused multiply‑add → bit‑identical result on any x86‑64‑v3 CPU.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_4x_f32_dual_avx2(
    weights: &[[f32; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut acc256 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        while i < len {
            let w128 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w256 = _mm256_broadcast_ps(&w128);
            let s_f0 = _mm_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm_set1_ps(*state_f1.get_unchecked(i));
            let s_blend = _mm256_set_m128(s_f1, s_f0);
            acc256 = _mm256_fmadd_ps(w256, s_blend, acc256);
            i += 1;
        }

        let acc_f0 = _mm256_extractf128_ps(acc256, 0);
        let acc_f1 = _mm256_extractf128_ps(acc256, 1);
        let mut out_f0 = [0.0f32; 4];
        let mut out_f1 = [0.0f32; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), acc_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), acc_f1);
        (out_f0, out_f1)
    }
}
