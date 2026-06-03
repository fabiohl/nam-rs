// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels Dot Product 4x — AVX-512 BF16 (VNNI BF16 native accumulation).
//!
//! Uses `_mm512_dpbf16_ps` and `_mm512_slli_epi32` to convert BF16→f32
//! and accumulate strictly in f32 SIMD registers, maintaining full
//! 24-bit mantissa precision throughout the dot product chain.

use core::arch::x86_64::*;

/// Helper: converts 4 adjacent BF16 values into a single __m128 of f32.
///
/// BF16 → f32 is done via: cast u16→u32, shift left 16 bits,
/// reinterpret as f32. The low 4 of the resulting __m256i are extracted
/// into __m128 for subsequent `_mm512_castps128_ps512` + `permutexvar`.
#[inline(always)]
unsafe fn bf16x4_to_f32x4(ptr: *const u16) -> __m128 {
    let v_u16 = _mm_loadl_epi64(ptr as *const __m128i);
    let v_u32 = _mm256_cvtepu16_epi32(v_u16);
    let v_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(v_u32, 16));
    _mm256_castps256_ps128(v_f32)
}

/// Helper: converts 16 adjacent BF16 values into a __m512 of f32.
#[inline(always)]
unsafe fn bf16x16_to_f32x16(ptr: *const u16) -> __m512 {
    let v_u16 = _mm256_loadu_si256(ptr as *const __m256i);
    let v_u32 = _mm512_cvtepu16_epi32(v_u16);
    _mm512_castsi512_ps(_mm512_slli_epi32(v_u32, 16))
}

/// Dot product interleaved 4x with BF16 state — AVX-512 native accumulation.
///
/// Weights are stored in `[[u16; 4]]` interleaved format (4 channels grouped).
/// State is `&[u16]` in BF16. Accumulation happens entirely in f32 ZMM registers.
///
/// Uses 8 alternating ZMM accumulators to break FMA dependency chains,
/// and permutexvar to broadcast each state value to all 4 channels.
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
pub unsafe fn dot_product_4x_interleaved_avx512_bf16(
    weights: &[[u16; 4]],
    state: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut i = 0;

    unsafe {
        let perm_idx = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

        let mut sum_a0 = _mm512_setzero_ps();
        let mut sum_a1 = _mm512_setzero_ps();
        let mut sum_a2 = _mm512_setzero_ps();
        let mut sum_a3 = _mm512_setzero_ps();
        let mut sum_b0 = _mm512_setzero_ps();
        let mut sum_b1 = _mm512_setzero_ps();
        let mut sum_b2 = _mm512_setzero_ps();
        let mut sum_b3 = _mm512_setzero_ps();

        while i + 32 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 48) as *const i8);

            let s_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i))),
            );
            let w_a = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            sum_a0 = _mm512_fmadd_ps(w_a, s_a, sum_a0);

            let s_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 4))),
            );
            let w_b = bf16x16_to_f32x16(weights.as_ptr().add(i + 4) as *const u16);
            sum_a1 = _mm512_fmadd_ps(w_b, s_b, sum_a1);

            let s_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 8))),
            );
            let w_c = bf16x16_to_f32x16(weights.as_ptr().add(i + 8) as *const u16);
            sum_a2 = _mm512_fmadd_ps(w_c, s_c, sum_a2);

            let s_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 12))),
            );
            let w_d = bf16x16_to_f32x16(weights.as_ptr().add(i + 12) as *const u16);
            sum_a3 = _mm512_fmadd_ps(w_d, s_d, sum_a3);

            let s_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 16))),
            );
            let w_e = bf16x16_to_f32x16(weights.as_ptr().add(i + 16) as *const u16);
            sum_b0 = _mm512_fmadd_ps(w_e, s_e, sum_b0);

            let s_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 20))),
            );
            let w_f = bf16x16_to_f32x16(weights.as_ptr().add(i + 20) as *const u16);
            sum_b1 = _mm512_fmadd_ps(w_f, s_f, sum_b1);

            let s_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 24))),
            );
            let w_g = bf16x16_to_f32x16(weights.as_ptr().add(i + 24) as *const u16);
            sum_b2 = _mm512_fmadd_ps(w_g, s_g, sum_b2);

            let s_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 28))),
            );
            let w_h = bf16x16_to_f32x16(weights.as_ptr().add(i + 28) as *const u16);
            sum_b3 = _mm512_fmadd_ps(w_h, s_h, sum_b3);

            i += 32;
        }

        while i + 16 <= len {
            let s_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i))),
            );
            let w_a = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            sum_a0 = _mm512_fmadd_ps(w_a, s_a, sum_a0);

            let s_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 4))),
            );
            let w_b = bf16x16_to_f32x16(weights.as_ptr().add(i + 4) as *const u16);
            sum_a1 = _mm512_fmadd_ps(w_b, s_b, sum_a1);

            let s_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 8))),
            );
            let w_c = bf16x16_to_f32x16(weights.as_ptr().add(i + 8) as *const u16);
            sum_a2 = _mm512_fmadd_ps(w_c, s_c, sum_a2);

            let s_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 12))),
            );
            let w_d = bf16x16_to_f32x16(weights.as_ptr().add(i + 12) as *const u16);
            sum_a3 = _mm512_fmadd_ps(w_d, s_d, sum_a3);

            i += 16;
        }

        while i + 4 <= len {
            let s = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i))),
            );
            let w = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            sum_a0 = _mm512_fmadd_ps(w, s, sum_a0);
            i += 4;
        }

        let s_ab = _mm512_add_ps(_mm512_add_ps(sum_a0, sum_a1), _mm512_add_ps(sum_a2, sum_a3));
        let s_all = _mm512_add_ps(
            s_ab,
            _mm512_add_ps(_mm512_add_ps(sum_b0, sum_b1), _mm512_add_ps(sum_b2, sum_b3)),
        );

        let lo = _mm512_extractf32x4_ps(s_all, 0);
        let hi0 = _mm512_extractf32x4_ps(s_all, 1);
        let hi1 = _mm512_extractf32x4_ps(s_all, 2);
        let hi2 = _mm512_extractf32x4_ps(s_all, 3);
        let mut sum128 = _mm_add_ps(_mm_add_ps(lo, hi0), _mm_add_ps(hi1, hi2));

        // Scalar tail: convert remaining BF16 state values to f32 and accumulate.
        // Each weight group has 4 interleaved channels.
        while i < len {
            let si = f32::from_bits((*state.get_unchecked(i) as u32) << 16);
            let w_ptr = weights.as_ptr().add(i) as *const u16;
            let w0 = f32::from_bits((*w_ptr as u32) << 16);
            let w1 = f32::from_bits((*w_ptr.add(1) as u32) << 16);
            let w2 = f32::from_bits((*w_ptr.add(2) as u32) << 16);
            let w3 = f32::from_bits((*w_ptr.add(3) as u32) << 16);
            let s_vec = _mm_set_ps(w3 * si, w2 * si, w1 * si, w0 * si);
            sum128 = _mm_add_ps(sum128, s_vec);
            i += 1;
        }

        let mut out = [0.0; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum128);
        out
    }
}

/// Dual-frame version of the BF16 interleaved dot product.
///
/// Reuses each weight load for both frames (f0 and f1), halving memory
/// bandwidth for the weight matrix.
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
pub unsafe fn dot_product_4x_interleaved_dual_frame_avx512_bf16(
    weights: &[[u16; 4]],
    state_f0: &[u16],
    state_f1: &[u16],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut i = 0;

    unsafe {
        let perm_idx = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

        let mut sum0_a0 = _mm512_setzero_ps();
        let mut sum0_a1 = _mm512_setzero_ps();
        let mut sum0_a2 = _mm512_setzero_ps();
        let mut sum0_a3 = _mm512_setzero_ps();
        let mut sum0_b0 = _mm512_setzero_ps();
        let mut sum0_b1 = _mm512_setzero_ps();
        let mut sum0_b2 = _mm512_setzero_ps();
        let mut sum0_b3 = _mm512_setzero_ps();

        let mut sum1_a0 = _mm512_setzero_ps();
        let mut sum1_a1 = _mm512_setzero_ps();
        let mut sum1_a2 = _mm512_setzero_ps();
        let mut sum1_a3 = _mm512_setzero_ps();
        let mut sum1_b0 = _mm512_setzero_ps();
        let mut sum1_b1 = _mm512_setzero_ps();
        let mut sum1_b2 = _mm512_setzero_ps();
        let mut sum1_b3 = _mm512_setzero_ps();

        while i + 32 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state_f0.as_ptr().add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(state_f1.as_ptr().add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 32) as *const i8);

            let w_a = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            let s0_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i))),
            );
            let s1_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w_a, s0_a, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w_a, s1_a, sum1_a0);

            let w_b = bf16x16_to_f32x16(weights.as_ptr().add(i + 4) as *const u16);
            let s0_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 4))),
            );
            let s1_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 4))),
            );
            sum0_a1 = _mm512_fmadd_ps(w_b, s0_b, sum0_a1);
            sum1_a1 = _mm512_fmadd_ps(w_b, s1_b, sum1_a1);

            let w_c = bf16x16_to_f32x16(weights.as_ptr().add(i + 8) as *const u16);
            let s0_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 8))),
            );
            let s1_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 8))),
            );
            sum0_a2 = _mm512_fmadd_ps(w_c, s0_c, sum0_a2);
            sum1_a2 = _mm512_fmadd_ps(w_c, s1_c, sum1_a2);

            let w_d = bf16x16_to_f32x16(weights.as_ptr().add(i + 12) as *const u16);
            let s0_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 12))),
            );
            let s1_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 12))),
            );
            sum0_a3 = _mm512_fmadd_ps(w_d, s0_d, sum0_a3);
            sum1_a3 = _mm512_fmadd_ps(w_d, s1_d, sum1_a3);

            let w_e = bf16x16_to_f32x16(weights.as_ptr().add(i + 16) as *const u16);
            let s0_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 16))),
            );
            let s1_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 16))),
            );
            sum0_b0 = _mm512_fmadd_ps(w_e, s0_e, sum0_b0);
            sum1_b0 = _mm512_fmadd_ps(w_e, s1_e, sum1_b0);

            let w_f = bf16x16_to_f32x16(weights.as_ptr().add(i + 20) as *const u16);
            let s0_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 20))),
            );
            let s1_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 20))),
            );
            sum0_b1 = _mm512_fmadd_ps(w_f, s0_f, sum0_b1);
            sum1_b1 = _mm512_fmadd_ps(w_f, s1_f, sum1_b1);

            let w_g = bf16x16_to_f32x16(weights.as_ptr().add(i + 24) as *const u16);
            let s0_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 24))),
            );
            let s1_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 24))),
            );
            sum0_b2 = _mm512_fmadd_ps(w_g, s0_g, sum0_b2);
            sum1_b2 = _mm512_fmadd_ps(w_g, s1_g, sum1_b2);

            let w_h = bf16x16_to_f32x16(weights.as_ptr().add(i + 28) as *const u16);
            let s0_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 28))),
            );
            let s1_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 28))),
            );
            sum0_b3 = _mm512_fmadd_ps(w_h, s0_h, sum0_b3);
            sum1_b3 = _mm512_fmadd_ps(w_h, s1_h, sum1_b3);

            i += 32;
        }

        while i + 16 <= len {
            let w_a = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            let s0_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i))),
            );
            let s1_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w_a, s0_a, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w_a, s1_a, sum1_a0);

            let w_b = bf16x16_to_f32x16(weights.as_ptr().add(i + 4) as *const u16);
            let s0_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 4))),
            );
            let s1_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 4))),
            );
            sum0_a1 = _mm512_fmadd_ps(w_b, s0_b, sum0_a1);
            sum1_a1 = _mm512_fmadd_ps(w_b, s1_b, sum1_a1);

            let w_c = bf16x16_to_f32x16(weights.as_ptr().add(i + 8) as *const u16);
            let s0_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 8))),
            );
            let s1_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 8))),
            );
            sum0_a2 = _mm512_fmadd_ps(w_c, s0_c, sum0_a2);
            sum1_a2 = _mm512_fmadd_ps(w_c, s1_c, sum1_a2);

            let w_d = bf16x16_to_f32x16(weights.as_ptr().add(i + 12) as *const u16);
            let s0_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i + 12))),
            );
            let s1_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i + 12))),
            );
            sum0_a3 = _mm512_fmadd_ps(w_d, s0_d, sum0_a3);
            sum1_a3 = _mm512_fmadd_ps(w_d, s1_d, sum1_a3);

            i += 16;
        }

        while i + 4 <= len {
            let w = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            let s0 = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f0.as_ptr().add(i))),
            );
            let s1 = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w, s0, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w, s1, sum1_a0);
            i += 4;
        }

        let s0_ab = _mm512_add_ps(
            _mm512_add_ps(sum0_a0, sum0_a1),
            _mm512_add_ps(sum0_a2, sum0_a3),
        );
        let s0_all = _mm512_add_ps(
            s0_ab,
            _mm512_add_ps(
                _mm512_add_ps(sum0_b0, sum0_b1),
                _mm512_add_ps(sum0_b2, sum0_b3),
            ),
        );
        let s1_ab = _mm512_add_ps(
            _mm512_add_ps(sum1_a0, sum1_a1),
            _mm512_add_ps(sum1_a2, sum1_a3),
        );
        let s1_all = _mm512_add_ps(
            s1_ab,
            _mm512_add_ps(
                _mm512_add_ps(sum1_b0, sum1_b1),
                _mm512_add_ps(sum1_b2, sum1_b3),
            ),
        );

        let lo0 = _mm512_extractf32x4_ps(s0_all, 0);
        let hi00 = _mm512_extractf32x4_ps(s0_all, 1);
        let hi01 = _mm512_extractf32x4_ps(s0_all, 2);
        let hi02 = _mm512_extractf32x4_ps(s0_all, 3);
        let mut sum128_f0 = _mm_add_ps(_mm_add_ps(lo0, hi00), _mm_add_ps(hi01, hi02));

        let lo1 = _mm512_extractf32x4_ps(s1_all, 0);
        let hi10 = _mm512_extractf32x4_ps(s1_all, 1);
        let hi11 = _mm512_extractf32x4_ps(s1_all, 2);
        let hi12 = _mm512_extractf32x4_ps(s1_all, 3);
        let mut sum128_f1 = _mm_add_ps(_mm_add_ps(lo1, hi10), _mm_add_ps(hi11, hi12));

        while i < len {
            let s0 = f32::from_bits((*state_f0.get_unchecked(i) as u32) << 16);
            let s1 = f32::from_bits((*state_f1.get_unchecked(i) as u32) << 16);
            let w_ptr = weights.as_ptr().add(i) as *const u16;
            let w0 = f32::from_bits((*w_ptr as u32) << 16);
            let w1 = f32::from_bits((*w_ptr.add(1) as u32) << 16);
            let w2 = f32::from_bits((*w_ptr.add(2) as u32) << 16);
            let w3 = f32::from_bits((*w_ptr.add(3) as u32) << 16);
            let s0_vec = _mm_set_ps(w3 * s0, w2 * s0, w1 * s0, w0 * s0);
            let s1_vec = _mm_set_ps(w3 * s1, w2 * s1, w1 * s1, w0 * s1);
            sum128_f0 = _mm_add_ps(sum128_f0, s0_vec);
            sum128_f1 = _mm_add_ps(sum128_f1, s1_vec);
            i += 1;
        }

        let mut out_f0 = [0.0; 4];
        let mut out_f1 = [0.0; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), sum128_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), sum128_f1);
        (out_f0, out_f1)
    }
}
