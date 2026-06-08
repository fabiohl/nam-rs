// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use core::arch::x86_64::*;

/// Helper: converts 4 adjacent BF16 values into a single __m128 of f32.
///
/// BF16 → f32 is done via: cast u16→u32, shift left 16 bits,
/// reinterpret as f32. The low 4 of the resulting __m256i are extracted
/// into __m128 for subsequent `_mm512_castps128_ps512` + `permutexvar`.
#[inline(always)]
pub(super) unsafe fn bf16x4_to_f32x4(ptr: *const u16) -> __m128 {
    let v_u16 = _mm_loadl_epi64(ptr as *const __m128i);
    let v_u32 = _mm256_cvtepu16_epi32(v_u16);
    let v_f32 = _mm256_castsi256_ps(_mm256_slli_epi32(v_u32, 16));
    _mm256_castps256_ps128(v_f32)
}

/// Helper: converts 16 adjacent BF16 values into a __m512 of f32.
#[inline(always)]
pub(super) unsafe fn bf16x16_to_f32x16(ptr: *const u16) -> __m512 {
    let v_u16 = _mm256_loadu_si256(ptr as *const __m256i);
    let v_u32 = _mm512_cvtepu16_epi32(v_u16);
    _mm512_castsi512_ps(_mm512_slli_epi32(v_u32, 16))
}
