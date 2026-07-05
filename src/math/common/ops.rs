// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Basic mathematical operations and low-level SIMD utilities.

use crate::math::common::half::f32_to_f16_bits;
use core::arch::x86_64::*;

/// Converts F32 to BF16 bits (simple truncation).
#[inline(always)]
pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// Quantizes an f32 weight to u16 (BF16 or F16 bits), based on `is_bf16`.
#[inline(always)]
#[deprecated(since = "2.0.0", note = "Quantization removed (SQ3); use f32 native storage")]
pub fn quantize_weight(f: f32, is_bf16: bool) -> u16 {
    if is_bf16 {
        f32_to_bf16(f)
    } else {
        f32_to_f16_bits(f)
    }
}

/// Converts a vector of f32 numbers (normal) to bf16 (compact) via AVX-512.
/// The bf16 format takes up half the space but maintains the range of f32 numbers,
/// making it ideal for fast AI models.
///
/// # Safety
/// `src` and `dest` must be valid slices. `dest` must have at least `src.len()` elements.
#[target_feature(enable = "avx512f,avx512vl")]
// SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
pub unsafe fn f32_to_bf16_avx512(src: &[f32], dest: &mut [u16]) {
    let n = core::cmp::min(src.len(), dest.len());
    let mut i = 0;
    // Process 16 conversions at once.
    while i + 16 <= n {
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            let v = _mm512_loadu_ps(src.as_ptr().add(i)); // Load 16 f32 numbers.
            let v_i = _mm512_castps_si512(v); // Treat as integers for bit manipulation.
            let v_shifted = _mm512_srli_epi32(v_i, 16); // Discard the least important part (extra precision).
            let packed = _mm512_cvtepi32_epi16(v_shifted); // Compact to 16 bits each.
            _mm256_storeu_si256(dest.as_mut_ptr().add(i) as *mut __m256i, packed); // Save 16 bf16 numbers.
        }
        i += 16;
    }
    // Convert the remainder manually.
    while i < n {
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            *dest.get_unchecked_mut(i) = ((*src.get_unchecked(i)).to_bits() >> 16) as u16;
        }
        i += 1;
    }
}

/// Adaptive prefetch based on dilation (Causal Conv1D).
///
/// # Safety
/// The `ptr` pointer must be valid or within the safe margin of the buffer.
#[inline(always)]
// SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
pub unsafe fn adaptive_prefetch_f32(ptr: *const f32, dilation: usize) {
    if dilation <= 8 {
        // Hardware prefetcher dominates.
    } else if dilation <= 64 {
        // Fetch to L1 (imminent reuse).
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(ptr as *const i8);
        }
    } else {
        // Massive dilations: fetch to L2 to spare L1 from aggressive eviction.
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            _mm_prefetch::<_MM_HINT_T1>(ptr as *const i8);
        }
    }
}

/// Adaptive 2-stage prefetch for extreme dilations (Causal Conv1D).
///
/// # Safety
/// The pointers must be valid or within the safe margin of the buffer.
#[inline(always)]
// SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
pub unsafe fn adaptive_prefetch_2stage_f32(
    ptr_next: *const f32,
    ptr_next_next: *const f32,
    dilation: usize,
) {
    if dilation >= 128 {
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            // Fetch the next tap to L1 (immediate use in the next k)
            _mm_prefetch::<_MM_HINT_T0>(ptr_next as *const i8);
            // Fetch the subsequent tap to L2 (prepares for k+2)
            _mm_prefetch::<_MM_HINT_T1>(ptr_next_next as *const i8);
        }
    } else {
        // Fallback to simple prefetch for smaller dilations
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            adaptive_prefetch_f32(ptr_next, dilation);
        }
    }
}

/// Unified signature for prefetch strategies.
pub type PrefetchFn =
    unsafe fn(base_ptr: *const f32, step: usize, k: usize, k_limit: usize, dilation: usize);

/// Safe speculative prefetch to L1 cache (T0 hint), avoiding pointer arithmetic UB.
///
/// # Safety
/// The caller must ensure that the base pointer is valid.
#[inline(always)]
pub unsafe fn prefetch_t0<T>(base: *const T, offset: usize) {
    // SAFETY: prefetch does not dereference the pointer, so using wrapping_add is UB-free even if the address goes out of bounds.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
            base.wrapping_add(offset).cast(),
        );
    }
}

/// Safe speculative prefetch to L2 cache (T1 hint), avoiding pointer arithmetic UB.
///
/// # Safety
/// The caller must ensure that the base pointer is valid.
#[inline(always)]
pub unsafe fn prefetch_t1<T>(base: *const T, offset: usize) {
    // SAFETY: prefetch does not dereference the pointer, so using wrapping_add is UB-free even if the address goes out of bounds.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T1 }>(
            base.wrapping_add(offset).cast(),
        );
    }
}

/// Simple prefetch strategy for small/medium dilations.
///
/// # Safety
/// The base pointer must be valid.
#[inline(always)]
pub unsafe fn prefetch_strategy_simple(
    base_ptr: *const f32,
    _step: usize,
    _k: usize,
    _k_limit: usize,
    _dilation: usize,
) {
    // SAFETY: Speculative prefetch using wrapping_add is safe from out-of-bounds UB.
    unsafe {
        prefetch_t0(base_ptr, 16);
    }
}

/// 2-stage prefetch strategy for extreme dilations.
///
/// # Safety
/// The base pointer and computed offsets must be valid.
#[inline(always)]
pub unsafe fn prefetch_strategy_2stage(
    base_ptr: *const f32,
    step: usize,
    k: usize,
    k_limit: usize,
    _dilation: usize,
) {
    if k + 1 < k_limit {
        // SAFETY: Speculative prefetch using wrapping_add is safe from out-of-bounds UB.
        unsafe {
            prefetch_t0(base_ptr, step);

            if k + 2 < k_limit {
                prefetch_t1(base_ptr, 2 * step);
            }
        }
    }
}

/// Enables DAZ (Denormals-Are-Zero) and FTZ (Flush-To-Zero) in the MXCSR register.
///
/// # Safety
/// SSE2 is implicit on x86-64.
pub unsafe fn set_daz_ftz() {
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    unsafe {
        let mut mxcsr: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut mxcsr);
        mxcsr |= 0x8040;
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &mxcsr);
    }
}
