// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use core::arch::x86_64::{_mm_loadu_ps, _mm_storeu_ps, _mm256_loadu_ps, _mm256_storeu_ps};

/// Stores 4 accumulator values back to the output buffer with
/// bounds‑check elision for the hot path. Extracted verbatim from single‑frame
/// convolution kernel.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path (`out_n` multiple of 4 or
/// `out_c + 3 < out_n`), `out_c + 3` must be < `out.len()`.
#[inline(always)]
pub(crate) unsafe fn store_4_accums(out: &mut [f32], out_c: usize, r: [f32; 4], out_n: usize) {
    if out_n.is_multiple_of(4) || out_c + 3 < out_n {
        let v = unsafe { _mm_loadu_ps(r.as_ptr()) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(out_c), v) };
    } else {
        unsafe { *out.get_unchecked_mut(out_c) = r[0] };
        if out_c + 1 < out_n {
            unsafe { *out.get_unchecked_mut(out_c + 1) = r[1] };
        }
        if out_c + 2 < out_n {
            unsafe { *out.get_unchecked_mut(out_c + 2) = r[2] };
        }
        if out_c + 3 < out_n {
            unsafe { *out.get_unchecked_mut(out_c + 3) = r[3] };
        }
    }
}

/// Stores 8 accumulator values back to the output buffer.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path, `out_c + 7 < out_n`.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) unsafe fn store_8_accums(out: &mut [f32], out_c: usize, r: [f32; 8], out_n: usize) {
    if out_n.is_multiple_of(8) || out_c + 7 < out_n {
        let v = unsafe { _mm256_loadu_ps(r.as_ptr()) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(out_c), v) };
    } else {
        unsafe { *out.get_unchecked_mut(out_c) = r[0] };
        for i in 1..8 {
            if out_c + i < out_n {
                unsafe { *out.get_unchecked_mut(out_c + i) = r[i] };
            }
        }
    }
}

/// Stores 16 accumulator values back to the output buffer.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path, `out_c + 15 < out_n`.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) unsafe fn store_16_accums(out: &mut [f32], out_c: usize, r: [f32; 16], out_n: usize) {
    if out_n.is_multiple_of(16) || out_c + 15 < out_n {
        let v0 = unsafe { _mm256_loadu_ps(r.as_ptr()) };
        let v1 = unsafe { _mm256_loadu_ps(r.as_ptr().add(8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(out_c), v0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(out_c + 8), v1) };
    } else {
        unsafe { *out.get_unchecked_mut(out_c) = r[0] };
        for i in 1..16 {
            if out_c + i < out_n {
                unsafe { *out.get_unchecked_mut(out_c + i) = r[i] };
            }
        }
    }
}
