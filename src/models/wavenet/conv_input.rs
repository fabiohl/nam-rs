// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// Loads 4 accumulator values from the output buffer with fallback for non‑multiple‑of‑4
/// OUT sizes. Extracted verbatim from single‑frame convolution kernel.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path (`out_n` multiple of 4 or
/// `out_c + 3 < out_n`), `out_c + 3` must be < `out.len()`.
#[inline(always)]
pub(crate) unsafe fn load_4_accums(out: &[f32], out_c: usize, out_n: usize) -> [f32; 4] {
    let r0 = unsafe { *out.get_unchecked(out_c) };
    if out_n.is_multiple_of(4) || out_c + 3 < out_n {
        let r1 = unsafe { *out.get_unchecked(out_c + 1) };
        let r2 = unsafe { *out.get_unchecked(out_c + 2) };
        let r3 = unsafe { *out.get_unchecked(out_c + 3) };
        [r0, r1, r2, r3]
    } else {
        let r1 = if out_c + 1 < out_n {
            unsafe { *out.get_unchecked(out_c + 1) }
        } else {
            0.0
        };
        let r2 = if out_c + 2 < out_n {
            unsafe { *out.get_unchecked(out_c + 2) }
        } else {
            0.0
        };
        let r3 = if out_c + 3 < out_n {
            unsafe { *out.get_unchecked(out_c + 3) }
        } else {
            0.0
        };
        [r0, r1, r2, r3]
    }
}

/// Stores 4 accumulator values back to the output buffer with
/// bounds‑check elision for the hot path. Extracted verbatim from single‑frame
/// convolution kernel.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path (`out_n` multiple of 4 or
/// `out_c + 3 < out_n`), `out_c + 3` must be < `out.len()`.
#[inline(always)]
pub(crate) unsafe fn store_4_accums(out: &mut [f32], out_c: usize, r: [f32; 4], out_n: usize) {
    unsafe { *out.get_unchecked_mut(out_c) = r[0] };
    if out_n.is_multiple_of(4) || out_c + 3 < out_n {
        unsafe {
            *out.get_unchecked_mut(out_c + 1) = r[1];
            *out.get_unchecked_mut(out_c + 2) = r[2];
            *out.get_unchecked_mut(out_c + 3) = r[3];
        }
    } else {
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

/// Loads 8 accumulator values from the output buffer.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path, `out_c + 7 < out_n`.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) unsafe fn load_8_accums(out: &[f32], out_c: usize, out_n: usize) -> [f32; 8] {
    let r0 = unsafe { *out.get_unchecked(out_c) };
    if out_n.is_multiple_of(8) || out_c + 7 < out_n {
        [
            r0,
            unsafe { *out.get_unchecked(out_c + 1) },
            unsafe { *out.get_unchecked(out_c + 2) },
            unsafe { *out.get_unchecked(out_c + 3) },
            unsafe { *out.get_unchecked(out_c + 4) },
            unsafe { *out.get_unchecked(out_c + 5) },
            unsafe { *out.get_unchecked(out_c + 6) },
            unsafe { *out.get_unchecked(out_c + 7) },
        ]
    } else {
        let mut r = [0.0f32; 8];
        r[0] = r0;
        for i in 1..8 {
            if out_c + i < out_n {
                unsafe { r[i] = *out.get_unchecked(out_c + i) };
            }
        }
        r
    }
}

/// Stores 8 accumulator values back to the output buffer.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path, `out_c + 7 < out_n`.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) unsafe fn store_8_accums(out: &mut [f32], out_c: usize, r: [f32; 8], out_n: usize) {
    unsafe { *out.get_unchecked_mut(out_c) = r[0] };
    if out_n.is_multiple_of(8) || out_c + 7 < out_n {
        unsafe {
            *out.get_unchecked_mut(out_c + 1) = r[1];
            *out.get_unchecked_mut(out_c + 2) = r[2];
            *out.get_unchecked_mut(out_c + 3) = r[3];
            *out.get_unchecked_mut(out_c + 4) = r[4];
            *out.get_unchecked_mut(out_c + 5) = r[5];
            *out.get_unchecked_mut(out_c + 6) = r[6];
            *out.get_unchecked_mut(out_c + 7) = r[7];
        }
    } else {
        for i in 1..8 {
            if out_c + i < out_n {
                unsafe { *out.get_unchecked_mut(out_c + i) = r[i] };
            }
        }
    }
}

/// Loads 16 accumulator values from the output buffer.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path, `out_c + 15 < out_n`.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) unsafe fn load_16_accums(out: &[f32], out_c: usize, out_n: usize) -> [f32; 16] {
    let r0 = unsafe { *out.get_unchecked(out_c) };
    if out_n.is_multiple_of(16) || out_c + 15 < out_n {
        let mut r = [0.0f32; 16];
        r[0] = r0;
        for i in 1..16 {
            unsafe { r[i] = *out.get_unchecked(out_c + i) };
        }
        r
    } else {
        let mut r = [0.0f32; 16];
        r[0] = r0;
        for i in 1..16 {
            if out_c + i < out_n {
                unsafe { r[i] = *out.get_unchecked(out_c + i) };
            }
        }
        r
    }
}

/// Stores 16 accumulator values back to the output buffer.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path, `out_c + 15 < out_n`.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub(crate) unsafe fn store_16_accums(out: &mut [f32], out_c: usize, r: [f32; 16], out_n: usize) {
    unsafe { *out.get_unchecked_mut(out_c) = r[0] };
    if out_n.is_multiple_of(16) || out_c + 15 < out_n {
        for i in 1..16 {
            unsafe { *out.get_unchecked_mut(out_c + i) = r[i] };
        }
    } else {
        for i in 1..16 {
            if out_c + i < out_n {
                unsafe { *out.get_unchecked_mut(out_c + i) = r[i] };
            }
        }
    }
}
