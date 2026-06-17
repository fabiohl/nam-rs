// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::SimdMath;

/// Initializes a block of 4 accumulator registers from bias and mixin vectors.
///
/// Unifies the 4-case initialization pattern (Some(mixin)+bias, Some(mixin) only,
/// bias only, zeros) shared between single-frame and dual-frame convolution kernels.
///
/// # Safety
/// `out_offset + 4` must not exceed the lengths of `bias` and `mixin` (when provided).
#[cfg_attr(not(test), allow(dead_code))]
#[inline(always)]
pub(crate) unsafe fn init_accum_with_bias_mixin<M: SimdMath>(
    acc: &mut [f32; 4],
    bias: &[f32],
    mixin: Option<&[f32]>,
    out_offset: usize,
    do_bias: bool,
) {
    if let Some(m) = mixin {
        if do_bias {
            acc.copy_from_slice(&bias[out_offset..out_offset + 4]);
            unsafe {
                M::accumulate_head(acc, &m[out_offset..out_offset + 4]);
            }
        } else {
            acc.copy_from_slice(&m[out_offset..out_offset + 4]);
        }
    } else if do_bias {
        acc.copy_from_slice(&bias[out_offset..out_offset + 4]);
    } else {
        acc.fill(0.0);
    }
}

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

/// Data Bridge (ConvInput):
/// This trait is a bridge that allows NAM-rs to use exactly the same code
/// for two number types: regular floats (f32) and compact numbers (u16/BF16).
/// This avoids duplicating complex logic and facilitates maintenance.
pub(crate) trait ConvInput: Copy + Default {
    /// 4x version: Computes 4 channels at once.
    unsafe fn dot_product_4x_interleaved<M: SimdMath>(
        weights: &[[u16; 4]],
        state: &[Self],
    ) -> [f32; 4];

    /// Dual Frame version: Computes 4 channels of TWO frames simultaneously.
    unsafe fn dot_product_4x_interleaved_dual_frame<M: SimdMath>(
        weights: &[[u16; 4]],
        state_f0: &[Self],
        state_f1: &[Self],
    ) -> ([f32; 4], [f32; 4]);

    /// Pointer Adjustment: Ensures the memory address follows the correct format.
    fn cast_ptr(ptr: *const Self) -> *const f32;
}

// 1. Full Precision Mode (f32):
// Used on computers that prioritize absolute sound fidelity.
impl ConvInput for f32 {
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved<M: SimdMath>(
        weights: &[[u16; 4]],
        state: &[Self],
    ) -> [f32; 4] {
        unsafe { M::dot_product_4x_interleaved(weights, state) }
    }
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame<M: SimdMath>(
        weights: &[[u16; 4]],
        state_f0: &[Self],
        state_f1: &[Self],
    ) -> ([f32; 4], [f32; 4]) {
        unsafe { M::dot_product_4x_interleaved_dual_frame(weights, state_f0, state_f1) }
    }
    #[inline(always)]
    fn cast_ptr(ptr: *const Self) -> *const f32 {
        ptr
    }
}

// 2. 'Turbo' Mode (u16/BF16):
// Used to gain speed. The BF16 format cuts the data size in half,
// allowing the processor to compute much faster with a quality loss
// imperceptible to the human ear.
impl ConvInput for u16 {
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved<M: SimdMath>(
        weights: &[[u16; 4]],
        state: &[Self],
    ) -> [f32; 4] {
        unsafe { M::dot_product_4x_interleaved_bf16(weights, state) }
    }
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame<M: SimdMath>(
        weights: &[[u16; 4]],
        state_f0: &[Self],
        state_f1: &[Self],
    ) -> ([f32; 4], [f32; 4]) {
        unsafe { M::dot_product_4x_interleaved_dual_frame_bf16(weights, state_f0, state_f1) }
    }
    #[inline(always)]
    fn cast_ptr(ptr: *const Self) -> *const f32 {
        ptr as *const f32
    }
}

/// F32-native 4-lane interleaved dot product (AVX2/FMA or AVX-512 kernel).
///
/// Computes Conv1D output directly from full-precision f32 weights.
///
/// Dispatches to `dot_product_4x_f32_avx512` when AVX-512F is detected
/// at runtime; otherwise falls back to `dot_product_4x_f32_avx2`.
///
/// # Bit‑exactness guarantee
/// Both the scalar reference (`mul_add`) and the SIMD kernels (`_mm_fmadd_ps`
/// / `_mm512_fmadd_ps`) use the same FMA3 fused multiply‑add → bit‑identical
/// result on any x86‑64‑v3 CPU.
#[inline(always)]
pub(crate) fn dot_product_4x(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    if is_x86_feature_detected!("avx512f") {
        unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(weights, state) }
    } else {
        unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx2(weights, state) }
    }
}

/// F32-native 4-lane interleaved dual-frame dot product
/// (AVX2/FMA or AVX-512 kernel).
///
/// Processes two independent state vectors against the same weight slice
/// simultaneously. Dispatches to `dot_product_4x_f32_dual_avx512` when
/// AVX-512F is detected at runtime; otherwise falls back to
/// `dot_product_4x_f32_dual_avx2`.
///
/// # Bit‑exactness guarantee
/// Both the scalar reference (`mul_add`) and the SIMD kernels (`_mm_fmadd_ps`
/// / `_mm512_fmadd_ps`) use the same FMA3 fused multiply‑add → bit‑identical
/// result on any x86‑64‑v3 CPU.
#[inline(always)]
pub(crate) fn dot_product_4x_dual(
    weights: &[[f32; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    if is_x86_feature_detected!("avx512f") {
        unsafe {
            crate::math::gemm::dot_4x::dot_product_4x_f32_dual_avx512(weights, state_f0, state_f1)
        }
    } else {
        unsafe {
            crate::math::gemm::dot_4x::dot_product_4x_f32_dual_avx2(weights, state_f0, state_f1)
        }
    }
}
