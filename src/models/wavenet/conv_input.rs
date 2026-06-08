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
