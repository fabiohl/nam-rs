// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::SimdMath;

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
