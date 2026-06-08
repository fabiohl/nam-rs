// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

macro_rules! impl_avx512vnni_gemv {
    () => {
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
            Avx512Math::dot_product(a, b)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe { crate::math::gemm::dot::dot_product_bf16_avx512(a, b) }
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
            Avx512Math::dot_product_4x_interleaved(weights, state)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
            Avx512Math::dot_product_4x_interleaved_bf16(weights, state)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_4x_interleaved_dual_frame(
            weights: &[[u16; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 4], [f32; 4]) {
            Avx512Math::dot_product_4x_interleaved_dual_frame(weights, state_f0, state_f1)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
            weights: &[[u16; 4]],
            state_f0: &[u16],
            state_f1: &[u16],
        ) -> ([f32; 4], [f32; 4]) {
            Avx512Math::dot_product_4x_interleaved_dual_frame_bf16(weights, state_f0, state_f1)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_bf16_4x(
            w0: &[u16],
            w1: &[u16],
            w2: &[u16],
            w3: &[u16],
            in_frame: &[u16],
        ) -> [f32; 4] {
            let mut out = [0.0; 4];
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::gemm::gemv_4gate::gemv_4gate_bf16_avx512(
                    in_frame, w0, w1, w2, w3, &[0.0; 4], &mut out, false,
                )
            }
            out
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn fused_add_gemv(
            in_frame: &[f32],
            weights: &[u16],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            Avx512Math::fused_add_gemv(in_frame, weights, bias, out_frame, do_bias)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn fused_add_gemm_batch(
            in_frames: &[f32],
            weights: &[u16],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            Avx512Math::fused_add_gemm_batch(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn fused_gemm_residual_batch(
            in_frames: &[f32],
            weights: &[u16],
            bias: &[f32],
            residual: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            Avx512Math::fused_gemm_residual_batch(
                in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn gemv_overwrite(
            in_frame: &[f32],
            weights: &[u16],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            Avx512Math::gemv_overwrite(in_frame, weights, bias, out_frame, do_bias)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn gemv_overwrite_bf16(
            in_frame: &[u16],
            weights: &[u16],
            bias: &[f32],
            out_frame: &mut [f32],
            do_bias: bool,
        ) {
            Avx512Math::gemv_overwrite_bf16(in_frame, weights, bias, out_frame, do_bias)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn gemv_overwrite_4gate(
            in_frame: &[f32],
            weights: &[u16],
            bias: &[f32],
            out_gates: &mut [f32],
            hidden_size: usize,
            do_bias: bool,
        ) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                Avx512Math::gemv_overwrite_4gate(
                    in_frame,
                    weights,
                    bias,
                    out_gates,
                    hidden_size,
                    do_bias,
                )
            }
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn gemv_overwrite_bf16_4gate(
            in_frame: &[u16],
            weights: &[u16],
            bias: &[f32],
            out_gates: &mut [f32],
            hidden_size: usize,
            do_bias: bool,
        ) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                Avx512Math::gemv_overwrite_bf16_4gate(
                    in_frame,
                    weights,
                    bias,
                    out_gates,
                    hidden_size,
                    do_bias,
                )
            }
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn gemv_overwrite_batch(
            in_frames: &[f32],
            weights: &[u16],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            Avx512Math::gemv_overwrite_batch(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn gemv_overwrite_batch_bf16(
            in_frames: &[u16],
            weights: &[u16],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            Avx512Math::gemv_overwrite_batch_bf16(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn gemv_with_bias_f32(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
        ) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::gemm::gemv::gemv_with_bias_f32_avx512(
                    in_frames, weights, bias, out_frames, num_frames,
                )
            }
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn gemv_no_bias_f32(
            in_frames: &[f32],
            weights: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
        ) {
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::gemm::gemv::gemv_no_bias_f32_avx512(
                    in_frames, weights, out_frames, num_frames,
                )
            }
        }
    };
}
