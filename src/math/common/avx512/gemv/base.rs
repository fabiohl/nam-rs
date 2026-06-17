// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

macro_rules! impl_avx512_gemv {
    () => {
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
            crate::math::gemm::dot::dot_product_avx512(a, b)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
            dot_product_bf16_fallback(a, b)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
            crate::math::gemm::dot_4x::dot_product_4x_interleaved_avx512(weights, state)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
            dot_product_4x_interleaved_bf16_fallback(weights, state)
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_4x_interleaved_dual_frame(
            weights: &[[u16; 4]],
            state_f0: &[f32],
            state_f1: &[f32],
        ) -> ([f32; 4], [f32; 4]) {
            crate::math::gemm::dot_4x::dot_product_4x_interleaved_dual_frame_avx512(
                weights, state_f0, state_f1,
            )
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
            weights: &[[u16; 4]],
            state_f0: &[u16],
            state_f1: &[u16],
        ) -> ([f32; 4], [f32; 4]) {
            dot_product_4x_interleaved_dual_frame_bf16_fallback(weights, state_f0, state_f1)
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
            dot_product_bf16_4x_fallback(w0, w1, w2, w3, in_frame)
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
            if out_frame.len() == 16 {
                crate::math::gemm::gemv::fused_add_gemv_avx512_small(
                    in_frame, weights, bias, out_frame, do_bias,
                )
            } else {
                fused_add_gemv_fallback(in_frame, weights, bias, out_frame, do_bias)
            }
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
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::gemm::gemm_batch::fused_add_gemm_batch_avx512(
                    in_frames, weights, bias, out_frames, num_frames, do_bias,
                )
            }
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
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::gemm::gemm_batch::fused_gemm_residual_batch_avx512(
                    in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
                )
            }
        }
        #[inline(always)]
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe fn fused_gemm_residual_batch_f32(
            in_frames: &[f32],
            weights: &[f32],
            bias: &[f32],
            residual: &[f32],
            out_frames: &mut [f32],
            num_frames: usize,
            do_bias: bool,
        ) {
            unsafe {
                crate::math::gemm::gemm_batch::fused_gemm_residual_batch_f32_avx512(
                    in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
                )
            }
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
            if out_frame.len() == 16 {
                crate::math::gemm::gemv::gemv_overwrite_avx512_small(
                    in_frame, weights, bias, out_frame, do_bias,
                )
            } else {
                gemv_overwrite_fallback(in_frame, weights, bias, out_frame, do_bias)
            }
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
            gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias)
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
            let ih = in_frame.len();
            let stride = ih * hidden_size;
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::gemm::gemv_4gate::gemv_4gate_avx512(
                    in_frame,
                    &weights[0..stride],
                    &weights[stride..2 * stride],
                    &weights[2 * stride..3 * stride],
                    &weights[3 * stride..4 * stride],
                    bias,
                    out_gates,
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
            let ih = in_frame.len();
            let stride = ih * hidden_size;
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::gemm::gemv_4gate::gemv_4gate_bf16_avx512(
                    in_frame,
                    &weights[0..stride],
                    &weights[stride..2 * stride],
                    &weights[2 * stride..3 * stride],
                    &weights[3 * stride..4 * stride],
                    bias,
                    out_gates,
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
            // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
            unsafe {
                crate::math::gemm::gemv::gemv_overwrite_batch_avx512(
                    in_frames, weights, bias, out_frames, num_frames, do_bias,
                )
            }
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
            let in_len = in_frames.len() / num_frames;
            let out_len = out_frames.len() / num_frames;
            for i in 0..num_frames {
                gemv_overwrite_bf16_fallback(
                    &in_frames[i * in_len..(i + 1) * in_len],
                    weights,
                    bias,
                    &mut out_frames[i * out_len..(i + 1) * out_len],
                    do_bias,
                );
            }
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
