// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::gemm::{gemv_overwrite_bf16_fallback, gemv_overwrite_fallback};

/// Scalar fallback for the 4 LSTM gates.
/// Each gate controls a different aspect: input, forget, content, and output.
/// Used directly by `avx512.rs` and `avx2.rs` for non-vectorized operations.
#[allow(clippy::too_many_arguments)]
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn gemv_4gate_fallback(
    in_frame: &[f32],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    unsafe {
        // Processes each of the 4 gates separately.
        gemv_overwrite_fallback(
            in_frame,
            w0,
            &bias[0..out_len],
            &mut out[0..out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w1,
            &bias[out_len..2 * out_len],
            &mut out[out_len..2 * out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w2,
            &bias[2 * out_len..3 * out_len],
            &mut out[2 * out_len..3 * out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w3,
            &bias[3 * out_len..4 * out_len],
            &mut out[3 * out_len..4 * out_len],
            do_bias,
        );
    }
}

/// BF16 version for the 4 LSTM gates.
#[allow(clippy::too_many_arguments)]
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn gemv_4gate_bf16_fallback(
    in_frame: &[u16],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
    unsafe {
        gemv_overwrite_bf16_fallback(
            in_frame,
            w0,
            &bias[0..out_len],
            &mut out[0..out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w1,
            &bias[out_len..2 * out_len],
            &mut out[out_len..2 * out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w2,
            &bias[2 * out_len..3 * out_len],
            &mut out[2 * out_len..3 * out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w3,
            &bias[3 * out_len..4 * out_len],
            &mut out[3 * out_len..4 * out_len],
            do_bias,
        );
    }
}
