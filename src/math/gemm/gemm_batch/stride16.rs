// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Pad-to-16 stride variants of CH=12 fused residual GEMM kernels.
//!
//! These kernels read/write buffers with 16-float stride (64 B = 1 cache line)
//! but only load/store 12 valid channels. Padding lanes 12-15 are preserved as
//! zero and never participate in FMAs that affect lanes 0-11.

/// Stride-16 variant of the CH=12 fused residual GEMM batch kernel.
///
/// Reads input and residual at 16-float stride, writes output at 16-float stride,
/// but only processes 12 logical channels (0-11). Padding lanes 12-15 are zero-preserving.
pub unsafe fn fused_gemm_residual_batch_f32_12x12_padded(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    debug_assert_eq!(in_frames.len(), num_frames * 16);
    debug_assert_eq!(out_frames.len(), num_frames * 16);
    debug_assert_eq!(residual.len(), num_frames * 16);
    debug_assert!(weights.len() >= 144);
    if do_bias {
        debug_assert!(bias.len() >= 12);
    }
    assert!(in_frames.len() == num_frames * 16);
    assert!(out_frames.len() == num_frames * 16);
    assert!(residual.len() == num_frames * 16);
    assert!(weights.len() >= 144);
    if do_bias {
        assert!(bias.len() >= 12);
    }

    use core::arch::x86_64::{
        _mm_add_ps, _mm_fmadd_ps, _mm_loadu_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
        _mm256_storeu_ps,
    };

    let mut f = 0;
    while f + 4 <= num_frames {
        let res0_lo = _mm256_loadu_ps(residual.as_ptr().add(f * 16));
        let res1_lo = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * 16));
        let res2_lo = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * 16));
        let res3_lo = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * 16));

        let b_lo = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };

        let mut acc0_lo = _mm256_add_ps(res0_lo, b_lo);
        let mut acc1_lo = _mm256_add_ps(res1_lo, b_lo);
        let mut acc2_lo = _mm256_add_ps(res2_lo, b_lo);
        let mut acc3_lo = _mm256_add_ps(res3_lo, b_lo);

        let res0_hi = _mm_loadu_ps(residual.as_ptr().add(f * 16 + 8));
        let res1_hi = _mm_loadu_ps(residual.as_ptr().add((f + 1) * 16 + 8));
        let res2_hi = _mm_loadu_ps(residual.as_ptr().add((f + 2) * 16 + 8));
        let res3_hi = _mm_loadu_ps(residual.as_ptr().add((f + 3) * 16 + 8));

        let b_hi = if do_bias {
            _mm_loadu_ps(bias.as_ptr().add(8))
        } else {
            _mm_setzero_ps()
        };

        let mut acc0_hi = _mm_add_ps(res0_hi, b_hi);
        let mut acc1_hi = _mm_add_ps(res1_hi, b_hi);
        let mut acc2_hi = _mm_add_ps(res2_hi, b_hi);
        let mut acc3_hi = _mm_add_ps(res3_hi, b_hi);

        for in_c in 0..12 {
            let wp_lo = weights.as_ptr().add(in_c * 12);
            let vw_lo = _mm256_loadu_ps(wp_lo);
            let vw_hi = _mm_loadu_ps(wp_lo.add(8));

            let vs0 = *in_frames.get_unchecked(f * 16 + in_c);
            let vs1 = *in_frames.get_unchecked((f + 1) * 16 + in_c);
            let vs2 = *in_frames.get_unchecked((f + 2) * 16 + in_c);
            let vs3 = *in_frames.get_unchecked((f + 3) * 16 + in_c);

            acc0_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs0), vw_lo, acc0_lo);
            acc1_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs1), vw_lo, acc1_lo);
            acc2_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs2), vw_lo, acc2_lo);
            acc3_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs3), vw_lo, acc3_lo);

            acc0_hi = _mm_fmadd_ps(_mm_set1_ps(vs0), vw_hi, acc0_hi);
            acc1_hi = _mm_fmadd_ps(_mm_set1_ps(vs1), vw_hi, acc1_hi);
            acc2_hi = _mm_fmadd_ps(_mm_set1_ps(vs2), vw_hi, acc2_hi);
            acc3_hi = _mm_fmadd_ps(_mm_set1_ps(vs3), vw_hi, acc3_hi);
        }

        _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * 16), acc0_lo);
        _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * 16), acc1_lo);
        _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * 16), acc2_lo);
        _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * 16), acc3_lo);

        _mm_storeu_ps(out_frames.as_mut_ptr().add(f * 16 + 8), acc0_hi);
        _mm_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * 16 + 8), acc1_hi);
        _mm_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * 16 + 8), acc2_hi);
        _mm_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * 16 + 8), acc3_hi);

        f += 4;
    }

    while f < num_frames {
        let res_lo = _mm256_loadu_ps(residual.as_ptr().add(f * 16));
        let b_lo = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };
        let mut acc_lo = _mm256_add_ps(res_lo, b_lo);

        let res_hi = _mm_loadu_ps(residual.as_ptr().add(f * 16 + 8));
        let b_hi = if do_bias {
            _mm_loadu_ps(bias.as_ptr().add(8))
        } else {
            _mm_setzero_ps()
        };
        let mut acc_hi = _mm_add_ps(res_hi, b_hi);

        for in_c in 0..12 {
            let wp_lo = weights.as_ptr().add(in_c * 12);
            let vw_lo = _mm256_loadu_ps(wp_lo);
            let vw_hi = _mm_loadu_ps(wp_lo.add(8));

            let vs = *in_frames.get_unchecked(f * 16 + in_c);
            acc_lo = _mm256_fmadd_ps(_mm256_set1_ps(vs), vw_lo, acc_lo);
            acc_hi = _mm_fmadd_ps(_mm_set1_ps(vs), vw_hi, acc_hi);
        }

        _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * 16), acc_lo);
        _mm_storeu_ps(out_frames.as_mut_ptr().add(f * 16 + 8), acc_hi);

        f += 1;
    }
}
