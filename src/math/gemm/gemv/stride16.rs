// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Pad-to-16 stride variants of CH=12 GEMV / broadcast_scale kernels.
//!
//! These kernels write 12 valid channels at stride 16, preserving zero in padding
//! lanes 12-15.

/// Broadcast (COND=1) with bias, stride-16 output.
///
/// For each condition frame `n`: `out[n*16 + oc] = bias[oc] + cond[n] * weights[oc]`
/// for `oc in 0..12`. Lanes 12-15 are unwritten (remain zero).
///
/// # Safety
/// AVX2+FMA ISA must be available (x86-64-v3).
/// `cond.len() * 16 == out_frames.len()`, `weights.len() >= 12`, `bias.len() >= 12`.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn broadcast_scale_with_bias_f32_avx2_padded(
    cond: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
) {
    debug_assert_eq!(out_frames.len(), num_frames * 16);
    debug_assert!(weights.len() >= 12);
    debug_assert!(bias.len() >= 12);

    use core::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskstore_ps, _mm256_set1_ps,
        _mm256_storeu_ps,
    };

    let mut w_buf = [0.0f32; 16];
    w_buf[..12].copy_from_slice(weights.get_unchecked(..12));
    let mut b_buf = [0.0f32; 16];
    b_buf[..12].copy_from_slice(bias.get_unchecked(..12));

    for n in 0..num_frames {
        let v_in = _mm256_set1_ps(*cond.get_unchecked(n));

        // lanes 0-7
        let v_w = _mm256_loadu_ps(w_buf.as_ptr());
        let v_b = _mm256_loadu_ps(b_buf.as_ptr());
        let v_out = _mm256_fmadd_ps(v_in, v_w, v_b);
        _mm256_storeu_ps(out_frames.as_mut_ptr().add(n * 16), v_out);

        // lanes 8-11 with masked store (only 4 valid elements)
        let v_w_hi = _mm256_loadu_ps(w_buf.as_ptr().add(8));
        let v_b_hi = _mm256_loadu_ps(b_buf.as_ptr().add(8));
        let v_out_hi = _mm256_fmadd_ps(v_in, v_w_hi, v_b_hi);

        let mut mask_buf = [0i32; 8];
        mask_buf[..4].fill(-1);
        let mask = _mm256_loadu_si256(mask_buf.as_ptr() as *const core::arch::x86_64::__m256i);
        _mm256_maskstore_ps(out_frames.as_mut_ptr().add(n * 16 + 8), mask, v_out_hi);
    }
}

/// Batched GEMV with bias, stride-16 input, natural stride output.
///
/// For each frame `f`: `out[f*OUT + oc] = bias[oc] + sum_{ic=0..11}(in[f*16+ic] * weights[ic*OUT+oc])`
///
/// # Safety
/// `in_frames.len() == num_frames * 16`, `out_frames.len() == num_frames * OUT`,
/// `weights.len() >= 12 * OUT`, `bias.len() >= OUT`.
/// AVX2+FMA ISA must be available (x86-64-v3).
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_with_bias_f32_avx2_padded(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    out_len: usize,
) {
    if num_frames == 0 {
        return;
    }
    debug_assert!(
        in_frames.len() >= num_frames * 16,
        "in_frames.len()={} < num_frames*16={}",
        in_frames.len(),
        num_frames * 16
    );
    debug_assert_eq!(out_frames.len(), num_frames * out_len);
    debug_assert!(weights.len() >= 12 * out_len);
    debug_assert!(bias.len() >= out_len);

    use core::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskstore_ps,
        _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    for f in 0..num_frames {
        let frame_base = f * 16;

        let mut oc = 0;
        while oc + 8 <= out_len {
            let w_ptr = weights.as_ptr().add(oc);
            let mut acc = _mm256_loadu_ps(bias.as_ptr().add(oc));

            for ic in 0..12 {
                let vs = *in_frames.get_unchecked(frame_base + ic);
                let vw = _mm256_loadu_ps(w_ptr.add(ic * out_len));
                acc = _mm256_fmadd_ps(_mm256_set1_ps(vs), vw, acc);
            }
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + oc), acc);
            oc += 8;
        }

        if oc < out_len {
            let rem = out_len - oc;
            let mut mask_buf = [0i32; 8];
            mask_buf[..rem].fill(-1);
            let mut acc = _mm256_setzero_ps();
            for ic in 0..12 {
                let vs = *in_frames.get_unchecked(frame_base + ic);
                let wp = weights.as_ptr().add(oc + ic * out_len);
                let mut w_buf = [0.0f32; 8];
                w_buf[..rem].copy_from_slice(std::slice::from_raw_parts(wp, rem));
                let vw = _mm256_loadu_ps(w_buf.as_ptr());
                acc = _mm256_fmadd_ps(_mm256_set1_ps(vs), vw, acc);
            }
            let mut b_buf = [0.0f32; 8];
            b_buf[..rem].copy_from_slice(std::slice::from_raw_parts(bias.as_ptr().add(oc), rem));
            let vb = _mm256_loadu_ps(b_buf.as_ptr());
            acc = _mm256_add_ps(acc, vb);
            let mask = _mm256_loadu_si256(mask_buf.as_ptr() as *const core::arch::x86_64::__m256i);
            _mm256_maskstore_ps(out_frames.as_mut_ptr().add(f * out_len + oc), mask, acc);
        }
    }
}
