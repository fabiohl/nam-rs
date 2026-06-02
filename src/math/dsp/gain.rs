// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! DSP gain operations, clipping detection, and stereo ramp.
//!
//! Dynamically dispatches to the configured SIMD backend.

use core::arch::x86_64::*;

/// Applies constant gain to a mono buffer via SIMD dispatch.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_gain(data: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain(data, gain))
}

/// Applies constant gain in stereo via SIMD dispatch.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain_stereo(left, right, gain))
}

/// Applies gain and detects clipping in stereo in a single pass.
/// Returns `true` if any resulting sample has `|x| > 1.0`.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_gain_and_detect_clipping_stereo(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_stereo(left, right, gain))
}

/// Applies linear gain ramp in stereo via SIMD dispatch.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    crate::math::common::dispatch_simd!(apply_ramp_stereo(left, right, start, step))
}

/// Applies linear gain ramp to a mono buffer via SIMD dispatch.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
    crate::math::common::dispatch_simd!(apply_ramp(data, start, step))
}

/// Applies the linear gain multiplier to the buffer (safe wrapper).
///
/// Fast-returns if `gain_linear` ~= 1.0 (fast-path bypass).
/// Routes to the SIMD backend via v-table.
pub fn apply_gain_simd(buffer: &mut [f32], gain_linear: f32) {
    if (gain_linear - 1.0).abs() < 1e-6 {
        return;
    }
    unsafe { apply_gain(buffer, gain_linear) };
}

/// Applies a linear gain ramp to the buffer (safe wrapper).
///
/// If the increment is negligible, applies constant gain instead.
/// Routes to the SIMD backend via v-table.
pub fn apply_ramp_simd(buffer: &mut [f32], start: f32, step: f32) {
    if step.abs() < 1e-9 {
        apply_gain_simd(buffer, start);
        return;
    }
    unsafe { apply_ramp(buffer, start, step) };
}

// ═══════════════════════════════════════════════════════════════
// Kernels AVX2
// ═══════════════════════════════════════════════════════════════

/// Applies constant gain to a mono buffer using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gain_avx2(data: &mut [f32], gain: f32) {
    let len = data.len();
    let vg = _mm256_set1_ps(gain);
    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_mul_ps(v, vg));
        i += 8;
    }
    while i < len {
        data[i] *= gain;
        i += 1;
    }
}

/// Applies gain and detects clipping in stereo in a single pass using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gain_and_detect_clipping_stereo_avx2(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let ymm_gain = _mm256_set1_ps(gain);
    let limit = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut any_clip = _mm256_setzero_ps();

    while i + 8 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);

        let vl = _mm256_loadu_ps(pl);
        let vr = _mm256_loadu_ps(pr);

        let gl = _mm256_mul_ps(vl, ymm_gain);
        let gr = _mm256_mul_ps(vr, ymm_gain);

        _mm256_storeu_ps(pl, gl);
        _mm256_storeu_ps(pr, gr);

        let abs_l = _mm256_andnot_ps(sign_mask, gl);
        let abs_r = _mm256_andnot_ps(sign_mask, gr);

        let cmp_l = _mm256_cmp_ps(abs_l, limit, _CMP_GT_OQ);
        let cmp_r = _mm256_cmp_ps(abs_r, limit, _CMP_GT_OQ);

        any_clip = _mm256_or_ps(any_clip, _mm256_or_ps(cmp_l, cmp_r));
        i += 8;
    }

    let mut clipped = _mm256_movemask_ps(any_clip) != 0;

    while i < n {
        let vl = *left.get_unchecked(i) * gain;
        let vr = *right.get_unchecked(i) * gain;
        *left.get_unchecked_mut(i) = vl;
        *right.get_unchecked_mut(i) = vr;
        if !clipped && (vl.abs() > 1.0 || vr.abs() > 1.0) {
            clipped = true;
        }
        i += 1;
    }
    clipped
}

/// Applies constant gain in stereo via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gain_stereo_avx2(left: &mut [f32], right: &mut [f32], gain: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let ymm_gain = _mm256_set1_ps(gain);
    while i + 8 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);
        _mm256_storeu_ps(pl, _mm256_mul_ps(_mm256_loadu_ps(pl), ymm_gain));
        _mm256_storeu_ps(pr, _mm256_mul_ps(_mm256_loadu_ps(pr), ymm_gain));
        i += 8;
    }
    while i < n {
        *left.get_unchecked_mut(i) *= gain;
        *right.get_unchecked_mut(i) *= gain;
        i += 1;
    }
}

/// Applies linear gain ramp in stereo via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_ramp_stereo_avx2(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let mut current_ramp = _mm256_set_ps(
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_8 = _mm256_set1_ps(8.0 * step);
    while i + 8 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);
        _mm256_storeu_ps(pl, _mm256_mul_ps(_mm256_loadu_ps(pl), current_ramp));
        _mm256_storeu_ps(pr, _mm256_mul_ps(_mm256_loadu_ps(pr), current_ramp));
        current_ramp = _mm256_add_ps(current_ramp, v_step_8);
        i += 8;
    }
    let mut g = start + (i as f32) * step;
    while i < n {
        *left.get_unchecked_mut(i) *= g;
        *right.get_unchecked_mut(i) *= g;
        g += step;
        i += 1;
    }
}

/// Applies linear gain ramp to a mono buffer via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_ramp_avx2(buffer: &mut [f32], start: f32, step: f32) {
    let len = buffer.len();
    let mut i = 0;
    let mut current_ramp = _mm256_set_ps(
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_8 = _mm256_set1_ps(8.0 * step);
    while i + 8 <= len {
        let ptr = buffer.as_mut_ptr().add(i);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(_mm256_loadu_ps(ptr), current_ramp));
        current_ramp = _mm256_add_ps(current_ramp, v_step_8);
        i += 8;
    }
    let mut m = start + (i as f32) * step;
    while i < len {
        *buffer.get_unchecked_mut(i) *= m;
        m += step;
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════
// Kernels AVX-512
// ═══════════════════════════════════════════════════════════════

/// Applies constant gain to a mono buffer using AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn apply_gain_avx512(data: &mut [f32], gain: f32) {
    let len = data.len();
    let vg = _mm512_set1_ps(gain);
    let mut i = 0;
    while i + 16 <= len {
        let v = _mm512_loadu_ps(data.as_ptr().add(i));
        _mm512_storeu_ps(data.as_mut_ptr().add(i), _mm512_mul_ps(v, vg));
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let v = _mm512_maskz_loadu_ps(mask, data.as_ptr().add(i));
        _mm512_mask_storeu_ps(data.as_mut_ptr().add(i), mask, _mm512_mul_ps(v, vg));
    }
}

/// Applies gain and detects clipping in stereo in a single pass using AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_gain_and_detect_clipping_stereo_avx512(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let v_gain = _mm512_set1_ps(gain);
    let v_limit = _mm512_set1_ps(1.0);
    let sign_mask = _mm512_set1_ps(-0.0f32);
    let mut k_clip = 0u16;

    while i + 16 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);

        let vl = _mm512_loadu_ps(pl);
        let vr = _mm512_loadu_ps(pr);

        let gl = _mm512_mul_ps(vl, v_gain);
        let gr = _mm512_mul_ps(vr, v_gain);

        _mm512_storeu_ps(pl, gl);
        _mm512_storeu_ps(pr, gr);

        let abs_l = _mm512_andnot_ps(sign_mask, gl);
        let abs_r = _mm512_andnot_ps(sign_mask, gr);

        let k_l = _mm512_cmp_ps_mask(abs_l, v_limit, _CMP_GT_OQ);
        let k_r = _mm512_cmp_ps_mask(abs_r, v_limit, _CMP_GT_OQ);

        k_clip |= k_l | k_r;
        i += 16;
    }

    let mut clipped = k_clip != 0;

    while i < n {
        let vl = *left.get_unchecked(i) * gain;
        let vr = *right.get_unchecked(i) * gain;
        *left.get_unchecked_mut(i) = vl;
        *right.get_unchecked_mut(i) = vr;
        if !clipped && (vl.abs() > 1.0 || vr.abs() > 1.0) {
            clipped = true;
        }
        i += 1;
    }
    clipped
}

/// Applies constant gain in stereo via AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_gain_stereo_avx512(left: &mut [f32], right: &mut [f32], gain: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let zmm_gain = _mm512_set1_ps(gain);
    while i + 16 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);
        _mm512_storeu_ps(pl, _mm512_mul_ps(_mm512_loadu_ps(pl), zmm_gain));
        _mm512_storeu_ps(pr, _mm512_mul_ps(_mm512_loadu_ps(pr), zmm_gain));
        i += 16;
    }
    while i < n {
        *left.get_unchecked_mut(i) *= gain;
        *right.get_unchecked_mut(i) *= gain;
        i += 1;
    }
}

/// Applies a smooth volume ramp to avoid audio "pops".
/// Volume starts at "start" and changes each sample by the "step" value.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_ramp_stereo_avx512(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    // Creates a ramp of 16 values to multiply at once.
    let mut current_ramp = _mm512_set_ps(
        start + 15.0 * step,
        start + 14.0 * step,
        start + 13.0 * step,
        start + 12.0 * step,
        start + 11.0 * step,
        start + 10.0 * step,
        start + 9.0 * step,
        start + 8.0 * step,
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_16 = _mm512_set1_ps(16.0 * step);
    while i + 16 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);
        // Multiplies 16 samples by the gradual volume.
        _mm512_storeu_ps(pl, _mm512_mul_ps(_mm512_loadu_ps(pl), current_ramp));
        _mm512_storeu_ps(pr, _mm512_mul_ps(_mm512_loadu_ps(pr), current_ramp));
        // Advances the ramp for the next 16.
        current_ramp = _mm512_add_ps(current_ramp, v_step_16);
        i += 16;
    }
    // Finishes the remainder.
    let mut g = start + (i as f32) * step;
    while i < n {
        *left.get_unchecked_mut(i) *= g;
        *right.get_unchecked_mut(i) *= g;
        g += step;
        i += 1;
    }
}

/// Applies linear gain ramp to a mono buffer via AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_ramp_avx512(buffer: &mut [f32], start: f32, step: f32) {
    let len = buffer.len();
    let mut i = 0;
    let mut current_ramp = _mm512_set_ps(
        start + 15.0 * step,
        start + 14.0 * step,
        start + 13.0 * step,
        start + 12.0 * step,
        start + 11.0 * step,
        start + 10.0 * step,
        start + 9.0 * step,
        start + 8.0 * step,
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_16 = _mm512_set1_ps(16.0 * step);
    while i + 16 <= len {
        let ptr = buffer.as_mut_ptr().add(i);
        _mm512_storeu_ps(ptr, _mm512_mul_ps(_mm512_loadu_ps(ptr), current_ramp));
        current_ramp = _mm512_add_ps(current_ramp, v_step_16);
        i += 16;
    }
    let mut m = start + (i as f32) * step;
    while i < len {
        *buffer.get_unchecked_mut(i) *= m;
        m += step;
        i += 1;
    }
}

#[cfg(test)]
#[path = "gain_test.rs"]
mod gain_test;
