// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use core::arch::x86_64::*;

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

/// Applies gain and detects clipping in mono in a single pass using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gain_and_detect_clipping_mono_avx2(data: &mut [f32], gain: f32) -> bool {
    let len = data.len();
    let mut i = 0;
    let vg = _mm256_set1_ps(gain);
    let limit = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut any_clip = _mm256_setzero_ps();

    while i + 8 <= len {
        let p = data.as_mut_ptr().add(i);
        let v = _mm256_loadu_ps(p);
        let g = _mm256_mul_ps(v, vg);
        _mm256_storeu_ps(p, g);
        let abs = _mm256_andnot_ps(sign_mask, g);
        let cmp = _mm256_cmp_ps(abs, limit, _CMP_GT_OQ);
        any_clip = _mm256_or_ps(any_clip, cmp);
        i += 8;
    }

    let mut clipped = _mm256_movemask_ps(any_clip) != 0;

    while i < len {
        let v = *data.get_unchecked(i) * gain;
        *data.get_unchecked_mut(i) = v;
        if !clipped && v.abs() > 1.0 {
            clipped = true;
        }
        i += 1;
    }
    clipped
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

/// Adds a broadcast constant to every element of a mono buffer using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_dither_add_avx2(data: &mut [f32], offset: f32) {
    let len = data.len();
    let voffset = _mm256_set1_ps(offset);
    let mut i = 0;
    while i + 8 <= len {
        let p = data.as_mut_ptr().add(i);
        _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), voffset));
        i += 8;
    }
    while i < len {
        *data.get_unchecked_mut(i) += offset;
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
