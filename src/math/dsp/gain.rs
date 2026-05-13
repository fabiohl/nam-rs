// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Operações DSP de ganho, detecção de clipping e rampa estéreo.
//!
//! Despacha dinamicamente para o backend SIMD configurado.

use core::arch::x86_64::*;

/// Aplica ganho constante em um buffer mono via despacho SIMD.
///
/// # Safety
/// O buffer deve ser válido.
pub unsafe fn apply_gain(data: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain(data, gain))
}

/// Aplica ganho constante em estéreo via despacho SIMD.
///
/// # Safety
/// Os buffers devem ser válidos e ter o mesmo tamanho.
pub unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain_stereo(left, right, gain))
}

/// Aplica ganho e detecta clipping em estéreo em uma única passagem.
/// Retorna `true` se qualquer amostra resultante possuir `|x| > 1.0`.
///
/// # Safety
/// Os buffers devem ser válidos e ter o mesmo tamanho.
pub unsafe fn apply_gain_and_detect_clipping_stereo(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_stereo(left, right, gain))
}

/// Aplica rampa linear de ganho em estéreo via despacho SIMD.
///
/// # Safety
/// Os buffers devem ser válidos e ter o mesmo tamanho.
pub unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    crate::math::common::dispatch_simd!(apply_ramp_stereo(left, right, start, step))
}

// ═══════════════════════════════════════════════════════════════
// Kernels AVX2
// ═══════════════════════════════════════════════════════════════

/// Aplica ganho constante em um buffer mono usando AVX2.
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

/// Aplica ganho e detecta clipping em estéreo em uma única passagem usando AVX2.
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

/// Aplica ganho constante em estéreo via AVX2.
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

/// Aplica rampa linear de ganho em estéreo via AVX2.
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

// ═══════════════════════════════════════════════════════════════
// Kernels AVX-512
// ═══════════════════════════════════════════════════════════════

/// Aplica ganho constante em um buffer mono usando AVX-512.
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

/// Aplica ganho e detecta clipping em estéreo em uma única passagem usando AVX-512.
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

/// Aplica ganho constante em estéreo via AVX-512.
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

/// Aplica uma rampa de volume suave para não dar "estalos" no áudio.
/// O volume começa em "start" e vai mudando a cada amostra pelo valor "step".
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_ramp_stereo_avx512(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    // Cria uma rampa de 16 valores para multiplicar de uma vez.
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
        // Multiplica 16 amostras pelo volume gradual.
        _mm512_storeu_ps(pl, _mm512_mul_ps(_mm512_loadu_ps(pl), current_ramp));
        _mm512_storeu_ps(pr, _mm512_mul_ps(_mm512_loadu_ps(pr), current_ramp));
        // Avança a rampa para os próximos 16.
        current_ramp = _mm512_add_ps(current_ramp, v_step_16);
        i += 16;
    }
    // Finaliza o que sobrou.
    let mut g = start + (i as f32) * step;
    while i < n {
        *left.get_unchecked_mut(i) *= g;
        *right.get_unchecked_mut(i) *= g;
        g += step;
        i += 1;
    }
}
