// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Operações DSP para processamento estéreo e medição de sinal.

use core::arch::x86_64::*;

/// Calcula a energia (Mean Square) de um bloco via AVX2.
/// $E = \frac{1}{N} \sum x_i^2$
///
/// # Safety
/// O slice `data` deve ser válido.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_energy_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();

        while i + 16 <= len {
            let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
            sum0 = _mm256_fmadd_ps(v0, v0, sum0);
            sum1 = _mm256_fmadd_ps(v1, v1, sum1);
            i += 16;
        }

        while i + 8 <= len {
            let v = _mm256_loadu_ps(data.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(v, v, sum0);
            i += 8;
        }

        let sum = _mm256_add_ps(sum0, sum1);
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let s128 = _mm_add_ps(lo, hi);

        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let r = _mm_add_ss(sums, shuf2);

        let mut total_sum = 0.0f32;
        _mm_store_ss(&mut total_sum, r);

        while i < len {
            total_sum += data[i] * data[i];
            i += 1;
        }

        total_sum / (len as f32)
    }
}

/// Calcula a diferença absoluta máxima entre dois blocos via AVX2.
/// $\max(|L_i - R_i|)$
///
/// # Safety
/// Os slices `a` e `b` devem ter o mesmo tamanho.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_max_diff_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    unsafe {
        let mut max_v = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0f32);

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            max_v = _mm256_max_ps(max_v, abs_diff);
            i += 8;
        }

        let hi = _mm256_extractf128_ps(max_v, 1);
        let lo = _mm256_castps256_ps128(max_v);
        let m128 = _mm_max_ps(lo, hi);

        let shuf = _mm_shuffle_ps(m128, m128, 0xEE);
        let m64 = _mm_max_ps(m128, shuf);
        let shuf2 = _mm_shuffle_ps(m64, m64, 0x55);
        let m32 = _mm_max_ps(m64, shuf2);

        let mut max_diff = 0.0f32;
        _mm_store_ss(&mut max_diff, m32);

        while i < len {
            let d = (a[i] - b[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
            i += 1;
        }

        max_diff
    }
}

/// Calcula o máximo da energia entre dois canais de áudio via despacho SIMD.
///
/// # Safety
/// Utiliza despacho dinâmico via v-table global.
pub unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_energy_stereo(l, r))
}
