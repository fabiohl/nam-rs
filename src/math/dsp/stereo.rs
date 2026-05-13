// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

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

/// Convolução estéreo (usada no resampler) via despacho SIMD.
/// Realiza o produto escalar entre um banco de coeficientes e dois buffers de entrada (L/R).
///
/// # Safety
/// `coeffs`, `input_l` e `input_r` devem ser ponteiros válidos para pelo menos `taps` elementos.
/// `coeffs` deve estar alinhado conforme o registrador SIMD.
pub unsafe fn convolve_stereo(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    crate::math::common::dispatch_simd!(convolve_stereo(coeffs, input_l, input_r, taps))
}

// ═══════════════════════════════════════════════════════════════
// Kernels AVX2
// ═══════════════════════════════════════════════════════════════

/// Convolução Stereo Interleaved AVX2.
/// Carrega coeficientes uma única vez e aplica a ambos os canais.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_stereo_avx2(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    unsafe {
        let mut sum_l0 = _mm256_setzero_ps();
        let mut sum_l1 = _mm256_setzero_ps();
        let mut sum_r0 = _mm256_setzero_ps();
        let mut sum_r1 = _mm256_setzero_ps();
        let mut i = 0;

        while i + 16 <= taps {
            let h0 = _mm256_load_ps(coeffs.add(i));
            let x0_l = _mm256_loadu_ps(input_l.add(i));
            let x0_r = _mm256_loadu_ps(input_r.add(i));
            sum_l0 = _mm256_fmadd_ps(h0, x0_l, sum_l0);
            sum_r0 = _mm256_fmadd_ps(h0, x0_r, sum_r0);

            let h1 = _mm256_load_ps(coeffs.add(i + 8));
            let x1_l = _mm256_loadu_ps(input_l.add(i + 8));
            let x1_r = _mm256_loadu_ps(input_r.add(i + 8));
            sum_l1 = _mm256_fmadd_ps(h1, x1_l, sum_l1);
            sum_r1 = _mm256_fmadd_ps(h1, x1_r, sum_r1);

            i += 16;
        }

        while i + 8 <= taps {
            let h = _mm256_load_ps(coeffs.add(i));
            let x_l = _mm256_loadu_ps(input_l.add(i));
            let x_r = _mm256_loadu_ps(input_r.add(i));
            sum_l0 = _mm256_fmadd_ps(h, x_l, sum_l0);
            sum_r0 = _mm256_fmadd_ps(h, x_r, sum_r0);
            i += 8;
        }

        // Redução horizontal L
        let sum_l = _mm256_add_ps(sum_l0, sum_l1);
        let hi128_l = _mm256_extractf128_ps(sum_l, 1);
        let lo128_l = _mm256_castps256_ps128(sum_l);
        let s128_l = _mm_add_ps(lo128_l, hi128_l);
        let shuf_l = _mm_movehdup_ps(s128_l);
        let sums_l = _mm_add_ps(s128_l, shuf_l);
        let shuf2_l = _mm_movehl_ps(sums_l, sums_l);
        let r_l = _mm_add_ss(sums_l, shuf2_l);
        let mut out_l = _mm_cvtss_f32(r_l);

        // Redução horizontal R
        let sum_r = _mm256_add_ps(sum_r0, sum_r1);
        let hi128_r = _mm256_extractf128_ps(sum_r, 1);
        let lo128_r = _mm256_castps256_ps128(sum_r);
        let s128_r = _mm_add_ps(lo128_r, hi128_r);
        let shuf_r = _mm_movehdup_ps(s128_r);
        let sums_r = _mm_add_ps(s128_r, shuf_r);
        let shuf2_r = _mm_movehl_ps(sums_r, sums_r);
        let r_r = _mm_add_ss(sums_r, shuf2_r);
        let mut out_r = _mm_cvtss_f32(r_r);

        while i < taps {
            let h = *coeffs.add(i);
            out_l += h * *input_l.add(i);
            out_r += h * *input_r.add(i);
            i += 1;
        }

        (out_l, out_r)
    }
}

/// Calcula o máximo da energia entre dois canais (Mean Square) via AVX2.
/// Funde as duas passagens em uma para economizar banda de memória.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_energy_stereo_avx2(l: &[f32], r: &[f32]) -> f32 {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    let mut sum_l0 = _mm256_setzero_ps();
    let mut sum_l1 = _mm256_setzero_ps();
    let mut sum_r0 = _mm256_setzero_ps();
    let mut sum_r1 = _mm256_setzero_ps();

    while i + 16 <= len {
        let vl0 = _mm256_loadu_ps(l.as_ptr().add(i));
        let vl1 = _mm256_loadu_ps(l.as_ptr().add(i + 8));
        let vr0 = _mm256_loadu_ps(r.as_ptr().add(i));
        let vr1 = _mm256_loadu_ps(r.as_ptr().add(i + 8));

        sum_l0 = _mm256_fmadd_ps(vl0, vl0, sum_l0);
        sum_l1 = _mm256_fmadd_ps(vl1, vl1, sum_l1);
        sum_r0 = _mm256_fmadd_ps(vr0, vr0, sum_r0);
        sum_r1 = _mm256_fmadd_ps(vr1, vr1, sum_r1);
        i += 16;
    }

    while i + 8 <= len {
        let vl = _mm256_loadu_ps(l.as_ptr().add(i));
        let vr = _mm256_loadu_ps(r.as_ptr().add(i));
        sum_l0 = _mm256_fmadd_ps(vl, vl, sum_l0);
        sum_r0 = _mm256_fmadd_ps(vr, vr, sum_r0);
        i += 8;
    }

    // Soma horizontal para L
    let sum_l = _mm256_add_ps(sum_l0, sum_l1);
    let hi_l = _mm256_extractf128_ps(sum_l, 1);
    let lo_l = _mm256_castps256_ps128(sum_l);
    let s128_l = _mm_add_ps(lo_l, hi_l);
    let shuf_l = _mm_movehdup_ps(s128_l);
    let sums_l = _mm_add_ps(s128_l, shuf_l);
    let shuf2_l = _mm_movehl_ps(sums_l, sums_l);
    let r_l = _mm_add_ss(sums_l, shuf2_l);
    let mut total_sum_l = 0.0f32;
    _mm_store_ss(&mut total_sum_l, r_l);

    // Soma horizontal para R
    let sum_r = _mm256_add_ps(sum_r0, sum_r1);
    let hi_r = _mm256_extractf128_ps(sum_r, 1);
    let lo_r = _mm256_castps256_ps128(sum_r);
    let s128_r = _mm_add_ps(lo_r, hi_r);
    let shuf_r = _mm_movehdup_ps(s128_r);
    let sums_r = _mm_add_ps(s128_r, shuf_r);
    let shuf2_r = _mm_movehl_ps(sums_r, sums_r);
    let r_r = _mm_add_ss(sums_r, shuf2_r);
    let mut total_sum_r = 0.0f32;
    _mm_store_ss(&mut total_sum_r, r_r);

    while i < len {
        total_sum_l += l[i] * l[i];
        total_sum_r += r[i] * r[i];
        i += 1;
    }

    let energy_l = total_sum_l / (len as f32);
    let energy_r = total_sum_r / (len as f32);
    energy_l.max(energy_r)
}

// ═══════════════════════════════════════════════════════════════
// Kernels AVX-512
// ═══════════════════════════════════════════════════════════════

/// Convolução Stereo: Aplica um filtro (coeficientes) em dois canais de áudio ao mesmo tempo.
/// É como passar o som por um equalizador ou simular uma sala (reverb).
#[target_feature(enable = "avx512f")]
pub unsafe fn convolve_stereo_avx512(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    unsafe {
        let mut sum_l0 = _mm512_setzero_ps();
        let mut sum_l1 = _mm512_setzero_ps();
        let mut sum_r0 = _mm512_setzero_ps();
        let mut sum_r1 = _mm512_setzero_ps();
        let mut i = 0;

        // Processa 32 amostras de áudio de uma vez só!
        while i + 32 <= taps {
            let h0 = _mm512_load_ps(coeffs.add(i)); // 16 coeficientes.
            let x0_l = _mm512_loadu_ps(input_l.add(i)); // 16 amostras da esquerda.
            let x0_r = _mm512_loadu_ps(input_r.add(i)); // 16 amostras da direita.
            sum_l0 = _mm512_fmadd_ps(h0, x0_l, sum_l0);
            sum_r0 = _mm512_fmadd_ps(h0, x0_r, sum_r0);

            let h1 = _mm512_load_ps(coeffs.add(i + 16));
            let x1_l = _mm512_loadu_ps(input_l.add(i + 16));
            let x1_r = _mm512_loadu_ps(input_r.add(i + 16));
            sum_l1 = _mm512_fmadd_ps(h1, x1_l, sum_l1);
            sum_r1 = _mm512_fmadd_ps(h1, x1_r, sum_r1);

            i += 32;
        }

        // Soma os acumuladores para obter o resultado final de cada canal.
        let sum_l = _mm512_add_ps(sum_l0, sum_l1);
        let sum_r = _mm512_add_ps(sum_r0, sum_r1);
        let mut out_l = _mm512_reduce_add_ps(sum_l);
        let mut out_r = _mm512_reduce_add_ps(sum_r);

        // Termina as amostras que sobraram.
        while i < taps {
            let h = *coeffs.add(i);
            out_l += h * *input_l.add(i);
            out_r += h * *input_r.add(i);
            i += 1;
        }

        (out_l, out_r)
    }
}

/// Calcula o máximo da energia entre dois canais (Mean Square) via AVX-512.
/// Funde as duas passagens em uma para economizar banda de memória.
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_energy_stereo_avx512(l: &[f32], r: &[f32]) -> f32 {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    let mut sum_lv = _mm512_setzero_ps();
    let mut sum_rv = _mm512_setzero_ps();

    while i + 16 <= len {
        let lv = _mm512_loadu_ps(l.as_ptr().add(i));
        let rv = _mm512_loadu_ps(r.as_ptr().add(i));
        sum_lv = _mm512_fmadd_ps(lv, lv, sum_lv);
        sum_rv = _mm512_fmadd_ps(rv, rv, sum_rv);
        i += 16;
    }

    let mut sum_l = crate::math::common::utility::hsum_avx512(sum_lv);
    let mut sum_r = crate::math::common::utility::hsum_avx512(sum_rv);

    while i < len {
        sum_l += l[i] * l[i];
        sum_r += r[i] * r[i];
        i += 1;
    }

    let energy_l = sum_l / (len as f32);
    let energy_r = sum_r / (len as f32);
    energy_l.max(energy_r)
}
