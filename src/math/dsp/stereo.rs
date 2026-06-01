// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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

    // Fallback escalar (cold-path) para blocos pequenos (comum no Bitwig/CLAP).
    // Evita o overhead das reduções horizontais SIMD para arrays muito curtos.
    if len < 8 {
        return compute_energy_scalar_cold(data);
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

    if len < 8 {
        return compute_max_diff_scalar_cold(a, b);
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

/// Calcula a diferença absoluta máxima entre dois blocos via despacho SIMD.
///
/// # Safety
/// Utiliza despacho dinâmico via v-table global.
pub unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_max_diff(a, b))
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

/// Convolução mono (usada no resampler) via despacho SIMD.
/// Realiza o produto escalar entre um banco de coeficientes e um buffer de entrada.
///
/// # Safety
/// `coeffs` e `input` devem ser ponteiros válidos para pelo menos `taps` elementos.
/// `coeffs` deve estar alinhado conforme o registrador SIMD.
pub unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    crate::math::common::dispatch_simd!(convolve_mono(coeffs, input, taps))
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

/// Convolução Stereo Dual AVX2.
/// Realiza duas convoluções estéreo (para dois conjuntos de coeficientes coeffs0 e coeffs1)
/// sobre os mesmos buffers de entrada input_l e input_r.
/// Carrega amostras de entrada uma única vez e aplica a ambos os conjuntos de coeficientes.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_stereo_dual_avx2(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> ((f32, f32), (f32, f32)) {
    unsafe {
        let mut sum0_l0 = _mm256_setzero_ps();
        let mut sum0_r0 = _mm256_setzero_ps();
        let mut sum0_l1 = _mm256_setzero_ps();
        let mut sum0_r1 = _mm256_setzero_ps();

        let mut sum1_l0 = _mm256_setzero_ps();
        let mut sum1_r0 = _mm256_setzero_ps();
        let mut sum1_l1 = _mm256_setzero_ps();
        let mut sum1_r1 = _mm256_setzero_ps();

        let mut i = 0;

        while i + 16 <= taps {
            let x0_l = _mm256_loadu_ps(input_l.add(i));
            let x0_r = _mm256_loadu_ps(input_r.add(i));

            let h0_0 = _mm256_load_ps(coeffs0.add(i));
            sum0_l0 = _mm256_fmadd_ps(h0_0, x0_l, sum0_l0);
            sum0_r0 = _mm256_fmadd_ps(h0_0, x0_r, sum0_r0);

            let h1_0 = _mm256_load_ps(coeffs1.add(i));
            sum1_l0 = _mm256_fmadd_ps(h1_0, x0_l, sum1_l0);
            sum1_r0 = _mm256_fmadd_ps(h1_0, x0_r, sum1_r0);

            let x1_l = _mm256_loadu_ps(input_l.add(i + 8));
            let x1_r = _mm256_loadu_ps(input_r.add(i + 8));

            let h0_1 = _mm256_load_ps(coeffs0.add(i + 8));
            sum0_l1 = _mm256_fmadd_ps(h0_1, x1_l, sum0_l1);
            sum0_r1 = _mm256_fmadd_ps(h0_1, x1_r, sum0_r1);

            let h1_1 = _mm256_load_ps(coeffs1.add(i + 8));
            sum1_l1 = _mm256_fmadd_ps(h1_1, x1_l, sum1_l1);
            sum1_r1 = _mm256_fmadd_ps(h1_1, x1_r, sum1_r1);

            i += 16;
        }

        while i + 8 <= taps {
            let x_l = _mm256_loadu_ps(input_l.add(i));
            let x_r = _mm256_loadu_ps(input_r.add(i));

            let h0 = _mm256_load_ps(coeffs0.add(i));
            sum0_l0 = _mm256_fmadd_ps(h0, x_l, sum0_l0);
            sum0_r0 = _mm256_fmadd_ps(h0, x_r, sum0_r0);

            let h1 = _mm256_load_ps(coeffs1.add(i));
            sum1_l0 = _mm256_fmadd_ps(h1, x_l, sum1_l0);
            sum1_r0 = _mm256_fmadd_ps(h1, x_r, sum1_r0);

            i += 8;
        }

        // Combine accumulators
        let sum0_l = _mm256_add_ps(sum0_l0, sum0_l1);
        let sum0_r = _mm256_add_ps(sum0_r0, sum0_r1);
        let sum1_l = _mm256_add_ps(sum1_l0, sum1_l1);
        let sum1_r = _mm256_add_ps(sum1_r0, sum1_r1);

        // Redução horizontal sum0_l
        let hi128_0l = _mm256_extractf128_ps(sum0_l, 1);
        let lo128_0l = _mm256_castps256_ps128(sum0_l);
        let s128_0l = _mm_add_ps(lo128_0l, hi128_0l);
        let shuf_0l = _mm_movehdup_ps(s128_0l);
        let sums_0l = _mm_add_ps(s128_0l, shuf_0l);
        let shuf2_0l = _mm_movehl_ps(sums_0l, sums_0l);
        let r_0l = _mm_add_ss(sums_0l, shuf2_0l);
        let mut out0_l = _mm_cvtss_f32(r_0l);

        // Redução horizontal sum0_r
        let hi128_0r = _mm256_extractf128_ps(sum0_r, 1);
        let lo128_0r = _mm256_castps256_ps128(sum0_r);
        let s128_0r = _mm_add_ps(lo128_0r, hi128_0r);
        let shuf_0r = _mm_movehdup_ps(s128_0r);
        let sums_0r = _mm_add_ps(s128_0r, shuf_0r);
        let shuf2_0r = _mm_movehl_ps(sums_0r, sums_0r);
        let r_0r = _mm_add_ss(sums_0r, shuf2_0r);
        let mut out0_r = _mm_cvtss_f32(r_0r);

        // Redução horizontal sum1_l
        let hi128_1l = _mm256_extractf128_ps(sum1_l, 1);
        let lo128_1l = _mm256_castps256_ps128(sum1_l);
        let s128_1l = _mm_add_ps(lo128_1l, hi128_1l);
        let shuf_1l = _mm_movehdup_ps(s128_1l);
        let sums_1l = _mm_add_ps(s128_1l, shuf_1l);
        let shuf2_1l = _mm_movehl_ps(sums_1l, sums_1l);
        let r_1l = _mm_add_ss(sums_1l, shuf2_1l);
        let mut out1_l = _mm_cvtss_f32(r_1l);

        // Redução horizontal sum1_r
        let hi128_1r = _mm256_extractf128_ps(sum1_r, 1);
        let lo128_1r = _mm256_castps256_ps128(sum1_r);
        let s128_1r = _mm_add_ps(lo128_1r, hi128_1r);
        let shuf_1r = _mm_movehdup_ps(s128_1r);
        let sums_1r = _mm_add_ps(s128_1r, shuf_1r);
        let shuf2_1r = _mm_movehl_ps(sums_1r, sums_1r);
        let r_1r = _mm_add_ss(sums_1r, shuf2_1r);
        let mut out1_r = _mm_cvtss_f32(r_1r);

        while i < taps {
            let h0 = *coeffs0.add(i);
            let h1 = *coeffs1.add(i);
            let xl = *input_l.add(i);
            let xr = *input_r.add(i);
            out0_l += h0 * xl;
            out0_r += h0 * xr;
            out1_l += h1 * xl;
            out1_r += h1 * xr;
            i += 1;
        }

        ((out0_l, out0_r), (out1_l, out1_r))
    }
}

/// Convolução Mono AVX2.
/// Carrega coeficientes e aplica a um único canal.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn convolve_mono_avx2(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut i = 0;

        while i + 16 <= taps {
            let h0 = _mm256_load_ps(coeffs.add(i));
            let x0 = _mm256_loadu_ps(input.add(i));
            sum0 = _mm256_fmadd_ps(h0, x0, sum0);

            let h1 = _mm256_load_ps(coeffs.add(i + 8));
            let x1 = _mm256_loadu_ps(input.add(i + 8));
            sum1 = _mm256_fmadd_ps(h1, x1, sum1);

            i += 16;
        }

        while i + 8 <= taps {
            let h = _mm256_load_ps(coeffs.add(i));
            let x = _mm256_loadu_ps(input.add(i));
            sum0 = _mm256_fmadd_ps(h, x, sum0);
            i += 8;
        }

        // Redução horizontal
        let sum = _mm256_add_ps(sum0, sum1);
        let hi128 = _mm256_extractf128_ps(sum, 1);
        let lo128 = _mm256_castps256_ps128(sum);
        let s128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let r = _mm_add_ss(sums, shuf2);
        let mut out = _mm_cvtss_f32(r);

        while i < taps {
            let h = *coeffs.add(i);
            out += h * *input.add(i);
            i += 1;
        }

        out
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

    if len < 8 {
        return compute_energy_stereo_scalar_cold(l, r);
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

/// Convolução Stereo Dual AVX-512.
/// Realiza duas convoluções estéreo (para dois conjuntos de coeficientes coeffs0 e coeffs1)
/// sobre os mesmos buffers de entrada input_l e input_r.
/// Carrega amostras de entrada uma única vez e aplica a ambos os conjuntos de coeficientes.
#[target_feature(enable = "avx512f")]
pub unsafe fn convolve_stereo_dual_avx512(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> ((f32, f32), (f32, f32)) {
    unsafe {
        let mut sum0_l = _mm512_setzero_ps();
        let mut sum0_r = _mm512_setzero_ps();
        let mut sum1_l = _mm512_setzero_ps();
        let mut sum1_r = _mm512_setzero_ps();
        let mut i = 0;

        // Processa 16 amostras de áudio de uma vez!
        while i + 16 <= taps {
            let x_l = _mm512_loadu_ps(input_l.add(i));
            let x_r = _mm512_loadu_ps(input_r.add(i));

            let h0 = _mm512_load_ps(coeffs0.add(i));
            sum0_l = _mm512_fmadd_ps(h0, x_l, sum0_l);
            sum0_r = _mm512_fmadd_ps(h0, x_r, sum0_r);

            let h1 = _mm512_load_ps(coeffs1.add(i));
            sum1_l = _mm512_fmadd_ps(h1, x_l, sum1_l);
            sum1_r = _mm512_fmadd_ps(h1, x_r, sum1_r);

            i += 16;
        }

        // Soma os acumuladores para obter o resultado final de cada canal.
        let mut out0_l = _mm512_reduce_add_ps(sum0_l);
        let mut out0_r = _mm512_reduce_add_ps(sum0_r);
        let mut out1_l = _mm512_reduce_add_ps(sum1_l);
        let mut out1_r = _mm512_reduce_add_ps(sum1_r);

        // Termina as amostras que sobraram.
        while i < taps {
            let h0 = *coeffs0.add(i);
            let h1 = *coeffs1.add(i);
            let xl = *input_l.add(i);
            let xr = *input_r.add(i);
            out0_l += h0 * xl;
            out0_r += h0 * xr;
            out1_l += h1 * xl;
            out1_r += h1 * xr;
            i += 1;
        }

        ((out0_l, out0_r), (out1_l, out1_r))
    }
}

/// Convolução Mono AVX-512.
/// Carrega coeficientes e aplica a um único canal.
#[target_feature(enable = "avx512f")]
pub unsafe fn convolve_mono_avx512(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut i = 0;

        while i + 32 <= taps {
            let h0 = _mm512_load_ps(coeffs.add(i));
            let x0 = _mm512_loadu_ps(input.add(i));
            sum0 = _mm512_fmadd_ps(h0, x0, sum0);

            let h1 = _mm512_load_ps(coeffs.add(i + 16));
            let x1 = _mm512_loadu_ps(input.add(i + 16));
            sum1 = _mm512_fmadd_ps(h1, x1, sum1);

            i += 32;
        }

        let sum = _mm512_add_ps(sum0, sum1);
        let mut out = _mm512_reduce_add_ps(sum);

        while i < taps {
            let h = *coeffs.add(i);
            out += h * *input.add(i);
            i += 1;
        }

        out
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

/// Calcula a energia (Mean Square) de um bloco via AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_energy_avx512(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();

    while i + 16 <= len {
        let v = _mm512_loadu_ps(data.as_ptr().add(i));
        sum_v = _mm512_fmadd_ps(v, v, sum_v);
        i += 16;
    }

    let mut total_sum = crate::math::common::utility::hsum_avx512(sum_v);

    while i < len {
        total_sum += data[i] * data[i];
        i += 1;
    }

    total_sum / (len as f32)
}

/// Calcula a diferença absoluta máxima entre dois blocos via AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_max_diff_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    let mut max_v = _mm512_setzero_ps();
    let sign_mask = _mm512_set1_ps(-0.0f32);

    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        let abs_diff = _mm512_andnot_ps(sign_mask, diff);
        max_v = _mm512_max_ps(max_v, abs_diff);
        i += 16;
    }

    let mut max_diff = _mm512_reduce_max_ps(max_v);

    while i < len {
        let d = (a[i] - b[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
        i += 1;
    }

    max_diff
}

// ═══════════════════════════════════════════════════════════════
// Fallbacks Escalares (Cold-Path)
// ═══════════════════════════════════════════════════════════════

#[cold]
#[inline(never)]
fn compute_energy_scalar_cold(data: &[f32]) -> f32 {
    let mut total_sum = 0.0;
    for &x in data {
        total_sum += x * x;
    }
    total_sum / (data.len() as f32)
}

#[cold]
#[inline(never)]
fn compute_max_diff_scalar_cold(a: &[f32], b: &[f32]) -> f32 {
    let mut max_diff = 0.0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    max_diff
}

#[cold]
#[inline(never)]
fn compute_energy_stereo_scalar_cold(l: &[f32], r: &[f32]) -> f32 {
    let mut total_sum_l = 0.0;
    let mut total_sum_r = 0.0;
    for i in 0..l.len() {
        total_sum_l += l[i] * l[i];
        total_sum_r += r[i] * r[i];
    }
    let energy_l = total_sum_l / (l.len() as f32);
    let energy_r = total_sum_r / (l.len() as f32);
    energy_l.max(energy_r)
}
