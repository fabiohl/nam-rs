// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

// Desativa alguns avisos do compilador para permitir o uso de código de baixo nível (unsafe)
// e funções com muitos argumentos, comuns em otimizações de áudio de alto desempenho.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Backends SIMD AVX-512.
//!
//! Kernels não-GEMM: convolução estéreo, ganho, detecção de clipping, rampa, energia estéreo, f32→bf16.
//! Kernels de álgebra linear (dot, gemv, gemm, gemv_4gate) foram movidos para `math::gemm/`.
//! Kernels wavenet (gated activation, head sum) foram movidos para `math::wavenet/`.
//! Kernels LSTM foram movidos para `math::lstm/`.

use core::arch::x86_64::*; // Acesso direto às instruções de hardware do processador (intrinsics).

// Re-exports de kernels GEMM (Tarefa 3.2 — mantém compatibilidade de paths explícitos)
pub use crate::math::gemm::dot::{dot_product_avx512, dot_product_bf16_avx512};
pub use crate::math::gemm::dot_4x::{
    dot_product_4x_interleaved_avx512, dot_product_4x_interleaved_dual_frame_avx512,
};
pub use crate::math::gemm::gemm_batch::{
    fused_add_gemm_batch_avx512, fused_gemm_residual_batch_avx512,
};
pub use crate::math::gemm::gemv::{
    fused_add_gemv_avx512, fused_add_gemv_avx512_small, gemv_overwrite_avx512,
    gemv_overwrite_avx512_small, gemv_overwrite_batch_avx512,
};
pub use crate::math::gemm::gemv_4gate::{gemv_4gate_avx512, gemv_4gate_bf16_avx512};

// Re-exports de kernels LSTM (Tarefa 3.3 — mantém compatibilidade de paths explícitos)
pub use crate::math::lstm::fused_lstm_gates_dyn_avx512;

// Re-exports de kernels Wavenet (Tarefa 3.4 — mantém compatibilidade de paths explícitos)
pub use crate::math::wavenet::accumulate::gated_activation_and_accumulate_block_avx512;
pub use crate::math::wavenet::head::{
    batch_wavenet_head_sum_avx512, batch_wavenet_head_sum_dyn_avx512,
};

/// Converte um vetor de números f32 (normais) para bf16 (compactos) via AVX-512.
/// O formato bf16 ocupa metade do espaço, mas mantém o alcance dos números f32,
/// sendo ideal para modelos de inteligência artificial rápidos.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn f32_to_bf16_avx512(src: &[f32], dest: &mut [u16]) {
    let n = core::cmp::min(src.len(), dest.len());
    let mut i = 0;
    // Processa 16 conversões de uma vez.
    while i + 16 <= n {
        let v = _mm512_loadu_ps(src.as_ptr().add(i)); // Carrega 16 números f32.
        let v_i = _mm512_castps_si512(v); // Trata como inteiros para manipular os bits.
        let v_shifted = _mm512_srli_epi32(v_i, 16); // Descarta a parte menos importante (precisão extra).
        let packed = _mm512_cvtepi32_epi16(v_shifted); // Compacta para 16 bits cada.
        _mm256_storeu_si256(dest.as_mut_ptr().add(i) as *mut __m256i, packed); // Salva 16 números bf16.
        i += 16;
    }
    // Converte o resto manualmente.
    while i < n {
        *dest.get_unchecked_mut(i) = (*src.get_unchecked(i)).to_bits() as u16;
        i += 1;
    }
}

// Re-export das structs de implementação (movidas para common/)
pub use crate::math::common::avx512_impl::{Avx512Math, Avx512VnniBf16Math, Avx512VnniMath};

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

/// Soma horizontal de um buffer f32 via AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn horizontal_sum_avx512(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();

    while i + 16 <= len {
        let v = _mm512_loadu_ps(ptr.add(i));
        sum_v = _mm512_add_ps(sum_v, v);
        i += 16;
    }

    let mut total = crate::math::common::utility::hsum_avx512(sum_v);

    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let v = _mm512_maskz_loadu_ps(mask, ptr.add(i));
        total += crate::math::common::utility::hsum_avx512(v);
    }

    total
}

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

#[cfg(test)]
#[path = "avx512_test.rs"]
mod avx512_test;
