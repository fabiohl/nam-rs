// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Backend de processamento ultra-rápido usando AVX2.
//!
//! Kernels não-GEMM: wavenet (head/accumulate/gated), LSTM gates, convolução estéreo,
//! ganho e detecção de clipping.
//! Kernels de álgebra linear (dot, gemv, gemm) foram movidos para `math::gemm/`.

use core::arch::x86_64::*;

// Re-exports de kernels GEMM (Tarefa 3.2 — mantém compatibilidade de paths explícitos)
pub use crate::math::gemm::dot::dot_product_avx2;
pub use crate::math::gemm::dot_4x::{
    dot_product_4x_avx2, dot_product_4x_interleaved_avx2,
    dot_product_4x_interleaved_dual_frame_avx2, dot_product_batch_4x_avx2,
};
pub use crate::math::gemm::gemm_batch::{
    fused_add_gemm_batch_avx2, fused_gemm_residual_batch_avx2,
};
pub use crate::math::gemm::gemv::{fused_add_gemv_avx2, gemv_overwrite_avx2};
pub use crate::math::gemm::gemv_4gate::gemv_4gate_avx2;

// Re-exports de kernels LSTM (Tarefa 3.3 — mantém compatibilidade de paths explícitos)
pub use crate::math::lstm::fused_lstm_gates_dyn_avx2;

// Re-export das structs de implementação (movidas para common/)
pub use crate::math::common::avx2_impl::{Avx2Math, Avx2VnniMath};

/// Acumula src em dest usando AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn accumulate_head_avx2(dest: &mut [f32], src: &[f32]) {
    let len = dest.len();
    let mut i = 0;
    while i + 8 <= len {
        let vs = _mm256_loadu_ps(src.as_ptr().add(i));
        let vd = _mm256_loadu_ps(dest.as_ptr().add(i));
        _mm256_storeu_ps(dest.as_mut_ptr().add(i), _mm256_add_ps(vd, vs));
        i += 8;
    }
    while i < len {
        dest[i] += src[i];
        i += 1;
    }
}

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

/// Aplica tanh in-place em block e acumula em head_input usando AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_accumulate_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vh = _mm256_loadu_ps(head_input.as_ptr().add(i));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vh, vt));
        i += 8;
    }
    while i < len {
        let val = block[i].tanh();
        block[i] = val;
        head_input[i] += val;
        i += 1;
    }
}

/// Aplica gated activation (tanh * sigmoid) in-place em block e acumula em head_input usando AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gated_activation_and_accumulate_block_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 8 <= ch {
            let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_avx2(z1, z2);
            let activated = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);

            let vh = _mm256_loadu_ps(head_input.as_ptr().add(head_offset + c));
            _mm256_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                _mm256_add_ps(vh, activated),
            );
            c += 8;
        }
        while c < ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            head_input[head_offset + c] += activated;
            c += 1;
        }
    }
}

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

/// Soma horizontal de um buffer f32 via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn horizontal_sum_avx2(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm256_setzero_ps();

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        sum_v = _mm256_add_ps(sum_v, v);
        i += 8;
    }

    let mut total = crate::math::common::utility::hsum_avx2(sum_v);

    while i < len {
        total += *ptr.add(i);
        i += 1;
    }

    total
}

/// Soma em lote (batch) das projeções Head do WaveNet usando AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn batch_wavenet_head_sum_avx2<const HEAD: usize>(
    head1: &[f32],
    head2: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let num_frames = output.len();
    for i in 0..num_frames {
        let ptr = head1.as_ptr().add(i * HEAD);
        // Soma os canais internos de cada quadro.
        let sum = horizontal_sum_avx2(ptr, HEAD);
        // Adiciona a entrada residual e aplica o volume (scale).
        *output.get_unchecked_mut(i) = (sum + *head2.get_unchecked(i)) * scale;
    }
}

#[cfg(test)]
#[path = "avx2_test.rs"]
mod avx2_test;
