// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels fundidos para portas LSTM (AVX2 e AVX-512).
//!
//! Extraídos de `activations/fused.rs` e `simd/avx2.rs`/`simd/avx512.rs`
//! durante a Tarefa 3.3.

use crate::math::activations::sigmoid::{
    simd_sigmoid_avx2, simd_sigmoid_avx512, simd_sigmoid_dual_avx2,
};
use crate::math::activations::tanh::{simd_tanh_avx2, simd_tanh_avx512};
use core::arch::x86_64::*;

/// Kernel fundido para portas LSTM (AVX2).
/// Computa:
///   new_cs = sig(gf) * cs + sig(gi) * tanh(gg)
///   hidden = sig(go) * tanh(new_cs)
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_lstm_gates_avx2(
    gf: __m256,
    gi: __m256,
    gg: __m256,
    go: __m256,
    cs: __m256,
) -> (__m256, __m256) {
    // Intercala sigmoides
    let (sig_f, sig_i) = unsafe { simd_sigmoid_dual_avx2(gf, gi) };
    let sig_o = unsafe { simd_sigmoid_avx2(go) };
    let tanh_g = unsafe { simd_tanh_avx2(gg) };

    let new_cs = _mm256_add_ps(_mm256_mul_ps(sig_f, cs), _mm256_mul_ps(sig_i, tanh_g));
    let hidden = _mm256_mul_ps(sig_o, unsafe { simd_tanh_avx2(new_cs) });

    (new_cs, hidden)
}

/// Kernel fundido para portas LSTM (AVX-512).
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_avx512(
    gf: __m512,
    gi: __m512,
    gg: __m512,
    go: __m512,
    cs: __m512,
) -> (__m512, __m512) {
    let sig_f = unsafe { simd_sigmoid_avx512(gf) };
    let sig_i = unsafe { simd_sigmoid_avx512(gi) };
    let sig_o = unsafe { simd_sigmoid_avx512(go) };
    let tanh_g = unsafe { simd_tanh_avx512(gg) };

    let new_cs = _mm512_add_ps(_mm512_mul_ps(sig_f, cs), _mm512_mul_ps(sig_i, tanh_g));
    let hidden = _mm512_mul_ps(sig_o, unsafe { simd_tanh_avx512(new_cs) });

    (new_cs, hidden)
}

/// Kernel fundido para processamento de portas LSTM dinâmicas via AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_lstm_gates_dyn_avx2(
    gates: &mut [f32],
    cell_state: &mut [f32],
    hidden_state: &mut [f32],
    hidden_size: usize,
) {
    let mut j = 0;
    while j + 8 <= hidden_size {
        let gi = _mm256_loadu_ps(gates.as_ptr().add(j));
        let gf = _mm256_loadu_ps(gates.as_ptr().add(j + hidden_size));
        let gg = _mm256_loadu_ps(gates.as_ptr().add(j + 2 * hidden_size));
        let go = _mm256_loadu_ps(gates.as_ptr().add(j + 3 * hidden_size));
        let cs = _mm256_loadu_ps(cell_state.as_ptr().add(j));

        let (new_cs, hidden) = fused_lstm_gates_avx2(gf, gi, gg, go, cs);

        _mm256_storeu_ps(cell_state.as_mut_ptr().add(j), new_cs);
        _mm256_storeu_ps(hidden_state.as_mut_ptr().add(j), hidden);

        j += 8;
    }
    while j < hidden_size {
        let sig_i = 1.0 / (1.0 + (-gates[j]).exp());
        let sig_f = 1.0 / (1.0 + (-gates[j + hidden_size]).exp());
        let tanh_g = gates[j + 2 * hidden_size].tanh();
        let sig_o = 1.0 / (1.0 + (-gates[j + 3 * hidden_size]).exp());

        let new_cs = sig_f * cell_state[j] + sig_i * tanh_g;
        cell_state[j] = new_cs;
        hidden_state[j] = sig_o * new_cs.tanh();
        j += 1;
    }
}

/// Kernel fundido para atualizar a memória (estado) de uma rede LSTM.
/// Esta função decide o que a rede deve "esquecer" do passado e o que "aprender" do presente,
/// atualizando os valores de uma só vez para 16 células de memória.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_dyn_avx512(
    gates: &mut [f32],
    cell_state: &mut [f32],
    hidden_state: &mut [f32],
    hidden_size: usize,
) {
    let mut j = 0;
    while j + 16 <= hidden_size {
        // Carrega as 4 decisões (esquecer, aprender, etc.) para 16 células.
        let gi = _mm512_loadu_ps(gates.as_ptr().add(j));
        let gf = _mm512_loadu_ps(gates.as_ptr().add(j + hidden_size));
        let gg = _mm512_loadu_ps(gates.as_ptr().add(j + 2 * hidden_size));
        let go = _mm512_loadu_ps(gates.as_ptr().add(j + 3 * hidden_size));
        let cs = _mm512_loadu_ps(cell_state.as_ptr().add(j));

        // Faz o cálculo da memória de forma fundida (fused).
        let (new_cs, hidden) = fused_lstm_gates_avx512(gf, gi, gg, go, cs);

        _mm512_storeu_ps(cell_state.as_mut_ptr().add(j), new_cs);
        _mm512_storeu_ps(hidden_state.as_mut_ptr().add(j), hidden);

        j += 16;
    }
    // Trata o resto.
    while j < hidden_size {
        let sig_i = 1.0 / (1.0 + (-gates[j]).exp());
        let sig_f = 1.0 / (1.0 + (-gates[j + hidden_size]).exp());
        let tanh_g = gates[j + 2 * hidden_size].tanh();
        let sig_o = 1.0 / (1.0 + (-gates[j + 3 * hidden_size]).exp());

        let new_cs = sig_f * cell_state[j] + sig_i * tanh_g;
        cell_state[j] = new_cs;
        hidden_state[j] = sig_o * new_cs.tanh();
        j += 1;
    }
}
