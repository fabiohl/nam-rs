// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Fused kernels for LSTM gates (AVX2 and AVX-512).
//!
//! Extracted from `activations/fused.rs` and `simd/avx2.rs`/`simd/avx512.rs`
//! during Task 3.3.

use crate::math::activations::sigmoid::{
    scalar_minimax_sigmoid, simd_sigmoid_avx2, simd_sigmoid_avx512, simd_sigmoid_dual_avx2,
};
use crate::math::activations::tanh::{scalar_pade_tanh, simd_tanh_avx2, simd_tanh_avx512};
use core::arch::x86_64::*;

/// Fused kernel for LSTM gates (AVX2).
/// Computes:
///   new_cs = sig(gf) * cs + sig(gi) * tanh(gg)
///   hidden = sig(go) * tanh(new_cs)
///
/// # Safety
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_lstm_gates_avx2(
    gf: __m256,
    gi: __m256,
    gg: __m256,
    go: __m256,
    cs: __m256,
) -> (__m256, __m256) {
    // Interleave sigmoids
    let (sig_f, sig_i) = unsafe { simd_sigmoid_dual_avx2(gf, gi) };
    let sig_o = unsafe { simd_sigmoid_avx2(go) };
    let tanh_g = unsafe { simd_tanh_avx2(gg) };

    let new_cs = _mm256_add_ps(_mm256_mul_ps(sig_f, cs), _mm256_mul_ps(sig_i, tanh_g));
    let hidden = _mm256_mul_ps(sig_o, unsafe { simd_tanh_avx2(new_cs) });

    (new_cs, hidden)
}

/// Fused kernel for LSTM gates (AVX-512).
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[inline]
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

/// Fused kernel for dynamic LSTM gate processing via AVX2.
#[inline]
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
        let sig_i = scalar_minimax_sigmoid(gates[j]);
        let sig_f = scalar_minimax_sigmoid(gates[j + hidden_size]);
        let tanh_g = scalar_pade_tanh(gates[j + 2 * hidden_size]);
        let sig_o = scalar_minimax_sigmoid(gates[j + 3 * hidden_size]);

        let new_cs = sig_f * cell_state[j] + sig_i * tanh_g;
        cell_state[j] = new_cs;
        hidden_state[j] = sig_o * scalar_pade_tanh(new_cs);
        j += 1;
    }
}

/// Fused kernel to update the memory (state) of an LSTM network.
/// This function decides what the network should "forget" from the past and what to "learn" from the present,
/// updating the values all at once for 16 memory cells.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_dyn_avx512(
    gates: &mut [f32],
    cell_state: &mut [f32],
    hidden_state: &mut [f32],
    hidden_size: usize,
) {
    let mut j = 0;
    while j + 16 <= hidden_size {
        // Load the 4 decisions (forget, learn, etc.) for 16 cells.
        let gi = _mm512_loadu_ps(gates.as_ptr().add(j));
        let gf = _mm512_loadu_ps(gates.as_ptr().add(j + hidden_size));
        let gg = _mm512_loadu_ps(gates.as_ptr().add(j + 2 * hidden_size));
        let go = _mm512_loadu_ps(gates.as_ptr().add(j + 3 * hidden_size));
        let cs = _mm512_loadu_ps(cell_state.as_ptr().add(j));

        // Perform the memory computation in a fused manner.
        let (new_cs, hidden) = fused_lstm_gates_avx512(gf, gi, gg, go, cs);

        _mm512_storeu_ps(cell_state.as_mut_ptr().add(j), new_cs);
        _mm512_storeu_ps(hidden_state.as_mut_ptr().add(j), hidden);

        j += 16;
    }
    // Handle the remainder.
    while j < hidden_size {
        let sig_i = scalar_minimax_sigmoid(gates[j]);
        let sig_f = scalar_minimax_sigmoid(gates[j + hidden_size]);
        let tanh_g = scalar_pade_tanh(gates[j + 2 * hidden_size]);
        let sig_o = scalar_minimax_sigmoid(gates[j + 3 * hidden_size]);

        let new_cs = sig_f * cell_state[j] + sig_i * tanh_g;
        cell_state[j] = new_cs;
        hidden_state[j] = sig_o * scalar_pade_tanh(new_cs);
        j += 1;
    }
}
