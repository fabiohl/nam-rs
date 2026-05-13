// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Implementações AVX-512 da trait `SimdMath`.
//!
//! Contém `Avx512Math`, `Avx512VnniMath` e `Avx512VnniBf16Math`.
//! `Avx512VnniMath` tem implementações reais (BF16 dot product nativo via `_mm512_dpbf16_ps`).
//! Os métodos delegam para funções-kernel em `math::gemm`, `math::wavenet`,
//! `math::lstm`, `math::dsp`, `math::common::ops` e `math::common::utility`.

use crate::math::common::scalar_ref::*;
use crate::math::common::traits::SimdMath;
use core::arch::x86_64::*;

/// Implementação SIMD via AVX-512.
/// Esta struct agrupa todas as funções matemáticas otimizadas para processadores que suportam AVX-512.
pub struct Avx512Math;

impl SimdMath for Avx512Math {
    type V = __m512; // O tipo de dado "vetor" usado aqui tem 512 bits (16 números f32).

    // Dot Product: Multiplica dois conjuntos de números e soma tudo num resultado só.
    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        super::super::gemm::dot::dot_product_avx512(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        dot_product_bf16_fallback(a, b)
    }

    // Versões intercaladas para processar 4 cálculos independentes ao mesmo tempo.
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        super::super::gemm::dot_4x::dot_product_4x_interleaved_avx512(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        dot_product_4x_interleaved_bf16_fallback(weights, state)
    }

    // Processa dois quadros de áudio simultaneamente.
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]) {
        super::super::gemm::dot_4x::dot_product_4x_interleaved_dual_frame_avx512(
            weights, state_f0, state_f1,
        )
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
        weights: &[[u16; 4]],
        state_f0: &[u16],
        state_f1: &[u16],
    ) -> ([f32; 4], [f32; 4]) {
        dot_product_4x_interleaved_dual_frame_bf16_fallback(weights, state_f0, state_f1)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        dot_product_bf16_4x_fallback(w0, w1, w2, w3, in_frame)
    }

    // Funções de matriz-vetor (GEMV) que vimos anteriormente.
    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        if out_frame.len() == 16 {
            super::super::gemm::gemv::fused_add_gemv_avx512_small(
                in_frame, weights, bias, out_frame, do_bias,
            )
        } else {
            fused_add_gemv_fallback(in_frame, weights, bias, out_frame, do_bias)
        }
    }

    #[inline(always)]
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        unsafe {
            super::super::gemm::gemm_batch::fused_add_gemm_batch_avx512(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        unsafe {
            super::super::gemm::gemm_batch::fused_gemm_residual_batch_avx512(
                in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        if out_frame.len() == 16 {
            super::super::gemm::gemv::gemv_overwrite_avx512_small(
                in_frame, weights, bias, out_frame, do_bias,
            )
        } else {
            gemv_overwrite_fallback(in_frame, weights, bias, out_frame, do_bias)
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias)
    }

    // LSTM Gates: Uma parte fundamental de modelos de memória (LSTM).
    #[inline(always)]
    unsafe fn gemv_overwrite_4gate(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        let ih = in_frame.len();
        let stride = ih * hidden_size;
        unsafe {
            super::super::gemm::gemv_4gate::gemv_4gate_avx512(
                in_frame,
                &weights[0..stride],
                &weights[stride..2 * stride],
                &weights[2 * stride..3 * stride],
                &weights[3 * stride..4 * stride],
                bias,
                out_gates,
                do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16_4gate(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        let ih = in_frame.len();
        let stride = ih * hidden_size;
        unsafe {
            super::super::gemm::gemv_4gate::gemv_4gate_bf16_avx512(
                in_frame,
                &weights[0..stride],
                &weights[stride..2 * stride],
                &weights[2 * stride..3 * stride],
                &weights[3 * stride..4 * stride],
                bias,
                out_gates,
                do_bias,
            )
        }
    }

    // Funções auxiliares para redes neurais (ativar e somar blocos).
    #[inline(always)]
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        accumulate_head_fallback(dest, src)
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        tanh_and_accumulate_block_fallback(head_input, block)
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        unsafe {
            super::super::wavenet::accumulate::gated_activation_and_accumulate_block_avx512(
                head_input, block, ch,
            )
        }
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        unsafe { super::ops::f32_to_bf16_avx512(src, dest) }
    }

    #[inline(always)]
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
        unsafe {
            let v_i = _mm512_castps_si512(v);
            let v_shifted = _mm512_srli_epi32(v_i, 16);
            let packed = _mm512_cvtepi32_epi16(v_shifted);
            _mm256_storeu_si256(ptr as *mut __m256i, packed);
        }
    }

    // Ativações matemáticas rápidas (Tangente Hiperbólica e Sigmóide).
    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        crate::math::activations::tanh_slice_avx512(slice)
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        crate::math::activations::sigmoid_slice_avx512(slice)
    }

    // Soma horizontal: Soma todos os números dentro de um único registrador.
    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { super::utility::horizontal_sum_avx512(ptr, N) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        crate::math::activations::tanh_slice_avx512(buf)
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        unsafe {
            super::super::lstm::fused_lstm_gates_dyn_avx512(
                gates,
                cell_state,
                hidden_state,
                hidden_size,
            )
        }
    }

    // Funções de áudio Stereo (Esquerda e Direita).
    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        unsafe { super::super::dsp::stereo::compute_energy_stereo_avx512(l, r) }
    }

    // Convolve Stereo: Aplica filtros de áudio (como um equalizador ou reverb).
    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { super::super::dsp::stereo::convolve_stereo_avx512(coeffs, input_l, input_r, taps) }
    }

    // Controle de ganho (volume) e detecção de "clipping" (quando o som distorce por ficar alto demais).
    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        unsafe {
            super::super::dsp::gain::apply_gain_and_detect_clipping_stereo_avx512(left, right, gain)
        }
    }

    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        unsafe { super::super::dsp::gain::apply_gain_stereo_avx512(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        unsafe { super::super::dsp::gain::apply_gain_avx512(data, gain) }
    }

    // Head Sum: Uma operação final usada no modelo WaveNet para gerar o som.
    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe {
            super::super::wavenet::head::batch_wavenet_head_sum_avx512::<HEAD>(
                head1, head2, output, scale,
            )
        }
    }

    // Ramp: Aumenta ou diminui o volume gradualmente para evitar estalos (cliques) no áudio.
    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        unsafe { super::super::dsp::gain::apply_ramp_stereo_avx512(left, right, start, step) }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        unsafe {
            super::super::gemm::gemv::gemv_overwrite_batch_avx512(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_batch_bf16(
        in_frames: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        let in_len = in_frames.len() / num_frames;
        let out_len = out_frames.len() / num_frames;
        for i in 0..num_frames {
            let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
            let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
            gemv_overwrite_bf16_fallback(in_slice, weights, bias, out_slice, do_bias);
        }
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum_dyn(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        head: usize,
        scale: f32,
    ) {
        unsafe {
            super::super::wavenet::head::batch_wavenet_head_sum_dyn_avx512(
                head1, head2, output, head, scale,
            )
        }
    }
}

/// Implementação estática para AVX-512 com suporte a VNNI.
pub struct Avx512VnniMath;

impl SimdMath for Avx512VnniMath {
    type V = __m512;

    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        unsafe { Avx512Math::compute_energy_stereo(l, r) }
    }

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        Avx512Math::dot_product(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { super::super::gemm::dot::dot_product_bf16_avx512(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        Avx512Math::dot_product_4x_interleaved(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        Avx512Math::dot_product_4x_interleaved_bf16(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]) {
        Avx512Math::dot_product_4x_interleaved_dual_frame(weights, state_f0, state_f1)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
        weights: &[[u16; 4]],
        state_f0: &[u16],
        state_f1: &[u16],
    ) -> ([f32; 4], [f32; 4]) {
        Avx512Math::dot_product_4x_interleaved_dual_frame_bf16(weights, state_f0, state_f1)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        let mut out = [0.0; 4];
        let bias = [0.0; 4]; // Dummy bias
        unsafe {
            super::super::gemm::gemv_4gate::gemv_4gate_bf16_avx512(
                in_frame, w0, w1, w2, w3, &bias, &mut out, false,
            );
        }
        out
    }

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::fused_add_gemv(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::fused_add_gemm_batch(in_frames, weights, bias, out_frames, num_frames, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::fused_gemm_residual_batch(
            in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
        )
    }

    #[inline(always)]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite_bf16(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_4gate(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        unsafe {
            Avx512Math::gemv_overwrite_4gate(
                in_frame,
                weights,
                bias,
                out_gates,
                hidden_size,
                do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16_4gate(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        unsafe {
            Avx512Math::gemv_overwrite_bf16_4gate(
                in_frame,
                weights,
                bias,
                out_gates,
                hidden_size,
                do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        Avx512Math::accumulate_head(dest, src)
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        Avx512Math::tanh_and_accumulate_block(head_input, block)
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        Avx512Math::gated_activation_and_accumulate_block(head_input, block, ch)
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        Avx512Math::f32_to_bf16(src, dest)
    }

    #[inline(always)]
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
        Avx512Math::store_bf16(ptr, v)
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        Avx512Math::tanh_slice(slice)
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        Avx512Math::sigmoid_slice(slice)
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        super::utility::horizontal_sum_avx512(ptr, N)
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        Avx512Math::activation_tanh_block(buf)
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        Avx512Math::fused_lstm_gates_dyn(gates, cell_state, hidden_state, hidden_size)
    }

    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { Avx512Math::convolve_stereo(coeffs, input_l, input_r, taps) }
    }
    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        Avx512Math::apply_gain_and_detect_clipping_stereo(left, right, gain)
    }

    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        Avx512Math::apply_gain_stereo(left, right, gain)
    }

    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        super::super::dsp::gain::apply_gain_avx512(data, gain)
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe {
            super::super::wavenet::head::batch_wavenet_head_sum_avx512::<HEAD>(
                head1, head2, output, scale,
            )
        }
    }

    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        Avx512Math::apply_ramp_stereo(left, right, start, step)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite_batch(in_frames, weights, bias, out_frames, num_frames, do_bias)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_batch_bf16(
        in_frames: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite_batch_bf16(
            in_frames, weights, bias, out_frames, num_frames, do_bias,
        )
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum_dyn(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        head: usize,
        scale: f32,
    ) {
        Avx512Math::batch_wavenet_head_sum_dyn(head1, head2, output, head, scale)
    }
}

/// Implementação estática para AVX-512 com suporte a VNNI e BF16.
pub struct Avx512VnniBf16Math;

impl SimdMath for Avx512VnniBf16Math {
    type V = __m512;
    const IS_BF16: bool = true;

    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        unsafe { Avx512Math::compute_energy_stereo(l, r) }
    }

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        dot_product_fallback(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { super::super::gemm::dot::dot_product_bf16_avx512(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        dot_product_4x_interleaved_fallback(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        dot_product_4x_interleaved_bf16_fallback(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]) {
        super::super::gemm::dot_4x::dot_product_4x_interleaved_dual_frame_avx512(
            weights, state_f0, state_f1,
        )
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
        weights: &[[u16; 4]],
        state_f0: &[u16],
        state_f1: &[u16],
    ) -> ([f32; 4], [f32; 4]) {
        dot_product_4x_interleaved_dual_frame_bf16_fallback(weights, state_f0, state_f1)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        dot_product_bf16_4x_fallback(w0, w1, w2, w3, in_frame)
    }

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::fused_add_gemv(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::fused_add_gemm_batch(in_frames, weights, bias, out_frames, num_frames, do_bias)
    }

    #[inline(always)]
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::fused_gemm_residual_batch(
            in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
        )
    }

    #[inline(always)]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite_bf16(in_frame, weights, bias, out_frame, do_bias)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_4gate(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        unsafe {
            Avx512Math::gemv_overwrite_4gate(
                in_frame,
                weights,
                bias,
                out_gates,
                hidden_size,
                do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16_4gate(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        unsafe {
            Avx512Math::gemv_overwrite_bf16_4gate(
                in_frame,
                weights,
                bias,
                out_gates,
                hidden_size,
                do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        Avx512Math::accumulate_head(dest, src)
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        Avx512Math::tanh_and_accumulate_block(head_input, block)
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        Avx512Math::gated_activation_and_accumulate_block(head_input, block, ch)
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        Avx512Math::f32_to_bf16(src, dest)
    }

    #[inline(always)]
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
        Avx512Math::store_bf16(ptr, v)
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        Avx512Math::tanh_slice(slice)
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        Avx512Math::sigmoid_slice(slice)
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        super::utility::horizontal_sum_avx512(ptr, N)
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        Avx512Math::activation_tanh_block(buf)
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        Avx512Math::fused_lstm_gates_dyn(gates, cell_state, hidden_state, hidden_size)
    }

    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { Avx512Math::convolve_stereo(coeffs, input_l, input_r, taps) }
    }
    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        Avx512Math::apply_gain_and_detect_clipping_stereo(left, right, gain)
    }

    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        Avx512Math::apply_gain_stereo(left, right, gain)
    }

    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        super::super::dsp::gain::apply_gain_avx512(data, gain)
    }

    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        Avx512Math::apply_ramp_stereo(left, right, start, step)
    }
    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe {
            super::super::wavenet::head::batch_wavenet_head_sum_avx512::<HEAD>(
                head1, head2, output, scale,
            )
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite_batch(in_frames, weights, bias, out_frames, num_frames, do_bias)
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_batch_bf16(
        in_frames: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        Avx512Math::gemv_overwrite_batch_bf16(
            in_frames, weights, bias, out_frames, num_frames, do_bias,
        )
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum_dyn(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        head: usize,
        scale: f32,
    ) {
        Avx512Math::batch_wavenet_head_sum_dyn(head1, head2, output, head, scale)
    }
}
