// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Implementações AVX2 da trait `SimdMath`.
//!
//! Este módulo contém as structs `Avx2Math` e `Avx2VnniMath` (type alias)
//! que implementam a trait `SimdMath` usando instruções AVX2/FMA.
//! Os métodos delegam para funções-kernel em `math::gemm`, `math::wavenet`,
//! `math::lstm`, `math::dsp` e `math::common::utility`.

use crate::math::common::scalar_ref::*;
use crate::math::common::traits::SimdMath;
use core::arch::x86_64::*;

/// Implementação concreta da trait SimdMath para processadores com suporte a AVX2 e FMA.
///
/// Aqui é onde "ligamos os fios": conectamos as operações matemáticas abstratas do sistema
/// às funções ultra-rápidas que documentamos acima. Esta estrutura garante que o NAM-rs
/// aproveite a força total do hardware moderno para processar áudio em tempo real.
pub struct Avx2Math;

impl SimdMath for Avx2Math {
    type V = __m256;

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        // Usa a função otimizada para AVX2 que criamos acima.
        unsafe { super::super::gemm::dot::dot_product_avx2(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        // Como o AVX2 puro não tem aceleração nativa para BF16, usamos a versão comum.
        unsafe { dot_product_bf16_fallback(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { super::super::gemm::dot_4x::dot_product_4x_interleaved_avx2(weights, state) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_bf16_fallback(weights, state) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]) {
        unsafe {
            super::super::gemm::dot_4x::dot_product_4x_interleaved_dual_frame_avx2(
                weights, state_f0, state_f1,
            )
        }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
        weights: &[[u16; 4]],
        state_f0: &[u16],
        state_f1: &[u16],
    ) -> ([f32; 4], [f32; 4]) {
        unsafe { dot_product_4x_interleaved_dual_frame_bf16_fallback(weights, state_f0, state_f1) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe { dot_product_bf16_4x_fallback(w0, w1, w2, w3, in_frame) }
    }

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe {
            super::super::gemm::gemv::fused_add_gemv_avx2(
                in_frame, weights, bias, out_frame, do_bias,
            )
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
            super::super::gemm::gemm_batch::fused_add_gemm_batch_avx2(
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
            super::super::gemm::gemm_batch::fused_gemm_residual_batch_avx2(
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
        unsafe {
            super::super::gemm::gemv::gemv_overwrite_avx2(
                in_frame, weights, bias, out_frame, do_bias,
            )
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
        unsafe { gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias) }
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
        let ih = in_frame.len();
        let stride = ih * hidden_size;
        unsafe {
            super::super::gemm::gemv_4gate::gemv_4gate_avx2(
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
            gemv_4gate_bf16_fallback(
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
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        unsafe { super::super::wavenet::accumulate::accumulate_head_avx2(dest, src) }
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        unsafe {
            super::super::wavenet::accumulate::tanh_and_accumulate_block_avx2(head_input, block)
        }
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        unsafe {
            super::super::wavenet::accumulate::gated_activation_and_accumulate_block_avx2(
                head_input, block, ch,
            )
        }
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        unsafe { f32_to_bf16_fallback(src, dest) }
    }

    #[inline(always)]
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
        unsafe {
            // Converte e armazena dados BF16 usando truques de bits em AVX2.
            let v_i = _mm256_castps_si256(v);
            let v_shifted = _mm256_srli_epi32(v_i, 16);
            let packed = _mm256_packus_epi32(v_shifted, v_shifted);
            let permuted = _mm256_permute4x64_epi64(packed, 8);
            let v_low = _mm256_castsi256_si128(permuted);
            _mm_storeu_si128(ptr as *mut __m128i, v_low);
        }
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::activations::tanh_slice_avx2(slice) }
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::activations::sigmoid_slice_avx2(slice) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { super::utility::horizontal_sum_avx2(ptr, N) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { crate::math::activations::tanh_slice_avx2(buf) }
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        unsafe {
            super::super::lstm::fused_lstm_gates_dyn_avx2(
                gates,
                cell_state,
                hidden_state,
                hidden_size,
            )
        }
    }

    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        unsafe { super::super::dsp::stereo::compute_energy_stereo_avx2(l, r) }
    }

    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { super::super::dsp::stereo::convolve_stereo_avx2(coeffs, input_l, input_r, taps) }
    }

    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        unsafe {
            super::super::dsp::gain::apply_gain_and_detect_clipping_stereo_avx2(left, right, gain)
        }
    }

    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        unsafe { super::super::dsp::gain::apply_gain_stereo_avx2(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        unsafe { super::super::dsp::gain::apply_gain_avx2(data, gain) }
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe {
            super::super::wavenet::head::batch_wavenet_head_sum_avx2::<HEAD>(
                head1, head2, output, scale,
            )
        }
    }

    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        unsafe { super::super::dsp::gain::apply_ramp_stereo_avx2(left, right, start, step) }
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
        let in_len = in_frames.len() / num_frames;
        let out_len = out_frames.len() / num_frames;
        for i in 0..num_frames {
            let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
            let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
            unsafe {
                super::super::gemm::gemv::gemv_overwrite_avx2(
                    in_slice, weights, bias, out_slice, do_bias,
                )
            };
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
            unsafe { gemv_overwrite_bf16_fallback(in_slice, weights, bias, out_slice, do_bias) };
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
            super::super::wavenet::head::batch_wavenet_head_sum_dyn_avx2(
                head1, head2, output, head, scale,
            )
        }
    }
}

/// AVX2 + VNNI: a instrução `VPDPBUSD` opera sobre inteiros de 8 bits,
/// sem benefício mensurável para kernels float do NAM-rs.
/// Delegação total para `Avx2Math` — type alias elimina ~300 linhas mortas.
///
/// Mantido como alias (não removido) para preservar compatibilidade com
/// `InstructionSet::Avx2Vnni` e o macro `dispatch_simd!`.
/// Futuro: remover também do enum quando a v-table for unificada.
pub type Avx2VnniMath = Avx2Math;
