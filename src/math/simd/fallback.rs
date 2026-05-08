// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! [T21] Implementações de fallback escalar para kernels SIMD.
//!
//! Estas funções são usadas quando o hardware não suporta extensões SIMD específicas
//! ou para validação cruzada (Golden Vectors).

use super::traits::SimdMath;

/// Fallback escalar para dot product f32 com pesos u16.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn dot_product_fallback(a: &[f32], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        unsafe {
            let fb = half::f16::from_bits(*b.get_unchecked(i)).to_f32();
            sum += *a.get_unchecked(i) * fb;
        }
    }
    sum
}

/// Fallback escalar para dot product BF16 com pesos u16.
///
/// # Safety
/// Os slices `a` e `b` devem ser válidos. Usa `get_unchecked` para acesso
/// sem verificação de limites no loop interno.
pub unsafe fn dot_product_bf16_fallback(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        unsafe {
            let fa = f32::from_bits((*a.get_unchecked(i) as u32) << 16);
            let fb = f32::from_bits((*b.get_unchecked(i) as u32) << 16);
            sum += fa * fb;
        }
    }
    sum
}

/// Fallback para dot product interleaved com entrada f32.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn dot_product_4x_interleaved_fallback(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];
    for i in 0..len {
        unsafe {
            let s = *state.get_unchecked(i);
            let w = weights.get_unchecked(i);
            sum[0] += half::f16::from_bits(w[0]).to_f32() * s;
            sum[1] += half::f16::from_bits(w[1]).to_f32() * s;
            sum[2] += half::f16::from_bits(w[2]).to_f32() * s;
            sum[3] += half::f16::from_bits(w[3]).to_f32() * s;
        }
    }
    sum
}

/// Fallback para dot product interleaved BF16.
///
/// # Safety
/// `weights` e `state` devem conter dados BF16 válidos e ter tamanhos consistentes.
pub unsafe fn dot_product_4x_interleaved_bf16_fallback(
    weights: &[[u16; 4]],
    state: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];
    for i in 0..len {
        unsafe {
            let s = f32::from_bits((*state.get_unchecked(i) as u32) << 16);
            let w = weights.get_unchecked(i);
            sum[0] += f32::from_bits((w[0] as u32) << 16) * s;
            sum[1] += f32::from_bits((w[1] as u32) << 16) * s;
            sum[2] += f32::from_bits((w[2] as u32) << 16) * s;
            sum[3] += f32::from_bits((w[3] as u32) << 16) * s;
        }
    }
    sum
}

/// Fallback para dot product interleaved com entrada f32 para dual frames.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn dot_product_4x_interleaved_dual_frame_fallback(
    weights: &[[u16; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut sum_f0 = [0.0f32; 4];
    let mut sum_f1 = [0.0f32; 4];
    for i in 0..len {
        unsafe {
            let s0 = *state_f0.get_unchecked(i);
            let s1 = *state_f1.get_unchecked(i);
            let w = weights.get_unchecked(i);

            let w0 = half::f16::from_bits(w[0]).to_f32();
            let w1 = half::f16::from_bits(w[1]).to_f32();
            let w2 = half::f16::from_bits(w[2]).to_f32();
            let w3 = half::f16::from_bits(w[3]).to_f32();

            sum_f0[0] += w0 * s0;
            sum_f0[1] += w1 * s0;
            sum_f0[2] += w2 * s0;
            sum_f0[3] += w3 * s0;

            sum_f1[0] += w0 * s1;
            sum_f1[1] += w1 * s1;
            sum_f1[2] += w2 * s1;
            sum_f1[3] += w3 * s1;
        }
    }
    (sum_f0, sum_f1)
}

/// Fallback para dot product interleaved BF16 para dual frames.
///
/// # Safety
/// `weights`, `state_f0` e `state_f1` devem conter dados válidos.
pub unsafe fn dot_product_4x_interleaved_dual_frame_bf16_fallback(
    weights: &[[u16; 4]],
    state_f0: &[u16],
    state_f1: &[u16],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut sum_f0 = [0.0f32; 4];
    let mut sum_f1 = [0.0f32; 4];
    for i in 0..len {
        unsafe {
            let s0 = f32::from_bits((*state_f0.get_unchecked(i) as u32) << 16);
            let s1 = f32::from_bits((*state_f1.get_unchecked(i) as u32) << 16);
            let w = weights.get_unchecked(i);

            let w0 = f32::from_bits((w[0] as u32) << 16);
            let w1 = f32::from_bits((w[1] as u32) << 16);
            let w2 = f32::from_bits((w[2] as u32) << 16);
            let w3 = f32::from_bits((w[3] as u32) << 16);

            sum_f0[0] += w0 * s0;
            sum_f0[1] += w1 * s0;
            sum_f0[2] += w2 * s0;
            sum_f0[3] += w3 * s0;

            sum_f1[0] += w0 * s1;
            sum_f1[1] += w1 * s1;
            sum_f1[2] += w2 * s1;
            sum_f1[3] += w3 * s1;
        }
    }
    (sum_f0, sum_f1)
}

/// Fallback para dot product BF16 em batch de 4.
///
/// # Safety
/// Todos os slices devem ser válidos e conter dados BF16. Delega para
/// `dot_product_bf16_fallback` que usa `get_unchecked`.
pub unsafe fn dot_product_bf16_4x_fallback(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    in_frame: &[u16],
) -> [f32; 4] {
    unsafe {
        [
            dot_product_bf16_fallback(in_frame, w0),
            dot_product_bf16_fallback(in_frame, w1),
            dot_product_bf16_fallback(in_frame, w2),
            dot_product_bf16_fallback(in_frame, w3),
        ]
    }
}

/// Fallback escalar para operação GEMM em batch.
///
/// # Safety
/// O chamador deve garantir que os buffers de entrada e saída tenham tamanhos compatíveis.
pub unsafe fn fused_add_gemm_batch_fallback(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    for f in 0..num_frames {
        unsafe {
            fused_add_gemv_fallback(
                in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
                weights,
                bias,
                out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
                do_bias,
            );
        }
    }
}

/// Fallback escalar para GEMV fundido (Acumula).
///
/// # Safety
/// Buffers de entrada e saída devem ser válidos.
pub unsafe fn fused_add_gemv_fallback(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            unsafe {
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) += sum;
        }
    }
}

/// Fallback escalar para GEMV com sobrescrita.
///
/// # Safety
/// Buffers de entrada e saída devem ser válidos.
pub unsafe fn gemv_overwrite_fallback(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            unsafe {
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Fallback escalar para GEMV residual em batch.
///
/// # Safety
/// Buffers de entrada e saída devem ser válidos.
pub unsafe fn fused_gemm_residual_batch_fallback(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    for frame_idx in 0..num_frames {
        unsafe {
            for (out_c, &b) in bias.iter().enumerate().take(out_len) {
                let mut sum = if do_bias { b } else { 0.0 };
                for in_c in 0..in_len {
                    let w_bits = *weights.get_unchecked(in_c * out_len + out_c);
                    let w = half::f16::from_bits(w_bits).to_f32();
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) =
                    sum + *residual.get_unchecked(frame_idx * out_len + out_c);
            }
        }
    }
}

/// Fallback escalar para GEMV com sobrescrita (entrada BF16).
///
/// # Safety
/// Buffers de entrada e saída devem ser válidos.
pub unsafe fn gemv_overwrite_bf16_fallback(
    in_frame: &[u16],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            unsafe {
                let s = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
                sum += s * f32::from_bits(
                    (*weights.get_unchecked(in_c * out_len + out_c) as u32) << 16,
                );
            }
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Fallback escalar para acumulação de cabeça.
///
/// # Safety
/// Buffers de entrada e saída devem ser válidos.
pub unsafe fn accumulate_head_fallback(dest: &mut [f32], src: &[f32]) {
    let len = core::cmp::min(dest.len(), src.len());
    for i in 0..len {
        unsafe {
            *dest.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
    }
}

/// Fallback escalar para Tanh + Head Accumulate.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn tanh_and_accumulate_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = v.tanh();
        block[i] = activated;
        head_input[i] += activated;
    }
}

/// Fallback escalar para Gated Activation + Head Accumulate.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn gated_activation_and_accumulate_block_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        for c in 0..ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp())); // tanh * sigmoid
            block[block_offset + c] = activated;
            head_input[head_offset + c] += activated;
        }
    }
}

/// Fallback escalar para conversão f32 -> bf16.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn f32_to_bf16_fallback(src: &[f32], dest: &mut [u16]) {
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        *d = (s.to_bits() >> 16) as u16;
    }
}

/// Fallback escalar para Tanh slice.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn tanh_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = v.tanh();
    }
}

/// Fallback escalar para Sigmoid slice.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn sigmoid_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Fallback escalar para soma horizontal.
///
/// # Safety
/// Buffers devem ser válidos.
pub unsafe fn horizontal_sum_fallback(ptr: *const f32, len: usize) -> f32 {
    let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
    slice.iter().sum()
}

/// Estrutura para despacho via trait para o backend de fallback.
pub struct FallbackMath;

impl SimdMath for FallbackMath {
    type V = f32; // Não utilizado no fallback

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { dot_product_fallback(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { dot_product_bf16_fallback(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_fallback(weights, state) }
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
        unsafe { dot_product_4x_interleaved_dual_frame_fallback(weights, state_f0, state_f1) }
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
        unsafe { fused_add_gemv_fallback(in_frame, weights, bias, out_frame, do_bias) }
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
            fused_add_gemm_batch_fallback(in_frames, weights, bias, out_frames, num_frames, do_bias)
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
            fused_gemm_residual_batch_fallback(
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
        unsafe { gemv_overwrite_fallback(in_frame, weights, bias, out_frame, do_bias) }
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
            gemv_4gate_fallback(
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
        unsafe { accumulate_head_fallback(dest, src) }
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        unsafe { tanh_and_accumulate_block_fallback(head_input, block) }
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        unsafe { gated_activation_and_accumulate_block_fallback(head_input, block, ch) }
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        unsafe { f32_to_bf16_fallback(src, dest) }
    }

    #[inline(always)]
    unsafe fn store_bf16(ptr: *mut u16, _v: Self::V) {
        // No fallback, store_bf16 is intended for SIMD registers
        // but for completeness we can implement for V=f32 if it was used.
        // Since FallbackMath::V = f32, we store one value.
        *ptr = (_v.to_bits() >> 16) as u16;
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { tanh_slice_fallback(slice) }
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { sigmoid_slice_fallback(slice) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { horizontal_sum_fallback(ptr, N) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { tanh_slice_fallback(buf) }
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        for j in 0..hidden_size {
            let sig_i = 1.0 / (1.0 + (-gates[j]).exp());
            let sig_f = 1.0 / (1.0 + (-gates[j + hidden_size]).exp());
            let tanh_g = gates[j + 2 * hidden_size].tanh();
            let sig_o = 1.0 / (1.0 + (-gates[j + 3 * hidden_size]).exp());

            let new_cs = sig_f * cell_state[j] + sig_i * tanh_g;
            cell_state[j] = new_cs;
            hidden_state[j] = sig_o * new_cs.tanh();
        }
    }
}

/// Fallback escalar para GEMV de 4 gates (LSTM).
/// Kernel GEMV 4-gate fallback para LSTM.
///
/// # Safety
/// Buffers devem ser válidos.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_fallback(
    in_frame: &[f32],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    unsafe {
        gemv_overwrite_fallback(
            in_frame,
            w0,
            &bias[0..out_len],
            &mut out[0..out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w1,
            &bias[out_len..2 * out_len],
            &mut out[out_len..2 * out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w2,
            &bias[2 * out_len..3 * out_len],
            &mut out[2 * out_len..3 * out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w3,
            &bias[3 * out_len..4 * out_len],
            &mut out[3 * out_len..4 * out_len],
            do_bias,
        );
    }
}

/// Fallback escalar para GEMV de 4 gates (LSTM) com entrada BF16.
/// Kernel GEMV 4-gate BF16 fallback para LSTM.
///
/// # Safety
/// Buffers devem ser válidos.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_bf16_fallback(
    in_frame: &[u16],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    unsafe {
        gemv_overwrite_bf16_fallback(
            in_frame,
            w0,
            &bias[0..out_len],
            &mut out[0..out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w1,
            &bias[out_len..2 * out_len],
            &mut out[out_len..2 * out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w2,
            &bias[2 * out_len..3 * out_len],
            &mut out[2 * out_len..3 * out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w3,
            &bias[3 * out_len..4 * out_len],
            &mut out[3 * out_len..4 * out_len],
            do_bias,
        );
    }
}

#[cfg(test)]
#[path = "fallback_test.rs"]
mod fallback_test;
