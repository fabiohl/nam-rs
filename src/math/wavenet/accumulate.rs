// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels de acúmulo e ativação para WaveNet — AVX2, AVX-512 e fallback escalar.
//!
//! Extraídos de `simd/avx2.rs`, `simd/avx512.rs` e `common/scalar_ref.rs`
//! durante a Tarefa 3.4.

use core::arch::x86_64::*;

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

/// Aplica a ativação das "portas" (tanh * sigmoid) em blocos de áudio.
/// Imagine que cada som passa por dois filtros: um que molda o timbre (tanh)
/// e outro que controla a intensidade (sigmoid). O resultado é somado ao "head_input".
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gated_activation_and_accumulate_block_avx512(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        // Processa 16 canais de uma vez.
        while c + 16 <= ch {
            let z1 = _mm512_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm512_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            // Aplica as funções matemáticas complexas de forma ultra rápida.
            let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_avx512(z1, z2);
            let activated = _mm512_mul_ps(tanh_z1, sig_z2);

            _mm512_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);

            let vh = _mm512_loadu_ps(head_input.as_ptr().add(head_offset + c));
            _mm512_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                _mm512_add_ps(vh, activated),
            );
            c += 16;
        }
        // Sobrou algum canal? Faz um por um.
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

/// Acumulação da "Cabeça" (Head) da rede.
/// Nas arquiteturas tipo WaveNet, as várias camadas (blocos) da rede contribuem
/// para um resultado comum chamado "head". Esta função apenas soma essas contribuições.
pub unsafe fn accumulate_head_fallback(dest: &mut [f32], src: &[f32]) {
    let len = core::cmp::min(dest.len(), src.len());
    for i in 0..len {
        unsafe {
            // Soma o conteúdo de 'src' (origem) no 'dest' (destino).
            *dest.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
    }
}

/// Aplica a ativação 'Tanh' e já acumula na saída principal.
/// Tanh (Tangente Hiperbólica) é uma função que "esmaga" qualquer número para
/// ficar entre -1.0 e 1.0. É muito comum em modelagem de amplificadores de guitarra.
pub unsafe fn tanh_and_accumulate_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = v.tanh(); // Aplica o "esmagamento".
        block[i] = activated; // Guarda o valor esmagado no bloco.
        head_input[i] += activated; // Soma o mesmo valor no acumulador da "cabeça".
    }
}

/// Ativação Gated (Com "Portão") + Acumulação.
/// Esta técnica usa dois sinais (z1 e z2):
/// 1. z1 passa por um Tanh (contém a "informação").
/// 2. z2 passa por um Sigmoid (serve como um "volume" ou "portão" para o z1).
///
/// No final, multiplicamos os dois. É como se z2 decidisse quanto de z1 vai passar.
pub unsafe fn gated_activation_and_accumulate_block_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize, // Número de canais.
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        for c in 0..ch {
            // O bloco contém z1 e z2 um do lado do outro.
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];

            // Ativação complexa: tanh(z1) * sigmoid(z2).
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));

            block[block_offset + c] = activated;
            head_input[head_offset + c] += activated;
        }
    }
}
