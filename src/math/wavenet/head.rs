// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels de soma Head (Batch) do WaveNet — AVX2, AVX-512 e dispatch dinâmico.
//!
//! Extraídos de `simd/avx2.rs` e `simd/avx512.rs` durante a Tarefa 3.4.
//! Depende de `common::utility::horizontal_sum_avx2` e `common::utility::horizontal_sum_avx512`
//! para as operações de soma horizontal.

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
        let sum = crate::math::common::utility::horizontal_sum_avx2(ptr, HEAD);
        // Adiciona a entrada residual e aplica o volume (scale).
        *output.get_unchecked_mut(i) = (sum + *head2.get_unchecked(i)) * scale;
    }
}

/// Dispatch dinâmico para `batch_wavenet_head_sum` via AVX2.
/// Despacha para o const generic apropriado (1 ou 16) ou usa fallback AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn batch_wavenet_head_sum_dyn_avx2(
    head1: &[f32],
    head2: &[f32],
    output: &mut [f32],
    head: usize,
    scale: f32,
) {
    match head {
        1 => batch_wavenet_head_sum_avx2::<1>(head1, head2, output, scale),
        16 => batch_wavenet_head_sum_avx2::<16>(head1, head2, output, scale),
        _ => {
            let num_frames = output.len();
            for i in 0..num_frames {
                let h1 = crate::math::common::utility::horizontal_sum_avx2(
                    head1.as_ptr().add(i * head),
                    head,
                );
                output[i] = (h1 + head2[i]) * scale;
            }
        }
    }
}

/// Kernel especializado para soma Head do WaveNet usando AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_wavenet_head_sum_avx512<const HEAD: usize>(
    head1: &[f32],
    head2: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let num_frames = output.len();
    for i in 0..num_frames {
        let h1 =
            crate::math::common::utility::horizontal_sum_avx512(head1.as_ptr().add(i * HEAD), HEAD);
        *output.get_unchecked_mut(i) = (h1 + *head2.get_unchecked(i)) * scale;
    }
}

/// Dispatch dinâmico para `batch_wavenet_head_sum` via AVX-512.
/// Despacha para o const generic apropriado (1 ou 16) ou usa fallback AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_wavenet_head_sum_dyn_avx512(
    head1: &[f32],
    head2: &[f32],
    output: &mut [f32],
    head: usize,
    scale: f32,
) {
    match head {
        1 => batch_wavenet_head_sum_avx512::<1>(head1, head2, output, scale),
        16 => batch_wavenet_head_sum_avx512::<16>(head1, head2, output, scale),
        _ => {
            let num_frames = output.len();
            for i in 0..num_frames {
                let h1 = crate::math::common::utility::horizontal_sum_avx512(
                    head1.as_ptr().add(i * head),
                    head,
                );
                output[i] = (h1 + head2[i]) * scale;
            }
        }
    }
}
