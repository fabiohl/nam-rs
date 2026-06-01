// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels de Produto Escalar (Dot Product) — AVX2 e AVX-512.

use core::arch::x86_64::*;

/// Calcula o Produto Escalar (Dot Product) de forma ultra-rápida usando aceleração de hardware (AVX2).
///
/// Esta função é o "coração" de muitos modelos de rede neural. Em vez de multiplicar e somar
/// um número por vez, ela processa blocos de dados simultaneamente (até 32 números de uma vez),
/// aproveitando ao máximo o poder do processador moderno.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;

    unsafe {
        // Prepara 4 "acumuladores" (baldes de soma) para trabalhar em paralelo.
        // Isso permite que o processador faça várias somas ao mesmo tempo sem esperar uma terminar.
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        // Loop Principal: Processa 32 números por vez (unrolling de 4x8).
        while i + 32 <= len {
            // Prefetch: Avisa o processador para já buscar os próximos dados na memória
            // antes mesmo de precisarmos deles, eliminando tempos de espera.
            _mm_prefetch::<_MM_HINT_T0>(a.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(b.as_ptr().add(i + 32) as *const i8);

            // Carrega e converte dados:
            // O vetor 'b' usa números comprimidos (f16/half), que são convertidos
            // para o formato de alta precisão (f32) instantaneamente pelo hardware.
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0); // Multiplica e Soma em um único passo (FMA)

            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + 8) as *const __m128i));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

            let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let vb2 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + 16) as *const __m128i));
            sum2 = _mm256_fmadd_ps(va2, vb2, sum2);

            let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            let vb3 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + 24) as *const __m128i));
            sum3 = _mm256_fmadd_ps(va3, vb3, sum3);

            i += 32;
        }

        // Trata os grupos restantes de 16 itens.
        while i + 16 <= len {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + 8) as *const __m128i));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

            i += 16;
        }

        // Trata os últimos grupos de 8 itens.
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(va, vb, sum0);
            i += 8;
        }

        // Combina os resultados dos 4 acumuladores paralelos em um só.
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum0, sum2);

        // Soma Horizontal: Junta os 8 valores parciais do registrador SIMD em um único número final.
        let mut scalar_sum = crate::math::common::utility::hsum_avx2(sum);

        // Limpeza Final: Processa os pouquíssimos itens que sobraram (menos de 8).
        while i < len {
            scalar_sum += a[i] * half::f16::from_bits(b[i]).to_f32();
            i += 1;
        }

        scalar_sum
    }
}

/// Dot product f32 com pesos u16 usando AVX-512.
/// Basicamente: (número_a1 * peso_b1) + (número_a2 * peso_b2) + ...
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();
    // Processa de 16 em 16.
    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_cvtph_ps(_mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i));
        sum_v = _mm512_fmadd_ps(va, vb, sum_v); // Multiplica e acumula.
        i += 16;
    }
    // Soma os resultados dentro do registrador e adiciona o resto.
    let mut sum = crate::math::common::utility::hsum_avx512(sum_v);
    while i < len {
        sum += *a.get_unchecked(i) * half::f16::from_bits(*b.get_unchecked(i)).to_f32();
        i += 1;
    }
    sum
}

/// Dot product BF16 usando AVX-512 BF16.
/// BF16 é um formato de número "cerebral" que foca no que importa para a IA.
/// Aqui o processador processa 32 números de uma só vez com uma única instrução (dpbf16_ps).
#[target_feature(enable = "avx512bf16,avx512vl")]
pub unsafe fn dot_product_bf16_avx512(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();
    while i + 32 <= len {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
        // Esta instrução é mágica: ela faz o cálculo de 32 pares de BF16 de uma vez.
        sum_v = _mm512_dpbf16_ps(
            sum_v,
            core::mem::transmute::<__m512i, __m512bh>(va),
            core::mem::transmute::<__m512i, __m512bh>(vb),
        );
        i += 32;
    }
    let mut sum = crate::math::common::utility::hsum_avx512(sum_v);
    // Trata o que sobrar de forma manual.
    while i < len {
        let fa = f32::from_bits((*a.get_unchecked(i) as u32) << 16);
        let fb = f32::from_bits((*b.get_unchecked(i) as u32) << 16);
        sum += fa * fb;
        i += 1;
    }
    sum
}
