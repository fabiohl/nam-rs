// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Backend de processamento ultra-rápido usando AVX2.
//!
//! Este módulo contém a "mágica" que faz o NAM-rs rodar tão leve.
//! Usamos instruções especiais do processador (SIMD) para fazer cálculos
//! matemáticos massivos em paralelo, processando vários sons de uma vez só.

use crate::math::common::scalar_ref::*; // Implementações escalares de referência usadas como oráculo de paridade.
use crate::math::common::traits::SimdMath;
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

/// Calcula 4 Produtos Escalares simultaneamente com o máximo de paralelismo (ILP) via AVX2.
///
/// Esta função é otimizada para situações onde precisamos multiplicar um mesmo "estado"
/// por 4 conjuntos diferentes de "pesos". Ao fazer os 4 cálculos de uma vez, mantemos
/// o processador ocupado e aproveitamos que o "estado" já está carregado para economizar
/// tempo de memória.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_4x_avx2(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    state: &[f32],
) -> [f32; 4] {
    let len = state.len();
    let mut i = 0;

    unsafe {
        // Usa 2 acumuladores para cada um dos 4 resultados (total 8 baldes de soma).
        // Isso maximiza o paralelismo, permitindo que o processador trabalhe em várias
        // frentes sem interrupções.
        let mut sum0_0 = _mm256_setzero_ps();
        let mut sum0_1 = _mm256_setzero_ps();
        let mut sum1_0 = _mm256_setzero_ps();
        let mut sum1_1 = _mm256_setzero_ps();
        let mut sum2_0 = _mm256_setzero_ps();
        let mut sum2_1 = _mm256_setzero_ps();
        let mut sum3_0 = _mm256_setzero_ps();
        let mut sum3_1 = _mm256_setzero_ps();

        // Loop Principal: Processa 16 números por vez para as 4 listas simultaneamente.
        while i + 16 <= len {
            // Antecipa a busca dos dados na memória (Prefetch).
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w0.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w1.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w2.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w3.as_ptr().add(i + 32) as *const i8);

            // Carrega o "estado" apenas uma vez para todos os pesos (reuso eficiente).
            let vs_0 = _mm256_loadu_ps(state.as_ptr().add(i));
            let vs_1 = _mm256_loadu_ps(state.as_ptr().add(i + 8));

            // Realiza o cálculo FMA (Multiplica e Soma) para os 4 vetores de pesos:
            let vw0_0 = _mm256_cvtph_ps(_mm_loadu_si128(w0.as_ptr().add(i) as *const __m128i));
            let vw0_1 = _mm256_cvtph_ps(_mm_loadu_si128(w0.as_ptr().add(i + 8) as *const __m128i));
            sum0_0 = _mm256_fmadd_ps(vw0_0, vs_0, sum0_0);
            sum0_1 = _mm256_fmadd_ps(vw0_1, vs_1, sum0_1);

            let vw1_0 = _mm256_cvtph_ps(_mm_loadu_si128(w1.as_ptr().add(i) as *const __m128i));
            let vw1_1 = _mm256_cvtph_ps(_mm_loadu_si128(w1.as_ptr().add(i + 8) as *const __m128i));
            sum1_0 = _mm256_fmadd_ps(vw1_0, vs_0, sum1_0);
            sum1_1 = _mm256_fmadd_ps(vw1_1, vs_1, sum1_1);

            let vw2_0 = _mm256_cvtph_ps(_mm_loadu_si128(w2.as_ptr().add(i) as *const __m128i));
            let vw2_1 = _mm256_cvtph_ps(_mm_loadu_si128(w2.as_ptr().add(i + 8) as *const __m128i));
            sum2_0 = _mm256_fmadd_ps(vw2_0, vs_0, sum2_0);
            sum2_1 = _mm256_fmadd_ps(vw2_1, vs_1, sum2_1);

            let vw3_0 = _mm256_cvtph_ps(_mm_loadu_si128(w3.as_ptr().add(i) as *const __m128i));
            let vw3_1 = _mm256_cvtph_ps(_mm_loadu_si128(w3.as_ptr().add(i + 8) as *const __m128i));
            sum3_0 = _mm256_fmadd_ps(vw3_0, vs_0, sum3_0);
            sum3_1 = _mm256_fmadd_ps(vw3_1, vs_1, sum3_1);

            i += 16;
        }

        // Trata os grupos restantes de 8 itens.
        while i + 8 <= len {
            let vs = _mm256_loadu_ps(state.as_ptr().add(i));

            let vw0 = _mm256_cvtph_ps(_mm_loadu_si128(w0.as_ptr().add(i) as *const __m128i));
            sum0_0 = _mm256_fmadd_ps(vw0, vs, sum0_0);

            let vw1 = _mm256_cvtph_ps(_mm_loadu_si128(w1.as_ptr().add(i) as *const __m128i));
            sum1_0 = _mm256_fmadd_ps(vw1, vs, sum1_0);

            let vw2 = _mm256_cvtph_ps(_mm_loadu_si128(w2.as_ptr().add(i) as *const __m128i));
            sum2_0 = _mm256_fmadd_ps(vw2, vs, sum2_0);

            let vw3 = _mm256_cvtph_ps(_mm_loadu_si128(w3.as_ptr().add(i) as *const __m128i));
            sum3_0 = _mm256_fmadd_ps(vw3, vs, sum3_0);

            i += 8;
        }

        // Consolida os acumuladores duplos de cada vetor.
        let sum0 = _mm256_add_ps(sum0_0, sum0_1);
        let sum1 = _mm256_add_ps(sum1_0, sum1_1);
        let sum2 = _mm256_add_ps(sum2_0, sum2_1);
        let sum3 = _mm256_add_ps(sum3_0, sum3_1);

        // Converte os resultados SIMD para números escalares finais.
        let mut s0: f32 = crate::math::common::utility::hsum_avx2(sum0);
        let mut s1: f32 = crate::math::common::utility::hsum_avx2(sum1);
        let mut s2: f32 = crate::math::common::utility::hsum_avx2(sum2);
        let mut s3: f32 = crate::math::common::utility::hsum_avx2(sum3);

        // Limpeza final para os poucos itens restantes.
        while i < len {
            s0 += half::f16::from_bits(w0[i]).to_f32() * state[i];
            s1 += half::f16::from_bits(w1[i]).to_f32() * state[i];
            s2 += half::f16::from_bits(w2[i]).to_f32() * state[i];
            s3 += half::f16::from_bits(w3[i]).to_f32() * state[i];
            i += 1;
        }

        [s0, s1, s2, s3]
    }
}

/// Calcula 4 Produtos Escalares simultaneamente usando o layout de "pesos interfolhados" via AVX2.
///
/// Neste formato, os pesos dos 4 cálculos estão organizados juntos na memória (em grupos de 4).
/// Esta função usa truques de "embaralhamento" (broadcast e blend) para alinhar os dados do
/// estado com esses pesos, permitindo um processamento extremamente veloz e eficiente em termos
/// de acesso à memória (cache friendly).
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_4x_interleaved_avx2(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len();
    let mut i = 0;

    unsafe {
        // Inicializa os acumuladores para os 4 resultados parciais.
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        // Loop Principal: Processa 8 blocos de pesos interfolhados por vez.
        while i + 8 <= len {
            // Antecipa os dados da memória (Prefetch).
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 8) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 16) as *const i8);

            // Broadcast e Blend:
            // Pega um valor do estado, duplica-o e o "mistura" (blend) para que ele
            // se alinhe com os 4 pesos interfolhados carregados do vetor de pesos.
            let s0 = _mm256_broadcast_ss(&state[i]);
            let s1 = _mm256_broadcast_ss(&state[i + 1]);
            let s01 = _mm256_blend_ps(s0, s1, 0b11110000);
            let w01 = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(w01, s01, sum0);

            let s2 = _mm256_broadcast_ss(&state[i + 2]);
            let s3 = _mm256_broadcast_ss(&state[i + 3]);
            let s23 = _mm256_blend_ps(s2, s3, 0b11110000);
            let w23 = _mm256_cvtph_ps(_mm_loadu_si128(
                weights.as_ptr().add(i + 2) as *const __m128i
            ));
            sum1 = _mm256_fmadd_ps(w23, s23, sum1);

            let s4 = _mm256_broadcast_ss(&state[i + 4]);
            let s5 = _mm256_broadcast_ss(&state[i + 5]);
            let s45 = _mm256_blend_ps(s4, s5, 0b11110000);
            let w45 = _mm256_cvtph_ps(_mm_loadu_si128(
                weights.as_ptr().add(i + 4) as *const __m128i
            ));
            sum2 = _mm256_fmadd_ps(w45, s45, sum2);

            let s6 = _mm256_broadcast_ss(&state[i + 6]);
            let s7 = _mm256_broadcast_ss(&state[i + 7]);
            let s67 = _mm256_blend_ps(s6, s7, 0b11110000);
            let w67 = _mm256_cvtph_ps(_mm_loadu_si128(
                weights.as_ptr().add(i + 6) as *const __m128i
            ));
            sum3 = _mm256_fmadd_ps(w67, s67, sum3);

            i += 8;
        }

        // Trata os grupos restantes de 2 itens.
        while i + 2 <= len {
            let s0 = _mm256_broadcast_ss(&state[i]);
            let s1 = _mm256_broadcast_ss(&state[i + 1]);
            let s01 = _mm256_blend_ps(s0, s1, 0b11110000);
            let w01 = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(w01, s01, sum0);
            i += 2;
        }

        // Soma os resultados parciais dos acumuladores.
        let sum01 = _mm256_add_ps(sum0, sum1);
        let sum23 = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum01, sum23);

        // Converte o registrador de 256 bits para 128 bits para finalizar.
        let lower = _mm256_castps256_ps128(sum);
        let upper = _mm256_extractf128_ps(sum, 1);
        let mut sum128 = _mm_add_ps(lower, upper);

        // Limpeza Final: Processa os itens que sobraram um por um.
        while i < len {
            let s0 = _mm_load1_ps(state.as_ptr().add(i));
            let w0 = _mm_cvtph_ps(_mm_loadu_si64(
                weights.as_ptr().add(i) as *const u16 as *const u8
            ));
            sum128 = _mm_fmadd_ps(w0, s0, sum128);
            i += 1;
        }

        // Salva o resultado final no array de saída.
        let mut out = [0.0; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum128);
        out
    }
}

/// Calcula 4 Produtos Escalares para dois quadros de áudio simultâneos (Dual Frame) via AVX2.
///
/// Esta é uma das funções mais eficientes do sistema. Ela aproveita que os pesos já foram
/// carregados na memória para aplicá-los em dois blocos de áudio diferentes (f0 e f1) ao
/// mesmo tempo. Isso dobra a produtividade do processador, pois cada peso lido é "reutilizado"
/// imediatamente para dois cálculos distintos.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_4x_interleaved_dual_frame_avx2(
    weights: &[[u16; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(state_f0.len(), state_f1.len());
    let mut i = 0;

    unsafe {
        // Inicializa 8 acumuladores: 4 para o primeiro quadro (f0) e 4 para o segundo (f1).
        let mut sum0_f0 = _mm256_setzero_ps();
        let mut sum1_f0 = _mm256_setzero_ps();
        let mut sum2_f0 = _mm256_setzero_ps();
        let mut sum3_f0 = _mm256_setzero_ps();

        let mut sum0_f1 = _mm256_setzero_ps();
        let mut sum1_f1 = _mm256_setzero_ps();
        let mut sum2_f1 = _mm256_setzero_ps();
        let mut sum3_f1 = _mm256_setzero_ps();

        // Loop Principal: Processa 8 blocos de pesos interfolhados para ambos os quadros.
        while i + 8 <= len {
            // Prepara os dados dos dois quadros usando broadcast e blend.
            let s0_f0 = _mm256_broadcast_ss(&state_f0[i]);
            let s1_f0 = _mm256_broadcast_ss(&state_f0[i + 1]);
            let s01_f0 = _mm256_blend_ps(s0_f0, s1_f0, 0b11110000);

            let s0_f1 = _mm256_broadcast_ss(&state_f1[i]);
            let s1_f1 = _mm256_broadcast_ss(&state_f1[i + 1]);
            let s01_f1 = _mm256_blend_ps(s0_f1, s1_f1, 0b11110000);

            // Carrega o peso uma única vez e aplica nos dois quadros (f0 e f1).
            let w01 = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
            sum0_f0 = _mm256_fmadd_ps(w01, s01_f0, sum0_f0);
            sum0_f1 = _mm256_fmadd_ps(w01, s01_f1, sum0_f1);

            let s2_f0 = _mm256_broadcast_ss(&state_f0[i + 2]);
            let s3_f0 = _mm256_broadcast_ss(&state_f0[i + 3]);
            let s23_f0 = _mm256_blend_ps(s2_f0, s3_f0, 0b11110000);

            let s2_f1 = _mm256_broadcast_ss(&state_f1[i + 2]);
            let s3_f1 = _mm256_broadcast_ss(&state_f1[i + 3]);
            let s23_f1 = _mm256_blend_ps(s2_f1, s3_f1, 0b11110000);

            let w23 = _mm256_cvtph_ps(_mm_loadu_si128(
                weights.as_ptr().add(i + 2) as *const __m128i
            ));
            sum1_f0 = _mm256_fmadd_ps(w23, s23_f0, sum1_f0);
            sum1_f1 = _mm256_fmadd_ps(w23, s23_f1, sum1_f1);

            let s4_f0 = _mm256_broadcast_ss(&state_f0[i + 4]);
            let s5_f0 = _mm256_broadcast_ss(&state_f0[i + 5]);
            let s45_f0 = _mm256_blend_ps(s4_f0, s5_f0, 0b11110000);

            let s4_f1 = _mm256_broadcast_ss(&state_f1[i + 4]);
            let s5_f1 = _mm256_broadcast_ss(&state_f1[i + 5]);
            let s45_f1 = _mm256_blend_ps(s4_f1, s5_f1, 0b11110000);

            let w45 = _mm256_cvtph_ps(_mm_loadu_si128(
                weights.as_ptr().add(i + 4) as *const __m128i
            ));
            sum2_f0 = _mm256_fmadd_ps(w45, s45_f0, sum2_f0);
            sum2_f1 = _mm256_fmadd_ps(w45, s45_f1, sum2_f1);

            let s6_f0 = _mm256_broadcast_ss(&state_f0[i + 6]);
            let s7_f0 = _mm256_broadcast_ss(&state_f0[i + 7]);
            let s67_f0 = _mm256_blend_ps(s6_f0, s7_f0, 0b11110000);

            let s6_f1 = _mm256_broadcast_ss(&state_f1[i + 6]);
            let s7_f1 = _mm256_broadcast_ss(&state_f1[i + 7]);
            let s67_f1 = _mm256_blend_ps(s6_f1, s7_f1, 0b11110000);

            let w67 = _mm256_cvtph_ps(_mm_loadu_si128(
                weights.as_ptr().add(i + 6) as *const __m128i
            ));
            sum3_f0 = _mm256_fmadd_ps(w67, s67_f0, sum3_f0);
            sum3_f1 = _mm256_fmadd_ps(w67, s67_f1, sum3_f1);

            i += 8;
        }

        // Trata os grupos restantes de 2 itens para ambos os quadros.
        while i + 2 <= len {
            let s0_f0 = _mm256_broadcast_ss(&state_f0[i]);
            let s1_f0 = _mm256_broadcast_ss(&state_f0[i + 1]);
            let s01_f0 = _mm256_blend_ps(s0_f0, s1_f0, 0b11110000);

            let s0_f1 = _mm256_broadcast_ss(&state_f1[i]);
            let s1_f1 = _mm256_broadcast_ss(&state_f1[i + 1]);
            let s01_f1 = _mm256_blend_ps(s0_f1, s1_f1, 0b11110000);

            let w01 = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
            sum0_f0 = _mm256_fmadd_ps(w01, s01_f0, sum0_f0);
            sum0_f1 = _mm256_fmadd_ps(w01, s01_f1, sum0_f1);
            i += 2;
        }

        // Consolida as somas parciais de cada quadro separadamente.
        let sum01_f0 = _mm256_add_ps(sum0_f0, sum1_f0);
        let sum23_f0 = _mm256_add_ps(sum2_f0, sum3_f0);
        let sum_f0 = _mm256_add_ps(sum01_f0, sum23_f0);

        let sum01_f1 = _mm256_add_ps(sum0_f1, sum1_f1);
        let sum23_f1 = _mm256_add_ps(sum2_f1, sum3_f1);
        let sum_f1 = _mm256_add_ps(sum01_f1, sum23_f1);

        // Finaliza o cálculo convertendo os registradores de 256 bits para 128 bits.
        let lower_f0 = _mm256_castps256_ps128(sum_f0);
        let upper_f0 = _mm256_extractf128_ps(sum_f0, 1);
        let mut sum128_f0 = _mm_add_ps(lower_f0, upper_f0);

        let lower_f1 = _mm256_castps256_ps128(sum_f1);
        let upper_f1 = _mm256_extractf128_ps(sum_f1, 1);
        let mut sum128_f1 = _mm_add_ps(lower_f1, upper_f1);

        // Limpeza Final: Processa o que sobrou individualmente para os dois quadros.
        while i < len {
            let s0_f0 = _mm_load1_ps(state_f0.as_ptr().add(i));
            let s0_f1 = _mm_load1_ps(state_f1.as_ptr().add(i));
            let w0 = _mm_cvtph_ps(_mm_loadu_si64(
                weights.as_ptr().add(i) as *const u16 as *const u8
            ));
            sum128_f0 = _mm_fmadd_ps(w0, s0_f0, sum128_f0);
            sum128_f1 = _mm_fmadd_ps(w0, s0_f1, sum128_f1);
            i += 1;
        }

        // Armazena os 4 resultados finais para cada quadro.
        let mut out_f0 = [0.0; 4];
        let mut out_f1 = [0.0; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), sum128_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), sum128_f1);
        (out_f0, out_f1)
    }
}

/// Kernel especializado para multiplicar pesos por 4 canais de áudio diferentes ao mesmo tempo.
/// Esta função é o "faz tudo" das redes neurais WaveNet e LSTM quando processamos
/// áudio em lote (batch). Ela economiza energia e tempo ao não precisar ler
/// os mesmos pesos da memória repetidamente.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_batch_4x_avx2(
    h0: &[f32],
    h1: &[f32],
    h2: &[f32],
    h3: &[f32],
    weights: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), h0.len());
    let mut i = 0;

    unsafe {
        // Prepara os 4 baldes de soma para os 4 sons diferentes.
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        // Loop Principal: Processa 16 números de cada som por vez.
        while i + 16 <= len {
            // Avisa o processador para já ir buscando os próximos dados.
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h0.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h1.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h2.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h3.as_ptr().add(i + 32) as *const i8);

            // Carrega o peso comprimido (f16) e expande para f32.
            let vw_0 = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));

            // Multiplica esse mesmo peso pelos 4 sons de entrada.
            // É como se um único professor desse aula para 4 alunos ao mesmo tempo.
            let vh0_0 = _mm256_loadu_ps(h0.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(vw_0, vh0_0, sum0);

            let vh1_0 = _mm256_loadu_ps(h1.as_ptr().add(i));
            sum1 = _mm256_fmadd_ps(vw_0, vh1_0, sum1);

            let vh2_0 = _mm256_loadu_ps(h2.as_ptr().add(i));
            sum2 = _mm256_fmadd_ps(vw_0, vh2_0, sum2);

            let vh3_0 = _mm256_loadu_ps(h3.as_ptr().add(i));
            sum3 = _mm256_fmadd_ps(vw_0, vh3_0, sum3);

            // Faz o mesmo para a segunda metade do bloco de 16.
            let vw_1 = _mm256_cvtph_ps(_mm_loadu_si128(
                weights.as_ptr().add(i + 8) as *const __m128i
            ));
            let vh0_1 = _mm256_loadu_ps(h0.as_ptr().add(i + 8));
            sum0 = _mm256_fmadd_ps(vw_1, vh0_1, sum0);
            let vh1_1 = _mm256_loadu_ps(h1.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(vw_1, vh1_1, sum1);
            let vh2_1 = _mm256_loadu_ps(h2.as_ptr().add(i + 8));
            sum2 = _mm256_fmadd_ps(vw_1, vh2_1, sum2);
            let vh3_1 = _mm256_loadu_ps(h3.as_ptr().add(i + 8));
            sum3 = _mm256_fmadd_ps(vw_1, vh3_1, sum3);

            i += 16;
        }

        // Trata blocos de 8 itens.
        while i + 8 <= len {
            let vw = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
            let vh0 = _mm256_loadu_ps(h0.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(vw, vh0, sum0);
            let vh1 = _mm256_loadu_ps(h1.as_ptr().add(i));
            sum1 = _mm256_fmadd_ps(vw, vh1, sum1);
            let vh2 = _mm256_loadu_ps(h2.as_ptr().add(i));
            sum2 = _mm256_fmadd_ps(vw, vh2, sum2);
            let vh3 = _mm256_loadu_ps(h3.as_ptr().add(i));
            sum3 = _mm256_fmadd_ps(vw, vh3, sum3);

            i += 8;
        }

        // Soma os resultados parciais de cada um dos 4 acumuladores.
        let mut s0 = crate::math::common::utility::hsum_avx2(sum0);
        let mut s1 = crate::math::common::utility::hsum_avx2(sum1);
        let mut s2 = crate::math::common::utility::hsum_avx2(sum2);
        let mut s3 = crate::math::common::utility::hsum_avx2(sum3);

        // Termina as amostras que sobraram (menos de 8).
        while i < len {
            let w = half::f16::from_bits(weights[i]).to_f32();
            s0 += w * h0[i];
            s1 += w * h1[i];
            s2 += w * h2[i];
            s3 += w * h3[i];
            i += 1;
        }

        [s0, s1, s2, s3]
    }
}

/// Realiza uma operação matemática combinada (fundida) de alta velocidade: Y = X_res + Bias + W * Z.
///
/// Esta função faz três coisas ao mesmo tempo: preserva o valor atual (residual), soma um
/// ajuste (bias) e adiciona o resultado de uma multiplicação de pesos por entrada. Fazer tudo
/// de uma vez evita que o processador precise ler e escrever na memória várias vezes, mantendo
/// os dados "quentes" e prontos para o próximo cálculo.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn fused_add_gemv_avx2(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    unsafe {
        let mut out_c = 0;
        // Processa 8 saídas de uma vez usando AVX2.
        while out_c + 8 <= out_len {
            // Carrega o valor atual (residual) que já estava no balde.
            let mut accum = _mm256_loadu_ps(out_frame.as_ptr().add(out_c));
            // Se tiver um ajuste (bias), soma-o agora.
            if do_bias {
                accum = _mm256_add_ps(accum, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
            }

            // Para cada entrada, multiplica pelo peso e soma no acumulador.
            for in_c in 0..in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                // Converte o peso comprimido (f16) para f32 na hora.
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));
                // A instrução 'fmadd' faz a multiplicação e a soma em um só golpe.
                accum = _mm256_fmadd_ps(vs, vw, accum);
            }

            // Salva o resultado final de volta no balde.
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 8;
        }

        // Trata o que sobrou um por um.
        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w = half::f16::from_bits(weights[in_c * out_len + out_c]).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) += sum;
            out_c += 1;
        }
    }
}

/// Realiza uma projeção linear (Y = Bias + W * Z), substituindo o conteúdo anterior.
///
/// Diferente da função anterior, esta limpa o "balde" de saída antes de começar, colocando
/// apenas o novo resultado da multiplicação (mais o ajuste opcional). É usada para iniciar
/// o cálculo de uma nova camada da rede neural do zero.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gemv_overwrite_avx2(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    unsafe {
        let mut out_c = 0;
        // Processa 8 saídas de uma vez.
        while out_c + 8 <= out_len {
            // Começa o balde do zero (ou com o ajuste/bias).
            let mut accum = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(out_c))
            } else {
                _mm256_setzero_ps()
            };

            for in_c in 0..in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));
                accum = _mm256_fmadd_ps(vs, vw, accum);
            }

            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 8;
        }

        // Finaliza o resto.
        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
    }
}

/// Processa vários quadros de áudio em lote (batch) usando a técnica fundida: Y = X_res + Bias + W * Z.
///
/// Esta é a versão mais potente da operação fundida. Ela organiza o trabalho em grupos de 4
/// quadros de áudio, permitindo que o processador reutilize os pesos da rede neural de forma
/// extremamente eficiente para todos eles antes de precisar ler novos dados da memória.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn fused_add_gemm_batch_avx2(
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

    unsafe {
        let mut f = 0;
        // Estratégia de Lote: Processa os dados em grupos de 4 quadros de áudio.
        // Isso permite que cada peso da rede neural seja lido uma vez e reutilizado
        // 4 vezes seguidas (uma para cada quadro), o que é extremamente eficiente.
        while f + 4 <= num_frames {
            let mut out_c = 0;
            while out_c + 8 <= out_len {
                // Carrega os resultados parciais (baldes) de 4 quadros simultaneamente.
                let mut acc0 = _mm256_loadu_ps(out_frames.as_ptr().add(f * out_len + out_c));
                let mut acc1 = _mm256_loadu_ps(out_frames.as_ptr().add((f + 1) * out_len + out_c));
                let mut acc2 = _mm256_loadu_ps(out_frames.as_ptr().add((f + 2) * out_len + out_c));
                let mut acc3 = _mm256_loadu_ps(out_frames.as_ptr().add((f + 3) * out_len + out_c));

                // Se houver Bias (ajuste), adiciona-o nos 4 quadros de uma só vez.
                if do_bias {
                    let b = _mm256_loadu_ps(bias.as_ptr().add(out_c));
                    acc0 = _mm256_add_ps(acc0, b);
                    acc1 = _mm256_add_ps(acc1, b);
                    acc2 = _mm256_add_ps(acc2, b);
                    acc3 = _mm256_add_ps(acc3, b);
                }

                // Loop de Cálculo: Multiplica a entrada pelos pesos.
                for in_c in 0..in_len {
                    // Lê o peso da memória apenas uma vez.
                    let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                    let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));

                    // Espalha a entrada correspondente de cada um dos 4 quadros.
                    let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                    let vs1 = _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                    let vs2 = _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                    let vs3 = _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));

                    // Multiplica e Soma (FMA) para os 4 quadros usando o mesmo peso lido.
                    acc0 = _mm256_fmadd_ps(vs0, vw, acc0);
                    acc1 = _mm256_fmadd_ps(vs1, vw, acc1);
                    acc2 = _mm256_fmadd_ps(vs2, vw, acc2);
                    acc3 = _mm256_fmadd_ps(vs3, vw, acc3);
                }

                // Salva os 4 novos resultados de volta na memória.
                _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
                _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
                out_c += 8;
            }

            // Trata as sobras de cada bloco de 4 quadros.
            while out_c < out_len {
                for i in 0..4 {
                    let frame_idx = f + i;
                    let mut sum = *out_frames.get_unchecked(frame_idx * out_len + out_c);
                    if do_bias {
                        sum += *bias.get_unchecked(out_c);
                    }
                    for in_c in 0..in_len {
                        let w_bits = *weights.get_unchecked(in_c * out_len + out_c);
                        let w = half::f16::from_bits(w_bits).to_f32();
                        sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                    }
                    *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
                }
                out_c += 1;
            }
            f += 4;
        }

        // Limpeza Final: Se sobrou algum quadro (menos de 4), processa um por um.
        while f < num_frames {
            fused_add_gemv_avx2(
                in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
                weights,
                bias,
                out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
                do_bias,
            );
            f += 1;
        }
    }
}

/// Realiza a projeção linear para as 4 "portas" de uma célula LSTM de forma simultânea via AVX2.
///
/// Em uma rede neural LSTM, cada passo exige o cálculo de 4 sub-resultados (portas). Esta
/// função executa todos esses cálculos de uma só vez, garantindo que a atualização da
/// "memória" da rede seja feita com o máximo de performance e o mínimo de latência.
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_avx2(
    in_frame: &[f32],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len() / 4;
    let in_len = in_frame.len();

    unsafe {
        let mut out_c = 0;
        // Processa as 4 portas do LSTM em paralelo, 8 elementos por vez.
        while out_c + 8 <= out_len {
            // Inicializa os acumuladores (baldes) com os valores de Bias de cada porta.
            let mut acc0 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(out_c))
            } else {
                _mm256_setzero_ps()
            };
            let mut acc1 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(out_len + out_c))
            } else {
                _mm256_setzero_ps()
            };
            let mut acc2 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(2 * out_len + out_c))
            } else {
                _mm256_setzero_ps()
            };
            let mut acc3 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(3 * out_len + out_c))
            } else {
                _mm256_setzero_ps()
            };

            // Loop de Cálculo Principal:
            for in_c in 0..in_len {
                // Pega um único valor de entrada e o "espalha" para usar em todas as portas.
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));

                // Multiplica a entrada pelos pesos de cada uma das 4 portas (acc0 a acc3).
                // Cada porta cuida de um aspecto diferente da "memória" do LSTM.
                let wp0 = w0.as_ptr().add(in_c * out_len + out_c);
                let vw0 = _mm256_cvtph_ps(_mm_loadu_si128(wp0 as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs, vw0, acc0);

                let wp1 = w1.as_ptr().add(in_c * out_len + out_c);
                let vw1 = _mm256_cvtph_ps(_mm_loadu_si128(wp1 as *const __m128i));
                acc1 = _mm256_fmadd_ps(vs, vw1, acc1);

                let wp2 = w2.as_ptr().add(in_c * out_len + out_c);
                let vw2 = _mm256_cvtph_ps(_mm_loadu_si128(wp2 as *const __m128i));
                acc2 = _mm256_fmadd_ps(vs, vw2, acc2);

                let wp3 = w3.as_ptr().add(in_c * out_len + out_c);
                let vw3 = _mm256_cvtph_ps(_mm_loadu_si128(wp3 as *const __m128i));
                acc3 = _mm256_fmadd_ps(vs, vw3, acc3);
            }

            // Salva os resultados finais de cada porta nos seus devidos lugares na memória.
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_len + out_c), acc1);
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(2 * out_len + out_c), acc2);
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(3 * out_len + out_c), acc3);
            out_c += 8;
        }

        // Limpeza Final: Processa os itens que sobraram individualmente (menos de 8).
        while out_c < out_len {
            let mut sum0 = if do_bias { bias[out_c] } else { 0.0 };
            let mut sum1 = if do_bias { bias[out_len + out_c] } else { 0.0 };
            let mut sum2 = if do_bias {
                bias[2 * out_len + out_c]
            } else {
                0.0
            };
            let mut sum3 = if do_bias {
                bias[3 * out_len + out_c]
            } else {
                0.0
            };

            for in_c in 0..in_len {
                let s = *in_frame.get_unchecked(in_c);
                sum0 +=
                    s * half::f16::from_bits(*w0.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum1 +=
                    s * half::f16::from_bits(*w1.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum2 +=
                    s * half::f16::from_bits(*w2.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum3 +=
                    s * half::f16::from_bits(*w3.get_unchecked(in_c * out_len + out_c)).to_f32();
            }

            *out_frame.get_unchecked_mut(out_c) = sum0;
            *out_frame.get_unchecked_mut(out_len + out_c) = sum1;
            *out_frame.get_unchecked_mut(2 * out_len + out_c) = sum2;
            *out_frame.get_unchecked_mut(3 * out_len + out_c) = sum3;
            out_c += 1;
        }
    }
}

/// Kernel GEMM com residual fundido via AVX2.
///
/// Esta função é o "motor principal" de muitas camadas de redes neurais modernas. Ela combina
/// a multiplicação de matrizes por vetores com a adição de uma "conexão residual" (um atalho
/// que ajuda a rede a manter informações importantes do passado). Ao fundir tudo isso em um
/// único passo vetorial, economizamos ciclos de memória valiosos.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn fused_gemm_residual_batch_avx2(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32],
    residual: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    let mut f = 0;
    // Estratégia de Lote: Processa 4 quadros de áudio simultaneamente para reuso de pesos.
    while f + 4 <= num_frames {
        let mut out_c = 0;
        while out_c + 8 <= out_len {
            // Inicializa os acumuladores com os valores da "Conexão Residual" (atalho).
            let mut acc0 = _mm256_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
            let mut acc1 = _mm256_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
            let mut acc2 = _mm256_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
            let mut acc3 = _mm256_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));

            // Se houver Bias, soma-o aos baldes residuais.
            if do_bias {
                let b = _mm256_loadu_ps(bias.as_ptr().add(out_c));
                acc0 = _mm256_add_ps(acc0, b);
                acc1 = _mm256_add_ps(acc1, b);
                acc2 = _mm256_add_ps(acc2, b);
                acc3 = _mm256_add_ps(acc3, b);
            }

            // Loop de Pesos: Multiplica e soma o resultado da matriz sobre os baldes.
            for in_c in 0..in_len {
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));

                acc0 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c)),
                    vw,
                    acc0,
                );
                acc1 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c)),
                    vw,
                    acc1,
                );
                acc2 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c)),
                    vw,
                    acc2,
                );
                acc3 = _mm256_fmadd_ps(
                    _mm256_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c)),
                    vw,
                    acc3,
                );
            }

            // Salva os 4 novos resultados finais (Residual + Bias + Multiplicação).
            _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
            _mm256_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
            out_c += 8;
        }

        // Limpeza final para o resto da largura da matriz em grupos de 4 quadros.
        while out_c < out_len {
            for i in 0..4 {
                let frame_idx = f + i;
                let mut sum = *residual.get_unchecked(frame_idx * out_len + out_c);
                if do_bias {
                    sum += *bias.get_unchecked(out_c);
                }
                for in_c in 0..in_len {
                    let w = half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c))
                        .to_f32();
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) = sum;
            }
            out_c += 1;
        }
        f += 4;
    }

    // Fallback: Se sobrar algum quadro isolado (menos de 4), processa-o individualmente.
    while f < num_frames {
        let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
        let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
        let res_frame = &residual[f * out_len..(f + 1) * out_len];

        let mut out_c = 0;
        while out_c + 8 <= out_len {
            let mut accum = _mm256_loadu_ps(res_frame.as_ptr().add(out_c));
            if do_bias {
                accum = _mm256_add_ps(accum, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
            }
            for in_c in 0..in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));
                accum = _mm256_fmadd_ps(vs, vw, accum);
            }
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 8;
        }
        while out_c < out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            sum += res_frame[out_c];
            for in_c in 0..in_len {
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
            *out_frame.get_unchecked_mut(out_c) = sum;
            out_c += 1;
        }
        f += 1;
    }
}

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
        unsafe { dot_product_avx2(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        // Como o AVX2 puro não tem aceleração nativa para BF16, usamos a versão comum.
        unsafe { dot_product_bf16_fallback(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_avx2(weights, state) }
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
        unsafe { dot_product_4x_interleaved_dual_frame_avx2(weights, state_f0, state_f1) }
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
        unsafe { fused_add_gemv_avx2(in_frame, weights, bias, out_frame, do_bias) }
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
            fused_add_gemm_batch_avx2(in_frames, weights, bias, out_frames, num_frames, do_bias)
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
            fused_gemm_residual_batch_avx2(
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
        unsafe { gemv_overwrite_avx2(in_frame, weights, bias, out_frame, do_bias) }
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
            gemv_4gate_avx2(
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
        unsafe { accumulate_head_avx2(dest, src) }
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        unsafe { tanh_and_accumulate_block_avx2(head_input, block) }
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        unsafe { gated_activation_and_accumulate_block_avx2(head_input, block, ch) }
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
        unsafe { crate::math::fastmath::tanh_slice_avx2(slice) }
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::sigmoid_slice_avx2(slice) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { horizontal_sum_avx2(ptr, N) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { crate::math::fastmath::tanh_slice_avx2(buf) }
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        unsafe { fused_lstm_gates_dyn_avx2(gates, cell_state, hidden_state, hidden_size) }
    }

    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        unsafe { compute_energy_stereo_avx2(l, r) }
    }

    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { convolve_stereo_avx2(coeffs, input_l, input_r, taps) }
    }

    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        unsafe { apply_gain_and_detect_clipping_stereo_avx2(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        unsafe { apply_gain_stereo_avx2(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        unsafe { apply_gain_avx2(data, gain) }
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe { batch_wavenet_head_sum_avx2::<HEAD>(head1, head2, output, scale) }
    }

    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        unsafe { apply_ramp_stereo_avx2(left, right, start, step) }
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
            gemv_overwrite_avx2(in_slice, weights, bias, out_slice, do_bias);
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
        match head {
            1 => Self::batch_wavenet_head_sum::<1>(head1, head2, output, scale),
            16 => Self::batch_wavenet_head_sum::<16>(head1, head2, output, scale),
            _ => {
                let num_frames = output.len();
                for i in 0..num_frames {
                    let h1 = horizontal_sum_avx2(head1.as_ptr().add(i * head), head);
                    output[i] = (h1 + head2[i]) * scale;
                }
            }
        }
    }
}

/// Implementação especializada para processadores que suportam AVX2 e instruções VNNI.
///
/// VNNI (Vector Neural Network Instructions) é uma tecnologia que acelera drasticamente
/// o processamento de redes neurais. Esta estrutura funciona como uma ponte de alta
/// performance para CPUs modernas, garantindo que o NAM-rs utilize o caminho mais
/// curto e eficiente oferecido pelo hardware Intel de gerações recentes.
pub struct Avx2VnniMath;

impl SimdMath for Avx2VnniMath {
    type V = __m256;

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        // Reutiliza a lógica ultra-rápida do AVX2 base.
        unsafe { Avx2Math::dot_product(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { Avx2Math::dot_product_bf16(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { Avx2Math::dot_product_4x_interleaved(weights, state) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        Avx2Math::dot_product_4x_interleaved_bf16(weights, state)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]) {
        Avx2Math::dot_product_4x_interleaved_dual_frame(weights, state_f0, state_f1)
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
        weights: &[[u16; 4]],
        state_f0: &[u16],
        state_f1: &[u16],
    ) -> ([f32; 4], [f32; 4]) {
        Avx2Math::dot_product_4x_interleaved_dual_frame_bf16(weights, state_f0, state_f1)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe { Avx2Math::dot_product_bf16_4x(w0, w1, w2, w3, in_frame) }
    }

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { Avx2Math::fused_add_gemv(in_frame, weights, bias, out_frame, do_bias) }
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
            Avx2Math::fused_add_gemm_batch(
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
            Avx2Math::fused_gemm_residual_batch(
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
        unsafe { Avx2Math::gemv_overwrite(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { Avx2Math::gemv_overwrite_bf16(in_frame, weights, bias, out_frame, do_bias) }
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
            Avx2Math::gemv_overwrite_4gate(in_frame, weights, bias, out_gates, hidden_size, do_bias)
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
            Avx2Math::gemv_overwrite_bf16_4gate(
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
        unsafe { Avx2Math::accumulate_head(dest, src) }
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        unsafe { Avx2Math::tanh_and_accumulate_block(head_input, block) }
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        unsafe { Avx2Math::gated_activation_and_accumulate_block(head_input, block, ch) }
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        unsafe { Avx2Math::f32_to_bf16(src, dest) }
    }

    #[inline(always)]
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
        unsafe { Avx2Math::store_bf16(ptr, v) }
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { Avx2Math::tanh_slice(slice) }
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { Avx2Math::sigmoid_slice(slice) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { Avx2Math::horizontal_sum::<N>(ptr) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { Avx2Math::activation_tanh_block(buf) }
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        unsafe { Avx2Math::fused_lstm_gates_dyn(gates, cell_state, hidden_state, hidden_size) }
    }

    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        unsafe { Avx2Math::compute_energy_stereo(l, r) }
    }

    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { Avx2Math::convolve_stereo(coeffs, input_l, input_r, taps) }
    }

    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        unsafe { Avx2Math::apply_gain_and_detect_clipping_stereo(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        unsafe { Avx2Math::apply_gain_stereo(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        unsafe { apply_gain_avx2(data, gain) }
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe { batch_wavenet_head_sum_avx2::<HEAD>(head1, head2, output, scale) }
    }

    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        unsafe { apply_ramp_stereo_avx2(left, right, start, step) }
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
        Avx2Math::gemv_overwrite_batch(in_frames, weights, bias, out_frames, num_frames, do_bias)
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
        Avx2Math::gemv_overwrite_batch_bf16(
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
        Avx2Math::batch_wavenet_head_sum_dyn(head1, head2, output, head, scale)
    }
}

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
        let vt = crate::math::fastmath::simd_tanh_avx2(vb);
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

            let tanh_z1 = crate::math::fastmath::simd_tanh_avx2(z1);
            let sig_z2 = crate::math::fastmath::simd_sigmoid_avx2(z2);
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

        let (new_cs, hidden) = crate::math::fastmath::fused_lstm_gates_avx2(gf, gi, gg, go, cs);

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
