// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels de Dot Product 4x (ILP interleaved, dual frame, batch) — AVX2 e AVX-512.

use core::arch::x86_64::*;

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

/// Dot product interleaved 4x com 8 acumuladores ZMM (2 conjuntos alternados de 4)
/// para quebrar cadeias de dependência no pipeline FMA.
///
/// Cada ZMM processa 4 valores de estado (16 lanes f32: 4 output channels × 4 replicações).
/// Os 2 conjuntos alternam a cada 16 elementos de estado para manter o pipeline FMA saturado.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_interleaved_avx512(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut i = 0;

    unsafe {
        let perm_idx = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

        let mut sum_a0 = _mm512_setzero_ps();
        let mut sum_a1 = _mm512_setzero_ps();
        let mut sum_a2 = _mm512_setzero_ps();
        let mut sum_a3 = _mm512_setzero_ps();
        let mut sum_b0 = _mm512_setzero_ps();
        let mut sum_b1 = _mm512_setzero_ps();
        let mut sum_b2 = _mm512_setzero_ps();
        let mut sum_b3 = _mm512_setzero_ps();

        while i + 32 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 48) as *const i8);

            let s_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i))),
            );
            let w_a =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            sum_a0 = _mm512_fmadd_ps(w_a, s_a, sum_a0);

            let s_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 4))),
            );
            let w_b = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
            sum_a1 = _mm512_fmadd_ps(w_b, s_b, sum_a1);

            let s_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 8))),
            );
            let w_c = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 8) as *const __m256i
            ));
            sum_a2 = _mm512_fmadd_ps(w_c, s_c, sum_a2);

            let s_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 12))),
            );
            let w_d = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 12) as *const __m256i
            ));
            sum_a3 = _mm512_fmadd_ps(w_d, s_d, sum_a3);

            let s_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 16))),
            );
            let w_e = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 16) as *const __m256i
            ));
            sum_b0 = _mm512_fmadd_ps(w_e, s_e, sum_b0);

            let s_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 20))),
            );
            let w_f = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 20) as *const __m256i
            ));
            sum_b1 = _mm512_fmadd_ps(w_f, s_f, sum_b1);

            let s_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 24))),
            );
            let w_g = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 24) as *const __m256i
            ));
            sum_b2 = _mm512_fmadd_ps(w_g, s_g, sum_b2);

            let s_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 28))),
            );
            let w_h = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 28) as *const __m256i
            ));
            sum_b3 = _mm512_fmadd_ps(w_h, s_h, sum_b3);

            i += 32;
        }

        while i + 16 <= len {
            let s_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i))),
            );
            let w_a =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            sum_a0 = _mm512_fmadd_ps(w_a, s_a, sum_a0);

            let s_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 4))),
            );
            let w_b = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
            sum_a1 = _mm512_fmadd_ps(w_b, s_b, sum_a1);

            let s_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 8))),
            );
            let w_c = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 8) as *const __m256i
            ));
            sum_a2 = _mm512_fmadd_ps(w_c, s_c, sum_a2);

            let s_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 12))),
            );
            let w_d = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 12) as *const __m256i
            ));
            sum_a3 = _mm512_fmadd_ps(w_d, s_d, sum_a3);

            i += 16;
        }

        while i + 4 <= len {
            let s = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i))),
            );
            let w = _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            sum_a0 = _mm512_fmadd_ps(w, s, sum_a0);
            i += 4;
        }

        let s_ab = _mm512_add_ps(_mm512_add_ps(sum_a0, sum_a1), _mm512_add_ps(sum_a2, sum_a3));
        let s_all = _mm512_add_ps(
            s_ab,
            _mm512_add_ps(_mm512_add_ps(sum_b0, sum_b1), _mm512_add_ps(sum_b2, sum_b3)),
        );

        let lo = _mm512_extractf32x4_ps(s_all, 0);
        let hi0 = _mm512_extractf32x4_ps(s_all, 1);
        let hi1 = _mm512_extractf32x4_ps(s_all, 2);
        let hi2 = _mm512_extractf32x4_ps(s_all, 3);
        let mut sum128 = _mm_add_ps(_mm_add_ps(lo, hi0), _mm_add_ps(hi1, hi2));

        while i < len {
            let s0 = _mm_load1_ps(state.as_ptr().add(i));
            let w0 = _mm_cvtph_ps(_mm_loadu_si64(
                weights.as_ptr().add(i) as *const u16 as *const u8
            ));
            sum128 = _mm_fmadd_ps(w0, s0, sum128);
            i += 1;
        }

        let mut out = [0.0; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum128);
        out
    }
}

/// Processa dois frames simultaneamente com 16 acumuladores ZMM (8 por frame)
/// em 2 conjuntos alternados para quebrar dependências FMA.
///
/// Reutiliza a mesma carga de pesos para ambos os frames (f0 e f1),
/// dobrando a eficiência de acesso à memória.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_interleaved_dual_frame_avx512(
    weights: &[[u16; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut i = 0;

    unsafe {
        let perm_idx = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

        let mut sum0_a0 = _mm512_setzero_ps();
        let mut sum0_a1 = _mm512_setzero_ps();
        let mut sum0_a2 = _mm512_setzero_ps();
        let mut sum0_a3 = _mm512_setzero_ps();
        let mut sum0_b0 = _mm512_setzero_ps();
        let mut sum0_b1 = _mm512_setzero_ps();
        let mut sum0_b2 = _mm512_setzero_ps();
        let mut sum0_b3 = _mm512_setzero_ps();

        let mut sum1_a0 = _mm512_setzero_ps();
        let mut sum1_a1 = _mm512_setzero_ps();
        let mut sum1_a2 = _mm512_setzero_ps();
        let mut sum1_a3 = _mm512_setzero_ps();
        let mut sum1_b0 = _mm512_setzero_ps();
        let mut sum1_b1 = _mm512_setzero_ps();
        let mut sum1_b2 = _mm512_setzero_ps();
        let mut sum1_b3 = _mm512_setzero_ps();

        while i + 32 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state_f0.as_ptr().add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(state_f1.as_ptr().add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 32) as *const i8);

            let w_a =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            let s0_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i))),
            );
            let s1_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w_a, s0_a, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w_a, s1_a, sum1_a0);

            let w_b = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
            let s0_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 4))),
            );
            let s1_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 4))),
            );
            sum0_a1 = _mm512_fmadd_ps(w_b, s0_b, sum0_a1);
            sum1_a1 = _mm512_fmadd_ps(w_b, s1_b, sum1_a1);

            let w_c = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 8) as *const __m256i
            ));
            let s0_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 8))),
            );
            let s1_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 8))),
            );
            sum0_a2 = _mm512_fmadd_ps(w_c, s0_c, sum0_a2);
            sum1_a2 = _mm512_fmadd_ps(w_c, s1_c, sum1_a2);

            let w_d = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 12) as *const __m256i
            ));
            let s0_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 12))),
            );
            let s1_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 12))),
            );
            sum0_a3 = _mm512_fmadd_ps(w_d, s0_d, sum0_a3);
            sum1_a3 = _mm512_fmadd_ps(w_d, s1_d, sum1_a3);

            let w_e = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 16) as *const __m256i
            ));
            let s0_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 16))),
            );
            let s1_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 16))),
            );
            sum0_b0 = _mm512_fmadd_ps(w_e, s0_e, sum0_b0);
            sum1_b0 = _mm512_fmadd_ps(w_e, s1_e, sum1_b0);

            let w_f = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 20) as *const __m256i
            ));
            let s0_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 20))),
            );
            let s1_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 20))),
            );
            sum0_b1 = _mm512_fmadd_ps(w_f, s0_f, sum0_b1);
            sum1_b1 = _mm512_fmadd_ps(w_f, s1_f, sum1_b1);

            let w_g = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 24) as *const __m256i
            ));
            let s0_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 24))),
            );
            let s1_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 24))),
            );
            sum0_b2 = _mm512_fmadd_ps(w_g, s0_g, sum0_b2);
            sum1_b2 = _mm512_fmadd_ps(w_g, s1_g, sum1_b2);

            let w_h = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 28) as *const __m256i
            ));
            let s0_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 28))),
            );
            let s1_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 28))),
            );
            sum0_b3 = _mm512_fmadd_ps(w_h, s0_h, sum0_b3);
            sum1_b3 = _mm512_fmadd_ps(w_h, s1_h, sum1_b3);

            i += 32;
        }

        while i + 16 <= len {
            let w_a =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            let s0_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i))),
            );
            let s1_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w_a, s0_a, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w_a, s1_a, sum1_a0);

            let w_b = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
            let s0_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 4))),
            );
            let s1_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 4))),
            );
            sum0_a1 = _mm512_fmadd_ps(w_b, s0_b, sum0_a1);
            sum1_a1 = _mm512_fmadd_ps(w_b, s1_b, sum1_a1);

            let w_c = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 8) as *const __m256i
            ));
            let s0_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 8))),
            );
            let s1_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 8))),
            );
            sum0_a2 = _mm512_fmadd_ps(w_c, s0_c, sum0_a2);
            sum1_a2 = _mm512_fmadd_ps(w_c, s1_c, sum1_a2);

            let w_d = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 12) as *const __m256i
            ));
            let s0_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 12))),
            );
            let s1_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 12))),
            );
            sum0_a3 = _mm512_fmadd_ps(w_d, s0_d, sum0_a3);
            sum1_a3 = _mm512_fmadd_ps(w_d, s1_d, sum1_a3);

            i += 16;
        }

        while i + 4 <= len {
            let w = _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            let s0 = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i))),
            );
            let s1 = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w, s0, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w, s1, sum1_a0);
            i += 4;
        }

        let s0_ab = _mm512_add_ps(
            _mm512_add_ps(sum0_a0, sum0_a1),
            _mm512_add_ps(sum0_a2, sum0_a3),
        );
        let s0_all = _mm512_add_ps(
            s0_ab,
            _mm512_add_ps(
                _mm512_add_ps(sum0_b0, sum0_b1),
                _mm512_add_ps(sum0_b2, sum0_b3),
            ),
        );
        let s1_ab = _mm512_add_ps(
            _mm512_add_ps(sum1_a0, sum1_a1),
            _mm512_add_ps(sum1_a2, sum1_a3),
        );
        let s1_all = _mm512_add_ps(
            s1_ab,
            _mm512_add_ps(
                _mm512_add_ps(sum1_b0, sum1_b1),
                _mm512_add_ps(sum1_b2, sum1_b3),
            ),
        );

        let lo0 = _mm512_extractf32x4_ps(s0_all, 0);
        let hi00 = _mm512_extractf32x4_ps(s0_all, 1);
        let hi01 = _mm512_extractf32x4_ps(s0_all, 2);
        let hi02 = _mm512_extractf32x4_ps(s0_all, 3);
        let mut sum128_f0 = _mm_add_ps(_mm_add_ps(lo0, hi00), _mm_add_ps(hi01, hi02));

        let lo1 = _mm512_extractf32x4_ps(s1_all, 0);
        let hi10 = _mm512_extractf32x4_ps(s1_all, 1);
        let hi11 = _mm512_extractf32x4_ps(s1_all, 2);
        let hi12 = _mm512_extractf32x4_ps(s1_all, 3);
        let mut sum128_f1 = _mm_add_ps(_mm_add_ps(lo1, hi10), _mm_add_ps(hi11, hi12));

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

        let mut out_f0 = [0.0; 4];
        let mut out_f1 = [0.0; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), sum128_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), sum128_f1);
        (out_f0, out_f1)
    }
}

#[cfg(test)]
#[path = "dot_4x_test.rs"]
mod dot_4x_test;
