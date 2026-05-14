// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels GEMM em Lote (batch) e GEMM com Residual Fundido — AVX2 e AVX-512.
//!
//! Processam múltiplos quadros de áudio simultaneamente para reuso eficiente de pesos.

use super::gemv::{fused_add_gemv_avx2, fused_add_gemv_avx512};
use core::arch::x86_64::*;

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

// ── AVX-512 ──────────────────────────────────────────────────────────────────

/// Versão em batch da operação fundida Y = X_res + Bias + W * Z via AVX-512.
/// Esta função é o "monstro" da performance. Ela processa 8 quadros de áudio simultaneamente,
/// cada um com 16 canais, totalizando 128 cálculos de uma vez só!
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemm_batch_avx512(
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

    let mut f = 0;
    // Tenta processar em grupos de 8 quadros de cada vez.
    while f + 8 <= num_frames {
        let mut out_c = 0;
        // Percorre os canais de 16 em 16.
        while out_c + 16 <= out_len {
            // Temos 8 baldes (acc0 a acc7), um para cada quadro sendo processado.
            let mut acc0 = _mm512_loadu_ps(out_frames.as_ptr().add(f * out_len + out_c));
            let mut acc1 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 1) * out_len + out_c));
            let mut acc2 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 2) * out_len + out_c));
            let mut acc3 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 3) * out_len + out_c));
            let mut acc4 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 4) * out_len + out_c));
            let mut acc5 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 5) * out_len + out_c));
            let mut acc6 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 6) * out_len + out_c));
            let mut acc7 = _mm512_loadu_ps(out_frames.as_ptr().add((f + 7) * out_len + out_c));

            if do_bias {
                let b = _mm512_loadu_ps(bias.as_ptr().add(out_c));
                // Soma o mesmo viés em todos os 8 baldes (economiza carregar o viés 8 vezes).
                acc0 = _mm512_add_ps(acc0, b);
                acc1 = _mm512_add_ps(acc1, b);
                acc2 = _mm512_add_ps(acc2, b);
                acc3 = _mm512_add_ps(acc3, b);
                acc4 = _mm512_add_ps(acc4, b);
                acc5 = _mm512_add_ps(acc5, b);
                acc6 = _mm512_add_ps(acc6, b);
                acc7 = _mm512_add_ps(acc7, b);
            }

            for in_c in 0..in_len {
                // Carrega 16 pesos comuns a todos os quadros.
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));

                // Pega 1 entrada de cada um dos 8 quadros.
                let vs0 = _mm512_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                let vs1 = _mm512_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c));
                let vs2 = _mm512_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c));
                let vs3 = _mm512_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c));
                let vs4 = _mm512_set1_ps(*in_frames.get_unchecked((f + 4) * in_len + in_c));
                let vs5 = _mm512_set1_ps(*in_frames.get_unchecked((f + 5) * in_len + in_c));
                let vs6 = _mm512_set1_ps(*in_frames.get_unchecked((f + 6) * in_len + in_c));
                let vs7 = _mm512_set1_ps(*in_frames.get_unchecked((f + 7) * in_len + in_c));

                // Multiplica a entrada pelo peso e soma no balde correspondente.
                acc0 = _mm512_fmadd_ps(vs0, vw, acc0);
                acc1 = _mm512_fmadd_ps(vs1, vw, acc1);
                acc2 = _mm512_fmadd_ps(vs2, vw, acc2);
                acc3 = _mm512_fmadd_ps(vs3, vw, acc3);
                acc4 = _mm512_fmadd_ps(vs4, vw, acc4);
                acc5 = _mm512_fmadd_ps(vs5, vw, acc5);
                acc6 = _mm512_fmadd_ps(vs6, vw, acc6);
                acc7 = _mm512_fmadd_ps(vs7, vw, acc7);
            }

            // Salva todos os 8 baldes (128 números f32 no total) de volta na memória.
            _mm512_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 4) * out_len + out_c), acc4);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 5) * out_len + out_c), acc5);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 6) * out_len + out_c), acc6);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 7) * out_len + out_c), acc7);
            out_c += 16;
        }

        // Trata o resto dos canais para os 8 quadros atuais.
        while out_c < out_len {
            for i in 0..8 {
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
        f += 8;
    }

    // Trata os quadros que sobraram (se o total não for múltiplo de 8).
    while f < num_frames {
        fused_add_gemv_avx512(
            in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
            weights,
            bias,
            out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
            do_bias,
        );
        f += 1;
    }
}

/// Kernel GEMM com residual fundido AVX-512.
/// Similar ao anterior, mas o "residual" (o áudio original sem efeito) vem de um local diferente.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_gemm_residual_batch_avx512(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32],
    residual: &[f32], // Entrada residual separada.
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    let mut f = 0;
    // Processamento em grupos de 8 quadros.
    while f + 8 <= num_frames {
        let mut out_c = 0;
        while out_c + 16 <= out_len {
            // Carrega o residual original para começar os baldes.
            let mut acc0 = _mm512_loadu_ps(residual.as_ptr().add(f * out_len + out_c));
            let mut acc1 = _mm512_loadu_ps(residual.as_ptr().add((f + 1) * out_len + out_c));
            let mut acc2 = _mm512_loadu_ps(residual.as_ptr().add((f + 2) * out_len + out_c));
            let mut acc3 = _mm512_loadu_ps(residual.as_ptr().add((f + 3) * out_len + out_c));
            let mut acc4 = _mm512_loadu_ps(residual.as_ptr().add((f + 4) * out_len + out_c));
            let mut acc5 = _mm512_loadu_ps(residual.as_ptr().add((f + 5) * out_len + out_c));
            let mut acc6 = _mm512_loadu_ps(residual.as_ptr().add((f + 6) * out_len + out_c));
            let mut acc7 = _mm512_loadu_ps(residual.as_ptr().add((f + 7) * out_len + out_c));

            if do_bias {
                let b = _mm512_loadu_ps(bias.as_ptr().add(out_c));
                acc0 = _mm512_add_ps(acc0, b);
                acc1 = _mm512_add_ps(acc1, b);
                acc2 = _mm512_add_ps(acc2, b);
                acc3 = _mm512_add_ps(acc3, b);
                acc4 = _mm512_add_ps(acc4, b);
                acc5 = _mm512_add_ps(acc5, b);
                acc6 = _mm512_add_ps(acc6, b);
                acc7 = _mm512_add_ps(acc7, b);
            }

            // Multiplica e acumula para os 8 quadros.
            for in_c in 0..in_len {
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));

                acc0 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked(f * in_len + in_c)),
                    vw,
                    acc0,
                );
                acc1 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 1) * in_len + in_c)),
                    vw,
                    acc1,
                );
                acc2 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 2) * in_len + in_c)),
                    vw,
                    acc2,
                );
                acc3 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 3) * in_len + in_c)),
                    vw,
                    acc3,
                );
                acc4 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 4) * in_len + in_c)),
                    vw,
                    acc4,
                );
                acc5 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 5) * in_len + in_c)),
                    vw,
                    acc5,
                );
                acc6 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 6) * in_len + in_c)),
                    vw,
                    acc6,
                );
                acc7 = _mm512_fmadd_ps(
                    _mm512_set1_ps(*in_frames.get_unchecked((f + 7) * in_len + in_c)),
                    vw,
                    acc7,
                );
            }

            // Salva o resultado final.
            _mm512_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 1) * out_len + out_c), acc1);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 2) * out_len + out_c), acc2);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 3) * out_len + out_c), acc3);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 4) * out_len + out_c), acc4);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 5) * out_len + out_c), acc5);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 6) * out_len + out_c), acc6);
            _mm512_storeu_ps(out_frames.as_mut_ptr().add((f + 7) * out_len + out_c), acc7);
            out_c += 16;
        }

        // Resto dos canais para os 8 quadros.
        while out_c < out_len {
            for i in 0..8 {
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
        f += 8;
    }

    // Resto dos quadros.
    while f < num_frames {
        let in_frame = &in_frames[f * in_len..(f + 1) * in_len];
        let out_frame = &mut out_frames[f * out_len..(f + 1) * out_len];
        let res_frame = &residual[f * out_len..(f + 1) * out_len];

        let mut out_c = 0;
        while out_c + 16 <= out_len {
            let mut accum = _mm512_loadu_ps(res_frame.as_ptr().add(out_c));
            if do_bias {
                accum = _mm512_add_ps(accum, _mm512_loadu_ps(bias.as_ptr().add(out_c)));
            }
            for in_c in 0..in_len {
                let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                accum = _mm512_fmadd_ps(
                    vs,
                    _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i)),
                    accum,
                );
            }
            _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
            out_c += 16;
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
