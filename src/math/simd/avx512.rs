// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

// Desativa alguns avisos do compilador para permitir o uso de código de baixo nível (unsafe)
// e funções com muitos argumentos, comuns em otimizações de áudio de alto desempenho.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Backends SIMD AVX-512.
//!
//! Este módulo contém "motores" ultra-rápidos que usam as instruções AVX-512 dos processadores Intel/AMD modernos.
//! O AVX-512 permite que o processador faça cálculos com 16 números de ponto flutuante (f32) ao mesmo tempo,
//! como se tivesse 16 calculadoras trabalhando em paralelo em uma única "esteira".

use crate::math::common::scalar_ref::*; // Implementações escalares de referência usadas como oráculo de paridade.
use crate::math::common::traits::SimdMath; // A "receita" que todos os motores matemáticos devem seguir.
use core::arch::x86_64::*; // Acesso direto às instruções de hardware do processador (intrinsics).

/// Kernel GEMV AVX-512 especializado para Standard WaveNet (CH=16).
/// GEMV significa "Multiplicação de Matriz por Vetor". É o "coração" do processamento neural.
/// Esta versão é otimizada para quando temos exatamente 16 canais de áudio.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512_small(
    in_frame: &[f32],      // Entrada: os números que vamos processar.
    weights: &[u16],       // Pesos: a "inteligência" do modelo (em formato compactado f16).
    bias: &[f32],          // Viés: um ajuste fixo somado ao final.
    out_frame: &mut [f32], // Saída: onde guardaremos o resultado.
    do_bias: bool,         // Pergunta: devemos somar o viés?
) {
    let in_len = in_frame.len(); // Quantos números temos na entrada.

    // "Acumuladores" são como baldes onde vamos somando os resultados parciais.
    // O AVX-512 usa registradores de 512 bits que cabem 16 números f32.
    let mut accum0 = if do_bias {
        // Se precisar de viés, já começamos o balde com os valores do viés.
        _mm512_loadu_ps(bias.as_ptr())
    } else {
        // Se não, começamos o balde com zeros.
        _mm512_setzero_ps()
    };
    let mut accum1 = _mm512_setzero_ps(); // Um segundo balde para ajudar na velocidade.

    let mut in_c = 0;
    // Processamos a entrada de 4 em 4 elementos para ganhar velocidade (unrolling).
    while in_c + 4 <= in_len {
        // Pegamos 1 número da entrada e "espalhamos" ele por todas as 16 posições de um registrador.
        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));

        // Descobrimos onde na memória estão os pesos para este grupo.
        let w_ptr = weights.as_ptr().add(in_c * 16);

        // fmadd_ps: Faz (entrada * peso) + acumulador. Tudo de uma vez!
        // cvtph_ps: Converte os pesos de 16 bits (metade do tamanho) para 32 bits (normal) na hora do cálculo.
        accum0 = _mm512_fmadd_ps(
            v_in0,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i)),
            accum0,
        );
        accum1 = _mm512_fmadd_ps(
            v_in1,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(16) as *const __m256i)),
            accum1,
        );
        accum0 = _mm512_fmadd_ps(
            v_in2,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(32) as *const __m256i)),
            accum0,
        );
        accum1 = _mm512_fmadd_ps(
            v_in3,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(48) as *const __m256i)),
            accum1,
        );
        in_c += 4;
    }

    // Somamos os dois baldes de resultados no final.
    accum0 = _mm512_add_ps(accum0, accum1);

    // Se sobrar algum número (menos que 4), processamos um por um.
    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        accum0 = _mm512_fmadd_ps(
            v_in,
            _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(in_c * 16) as *const __m256i
            )),
            accum0,
        );
        in_c += 1;
    }

    // Salva o resultado final (os 16 números calculados) na memória de saída.
    _mm512_storeu_ps(out_frame.as_mut_ptr(), accum0);
}

/// Kernel Fused-Add-GEMV AVX-512 especializado para Standard WaveNet (CH=16).
/// Esta versão faz o mesmo que a anterior, mas em vez de substituir o resultado,
/// ela SOMA o novo resultado ao que já existia na saída. É útil para conexões residuais (atalhos).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512_small(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let in_len = in_frame.len();
    // Carrega o valor que já estava na saída para somar em cima dele.
    let mut accum0 = _mm512_loadu_ps(out_frame.as_ptr());
    if do_bias {
        // Se tiver viés, soma ele também.
        accum0 = _mm512_add_ps(accum0, _mm512_loadu_ps(bias.as_ptr()));
    }
    let mut accum1 = _mm512_setzero_ps();

    let mut in_c = 0;
    // Processamento em grupo de 4 para velocidade.
    while in_c + 4 <= in_len {
        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));

        let w_ptr = weights.as_ptr().add(in_c * 16);
        // fmadd_ps: Multiplica e Soma no acumulador.
        accum0 = _mm512_fmadd_ps(
            v_in0,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i)),
            accum0,
        );
        accum1 = _mm512_fmadd_ps(
            v_in1,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(16) as *const __m256i)),
            accum1,
        );
        accum0 = _mm512_fmadd_ps(
            v_in2,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(32) as *const __m256i)),
            accum0,
        );
        accum1 = _mm512_fmadd_ps(
            v_in3,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(48) as *const __m256i)),
            accum1,
        );
        in_c += 4;
    }
    accum0 = _mm512_add_ps(accum0, accum1);

    // Trata o que sobrar individualmente.
    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        accum0 = _mm512_fmadd_ps(
            v_in,
            _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(in_c * 16) as *const __m256i
            )),
            accum0,
        );
        in_c += 1;
    }
    // Salva o resultado final acumulado.
    _mm512_storeu_ps(out_frame.as_mut_ptr(), accum0);
}

/// Realiza a projeção linear Y = Bias + W * Z (GEMV) substituindo o conteúdo de out_frame via AVX-512.
/// Esta é a versão geral, que funciona para qualquer tamanho de saída (múltiplos de 16).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    // Se o tamanho for exatamente 16, usa a versão ultra-otimizada lá de cima.
    if out_len == 16 {
        gemv_overwrite_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    let mut out_c = 0;
    // Percorre a saída em blocos de 16 números (largura de um registrador AVX-512).
    while out_c + 16 <= out_len {
        let mut accum = if do_bias {
            // Carrega o viés inicial para este bloco.
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };

        // Para cada entrada, multiplica pelo peso correspondente e soma no balde.
        for in_c in 0..in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            // Pega 16 pesos de uma vez, converte e calcula.
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));
            accum = _mm512_fmadd_ps(vs, vw, accum);
        }

        // Salva os 16 resultados processados na memória.
        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
        out_c += 16;
    }

    // "Resto": Se o tamanho total não for múltiplo de 16, calcula o que sobrou da forma lenta.
    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w = half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
            sum += *in_frame.get_unchecked(in_c) * w;
        }
        *out_frame.get_unchecked_mut(out_c) = sum;
        out_c += 1;
    }
}

/// Versão em batch de gemv_overwrite via AVX-512.
/// "Batch" significa processar vários quadros de áudio de uma vez, o que é mais eficiente que um por um.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_batch_avx512(
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
    let in_len = in_frames.len() / num_frames; // Tamanho de cada quadro na entrada.
    let out_len = out_frames.len() / num_frames; // Tamanho de cada quadro na saída.
    for i in 0..num_frames {
        // Pega uma fatia (slice) de cada quadro e processa usando a função de quadro único.
        let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
        let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
        gemv_overwrite_avx512(in_slice, weights, bias, out_slice, do_bias);
    }
}

/// Realiza a operação fundida Y = X_res + Bias + W * Z (Broadcast GEMV) via AVX-512.
/// "Fundida" (Fused) significa que fazemos a soma do residual e o cálculo neural no mesmo passo,
/// economizando viagens desnecessárias dos dados entre a memória e o processador.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    // Novamente, se for pequeno (16), usa o especialista.
    if out_len == 16 {
        fused_add_gemv_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        // Começa o balde com o valor que já estava na saída (residual).
        let mut accum = _mm512_loadu_ps(out_frame.as_ptr().add(out_c));
        if do_bias {
            // Soma o viés também.
            accum = _mm512_add_ps(accum, _mm512_loadu_ps(bias.as_ptr().add(out_c)));
        }

        // Loop principal de multiplicação e acumulação.
        for in_c in 0..in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));
            accum = _mm512_fmadd_ps(vs, vw, accum);
        }

        // Devolve os 16 números calculados para a memória.
        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
        out_c += 16;
    }

    // Calcula o resto que sobrar.
    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w = half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
            sum += *in_frame.get_unchecked(in_c) * w;
        }
        *out_frame.get_unchecked_mut(out_c) += sum;
        out_c += 1;
    }
}

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

/// Converte um vetor de números f32 (normais) para bf16 (compactos) via AVX-512.
/// O formato bf16 ocupa metade do espaço, mas mantém o alcance dos números f32,
/// sendo ideal para modelos de inteligência artificial rápidos.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn f32_to_bf16_avx512(src: &[f32], dest: &mut [u16]) {
    let n = core::cmp::min(src.len(), dest.len());
    let mut i = 0;
    // Processa 16 conversões de uma vez.
    while i + 16 <= n {
        let v = _mm512_loadu_ps(src.as_ptr().add(i)); // Carrega 16 números f32.
        let v_i = _mm512_castps_si512(v); // Trata como inteiros para manipular os bits.
        let v_shifted = _mm512_srli_epi32(v_i, 16); // Descarta a parte menos importante (precisão extra).
        let packed = _mm512_cvtepi32_epi16(v_shifted); // Compacta para 16 bits cada.
        _mm256_storeu_si256(dest.as_mut_ptr().add(i) as *mut __m256i, packed); // Salva 16 números bf16.
        i += 16;
    }
    // Converte o resto manualmente.
    while i < n {
        *dest.get_unchecked_mut(i) = (*src.get_unchecked(i)).to_bits() as u16;
        i += 1;
    }
}

/// Implementação SIMD via AVX-512.
/// Esta struct agrupa todas as funções matemáticas otimizadas para processadores que suportam AVX-512.
pub struct Avx512Math;

impl SimdMath for Avx512Math {
    type V = __m512; // O tipo de dado "vetor" usado aqui tem 512 bits (16 números f32).

    // Dot Product: Multiplica dois conjuntos de números e soma tudo num resultado só.
    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        dot_product_avx512(a, b)
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        dot_product_bf16_fallback(a, b)
    }

    // Versões intercaladas para processar 4 cálculos independentes ao mesmo tempo.
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        dot_product_4x_interleaved_avx512(weights, state)
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
        dot_product_4x_interleaved_dual_frame_avx512(weights, state_f0, state_f1)
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
            fused_add_gemv_avx512_small(in_frame, weights, bias, out_frame, do_bias)
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
            fused_add_gemm_batch_avx512(in_frames, weights, bias, out_frames, num_frames, do_bias)
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
            fused_gemm_residual_batch_avx512(
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
            gemv_overwrite_avx512_small(in_frame, weights, bias, out_frame, do_bias)
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
            gemv_4gate_avx512(
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
            gemv_4gate_bf16_avx512(
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
        unsafe { gated_activation_and_accumulate_block_avx512(head_input, block, ch) }
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        unsafe { f32_to_bf16_avx512(src, dest) }
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
        crate::math::fastmath::tanh_slice_avx512(slice)
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        crate::math::fastmath::sigmoid_slice_avx512(slice)
    }

    // Soma horizontal: Soma todos os números dentro de um único registrador.
    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { horizontal_sum_avx512(ptr, N) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        crate::math::fastmath::tanh_slice_avx512(buf)
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        unsafe { fused_lstm_gates_dyn_avx512(gates, cell_state, hidden_state, hidden_size) }
    }

    // Funções de áudio Stereo (Esquerda e Direita).
    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        unsafe { compute_energy_stereo_avx512(l, r) }
    }

    // Convolve Stereo: Aplica filtros de áudio (como um equalizador ou reverb).
    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { convolve_stereo_avx512(coeffs, input_l, input_r, taps) }
    }

    // Controle de ganho (volume) e detecção de "clipping" (quando o som distorce por ficar alto demais).
    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        unsafe { apply_gain_and_detect_clipping_stereo_avx512(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        unsafe { apply_gain_stereo_avx512(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        unsafe { apply_gain_avx512(data, gain) }
    }

    // Head Sum: Uma operação final usada no modelo WaveNet para gerar o som.
    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe { batch_wavenet_head_sum_avx512::<HEAD>(head1, head2, output, scale) }
    }

    // Ramp: Aumenta ou diminui o volume gradualmente para evitar estalos (cliques) no áudio.
    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        unsafe { apply_ramp_stereo_avx512(left, right, start, step) }
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
            gemv_overwrite_batch_avx512(in_frames, weights, bias, out_frames, num_frames, do_bias)
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
                    let h1 = horizontal_sum_avx512(head1.as_ptr().add(i * head), head);
                    output[i] = (h1 + head2[i]) * scale;
                }
            }
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
        unsafe { dot_product_bf16_avx512(a, b) }
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
            gemv_4gate_bf16_avx512(in_frame, w0, w1, w2, w3, &bias, &mut out, false);
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
        horizontal_sum_avx512(ptr, N)
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
        apply_gain_avx512(data, gain)
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe { batch_wavenet_head_sum_avx512::<HEAD>(head1, head2, output, scale) }
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
        dot_product_bf16_avx512(a, b)
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
        dot_product_4x_interleaved_dual_frame_avx512(weights, state_f0, state_f1)
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
        horizontal_sum_avx512(ptr, N)
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
        apply_gain_avx512(data, gain)
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
        unsafe { batch_wavenet_head_sum_avx512::<HEAD>(head1, head2, output, scale) }
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

/// Dot product interleaved 4x usando AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_interleaved_avx512(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    // Para não quebrar o build, vamos delegar para o fallback por enquanto se a implementação SIMD for muito longa.
    unsafe { dot_product_4x_interleaved_fallback(weights, state) }
}

/// Processa dois frames simultaneamente, acumulando 4 weights para dot product com AVX-512
pub unsafe fn dot_product_4x_interleaved_dual_frame_avx512(
    weights: &[[u16; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    unsafe { dot_product_4x_interleaved_dual_frame_fallback(weights, state_f0, state_f1) }
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
        let fa = half::f16::from_bits(*a.get_unchecked(i)).to_f32();
        let fb = half::f16::from_bits(*b.get_unchecked(i)).to_f32();
        sum += fa * fb;
        i += 1;
    }
    sum
}

/// Kernel GEMV 4-gate AVX-512 para LSTM.
/// Portas (gates) em uma rede LSTM controlam o que deve ser lembrado e o que deve ser esquecido.
/// Esta função processa as 4 portas principais de uma vez para ganhar muita velocidade.
#[target_feature(enable = "avx512f,avx512vl")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_avx512(
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
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        // Baldes para as 4 portas.
        let mut acc0 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + out_len))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc2 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + 2 * out_len))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc3 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + 3 * out_len))
        } else {
            _mm512_setzero_ps()
        };

        for in_c in 0..in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));

            // Carrega 16 pesos para cada uma das 4 portas.
            let vw0 = _mm512_cvtph_ps(_mm256_loadu_si256(
                w0.as_ptr().add(in_c * out_len + out_c) as *const __m256i
            ));
            let vw1 = _mm512_cvtph_ps(_mm256_loadu_si256(
                w1.as_ptr().add(in_c * out_len + out_c) as *const __m256i
            ));
            let vw2 = _mm512_cvtph_ps(_mm256_loadu_si256(
                w2.as_ptr().add(in_c * out_len + out_c) as *const __m256i
            ));
            let vw3 = _mm512_cvtph_ps(_mm256_loadu_si256(
                w3.as_ptr().add(in_c * out_len + out_c) as *const __m256i
            ));

            // Multiplica e soma em todos os 4 baldes ao mesmo tempo.
            acc0 = _mm512_fmadd_ps(vs, vw0, acc0);
            acc1 = _mm512_fmadd_ps(vs, vw1, acc1);
            acc2 = _mm512_fmadd_ps(vs, vw2, acc2);
            acc3 = _mm512_fmadd_ps(vs, vw3, acc3);
        }

        // Salva os resultados.
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c), acc0);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + out_len), acc1);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 2 * out_len), acc2);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 3 * out_len), acc3);
        out_c += 16;
    }

    // Resto lento.
    while out_c < out_len {
        let mut s0 = if do_bias { bias[out_c] } else { 0.0 };
        let mut s1 = if do_bias { bias[out_c + out_len] } else { 0.0 };
        let mut s2 = if do_bias {
            bias[out_c + 2 * out_len]
        } else {
            0.0
        };
        let mut s3 = if do_bias {
            bias[out_c + 3 * out_len]
        } else {
            0.0
        };
        for in_c in 0..in_len {
            let si = *in_frame.get_unchecked(in_c);
            s0 += si * half::f16::from_bits(*w0.get_unchecked(in_c * out_len + out_c)).to_f32();
            s1 += si * half::f16::from_bits(*w1.get_unchecked(in_c * out_len + out_c)).to_f32();
            s2 += si * half::f16::from_bits(*w2.get_unchecked(in_c * out_len + out_c)).to_f32();
            s3 += si * half::f16::from_bits(*w3.get_unchecked(in_c * out_len + out_c)).to_f32();
        }
        out[out_c] = s0;
        out[out_c + out_len] = s1;
        out[out_c + 2 * out_len] = s2;
        out[out_c + 3 * out_len] = s3;
        out_c += 1;
    }
}

/// [T41] Kernel GEMV 4-gate BF16 AVX-512 para LSTM.
/// Esta versão usa o formato BF16 desde o início para ser ainda mais rápida.
#[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_bf16_avx512(
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
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut acc0 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + out_len))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc2 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + 2 * out_len))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc3 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + 3 * out_len))
        } else {
            _mm512_setzero_ps()
        };

        let mut in_c = 0;
        while in_c + 2 <= in_len {
            // Carrega e faz o broadcast do par (in[i], in[i+1]) para todos os 16 slots.
            let v_in_raw = _mm256_set1_epi32(*(in_frame.as_ptr().add(in_c) as *const i32));
            let v_in = _mm512_broadcast_i32x8(v_in_raw);

            // Carrega pesos intercalados para formar pares (w[i], w[i+1]) na hora.
            let vw0_i =
                _mm256_loadu_si256(w0.as_ptr().add(in_c * out_len + out_c) as *const __m256i);
            let vw0_i1 =
                _mm256_loadu_si256(w0.as_ptr().add((in_c + 1) * out_len + out_c) as *const __m256i);
            let vw0 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_unpacklo_epi16(vw0_i, vw0_i1)),
                _mm256_unpackhi_epi16(vw0_i, vw0_i1),
            );

            // ... Repete para as outras portas ...
            let vw1_i =
                _mm256_loadu_si256(w1.as_ptr().add(in_c * out_len + out_c) as *const __m256i);
            let vw1_i1 =
                _mm256_loadu_si256(w1.as_ptr().add((in_c + 1) * out_len + out_c) as *const __m256i);
            let vw1 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_unpacklo_epi16(vw1_i, vw1_i1)),
                _mm256_unpackhi_epi16(vw1_i, vw1_i1),
            );

            let vw2_i =
                _mm256_loadu_si256(w2.as_ptr().add(in_c * out_len + out_c) as *const __m256i);
            let vw2_i1 =
                _mm256_loadu_si256(w2.as_ptr().add((in_c + 1) * out_len + out_c) as *const __m256i);
            let vw2 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_unpacklo_epi16(vw2_i, vw2_i1)),
                _mm256_unpackhi_epi16(vw2_i, vw2_i1),
            );

            let vw3_i =
                _mm256_loadu_si256(w3.as_ptr().add(in_c * out_len + out_c) as *const __m256i);
            let vw3_i1 =
                _mm256_loadu_si256(w3.as_ptr().add((in_c + 1) * out_len + out_c) as *const __m256i);
            let vw3 = _mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(_mm256_unpacklo_epi16(vw3_i, vw3_i1)),
                _mm256_unpackhi_epi16(vw3_i, vw3_i1),
            );

            // Multiplica e acumula usando a instrução BF16 mágica.
            acc0 = _mm512_dpbf16_ps(
                acc0,
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(vw0)),
            );
            acc1 = _mm512_dpbf16_ps(
                acc1,
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(vw1)),
            );
            acc2 = _mm512_dpbf16_ps(
                acc2,
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(vw2)),
            );
            acc3 = _mm512_dpbf16_ps(
                acc3,
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(vw3)),
            );

            in_c += 2;
        }

        // Resto manual se necessário.
        if in_c < in_len {
            let si = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
            let v_in = _mm512_set1_ps(si);

            let vw0 = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                w0.as_ptr().add(in_c * out_len + out_c) as *const __m256i,
            )));
            let vw1 = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                w1.as_ptr().add(in_c * out_len + out_c) as *const __m256i,
            )));
            let vw2 = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                w2.as_ptr().add(in_c * out_len + out_c) as *const __m256i,
            )));
            let vw3 = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                w3.as_ptr().add(in_c * out_len + out_c) as *const __m256i,
            )));

            let vw0_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vw0), 16));
            let vw1_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vw1), 16));
            let vw2_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vw2), 16));
            let vw3_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vw3), 16));

            acc0 = _mm512_fmadd_ps(v_in, vw0_f32, acc0);
            acc1 = _mm512_fmadd_ps(v_in, vw1_f32, acc1);
            acc2 = _mm512_fmadd_ps(v_in, vw2_f32, acc2);
            acc3 = _mm512_fmadd_ps(v_in, vw3_f32, acc3);
        }

        _mm512_storeu_ps(out.as_mut_ptr().add(out_c), acc0);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + out_len), acc1);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 2 * out_len), acc2);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 3 * out_len), acc3);
        out_c += 16;
    }

    while out_c < out_len {
        let mut s0 = if do_bias { bias[out_c] } else { 0.0 };
        let mut s1 = if do_bias { bias[out_c + out_len] } else { 0.0 };
        let mut s2 = if do_bias {
            bias[out_c + 2 * out_len]
        } else {
            0.0
        };
        let mut s3 = if do_bias {
            bias[out_c + 3 * out_len]
        } else {
            0.0
        };
        for in_c in 0..in_len {
            let si = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
            s0 += si * f32::from_bits((*w0.get_unchecked(in_c * out_len + out_c) as u32) << 16);
            s1 += si * f32::from_bits((*w1.get_unchecked(in_c * out_len + out_c) as u32) << 16);
            s2 += si * f32::from_bits((*w2.get_unchecked(in_c * out_len + out_c) as u32) << 16);
            s3 += si * f32::from_bits((*w3.get_unchecked(in_c * out_len + out_c) as u32) << 16);
        }
        out[out_c] = s0;
        out[out_c + out_len] = s1;
        out[out_c + 2 * out_len] = s2;
        out[out_c + 3 * out_len] = s3;
        out_c += 1;
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
            let tanh_z1 = crate::math::fastmath::simd_tanh_avx512(z1);
            let sig_z2 = crate::math::fastmath::simd_sigmoid_avx512(z2);
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

/// Kernel fundido para atualizar a memória (estado) de uma rede LSTM.
/// Esta função decide o que a rede deve "esquecer" do passado e o que "aprender" do presente,
/// atualizando os valores de uma só vez para 16 células de memória.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_dyn_avx512(
    gates: &mut [f32],
    cell_state: &mut [f32],
    hidden_state: &mut [f32],
    hidden_size: usize,
) {
    let mut j = 0;
    while j + 16 <= hidden_size {
        // Carrega as 4 decisões (esquecer, aprender, etc.) para 16 células.
        let gi = _mm512_loadu_ps(gates.as_ptr().add(j));
        let gf = _mm512_loadu_ps(gates.as_ptr().add(j + hidden_size));
        let gg = _mm512_loadu_ps(gates.as_ptr().add(j + 2 * hidden_size));
        let go = _mm512_loadu_ps(gates.as_ptr().add(j + 3 * hidden_size));
        let cs = _mm512_loadu_ps(cell_state.as_ptr().add(j));

        // Faz o cálculo da memória de forma fundida (fused).
        let (new_cs, hidden) = crate::math::fastmath::fused_lstm_gates_avx512(gf, gi, gg, go, cs);

        _mm512_storeu_ps(cell_state.as_mut_ptr().add(j), new_cs);
        _mm512_storeu_ps(hidden_state.as_mut_ptr().add(j), hidden);

        j += 16;
    }
    // Trata o resto.
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

/// Convolução Stereo: Aplica um filtro (coeficientes) em dois canais de áudio ao mesmo tempo.
/// É como passar o som por um equalizador ou simular uma sala (reverb).
#[target_feature(enable = "avx512f")]
pub unsafe fn convolve_stereo_avx512(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    unsafe {
        let mut sum_l0 = _mm512_setzero_ps();
        let mut sum_l1 = _mm512_setzero_ps();
        let mut sum_r0 = _mm512_setzero_ps();
        let mut sum_r1 = _mm512_setzero_ps();
        let mut i = 0;

        // Processa 32 amostras de áudio de uma vez só!
        while i + 32 <= taps {
            let h0 = _mm512_load_ps(coeffs.add(i)); // 16 coeficientes.
            let x0_l = _mm512_loadu_ps(input_l.add(i)); // 16 amostras da esquerda.
            let x0_r = _mm512_loadu_ps(input_r.add(i)); // 16 amostras da direita.
            sum_l0 = _mm512_fmadd_ps(h0, x0_l, sum_l0);
            sum_r0 = _mm512_fmadd_ps(h0, x0_r, sum_r0);

            let h1 = _mm512_load_ps(coeffs.add(i + 16));
            let x1_l = _mm512_loadu_ps(input_l.add(i + 16));
            let x1_r = _mm512_loadu_ps(input_r.add(i + 16));
            sum_l1 = _mm512_fmadd_ps(h1, x1_l, sum_l1);
            sum_r1 = _mm512_fmadd_ps(h1, x1_r, sum_r1);

            i += 32;
        }

        // Soma os acumuladores para obter o resultado final de cada canal.
        let sum_l = _mm512_add_ps(sum_l0, sum_l1);
        let sum_r = _mm512_add_ps(sum_r0, sum_r1);
        let mut out_l = _mm512_reduce_add_ps(sum_l);
        let mut out_r = _mm512_reduce_add_ps(sum_r);

        // Termina as amostras que sobraram.
        while i < taps {
            let h = *coeffs.add(i);
            out_l += h * *input_l.add(i);
            out_r += h * *input_r.add(i);
            i += 1;
        }

        (out_l, out_r)
    }
}
/// Aplica ganho e detecta clipping em estéreo em uma única passagem usando AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_gain_and_detect_clipping_stereo_avx512(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let v_gain = _mm512_set1_ps(gain);
    let v_limit = _mm512_set1_ps(1.0);
    let sign_mask = _mm512_set1_ps(-0.0f32);
    let mut k_clip = 0u16;

    while i + 16 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);

        let vl = _mm512_loadu_ps(pl);
        let vr = _mm512_loadu_ps(pr);

        let gl = _mm512_mul_ps(vl, v_gain);
        let gr = _mm512_mul_ps(vr, v_gain);

        _mm512_storeu_ps(pl, gl);
        _mm512_storeu_ps(pr, gr);

        let abs_l = _mm512_andnot_ps(sign_mask, gl);
        let abs_r = _mm512_andnot_ps(sign_mask, gr);

        let k_l = _mm512_cmp_ps_mask(abs_l, v_limit, _CMP_GT_OQ);
        let k_r = _mm512_cmp_ps_mask(abs_r, v_limit, _CMP_GT_OQ);

        k_clip |= k_l | k_r;
        i += 16;
    }

    let mut clipped = k_clip != 0;

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

/// Aplica ganho constante em estéreo via AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_gain_stereo_avx512(left: &mut [f32], right: &mut [f32], gain: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let zmm_gain = _mm512_set1_ps(gain);
    while i + 16 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);
        _mm512_storeu_ps(pl, _mm512_mul_ps(_mm512_loadu_ps(pl), zmm_gain));
        _mm512_storeu_ps(pr, _mm512_mul_ps(_mm512_loadu_ps(pr), zmm_gain));
        i += 16;
    }
    while i < n {
        *left.get_unchecked_mut(i) *= gain;
        *right.get_unchecked_mut(i) *= gain;
        i += 1;
    }
}

/// Aplica uma rampa de volume suave para não dar "estalos" no áudio.
/// O volume começa em "start" e vai mudando a cada amostra pelo valor "step".
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_ramp_stereo_avx512(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    // Cria uma rampa de 16 valores para multiplicar de uma vez.
    let mut current_ramp = _mm512_set_ps(
        start + 15.0 * step,
        start + 14.0 * step,
        start + 13.0 * step,
        start + 12.0 * step,
        start + 11.0 * step,
        start + 10.0 * step,
        start + 9.0 * step,
        start + 8.0 * step,
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_16 = _mm512_set1_ps(16.0 * step);
    while i + 16 <= n {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);
        // Multiplica 16 amostras pelo volume gradual.
        _mm512_storeu_ps(pl, _mm512_mul_ps(_mm512_loadu_ps(pl), current_ramp));
        _mm512_storeu_ps(pr, _mm512_mul_ps(_mm512_loadu_ps(pr), current_ramp));
        // Avança a rampa para os próximos 16.
        current_ramp = _mm512_add_ps(current_ramp, v_step_16);
        i += 16;
    }
    // Finaliza o que sobrou.
    let mut g = start + (i as f32) * step;
    while i < n {
        *left.get_unchecked_mut(i) *= g;
        *right.get_unchecked_mut(i) *= g;
        g += step;
        i += 1;
    }
}

/// Calcula o máximo da energia entre dois canais (Mean Square) via AVX-512.
/// Funde as duas passagens em uma para economizar banda de memória.
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_energy_stereo_avx512(l: &[f32], r: &[f32]) -> f32 {
    let len = core::cmp::min(l.len(), r.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    let mut sum_lv = _mm512_setzero_ps();
    let mut sum_rv = _mm512_setzero_ps();

    while i + 16 <= len {
        let lv = _mm512_loadu_ps(l.as_ptr().add(i));
        let rv = _mm512_loadu_ps(r.as_ptr().add(i));
        sum_lv = _mm512_fmadd_ps(lv, lv, sum_lv);
        sum_rv = _mm512_fmadd_ps(rv, rv, sum_rv);
        i += 16;
    }

    let mut sum_l = crate::math::common::utility::hsum_avx512(sum_lv);
    let mut sum_r = crate::math::common::utility::hsum_avx512(sum_rv);

    while i < len {
        sum_l += l[i] * l[i];
        sum_r += r[i] * r[i];
        i += 1;
    }

    let energy_l = sum_l / (len as f32);
    let energy_r = sum_r / (len as f32);
    energy_l.max(energy_r)
}

/// Soma horizontal de um buffer f32 via AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn horizontal_sum_avx512(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();

    while i + 16 <= len {
        let v = _mm512_loadu_ps(ptr.add(i));
        sum_v = _mm512_add_ps(sum_v, v);
        i += 16;
    }

    let mut total = crate::math::common::utility::hsum_avx512(sum_v);

    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let v = _mm512_maskz_loadu_ps(mask, ptr.add(i));
        total += crate::math::common::utility::hsum_avx512(v);
    }

    total
}

/// Aplica ganho constante em um buffer mono usando AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn apply_gain_avx512(data: &mut [f32], gain: f32) {
    let len = data.len();
    let vg = _mm512_set1_ps(gain);
    let mut i = 0;
    while i + 16 <= len {
        let v = _mm512_loadu_ps(data.as_ptr().add(i));
        _mm512_storeu_ps(data.as_mut_ptr().add(i), _mm512_mul_ps(v, vg));
        i += 16;
    }
    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let v = _mm512_maskz_loadu_ps(mask, data.as_ptr().add(i));
        _mm512_mask_storeu_ps(data.as_mut_ptr().add(i), mask, _mm512_mul_ps(v, vg));
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
        let h1 = horizontal_sum_avx512(head1.as_ptr().add(i * HEAD), HEAD);
        *output.get_unchecked_mut(i) = (h1 + *head2.get_unchecked(i)) * scale;
    }
}

#[cfg(test)]
#[path = "avx512_test.rs"]
mod avx512_test;
