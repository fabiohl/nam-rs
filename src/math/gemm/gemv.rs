// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels GEMV (Multiplicação de Matriz por Vetor) — AVX2 e AVX-512.
//!
//! Inclui variantes `_small` especializadas para Standard WaveNet (CH=16),
//! versões em batch e a operação fundida `fused_add_gemv`.

use core::arch::x86_64::*;

// ── AVX2 ──────────────────────────────────────────────────────────────────────

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

// ── AVX-512 Small (CH=16 especializado) ──────────────────────────────────────

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

// ── AVX-512 Geral ────────────────────────────────────────────────────────────

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
