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
//!
//! # Estratégia de Paralelismo
//! - AVX2: 4 acumuladores YMM (4×8 = 32 lanes), loop interno com passo 4.
//! - AVX-512: 8 acumuladores ZMM (8×16 = 128 lanes), loop interno com passo 8.
//! - Quebra de cadeias de dependência FMA via múltiplos acumuladores.
//! - Software prefetch em in_frame para reduzir latência de cache miss.

use core::arch::x86_64::*;

// ── AVX2 ──────────────────────────────────────────────────────────────────────

/// Realiza uma operação matemática combinada (fundida) de alta velocidade: Y = X_res + Bias + W * Z.
///
/// Esta função faz três coisas ao mesmo tempo: preserva o valor atual (residual), soma um
/// ajuste (bias) e adiciona o resultado de uma multiplicação de pesos por entrada. Fazer tudo
/// de uma vez evita que o processador precise ler e escrever na memória várias vezes, mantendo
/// os dados "quentes" e prontos para o próximo cálculo.
///
/// Utiliza 4 acumuladores independentes para quebrar a cadeia de dependência do pipeline FMA,
/// permitindo que o processador execute até 4 FMAs em paralelo.
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
        while out_c + 8 <= out_len {
            let mut acc0 = _mm256_loadu_ps(out_frame.as_ptr().add(out_c));
            if do_bias {
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(bias.as_ptr().add(out_c)));
            }
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut in_c = 0;
            while in_c + 4 <= in_len {
                _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 32) as *const i8);

                let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 1));
                let vs2 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 2));
                let vs3 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 3));

                let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs0, w0, acc0);

                let w1 =
                    _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(out_len) as *const __m128i));
                acc1 = _mm256_fmadd_ps(vs1, w1, acc1);

                let w2 =
                    _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(2 * out_len) as *const __m128i));
                acc2 = _mm256_fmadd_ps(vs2, w2, acc2);

                let w3 =
                    _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(3 * out_len) as *const __m128i));
                acc3 = _mm256_fmadd_ps(vs3, w3, acc3);

                in_c += 4;
            }

            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);

            while in_c < in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs, vw, acc0);
                in_c += 1;
            }

            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
            out_c += 8;
        }

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
/// Utiliza 4 acumuladores independentes para quebrar a cadeia de dependência do pipeline FMA.
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
        while out_c + 8 <= out_len {
            let mut acc0 = if do_bias {
                _mm256_loadu_ps(bias.as_ptr().add(out_c))
            } else {
                _mm256_setzero_ps()
            };
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut in_c = 0;
            while in_c + 4 <= in_len {
                _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 32) as *const i8);

                let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 1));
                let vs2 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 2));
                let vs3 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 3));

                let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs0, w0, acc0);

                let w1 =
                    _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(out_len) as *const __m128i));
                acc1 = _mm256_fmadd_ps(vs1, w1, acc1);

                let w2 =
                    _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(2 * out_len) as *const __m128i));
                acc2 = _mm256_fmadd_ps(vs2, w2, acc2);

                let w3 =
                    _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(3 * out_len) as *const __m128i));
                acc3 = _mm256_fmadd_ps(vs3, w3, acc3);

                in_c += 4;
            }

            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc0 = _mm256_add_ps(acc0, acc2);

            while in_c < in_len {
                let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                let vw = _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i));
                acc0 = _mm256_fmadd_ps(vs, vw, acc0);
                in_c += 1;
            }

            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
            out_c += 8;
        }

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
///
/// Utiliza 8 acumuladores ZMM independentes (8×16 = 128 lanes) e loop interno
/// com passo 8, quebrando a cadeia de dependência FMA e saturando as portas
/// de execução AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512_small(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let in_len = in_frame.len();

    let mut acc0 = if do_bias {
        _mm512_loadu_ps(bias.as_ptr())
    } else {
        _mm512_setzero_ps()
    };
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();

    let mut in_c = 0;
    while in_c + 8 <= in_len {
        _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 64) as *const i8);

        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
        let v_in4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
        let v_in5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
        let v_in6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
        let v_in7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

        let w_ptr = weights.as_ptr().add(in_c * 16);

        acc0 = _mm512_fmadd_ps(
            v_in0,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i)),
            acc0,
        );
        acc1 = _mm512_fmadd_ps(
            v_in1,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(16) as *const __m256i)),
            acc1,
        );
        acc2 = _mm512_fmadd_ps(
            v_in2,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(32) as *const __m256i)),
            acc2,
        );
        acc3 = _mm512_fmadd_ps(
            v_in3,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(48) as *const __m256i)),
            acc3,
        );
        acc4 = _mm512_fmadd_ps(
            v_in4,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(64) as *const __m256i)),
            acc4,
        );
        acc5 = _mm512_fmadd_ps(
            v_in5,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(80) as *const __m256i)),
            acc5,
        );
        acc6 = _mm512_fmadd_ps(
            v_in6,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(96) as *const __m256i)),
            acc6,
        );
        acc7 = _mm512_fmadd_ps(
            v_in7,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(112) as *const __m256i)),
            acc7,
        );
        in_c += 8;
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc4 = _mm512_add_ps(acc4, acc5);
    acc6 = _mm512_add_ps(acc6, acc7);
    acc0 = _mm512_add_ps(acc0, acc2);
    acc4 = _mm512_add_ps(acc4, acc6);
    acc0 = _mm512_add_ps(acc0, acc4);

    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        acc0 = _mm512_fmadd_ps(
            v_in,
            _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(in_c * 16) as *const __m256i,
            )),
            acc0,
        );
        in_c += 1;
    }

    _mm512_storeu_ps(out_frame.as_mut_ptr(), acc0);
}

/// Kernel Fused-Add-GEMV AVX-512 especializado para Standard WaveNet (CH=16).
///
/// 8 acumuladores ZMM independentes com passo 8 no loop interno.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512_small(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let in_len = in_frame.len();

    let mut acc0 = _mm512_loadu_ps(out_frame.as_ptr());
    if do_bias {
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(bias.as_ptr()));
    }
    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();

    let mut in_c = 0;
    while in_c + 8 <= in_len {
        _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 64) as *const i8);

        let v_in0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        let v_in1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
        let v_in2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
        let v_in3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
        let v_in4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
        let v_in5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
        let v_in6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
        let v_in7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

        let w_ptr = weights.as_ptr().add(in_c * 16);

        acc0 = _mm512_fmadd_ps(
            v_in0,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i)),
            acc0,
        );
        acc1 = _mm512_fmadd_ps(
            v_in1,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(16) as *const __m256i)),
            acc1,
        );
        acc2 = _mm512_fmadd_ps(
            v_in2,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(32) as *const __m256i)),
            acc2,
        );
        acc3 = _mm512_fmadd_ps(
            v_in3,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(48) as *const __m256i)),
            acc3,
        );
        acc4 = _mm512_fmadd_ps(
            v_in4,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(64) as *const __m256i)),
            acc4,
        );
        acc5 = _mm512_fmadd_ps(
            v_in5,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(80) as *const __m256i)),
            acc5,
        );
        acc6 = _mm512_fmadd_ps(
            v_in6,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(96) as *const __m256i)),
            acc6,
        );
        acc7 = _mm512_fmadd_ps(
            v_in7,
            _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(112) as *const __m256i)),
            acc7,
        );
        in_c += 8;
    }

    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc4 = _mm512_add_ps(acc4, acc5);
    acc6 = _mm512_add_ps(acc6, acc7);
    acc0 = _mm512_add_ps(acc0, acc2);
    acc4 = _mm512_add_ps(acc4, acc6);
    acc0 = _mm512_add_ps(acc0, acc4);

    while in_c < in_len {
        let v_in = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
        acc0 = _mm512_fmadd_ps(
            v_in,
            _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(in_c * 16) as *const __m256i,
            )),
            acc0,
        );
        in_c += 1;
    }

    _mm512_storeu_ps(out_frame.as_mut_ptr(), acc0);
}

// ── AVX-512 Geral ────────────────────────────────────────────────────────────

/// Realiza a projeção linear Y = Bias + W * Z (GEMV) substituindo o conteúdo de out_frame via AVX-512.
///
/// Utiliza 8 acumuladores ZMM independentes (8×16 = 128 lanes) com loop interno
/// de passo 8 para quebrar a cadeia de dependência FMA.
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

    if out_len == 16 {
        gemv_overwrite_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut acc0 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();

        let mut in_c = 0;
        while in_c + 8 <= in_len {
            _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 64) as *const i8);

            let vs0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let vs1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
            let vs2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
            let vs3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
            let vs4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
            let vs5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
            let vs6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
            let vs7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

            let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);

            let w0 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i));
            acc0 = _mm512_fmadd_ps(vs0, w0, acc0);

            let w1 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(out_len) as *const __m256i));
            acc1 = _mm512_fmadd_ps(vs1, w1, acc1);

            let w2 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(2 * out_len) as *const __m256i));
            acc2 = _mm512_fmadd_ps(vs2, w2, acc2);

            let w3 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(3 * out_len) as *const __m256i));
            acc3 = _mm512_fmadd_ps(vs3, w3, acc3);

            let w4 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(4 * out_len) as *const __m256i));
            acc4 = _mm512_fmadd_ps(vs4, w4, acc4);

            let w5 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(5 * out_len) as *const __m256i));
            acc5 = _mm512_fmadd_ps(vs5, w5, acc5);

            let w6 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(6 * out_len) as *const __m256i));
            acc6 = _mm512_fmadd_ps(vs6, w6, acc6);

            let w7 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(7 * out_len) as *const __m256i));
            acc7 = _mm512_fmadd_ps(vs7, w7, acc7);

            in_c += 8;
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc4 = _mm512_add_ps(acc4, acc5);
        acc6 = _mm512_add_ps(acc6, acc7);
        acc0 = _mm512_add_ps(acc0, acc2);
        acc4 = _mm512_add_ps(acc4, acc6);
        acc0 = _mm512_add_ps(acc0, acc4);

        while in_c < in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));
            acc0 = _mm512_fmadd_ps(vs, vw, acc0);
            in_c += 1;
        }

        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
        out_c += 16;
    }

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

/// Versão em batch de gemv_overwrite via AVX-512.
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
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;
    for i in 0..num_frames {
        let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
        let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
        gemv_overwrite_avx512(in_slice, weights, bias, out_slice, do_bias);
    }
}

/// Realiza a operação fundida Y = X_res + Bias + W * Z (Broadcast GEMV) via AVX-512.
///
/// 8 acumuladores ZMM independentes com passo 8 no loop interno.
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

    if out_len == 16 {
        fused_add_gemv_avx512_small(in_frame, weights, bias, out_frame, do_bias);
        return;
    }

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut acc0 = _mm512_loadu_ps(out_frame.as_ptr().add(out_c));
        if do_bias {
            acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(bias.as_ptr().add(out_c)));
        }
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();

        let mut in_c = 0;
        while in_c + 8 <= in_len {
            _mm_prefetch::<_MM_HINT_T0>(in_frame.as_ptr().add(in_c + 64) as *const i8);

            let vs0 = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let vs1 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 1));
            let vs2 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 2));
            let vs3 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 3));
            let vs4 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 4));
            let vs5 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 5));
            let vs6 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 6));
            let vs7 = _mm512_set1_ps(*in_frame.get_unchecked(in_c + 7));

            let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);

            let w0 = _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr as *const __m256i));
            acc0 = _mm512_fmadd_ps(vs0, w0, acc0);

            let w1 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(out_len) as *const __m256i));
            acc1 = _mm512_fmadd_ps(vs1, w1, acc1);

            let w2 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(2 * out_len) as *const __m256i));
            acc2 = _mm512_fmadd_ps(vs2, w2, acc2);

            let w3 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(3 * out_len) as *const __m256i));
            acc3 = _mm512_fmadd_ps(vs3, w3, acc3);

            let w4 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(4 * out_len) as *const __m256i));
            acc4 = _mm512_fmadd_ps(vs4, w4, acc4);

            let w5 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(5 * out_len) as *const __m256i));
            acc5 = _mm512_fmadd_ps(vs5, w5, acc5);

            let w6 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(6 * out_len) as *const __m256i));
            acc6 = _mm512_fmadd_ps(vs6, w6, acc6);

            let w7 =
                _mm512_cvtph_ps(_mm256_loadu_si256(w_ptr.add(7 * out_len) as *const __m256i));
            acc7 = _mm512_fmadd_ps(vs7, w7, acc7);

            in_c += 8;
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc4 = _mm512_add_ps(acc4, acc5);
        acc6 = _mm512_add_ps(acc6, acc7);
        acc0 = _mm512_add_ps(acc0, acc2);
        acc4 = _mm512_add_ps(acc4, acc6);
        acc0 = _mm512_add_ps(acc0, acc4);

        while in_c < in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));
            let weight_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i));
            acc0 = _mm512_fmadd_ps(vs, vw, acc0);
            in_c += 1;
        }

        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
        out_c += 16;
    }

    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w =
                half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
            sum += *in_frame.get_unchecked(in_c) * w;
        }
        *out_frame.get_unchecked_mut(out_c) += sum;
        out_c += 1;
    }
}
