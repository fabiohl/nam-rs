// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels GEMV 4-Gate para LSTM — AVX2 e AVX-512 (incluindo BF16).

use core::arch::x86_64::*;

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

/// Kernel GEMV 4-gate BF16 AVX-512 para LSTM.
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
