// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Engenharia de Registradores baseada em instruções explícitas x86_64.
//!
//! Este módulo exporta funções analíticas implementadas com instrinsics de AVX2 e FMA,
//! otimizando os cálculos críticos (como Fused Multiply-Add) limitando os
//! desvios matemáticos inerentes aos loops comuns e reduzindo latência nas CNNs.

use core::arch::x86_64::*;

/// Habilita DAZ (Denormals-Are-Zero) e FTZ (Flush-To-Zero) no registrador MXCSR.
///
/// Em CPUs x86-64, operações sobre números subnormais (denormals) incorrem em
/// penalidade de micro-código na FPU que pode exceder o orçamento temporal do
/// callback `SCHED_FIFO`. Silêncio prolongado de instrumento gera decaimentos
/// exponenciais nos estados LSTM e buffers WaveNet que convergem para denormals.
///
/// Esta função seta os bits 6 (DAZ) e 15 (FTZ) do MXCSR via instruções
/// `stmxcsr` / `ldmxcsr`.
///
/// - **FTZ (bit 15):** Resultados subnormais são truncados para zero.
/// - **DAZ (bit 6):** Operandos subnormais são tratados como zero.
///
/// # Safety
/// O chamador deve assegurar que a CPU é x86_64 com suporte SSE2+.
/// Esta função altera estado global do processador (MXCSR) — deve ser
/// chamada apenas uma vez por thread (tipicamente no início do callback RT).
pub unsafe fn set_daz_ftz() {
    // 0x8040 = bit 15 (FTZ) | bit 6 (DAZ)
    unsafe {
        let mut mxcsr: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut mxcsr);
        mxcsr |= 0x8040;
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &mxcsr);
    }
}

/// Calcula o Dot Product (Produto Escalar) de duas fatias via AVX2 e FMA.
///
/// ## Otimização: 4 Acumuladores Independentes (ILP)
///
/// CPUs modernas (Zen4, Skylake+) possuem 2 portas de FMA com throughput de
/// 0.5 ciclos/instrução, mas **latência de 4–5 ciclos** por FMA. Um único
/// acumulador cria uma cadeia de dependência serial que desperdiça ~87% do
/// pipeline. Os 4 acumuladores independentes (`sum0..sum3`) quebram essa cadeia,
/// permitindo ao scheduler despachar 4 FMAs em paralelo nos 2 ports — saturando
/// o throughput teórico do processador.
///
/// Para vetores curtos (H=8..16, típicos de LSTM/WaveNet NAM), o loop de 8-em-8
/// com 2 acumuladores captura a maior parte do ganho sem overhead excessivo.
///
/// # Safety
/// O chamador deve assegurar que a CPU suporta os recursos "avx2" e "fma".
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;

    unsafe {
        // 4 acumuladores independentes — quebra cadeia de dependência FMA
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        // Loop principal: 4×8 = 32 floats/iteração (throughput-bound)
        while i + 32 <= len {
            // Software prefetch 2 cache lines ahead (128 bytes = 32 floats)
            _mm_prefetch::<_MM_HINT_T0>(a.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(b.as_ptr().add(i + 32) as *const i8);

            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

            let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
            sum2 = _mm256_fmadd_ps(va2, vb2, sum2);

            let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
            sum3 = _mm256_fmadd_ps(va3, vb3, sum3);

            i += 32;
        }

        // Remainder: 8-em-8 com 2 acumuladores (vetores curtos H=8..16)
        while i + 16 <= len {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

            i += 16;
        }

        // Remainder: 8-em-8 simples
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(va, vb, sum0);
            i += 8;
        }

        // Redução: combina 4 acumuladores → 1
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum0, sum2);

        // Horizontal sum: extrair e somar os 8 floats do YMM
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum);

        let mut scalar_sum = temp.iter().sum();

        // Loop tail escalar
        while i < len {
            scalar_sum += a[i] * b[i];
            i += 1;
        }

        scalar_sum
    }
}

/// Calcula o Dot Product (Produto Escalar) de duas fatias via AVX-512 (ZMM).
///
/// ## Otimização: 2 Acumuladores ZMM Independentes (ILP)
///
/// Mesma motivação do `dot_product_avx2`: quebra a cadeia de dependência FMA
/// usando 2 acumuladores de 512 bits. Com ZMM (16 floats), 2 acumuladores
/// processam 32 floats/iteração e são suficientes para saturar o pipeline
/// AVX-512 (que tipicamente tem 1–2 FMA ports em Zen4/Sapphire Rapids).
///
/// # Safety
/// O chamador deve assegurar que a CPU suporta os recursos "avx512f" e "avx512vl".
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;

    unsafe {
        // 2 acumuladores ZMM independentes (32 floats/iter)
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();

        // Loop principal: 2×16 = 32 floats/iteração
        while i + 32 <= len {
            let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm512_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm512_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
            let vb1 = _mm512_loadu_ps(b.as_ptr().add(i + 16));
            sum1 = _mm512_fmadd_ps(va1, vb1, sum1);

            i += 32;
        }

        // Remainder: 16-em-16
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            sum0 = _mm512_fmadd_ps(va, vb, sum0);
            i += 16;
        }

        // Redução: combina 2 acumuladores → 1
        let sum = _mm512_add_ps(sum0, sum1);
        let mut scalar_sum = _mm512_reduce_add_ps(sum);

        // Loop tail escalar
        while i < len {
            scalar_sum += a[i] * b[i];
            i += 1;
        }

        scalar_sum
    }
}

/// Calcula 4 Dot Products simultâneos (ILP máximo) reutilizando o mesmo carregamento do vetor state.
///
/// Otimizado especificamente para as 4 portas do LSTM (Input, Forget, Cell, Output).
/// # Safety
/// O chamador deve assegurar que a CPU suporta os recursos "avx2" e "fma".
pub unsafe fn dot_product_4x_avx2(
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    state: &[f32],
) -> [f32; 4] {
    let len = state.len();
    let mut i = 0;

    unsafe {
        let mut sum0_0 = _mm256_setzero_ps();
        let mut sum0_1 = _mm256_setzero_ps();
        let mut sum1_0 = _mm256_setzero_ps();
        let mut sum1_1 = _mm256_setzero_ps();
        let mut sum2_0 = _mm256_setzero_ps();
        let mut sum2_1 = _mm256_setzero_ps();
        let mut sum3_0 = _mm256_setzero_ps();
        let mut sum3_1 = _mm256_setzero_ps();

        while i + 16 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w0.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w1.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w2.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w3.as_ptr().add(i + 32) as *const i8);

            let vs_0 = _mm256_loadu_ps(state.as_ptr().add(i));
            let vs_1 = _mm256_loadu_ps(state.as_ptr().add(i + 8));

            let vw0_0 = _mm256_loadu_ps(w0.as_ptr().add(i));
            let vw0_1 = _mm256_loadu_ps(w0.as_ptr().add(i + 8));
            sum0_0 = _mm256_fmadd_ps(vw0_0, vs_0, sum0_0);
            sum0_1 = _mm256_fmadd_ps(vw0_1, vs_1, sum0_1);

            let vw1_0 = _mm256_loadu_ps(w1.as_ptr().add(i));
            let vw1_1 = _mm256_loadu_ps(w1.as_ptr().add(i + 8));
            sum1_0 = _mm256_fmadd_ps(vw1_0, vs_0, sum1_0);
            sum1_1 = _mm256_fmadd_ps(vw1_1, vs_1, sum1_1);

            let vw2_0 = _mm256_loadu_ps(w2.as_ptr().add(i));
            let vw2_1 = _mm256_loadu_ps(w2.as_ptr().add(i + 8));
            sum2_0 = _mm256_fmadd_ps(vw2_0, vs_0, sum2_0);
            sum2_1 = _mm256_fmadd_ps(vw2_1, vs_1, sum2_1);

            let vw3_0 = _mm256_loadu_ps(w3.as_ptr().add(i));
            let vw3_1 = _mm256_loadu_ps(w3.as_ptr().add(i + 8));
            sum3_0 = _mm256_fmadd_ps(vw3_0, vs_0, sum3_0);
            sum3_1 = _mm256_fmadd_ps(vw3_1, vs_1, sum3_1);

            i += 16;
        }

        while i + 8 <= len {
            let vs = _mm256_loadu_ps(state.as_ptr().add(i));

            let vw0 = _mm256_loadu_ps(w0.as_ptr().add(i));
            sum0_0 = _mm256_fmadd_ps(vw0, vs, sum0_0);

            let vw1 = _mm256_loadu_ps(w1.as_ptr().add(i));
            sum1_0 = _mm256_fmadd_ps(vw1, vs, sum1_0);

            let vw2 = _mm256_loadu_ps(w2.as_ptr().add(i));
            sum2_0 = _mm256_fmadd_ps(vw2, vs, sum2_0);

            let vw3 = _mm256_loadu_ps(w3.as_ptr().add(i));
            sum3_0 = _mm256_fmadd_ps(vw3, vs, sum3_0);

            i += 8;
        }

        let sum0 = _mm256_add_ps(sum0_0, sum0_1);
        let sum1 = _mm256_add_ps(sum1_0, sum1_1);
        let sum2 = _mm256_add_ps(sum2_0, sum2_1);
        let sum3 = _mm256_add_ps(sum3_0, sum3_1);

        // Horizontal sum para cada acumulador
        let mut temp0 = [0.0f32; 8];
        let mut temp1 = [0.0f32; 8];
        let mut temp2 = [0.0f32; 8];
        let mut temp3 = [0.0f32; 8];

        _mm256_storeu_ps(temp0.as_mut_ptr(), sum0);
        _mm256_storeu_ps(temp1.as_mut_ptr(), sum1);
        _mm256_storeu_ps(temp2.as_mut_ptr(), sum2);
        _mm256_storeu_ps(temp3.as_mut_ptr(), sum3);

        let mut s0: f32 = temp0.iter().sum();
        let mut s1: f32 = temp1.iter().sum();
        let mut s2: f32 = temp2.iter().sum();
        let mut s3: f32 = temp3.iter().sum();

        while i < len {
            s0 += w0[i] * state[i];
            s1 += w1[i] * state[i];
            s2 += w2[i] * state[i];
            s3 += w3[i] * state[i];
            i += 1;
        }

        [s0, s1, s2, s3]
    }
}

/// Calcula 4 Dot Products simultâneos (ILP máximo) via AVX-512 reutilizando o state.
///
/// # Safety
/// O chamador deve assegurar que a CPU suporta os recursos "avx512f" e "avx512vl".
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_avx512(
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    state: &[f32],
) -> [f32; 4] {
    let len = state.len();
    let mut i = 0;

    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();

        while i + 16 <= len {
            let vs = _mm512_loadu_ps(state.as_ptr().add(i));

            let vw0 = _mm512_loadu_ps(w0.as_ptr().add(i));
            sum0 = _mm512_fmadd_ps(vw0, vs, sum0);

            let vw1 = _mm512_loadu_ps(w1.as_ptr().add(i));
            sum1 = _mm512_fmadd_ps(vw1, vs, sum1);

            let vw2 = _mm512_loadu_ps(w2.as_ptr().add(i));
            sum2 = _mm512_fmadd_ps(vw2, vs, sum2);

            let vw3 = _mm512_loadu_ps(w3.as_ptr().add(i));
            sum3 = _mm512_fmadd_ps(vw3, vs, sum3);

            i += 16;
        }

        let mut s0 = _mm512_reduce_add_ps(sum0);
        let mut s1 = _mm512_reduce_add_ps(sum1);
        let mut s2 = _mm512_reduce_add_ps(sum2);
        let mut s3 = _mm512_reduce_add_ps(sum3);

        while i < len {
            s0 += w0[i] * state[i];
            s1 += w1[i] * state[i];
            s2 += w2[i] * state[i];
            s3 += w3[i] * state[i];
            i += 1;
        }

        [s0, s1, s2, s3]
    }
}

/// Calcula 4 Dot Products simultâneos (ILP máximo) reutilizando o mesmo carregamento do vetor state.
/// Otimizado especificamente para as 4 portas do LSTM interfolhadas (Input, Forget, Cell, Output).
///
/// # Safety
/// O chamador deve assegurar que a CPU suporta os recursos "avx2" e "fma".
pub unsafe fn dot_product_4x_interleaved_avx2(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len();
    let mut i = 0;

    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        while i + 8 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 8) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 16) as *const i8);

            let s0 = _mm256_broadcast_ss(&state[i]);
            let s1 = _mm256_broadcast_ss(&state[i + 1]);
            let s01 = _mm256_blend_ps(s0, s1, 0b11110000);
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            sum0 = _mm256_fmadd_ps(w01, s01, sum0);

            let s2 = _mm256_broadcast_ss(&state[i + 2]);
            let s3 = _mm256_broadcast_ss(&state[i + 3]);
            let s23 = _mm256_blend_ps(s2, s3, 0b11110000);
            let w23 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            sum1 = _mm256_fmadd_ps(w23, s23, sum1);

            let s4 = _mm256_broadcast_ss(&state[i + 4]);
            let s5 = _mm256_broadcast_ss(&state[i + 5]);
            let s45 = _mm256_blend_ps(s4, s5, 0b11110000);
            let w45 = _mm256_loadu_ps(weights.as_ptr().add(i + 4) as *const f32);
            sum2 = _mm256_fmadd_ps(w45, s45, sum2);

            let s6 = _mm256_broadcast_ss(&state[i + 6]);
            let s7 = _mm256_broadcast_ss(&state[i + 7]);
            let s67 = _mm256_blend_ps(s6, s7, 0b11110000);
            let w67 = _mm256_loadu_ps(weights.as_ptr().add(i + 6) as *const f32);
            sum3 = _mm256_fmadd_ps(w67, s67, sum3);

            i += 8;
        }

        while i + 2 <= len {
            let s0 = _mm256_broadcast_ss(&state[i]);
            let s1 = _mm256_broadcast_ss(&state[i + 1]);
            let s01 = _mm256_blend_ps(s0, s1, 0b11110000);
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            sum0 = _mm256_fmadd_ps(w01, s01, sum0);
            i += 2;
        }

        let sum01 = _mm256_add_ps(sum0, sum1);
        let sum23 = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum01, sum23);

        let lower = _mm256_castps256_ps128(sum);
        let upper = _mm256_extractf128_ps(sum, 1);
        let mut sum128 = _mm_add_ps(lower, upper);

        while i < len {
            let s0 = _mm_load1_ps(state.as_ptr().add(i));
            let w0 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            sum128 = _mm_fmadd_ps(w0, s0, sum128);
            i += 1;
        }

        let mut out = [0.0; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum128);
        out
    }
}

/// Calcula 4 Dot Products simultâneos (ILP máximo) via AVX-512 reutilizando o state.
/// Otimizado especificamente para as 4 portas do LSTM interfolhadas.
///
/// # Safety
/// O chamador deve assegurar que a CPU suporta os recursos "avx512f" e "avx512vl".
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_interleaved_avx512(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len();
    let mut i = 0;

    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();

        while i + 8 <= len {
            let s0 = state[i];
            let s1 = state[i + 1];
            let s2 = state[i + 2];
            let s3 = state[i + 3];
            let vs0 = _mm512_set_ps(
                s3, s3, s3, s3, s2, s2, s2, s2, s1, s1, s1, s1, s0, s0, s0, s0,
            );
            let vw0 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            sum0 = _mm512_fmadd_ps(vw0, vs0, sum0);

            let s4 = state[i + 4];
            let s5 = state[i + 5];
            let s6 = state[i + 6];
            let s7 = state[i + 7];
            let vs1 = _mm512_set_ps(
                s7, s7, s7, s7, s6, s6, s6, s6, s5, s5, s5, s5, s4, s4, s4, s4,
            );
            let vw1 = _mm512_loadu_ps(weights.as_ptr().add(i + 4) as *const f32);
            sum1 = _mm512_fmadd_ps(vw1, vs1, sum1);

            i += 8;
        }

        while i + 4 <= len {
            let s0 = state[i];
            let s1 = state[i + 1];
            let s2 = state[i + 2];
            let s3 = state[i + 3];
            let vs0 = _mm512_set_ps(
                s3, s3, s3, s3, s2, s2, s2, s2, s1, s1, s1, s1, s0, s0, s0, s0,
            );
            let vw0 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            sum0 = _mm512_fmadd_ps(vw0, vs0, sum0);
            i += 4;
        }

        let sum = _mm512_add_ps(sum0, sum1);

        let lower256 = _mm512_castps512_ps256(sum);
        let upper256 = _mm512_extractf32x8_ps(sum, 1);
        let sum256 = _mm256_add_ps(lower256, upper256);

        let lower128 = _mm256_castps256_ps128(sum256);
        let upper128 = _mm256_extractf128_ps(sum256, 1);
        let mut sum128 = _mm_add_ps(lower128, upper128);

        while i < len {
            let s0 = _mm_load1_ps(state.as_ptr().add(i));
            let w0 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            sum128 = _mm_fmadd_ps(w0, s0, sum128);
            i += 1;
        }

        let mut out = [0.0; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum128);
        out
    }
}

/// Despacho Dinâmico Global de Funções Matemáticas SIMD.
/// Resolve o multiversionamento (AVX2/AVX-512) para a inferência sem causar alocações.
///
/// ## Motivação: Caching via `LazyLock`
///
/// A detecção de features SIMD (`is_x86_feature_detected!`) realiza internamente
/// uma leitura atômica de um `OnceLock` global na stdlib. Embora individual seja
/// barata (~2 loads + branch), ela é invocada **a cada bloco DSP** em todos os
/// modelos (WaveNet, LSTM, WaveNet Dyn, LSTM Dyn). O `SIMD_MATH_CONFIG` global
/// (`LazyLock`) resolve a v-table **uma única vez** no startup e expõe via
/// `SimdMathConfig::get()` uma referência `&'static` com overhead efetivo zero
/// no hot-path RT.
/// Assinatura da função para 4 Dot Products simultâneos (ILP máximo) reutilizando o state.
pub type DotProduct4xFn = unsafe fn(&[f32], &[f32], &[f32], &[f32], &[f32]) -> [f32; 4];

/// Assinatura da função para 4 Dot Products simultâneos para pesos interfolhados.
pub type DotProduct4xInterleavedFn = unsafe fn(&[[f32; 4]], &[f32]) -> [f32; 4];

/// Despacho Dinâmico Global de Funções Matemáticas SIMD.
/// Resolve o multiversionamento (AVX2/AVX-512) para a inferência sem causar alocações.
///
/// **Design:** Singleton resolvido tardiamente (Lazily Evaluated via `SimdMathConfig::get()`).
/// Toda a API `unsafe` tem garantia arquitetural no init estático via `is_x86_feature_detected!`.
#[derive(Clone, Copy)]
pub struct SimdMathConfig {
    /// Função inlined dinamicamente agendada para computar fma vetorial.
    pub dot_product: unsafe fn(&[f32], &[f32]) -> f32,
    /// Fused GEMV de 4 portas (para Conv1d do WaveNet).
    pub dot_product_4x: DotProduct4xFn,
    /// Fused GEMV de 4 portas interfolhadas (para LSTM).
    pub dot_product_4x_interleaved: DotProduct4xInterleavedFn,
    /// Loop ativado via fptr para iterar `tanh(x)` na matriz especificada.
    pub tanh_slice: unsafe fn(&mut [f32]),
    /// Loop ativado via fptr para iterar `sigmoid(x)` na matriz especificada.
    pub sigmoid_slice: unsafe fn(&mut [f32]),
}

/// V-Table SIMD global, inicializada uma única vez via `LazyLock`.
///
/// Elimina a necessidade de `is_x86_feature_detected!` (leitura atômica de `OnceLock`)
/// em cada chamada `process()` dos modelos. Após a primeira avaliação, acesso via
/// `SimdMathConfig::get()` se resolve em um único load de ponteiro estático.
static SIMD_MATH_CONFIG: std::sync::LazyLock<SimdMathConfig> =
    std::sync::LazyLock::new(SimdMathConfig::current);

impl SimdMathConfig {
    /// Inicia e aloca a v-table matemática inspecionando nativamente as capabilities da CPU.
    pub fn current() -> Self {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            return Self {
                dot_product: <Avx512Math as SimdMath>::dot_product,
                dot_product_4x: <Avx512Math as SimdMath>::dot_product_4x,
                dot_product_4x_interleaved: <Avx512Math as SimdMath>::dot_product_4x_interleaved,
                tanh_slice: <Avx512Math as SimdMath>::tanh_slice,
                sigmoid_slice: <Avx512Math as SimdMath>::sigmoid_slice,
            };
        }

        Self {
            dot_product: dot_product_avx2,
            dot_product_4x: dot_product_4x_avx2,
            dot_product_4x_interleaved: dot_product_4x_interleaved_avx2,
            tanh_slice: crate::math::fastmath::tanh_slice_avx2,
            sigmoid_slice: crate::math::fastmath::sigmoid_slice_avx2,
        }
    }

    /// Retorna referência estática à v-table SIMD resolvida no startup.
    ///
    /// Zero overhead após a primeira chamada — nenhuma leitura atômica, nenhum
    /// branch de CPUID. Preferir esta API em todos os hot-paths (DSP, modelos)
    /// em vez de `SimdMathConfig::current()` que repete a detecção de features.
    #[inline(always)]
    pub fn get() -> &'static Self {
        &SIMD_MATH_CONFIG
    }
}

/// Trait de abstração para despacho estático de operações matemáticas SIMD.
pub trait SimdMath {
    /// Calcula o produto escalar entre dois vetores.
    ///
    /// # Safety
    /// Depende da arquitetura SIMD alvo suportar as instruções emitidas e os slices terem o mesmo tamanho.
    unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32;
    /// Calcula 4 produtos escalares SIMD em paralelo (Loop Unrolling otimizado) para WaveNet.
    ///
    /// # Safety
    /// Depende de HW support, slices dimensionados corretamente, e ponteiros alinhados.
    unsafe fn dot_product_4x(
        w0: &[f32],
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        state: &[f32],
    ) -> [f32; 4];
    /// Calcula 4 produtos escalares SIMD em paralelo para LSTM interfolhado.
    ///
    /// # Safety
    /// Depende de HW support, fatias com tamanhos compatíveis e ponteiros alinhados.
    unsafe fn dot_product_4x_interleaved(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4];
    /// Aplica Tanh em-lugar no slice usando aproximação minimax polinomial fastmath.
    ///
    /// # Safety
    /// Depende do suporte de hardware correspondente para iterações vetorizadas.
    unsafe fn tanh_slice(slice: &mut [f32]);
    /// Aplica Sigmoid em-lugar no slice via fastmath.
    ///
    /// # Safety
    /// Hardware SIMD requerido e iteradores vetorizados inseguros sem bounds checking.
    unsafe fn sigmoid_slice(slice: &mut [f32]);
}

/// Implementação estática para microarquitetura x86-64-v3 (AVX2/FMA).
pub struct Avx2Math;
impl SimdMath for Avx2Math {
    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        unsafe { dot_product_avx2(a, b) }
    }
    #[inline(always)]
    unsafe fn dot_product_4x(
        w0: &[f32],
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        state: &[f32],
    ) -> [f32; 4] {
        unsafe { dot_product_4x_avx2(w0, w1, w2, w3, state) }
    }
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_avx2(weights, state) }
    }
    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::tanh_slice_avx2(slice) }
    }
    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::sigmoid_slice_avx2(slice) }
    }
}

/// Implementação estática para microarquitetura x86-64-v4 (AVX-512).
pub struct Avx512Math;
impl SimdMath for Avx512Math {
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        unsafe { dot_product_avx512(a, b) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product_4x(
        w0: &[f32],
        w1: &[f32],
        w2: &[f32],
        w3: &[f32],
        state: &[f32],
    ) -> [f32; 4] {
        unsafe { dot_product_4x_avx512(w0, w1, w2, w3, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product_4x_interleaved(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_avx512(weights, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::tanh_slice_avx512(slice) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::sigmoid_slice_avx512(slice) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_avx2_fma() {
        let vec_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let vec_b = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

        let result = unsafe { dot_product_avx2(&vec_a, &vec_b) };

        // Expected = (1*2 + 2*2 ... + 8*2) + 9*2
        // 72 * 2 + 18 = 144 + 18 = 90
        let expected: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();

        // FMA math is accurate but compare with epsilon
        assert!(
            (result - expected).abs() < 1e-4,
            "Resultado divergente: esperado {}, obtido {}",
            expected,
            result
        );
    }

    #[test]
    fn test_dot_product_avx512() {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            let vec_a = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0,
            ];
            let vec_b = vec![
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            ];

            let result = unsafe { dot_product_avx512(&vec_a, &vec_b) };
            let expected: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();

            assert!(
                (result - expected).abs() < 1e-4,
                "Resultado divergente: esperado {}, obtido {}",
                expected,
                result
            );
        }
    }

    /// Verifica que `set_daz_ftz` seta corretamente os bits DAZ (6) e FTZ (15) no MXCSR.
    #[test]
    fn test_set_daz_ftz() {
        unsafe {
            // Ler MXCSR atual e limpar DAZ+FTZ para verificar que a função os seta
            let mut before: u32 = 0;
            core::arch::asm!("stmxcsr [{0}]", in(reg) &mut before);
            let cleared = before & !0x8040;
            core::arch::asm!("ldmxcsr [{0}]", in(reg) &cleared);

            set_daz_ftz();

            let mut after: u32 = 0;
            core::arch::asm!("stmxcsr [{0}]", in(reg) &mut after);
            assert!(
                (after & 0x8040) == 0x8040,
                "set_daz_ftz() não setou DAZ+FTZ: MXCSR=0x{:08X}",
                after
            );

            // Restaurar MXCSR original
            core::arch::asm!("ldmxcsr [{0}]", in(reg) &before);
        }
    }
}
