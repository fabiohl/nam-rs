// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Engenharia de Registradores baseada em instruções explícitas x86_64.
//!
//! Este módulo exporta funções analíticas implementadas com instrinsics de AVX2 e FMA,
//! otimizando os cálculos críticos (como Fused Multiply-Add) limitando os
//! desvios matemáticos inerentes aos loops comuns e reduzindo latência nas CNNs.

/// Converte F32 para os bits de um BF16 (truncamento simples).
#[inline(always)]
pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

use core::arch::x86_64::*;

/// Prefetch adaptativo baseado na dilatação (Causal Conv1D).
///
/// O prefetcher nativo (hardware) do x86-64 lida bem com acessos sequenciais (dilatações baixas).
/// Para dilatações maiores, o salto de memória excede a janela do hardware prefetcher, exigindo
/// hints explícitos para evitar stalls no pipeline FMA.
///
/// - **D <= 8**: O hardware prefetcher resolve de forma ótima. Nenhum hint emitido.
/// - **16 <= D <= 64**: Dados "quentes" necessários em breve. Hint `T0` (L1).
/// - **D >= 128**: Acessos esparsos e longínquos. Hint `T1` (L2) para evitar L1 thrashing.
///
/// # Safety
/// O ponteiro `ptr` deve ser válido ou estar dentro da margem de segurança do buffer.
/// Como `_mm_prefetch` é apenas um hint, não causa falha de segmentação se o ponteiro for inválido,
/// mas deve ser usado com cautela em regiões críticas.
#[inline(always)]
pub unsafe fn adaptive_prefetch_f32(ptr: *const f32, dilation: usize) {
    if dilation <= 8 {
        // Hardware prefetcher domina.
    } else if dilation <= 64 {
        // Traz para L1 (reuso iminente).
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(ptr as *const i8);
        }
    } else {
        // Dilatações massivas: traz para L2 para poupar o L1 de evicção agressiva.
        unsafe {
            _mm_prefetch::<_MM_HINT_T1>(ptr as *const i8);
        }
    }
}

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
/// Esta função altera estado global do processador (MXCSR) — deve ser
/// chamada apenas uma vez por thread (tipicamente no início do callback RT).
///
/// # Safety
/// O chamador deve garantir que a CPU suporte SSE2 (implícito em x86-64).
/// Altera estado global (MXCSR), portanto deve ser chamada antes de iniciar
/// processamento em paralelo na mesma thread.
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
/// Requer CPU com suporte a AVX2, FMA e F16C. Os slices `a` e `b` devem ser
/// válidos e acessíveis. O cálculo usa `get_unchecked` no tail escalar.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[u16]) -> f32 {
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
            let vb0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

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

        // Remainder: 8-em-8 com 2 acumuladores (vetores curtos H=8..16)
        while i + 16 <= len {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i + 8) as *const __m128i));
            sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

            i += 16;
        }

        // Remainder: 8-em-8 simples
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_cvtph_ps(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            sum0 = _mm256_fmadd_ps(va, vb, sum0);
            i += 8;
        }

        // Redução: combina 4 acumuladores → 1
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum0, sum2);

        // Horizontal sum otimizado: redução via intrínsecos (sem spill YMM→stack)
        let hi128 = _mm256_extractf128_ps(sum, 1);
        let lo128 = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let final_ss = _mm_add_ss(sums, shuf2);
        let mut scalar_sum = 0.0f32;
        _mm_store_ss(&mut scalar_sum, final_ss);

        // Loop tail escalar
        while i < len {
            scalar_sum += a[i] * half::f16::from_bits(b[i]).to_f32();
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
/// Requer CPU com suporte a AVX-512F, AVX-512VL e F16C. Os slices `a` e `b`
/// devem ser válidos.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;

    unsafe {
        // 2 acumuladores ZMM independentes (32 floats/iter)
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();

        // Loop principal: 2×16 = 32 floats/iteração
        while i + 32 <= len {
            let va0 = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm512_cvtph_ps(_mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i));
            sum0 = _mm512_fmadd_ps(va0, vb0, sum0);

            let va1 = _mm512_loadu_ps(a.as_ptr().add(i + 16));
            let vb1 = _mm512_cvtph_ps(_mm256_loadu_si256(b.as_ptr().add(i + 16) as *const __m256i));
            sum1 = _mm512_fmadd_ps(va1, vb1, sum1);

            i += 32;
        }

        // Remainder: 16-em-16
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_cvtph_ps(_mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i));
            sum0 = _mm512_fmadd_ps(va, vb, sum0);
            i += 16;
        }

        // Redução: combina 2 acumuladores → 1
        let sum = _mm512_add_ps(sum0, sum1);
        let mut scalar_sum = _mm512_reduce_add_ps(sum);

        // Loop tail escalar
        while i < len {
            scalar_sum += a[i] * half::f16::from_bits(b[i]).to_f32();
            i += 1;
        }

        scalar_sum
    }
}

/// Calcula o Dot Product (Produto Escalar) de duas fatias BF16 via AVX-512 BF16 nativo.
///
/// ## Otimização: VDPBF16PS (BF16 Pairs)
///
/// Instrução `VDPBF16PS` processa 32 valores BF16 (16 pares) por registro ZMM,
/// realizando a operação `acc[j] += a[2j]*b[2j] + a[2j+1]*b[2j+1]` em um único ciclo.
/// Isso dobra o throughput em relação ao FMA32 (que processa 16 valores/ciclo).
///
/// # Safety
/// Requer CPU com suporte a AVX-512F, AVX-512VL e AVX-512BF16.
/// Os slices `a` e `b` contêm valores BF16 empacotados como u16.
#[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
pub unsafe fn dot_product_bf16_native_avx512(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;
    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();

        // Loop principal: 2×32 = 64 BF16/iteração
        while i + 64 <= len {
            let va0 = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let vb0 = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            sum0 = _mm512_dpbf16_ps(
                sum0,
                core::mem::transmute::<__m512i, __m512bh>(va0),
                core::mem::transmute::<__m512i, __m512bh>(vb0),
            );

            let va1 = _mm512_loadu_si512(a.as_ptr().add(i + 32) as *const __m512i);
            let vb1 = _mm512_loadu_si512(b.as_ptr().add(i + 32) as *const __m512i);
            sum1 = _mm512_dpbf16_ps(
                sum1,
                core::mem::transmute::<__m512i, __m512bh>(va1),
                core::mem::transmute::<__m512i, __m512bh>(vb1),
            );

            i += 64;
        }

        while i + 32 <= len {
            let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const __m512i);
            sum0 = _mm512_dpbf16_ps(
                sum0,
                core::mem::transmute::<__m512i, __m512bh>(va),
                core::mem::transmute::<__m512i, __m512bh>(vb),
            );
            i += 32;
        }

        let sum = _mm512_add_ps(sum0, sum1);
        // Redução horizontal 512 -> 256 -> 128 -> escalar
        let v256 = _mm256_add_ps(
            _mm512_extractf32x8_ps::<0>(sum),
            _mm512_extractf32x8_ps::<1>(sum),
        );
        let v128 = _mm_add_ps(
            _mm256_extractf128_ps::<0>(v256),
            _mm256_extractf128_ps::<1>(v256),
        );
        let v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
        let v32 = _mm_add_ss(v64, _mm_shuffle_ps::<0x55>(v64, v64));
        let mut scalar_sum = _mm_cvtss_f32(v32);

        // Tail escalar (conversão manual BF16 -> F32)
        while i < len {
            let fa = f32::from_bits((a[i] as u32) << 16);
            let fb = f32::from_bits((b[i] as u32) << 16);
            scalar_sum += fa * fb;
            i += 1;
        }

        scalar_sum
    }
}

/// Calcula 4 Dot Products simultâneos (ILP máximo) via AVX-512 BF16 nativo.
///
/// # Safety
/// Requer CPU com suporte a AVX-512F, AVX-512VL e AVX-512BF16.
#[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
pub unsafe fn dot_product_bf16_4x_native_avx512(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    in_frame: &[u16],
) -> [f32; 4] {
    let len = in_frame.len();
    let mut i = 0;
    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();

        while i + 32 <= len {
            let vi = _mm512_loadu_si512(in_frame.as_ptr().add(i) as *const __m512i);
            let vi_bh = core::mem::transmute::<__m512i, __m512bh>(vi);

            let vw0 = _mm512_loadu_si512(w0.as_ptr().add(i) as *const __m512i);
            sum0 = _mm512_dpbf16_ps(sum0, vi_bh, core::mem::transmute::<__m512i, __m512bh>(vw0));

            let vw1 = _mm512_loadu_si512(w1.as_ptr().add(i) as *const __m512i);
            sum1 = _mm512_dpbf16_ps(sum1, vi_bh, core::mem::transmute::<__m512i, __m512bh>(vw1));

            let vw2 = _mm512_loadu_si512(w2.as_ptr().add(i) as *const __m512i);
            sum2 = _mm512_dpbf16_ps(sum2, vi_bh, core::mem::transmute::<__m512i, __m512bh>(vw2));

            let vw3 = _mm512_loadu_si512(w3.as_ptr().add(i) as *const __m512i);
            sum3 = _mm512_dpbf16_ps(sum3, vi_bh, core::mem::transmute::<__m512i, __m512bh>(vw3));

            i += 32;
        }

        // Redução horizontal auxiliar
        #[inline(always)]
        unsafe fn hsum512(sum: __m512) -> f32 {
            unsafe {
                let v256 = _mm256_add_ps(
                    _mm512_extractf32x8_ps::<0>(sum),
                    _mm512_extractf32x8_ps::<1>(sum),
                );
                let v128 = _mm_add_ps(
                    _mm256_extractf128_ps::<0>(v256),
                    _mm256_extractf128_ps::<1>(v256),
                );
                let v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
                let v32 = _mm_add_ss(v64, _mm_shuffle_ps::<0x55>(v64, v64));
                _mm_cvtss_f32(v32)
            }
        }

        let mut s0 = hsum512(sum0);
        let mut s1 = hsum512(sum1);
        let mut s2 = hsum512(sum2);
        let mut s3 = hsum512(sum3);

        while i < len {
            let vi = f32::from_bits((in_frame[i] as u32) << 16);
            s0 += f32::from_bits((w0[i] as u32) << 16) * vi;
            s1 += f32::from_bits((w1[i] as u32) << 16) * vi;
            s2 += f32::from_bits((w2[i] as u32) << 16) * vi;
            s3 += f32::from_bits((w3[i] as u32) << 16) * vi;
            i += 1;
        }

        [s0, s1, s2, s3]
    }
}

/// Calcula 4 produtos escalares interfolhados BF16 usando AVX-512 VNNI.
///
/// ## Mecânica: 1:4 Broadcast + VDPBF16PS
///
/// Processa 8 linhas de 4 pesos ([I, F, C, O]) por iteração.
/// O estado é carregado em 128 bits e expandido para 512 bits (broadcast 1:4 por elemento).
/// A instrução `VDPBF16PS` então realiza o dot product de pares BF16, acumulando em F32.
///
/// # Safety
/// Requer CPU com AVX-512F, AVX-512VL, AVX-512BW e AVX-512BF16.
/// `weights` e `state` devem conter valores BF16 válidos.
#[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
pub unsafe fn dot_product_4x_interleaved_bf16_avx512(
    weights: &[[u16; 4]],
    state: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut i = 0;
    let mut sum0 = _mm512_setzero_ps();

    // Índice para broadcast 1:4 de u16: [0,0,0,0, 1,1,1,1, ..., 7,7,7,7]
    let idx = _mm512_set_epi16(
        7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,
        0, 0,
    );

    while i + 8 <= len {
        unsafe {
            let vw = _mm512_loadu_si512(weights.as_ptr().add(i) as *const __m512i);
            let vs_raw = _mm_loadu_si128(state.as_ptr().add(i) as *const __m128i);
            let vs = _mm512_permutexvar_epi16(idx, _mm512_castsi128_si512(vs_raw));

            sum0 = _mm512_dpbf16_ps(
                sum0,
                core::mem::transmute::<__m512i, __m512bh>(vw),
                core::mem::transmute::<__m512i, __m512bh>(vs),
            );
        }
        i += 8;
    }

    let mut final_sum = [0.0f32; 4];
    let mut res = [0.0f32; 16];
    unsafe {
        _mm512_storeu_ps(res.as_mut_ptr(), sum0);
    }

    for j in 0..4 {
        final_sum[j] = res[j] + res[j + 4] + res[j + 8] + res[j + 12];
    }

    // Tail escalar
    while i < len {
        let s = f32::from_bits((state[i] as u32) << 16);
        let w = weights[i];
        final_sum[0] += f32::from_bits((w[0] as u32) << 16) * s;
        final_sum[1] += f32::from_bits((w[1] as u32) << 16) * s;
        final_sum[2] += f32::from_bits((w[2] as u32) << 16) * s;
        final_sum[3] += f32::from_bits((w[3] as u32) << 16) * s;
        i += 1;
    }

    final_sum
}

/// Fallback escalar para dot product BF16.
///
/// # Safety
/// Os slices `a` e `b` devem ser válidos. Usa `get_unchecked` para acesso
/// sem verificação de limites no loop interno.
pub unsafe fn dot_product_bf16_fallback(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..len {
        let fa = f32::from_bits(unsafe { *a.get_unchecked(i) as u32 } << 16);
        let fb = f32::from_bits(unsafe { *b.get_unchecked(i) as u32 } << 16);
        sum += fa * fb;
    }
    sum
}

/// Fallback para dot product interleaved BF16.
///
/// # Safety
/// `weights` e `state` devem conter dados BF16 válidos e ter tamanhos consistentes.
pub unsafe fn dot_product_4x_interleaved_bf16_fallback(
    weights: &[[u16; 4]],
    state: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];
    for i in 0..len {
        let s = f32::from_bits((state[i] as u32) << 16);
        let w = weights[i];
        sum[0] += f32::from_bits((w[0] as u32) << 16) * s;
        sum[1] += f32::from_bits((w[1] as u32) << 16) * s;
        sum[2] += f32::from_bits((w[2] as u32) << 16) * s;
        sum[3] += f32::from_bits((w[3] as u32) << 16) * s;
    }
    sum
}

/// Fallback para dot product BF16 em batch de 4.
///
/// # Safety
/// Todos os slices devem ser válidos e conter dados BF16. Delega para
/// `dot_product_bf16_fallback` que usa `get_unchecked`.
pub unsafe fn dot_product_bf16_batch_4x_fallback(
    h0: &[u16],
    h1: &[u16],
    h2: &[u16],
    h3: &[u16],
    w: &[u16],
) -> [f32; 4] {
    [
        unsafe { dot_product_bf16_fallback(h0, w) },
        unsafe { dot_product_bf16_fallback(h1, w) },
        unsafe { dot_product_bf16_fallback(h2, w) },
        unsafe { dot_product_bf16_fallback(h3, w) },
    ]
}

/// Calcula 4 Dot Products simultâneos (ILP máximo) reutilizando o mesmo carregamento do vetor state.
/// Otimizado especificamente para as 4 portas do LSTM (Input, Forget, Cell, Output).
///
/// # Safety
/// Requer CPU com AVX2, FMA e F16C. Todos os slices de pesos (`w0`–`w3`)
/// e `state` devem ter tamanhos compatíveis.
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

        let sum0 = _mm256_add_ps(sum0_0, sum0_1);
        let sum1 = _mm256_add_ps(sum1_0, sum1_1);
        let sum2 = _mm256_add_ps(sum2_0, sum2_1);
        let sum3 = _mm256_add_ps(sum3_0, sum3_1);

        // Horizontal sum otimizado: redução via intrínsecos (sem spill YMM→stack)
        #[inline(always)]
        unsafe fn hsum_avx2(v: __m256) -> f32 {
            unsafe {
                let hi = _mm256_extractf128_ps(v, 1);
                let lo = _mm256_castps256_ps128(v);
                let s128 = _mm_add_ps(lo, hi);
                let shuf = _mm_movehdup_ps(s128);
                let sums = _mm_add_ps(s128, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let r = _mm_add_ss(sums, shuf2);
                let mut out = 0.0f32;
                _mm_store_ss(&mut out, r);
                out
            }
        }

        let mut s0: f32 = hsum_avx2(sum0);
        let mut s1: f32 = hsum_avx2(sum1);
        let mut s2: f32 = hsum_avx2(sum2);
        let mut s3: f32 = hsum_avx2(sum3);

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

/// Calcula 4 Dot Products simultâneos (ILP máximo) via AVX-512 reutilizando o state.
///
/// # Safety
/// Requer CPU com AVX-512F e AVX-512VL. Os slices `w0`–`w3` (F16C) e `state`
/// (f32) devem ser válidos e ter tamanhos compatíveis.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_avx512(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
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

            let vw0 = _mm512_cvtph_ps(_mm256_loadu_si256(w0.as_ptr().add(i) as *const __m256i));
            sum0 = _mm512_fmadd_ps(vw0, vs, sum0);

            let vw1 = _mm512_cvtph_ps(_mm256_loadu_si256(w1.as_ptr().add(i) as *const __m256i));
            sum1 = _mm512_fmadd_ps(vw1, vs, sum1);

            let vw2 = _mm512_cvtph_ps(_mm256_loadu_si256(w2.as_ptr().add(i) as *const __m256i));
            sum2 = _mm512_fmadd_ps(vw2, vs, sum2);

            let vw3 = _mm512_cvtph_ps(_mm256_loadu_si256(w3.as_ptr().add(i) as *const __m256i));
            sum3 = _mm512_fmadd_ps(vw3, vs, sum3);

            i += 16;
        }

        let mut s0 = _mm512_reduce_add_ps(sum0);
        let mut s1 = _mm512_reduce_add_ps(sum1);
        let mut s2 = _mm512_reduce_add_ps(sum2);
        let mut s3 = _mm512_reduce_add_ps(sum3);

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

/// Calcula 4 Dot Products simultâneos (ILP máximo) reutilizando o mesmo carregamento do vetor state.
/// Otimizado especificamente para as 4 portas do LSTM interfolhadas (Input, Forget, Cell, Output).
///
/// # Safety
/// Requer CPU com AVX2, FMA e F16C. `weights.len()` e `state.len()` devem ser
/// consistentes. O layout `[u16; 4]` representa pesos interfolhados (I,F,C,O).
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_4x_interleaved_avx2(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
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

        while i + 2 <= len {
            let s0 = _mm256_broadcast_ss(&state[i]);
            let s1 = _mm256_broadcast_ss(&state[i + 1]);
            let s01 = _mm256_blend_ps(s0, s1, 0b11110000);
            let w01 = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
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
            let w0 = _mm_cvtph_ps(_mm_loadu_si64(weights.as_ptr().add(i) as *const u8));
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
/// Requer CPU com AVX-512F e AVX-512VL. `weights` e `state` devem ter
/// tamanhos consistentes.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_interleaved_avx512(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
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
            let vw0 =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            sum0 = _mm512_fmadd_ps(vw0, vs0, sum0);

            let s4 = state[i + 4];
            let s5 = state[i + 5];
            let s6 = state[i + 6];
            let s7 = state[i + 7];
            let vs1 = _mm512_set_ps(
                s7, s7, s7, s7, s6, s6, s6, s6, s5, s5, s5, s5, s4, s4, s4, s4,
            );
            let vw1 = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
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
            let vw0 =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
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
            let w0 = _mm_cvtph_ps(_mm_loadu_si64(weights.as_ptr().add(i) as *const u8));
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
pub type DotProduct4xFn = unsafe fn(&[u16], &[u16], &[u16], &[u16], &[f32]) -> [f32; 4];

/// Assinatura da função para 4 Dot Products simultâneos para pesos interfolhados.
pub type DotProduct4xInterleavedFn = unsafe fn(&[[u16; 4]], &[f32]) -> [f32; 4];

/// Despacho Dinâmico Global de Funções Matemáticas SIMD.
/// Resolve o multiversionamento (AVX2/AVX-512) para a inferência sem causar alocações.
///
/// **Design:** Singleton resolvido tardiamente (Lazily Evaluated via `SimdMathConfig::get()`).
/// Toda a API `unsafe` tem garantia arquitetural no init estático via `is_x86_feature_detected!`.
#[derive(Clone, Copy)]
pub struct SimdMathConfig {
    /// Função inlined dinamicamente agendada para computar fma vetorial.
    pub dot_product: unsafe fn(&[f32], &[u16]) -> f32,
    /// Fused GEMV de 4 portas (para Conv1d do WaveNet).
    pub dot_product_4x: DotProduct4xFn,
    /// Fused GEMV de 4 portas interfolhadas (para LSTM).
    pub dot_product_4x_interleaved: DotProduct4xInterleavedFn,
    /// Loop ativado via fptr para iterar `tanh(x)` na matriz especificada.
    pub tanh_slice: unsafe fn(&mut [f32]),
    /// Loop ativado via fptr para iterar `sigmoid(x)` na matriz especificada.
    pub sigmoid_slice: unsafe fn(&mut [f32]),
    /// Loop ativado via fptr para aplicar tanh em bloco pequeno com padding.
    pub activation_tanh_block: unsafe fn(&mut [f32]),
    /// Conjunto de instruções SIMD detectado.
    /// Define a trait matemática exata no macro `dispatch_simd!`.
    pub instruction_set: SimdInstructionSet,
    /// Cache do suporte a AVX-512 para despacho rápido fora do macro `dispatch_simd!`.
    pub is_avx512: bool,
}

/// Conjuntos de instruções SIMD suportados e detectáveis na inicialização.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdInstructionSet {
    /// Baseline (x86-64-v3): AVX2 + FMA
    Avx2,
    /// AVX2 com VNNI (Vector Neural Network Instructions)
    Avx2Vnni,
    /// AVX-512 (x86-64-v4)
    Avx512,
    /// AVX-512 com VNNI
    Avx512Vnni,
    /// AVX-512 com VNNI e BF16 (x86-64-v4 + BF16)
    Avx512VnniBf16,
}

impl SimdInstructionSet {
    /// Retorna `true` se o conjunto de instruções suporta AVX-512 (Foundation).
    #[inline(always)]
    pub fn is_avx512(&self) -> bool {
        matches!(self, Self::Avx512 | Self::Avx512Vnni | Self::Avx512VnniBf16)
    }
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
        let has_avx512 =
            std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl");
        let has_avx512_vnni = has_avx512 && std::is_x86_feature_detected!("avx512vnni");
        let has_avx512_bf16 = has_avx512_vnni && std::is_x86_feature_detected!("avx512bf16");
        let has_avx2_vnni = std::is_x86_feature_detected!("avxvnni");

        if has_avx512_bf16 {
            return Self {
                dot_product: <Avx512VnniBf16Math as SimdMath>::dot_product,
                dot_product_4x: <Avx512VnniBf16Math as SimdMath>::dot_product_4x,
                dot_product_4x_interleaved:
                    <Avx512VnniBf16Math as SimdMath>::dot_product_4x_interleaved,
                tanh_slice: <Avx512VnniBf16Math as SimdMath>::tanh_slice,
                sigmoid_slice: <Avx512VnniBf16Math as SimdMath>::sigmoid_slice,
                activation_tanh_block: <Avx512VnniBf16Math as SimdMath>::activation_tanh_block,
                instruction_set: SimdInstructionSet::Avx512VnniBf16,
                is_avx512: true,
            };
        }

        if has_avx512_vnni {
            return Self {
                dot_product: <Avx512VnniMath as SimdMath>::dot_product,
                dot_product_4x: <Avx512VnniMath as SimdMath>::dot_product_4x,
                dot_product_4x_interleaved:
                    <Avx512VnniMath as SimdMath>::dot_product_4x_interleaved,
                tanh_slice: <Avx512VnniMath as SimdMath>::tanh_slice,
                sigmoid_slice: <Avx512VnniMath as SimdMath>::sigmoid_slice,
                activation_tanh_block: <Avx512VnniMath as SimdMath>::activation_tanh_block,
                instruction_set: SimdInstructionSet::Avx512Vnni,
                is_avx512: true,
            };
        }

        if has_avx512 {
            return Self {
                dot_product: <Avx512Math as SimdMath>::dot_product,
                dot_product_4x: <Avx512Math as SimdMath>::dot_product_4x,
                dot_product_4x_interleaved: <Avx512Math as SimdMath>::dot_product_4x_interleaved,
                tanh_slice: <Avx512Math as SimdMath>::tanh_slice,
                sigmoid_slice: <Avx512Math as SimdMath>::sigmoid_slice,
                activation_tanh_block: <Avx512Math as SimdMath>::activation_tanh_block,
                instruction_set: SimdInstructionSet::Avx512,
                is_avx512: true,
            };
        }

        if has_avx2_vnni {
            return Self {
                dot_product: <Avx2VnniMath as SimdMath>::dot_product,
                dot_product_4x: <Avx2VnniMath as SimdMath>::dot_product_4x,
                dot_product_4x_interleaved: <Avx2VnniMath as SimdMath>::dot_product_4x_interleaved,
                tanh_slice: <Avx2VnniMath as SimdMath>::tanh_slice,
                sigmoid_slice: <Avx2VnniMath as SimdMath>::sigmoid_slice,
                activation_tanh_block: <Avx2VnniMath as SimdMath>::activation_tanh_block,
                instruction_set: SimdInstructionSet::Avx2Vnni,
                is_avx512: false,
            };
        }

        Self {
            dot_product: dot_product_avx2,
            dot_product_4x: dot_product_4x_avx2,
            dot_product_4x_interleaved: dot_product_4x_interleaved_avx2,
            tanh_slice: crate::math::fastmath::tanh_slice_avx2,
            sigmoid_slice: crate::math::fastmath::sigmoid_slice_avx2,
            activation_tanh_block: <Avx2Math as SimdMath>::activation_tanh_block,
            instruction_set: SimdInstructionSet::Avx2,
            is_avx512: false,
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
///
/// # Safety
/// Todas as implementações deste trait utilizam intrinsics SIMD x86-64 que requerem
/// features de CPU específicas (AVX2/FMA mínimo). O chamador deve garantir que a CPU
/// suporte as features declaradas via `#[target_feature]` na implementação concreta.
/// Os slices passados devem ser válidos e acessíveis para leitura/escrita conforme indicado.
pub trait SimdMath {
    /// Tipo de registrador SIMD utilizado (ex: __m256 ou __m512).
    type V;

    /// Indica se esta implementação utiliza pesos e sinais em formato BF16.
    const IS_BF16: bool = false;
    /// Calcula o produto escalar entre dois vetores.
    ///
    /// # Safety
    /// `a` (f32) e `b` (f16 como u16) devem ser válidos. Requer target feature da implementação.
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32;
    /// Converte f32 para BF16 (u16) de forma vetorizada.
    ///
    /// # Safety
    /// `src` e `dst` devem ser válidos e `dst` deve ter ao menos `src.len()` elementos.
    unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]);

    /// Produto escalar 1x entre dois vetores BF16 (u16 bits).
    ///
    /// # Safety
    /// `a` e `b` devem conter dados BF16 válidos empacotados como u16.
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32;

    /// Produto escalar 4x entre vetores BF16 (u16 bits).
    ///
    /// # Safety
    /// Todos os slices devem conter dados BF16 válidos e ter tamanhos consistentes.
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4];

    /// Calcula 4 produtos escalares SIMD em paralelo (Loop Unrolling otimizado) para WaveNet.
    ///
    /// # Safety
    /// `w0`–`w3` (f16 como u16) e `state` (f32) devem ter tamanhos compatíveis.
    unsafe fn dot_product_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        state: &[f32],
    ) -> [f32; 4];
    /// Calcula 4 produtos escalares SIMD em paralelo para LSTM interfolhado.
    ///
    /// # Safety
    /// `weights` e `state` devem ter tamanhos consistentes. Layout `[u16; 4]` = (I,F,C,O).
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4];
    /// Calcula 4 produtos escalares SIMD em paralelo para LSTM interfolhado BF16.
    ///
    /// # Safety
    /// `weights` e `state` devem conter dados BF16 válidos e ter tamanhos consistentes.
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4];
    /// Aplica Tanh em-lugar no slice usando aproximação minimax polinomial fastmath.
    ///
    /// # Safety
    /// `slice` deve ser válido e acessível para leitura e escrita.
    unsafe fn tanh_slice(slice: &mut [f32]);
    /// Aplica Sigmoid em-lugar no slice via fastmath.
    ///
    /// # Safety
    /// `slice` deve ser válido e acessível para leitura e escrita.
    unsafe fn sigmoid_slice(slice: &mut [f32]);

    /// Realiza a operação fundida Y = X_res + Bias + W * Z (Broadcast GEMV).
    /// out_frame: vetor Y e X_res (in-place).
    ///
    /// # Safety
    /// `weights` deve ter `in_len * out_len` elementos no layout `[IN][OUT]`.
    /// `bias` deve ter ao menos `out_len` elementos. `out_frame` é lido e escrito.
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// Realiza a projeção linear Y = Bias + W * Z (GEMV) substituindo o conteúdo de out_frame.
    /// out_frame: vetor Y (overwrite).
    ///
    /// # Safety
    /// `weights` deve ter `in_len * out_len` elementos no layout `[IN][OUT]`.
    /// `bias` deve ter ao menos `out_len` elementos.
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// Realiza a projeção linear Y = Bias + W * Z (GEMV) para BF16.
    ///
    /// # Safety
    /// `weights` deve ter `in_len * out_len` elementos no layout `[IN][OUT]`.
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// Aplica Tanh em um bloco pequeno (CH sized) com padding para evitar loops escalares.
    /// Otimizado para ativações WaveNet onde CH é tipicamente 4, 8, 12 ou 16.
    ///
    /// # Safety
    /// `buf` deve ser válido e acessível para leitura e escrita.
    unsafe fn activation_tanh_block(buf: &mut [f32]);

    /// Calcula a soma horizontal de N elementos a partir de um ponteiro.
    ///
    /// # Safety
    /// `ptr` deve ser válido para leitura de N floats.
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32;

    /// Executa a ativação fundida dos gates LSTM.
    ///
    /// # Safety
    /// Os argumentos devem ser vetores válidos.
    unsafe fn fused_lstm_gates(
        gf: Self::V,
        gi: Self::V,
        gg: Self::V,
        go: Self::V,
        cs: Self::V,
    ) -> (Self::V, Self::V);
}

/// Implementação estática para microarquitetura x86-64-v3 (AVX2/FMA).
pub struct Avx2Math;
impl SimdMath for Avx2Math {
    type V = __m256;
    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { dot_product_avx2(a, b) }
    }
    #[inline(always)]
    unsafe fn dot_product_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        state: &[f32],
    ) -> [f32; 4] {
        unsafe { dot_product_4x_avx2(w0, w1, w2, w3, state) }
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
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::tanh_slice_avx2(slice) }
    }
    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::sigmoid_slice_avx2(slice) }
    }
    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
        let len = core::cmp::min(src.len(), dst.len());
        for i in 0..len {
            unsafe {
                *dst.get_unchecked_mut(i) = (src.get_unchecked(i).to_bits() >> 16) as u16;
            }
        }
    }
    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { dot_product_bf16_fallback(a, b) }
    }
    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe {
            let r0 = dot_product_bf16_fallback(w0, in_frame);
            let r1 = dot_product_bf16_fallback(w1, in_frame);
            let r2 = dot_product_bf16_fallback(w2, in_frame);
            let r3 = dot_product_bf16_fallback(w3, in_frame);
            [r0, r1, r2, r3]
        }
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
    unsafe fn fused_lstm_gates(
        gf: Self::V,
        gi: Self::V,
        gg: Self::V,
        go: Self::V,
        cs: Self::V,
    ) -> (Self::V, Self::V) {
        unsafe { crate::math::fastmath::fused_lstm_gates_avx2(gf, gi, gg, go, cs) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        let len = buf.len();
        if len <= 8 {
            let mut tmp = [0.0f32; 8];
            tmp[..len].copy_from_slice(buf);
            unsafe {
                let v = _mm256_loadu_ps(tmp.as_ptr());
                let res = crate::math::fastmath::simd_tanh(v);
                _mm256_storeu_ps(tmp.as_mut_ptr(), res);
            }
            buf.copy_from_slice(&tmp[..len]);
        } else if len <= 16 {
            let mut tmp = [0.0f32; 16];
            tmp[..len].copy_from_slice(buf);
            unsafe {
                let v0 = _mm256_loadu_ps(tmp.as_ptr());
                let v1 = _mm256_loadu_ps(tmp.as_ptr().add(8));
                let res0 = crate::math::fastmath::simd_tanh(v0);
                let res1 = crate::math::fastmath::simd_tanh(v1);
                _mm256_storeu_ps(tmp.as_mut_ptr(), res0);
                _mm256_storeu_ps(tmp.as_mut_ptr().add(8), res1);
            }
            buf.copy_from_slice(&tmp[..len]);
        } else {
            unsafe { Self::tanh_slice(buf) };
        }
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        if N == 0 {
            return 0.0;
        }
        if N == 1 {
            return unsafe { *ptr };
        }

        unsafe {
            if N <= 4 {
                let v = if N == 4 {
                    _mm_loadu_ps(ptr)
                } else {
                    let mut tmp = [0.0f32; 4];
                    core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr(), N);
                    _mm_loadu_ps(tmp.as_ptr())
                };
                let h1 = _mm_hadd_ps(v, v);
                let h2 = _mm_hadd_ps(h1, h1);
                _mm_cvtss_f32(h2)
            } else if N <= 8 {
                let v = if N == 8 {
                    _mm256_loadu_ps(ptr)
                } else {
                    let mut tmp = [0.0f32; 8];
                    core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr(), N);
                    _mm256_loadu_ps(tmp.as_ptr())
                };
                let h1 = _mm256_hadd_ps(v, v);
                let h2 = _mm256_hadd_ps(h1, h1);
                let lo = _mm256_castps256_ps128(h2);
                let hi = _mm256_extractf128_ps::<1>(h2);
                let sum128 = _mm_add_ss(lo, hi);
                _mm_cvtss_f32(sum128)
            } else {
                let mut sum_v = _mm256_setzero_ps();
                let mut i = 0;
                while i + 8 <= N {
                    let v = _mm256_loadu_ps(ptr.add(i));
                    sum_v = _mm256_add_ps(sum_v, v);
                    i += 8;
                }

                let h1 = _mm256_hadd_ps(sum_v, sum_v);
                let h2 = _mm256_hadd_ps(h1, h1);
                let lo = _mm256_castps256_ps128(h2);
                let hi = _mm256_extractf128_ps::<1>(h2);
                let sum128 = _mm_add_ss(lo, hi);
                let mut total = _mm_cvtss_f32(sum128);

                while i < N {
                    total += *ptr.add(i);
                    i += 1;
                }
                total
            }
        }
    }
}

/// Implementação estática para microarquitetura x86-64-v3 com AVX-VNNI (Alder Lake+).
pub struct Avx2VnniMath;
impl SimdMath for Avx2VnniMath {
    type V = __m256;
    #[target_feature(enable = "avxvnni")]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { dot_product_avx2(a, b) }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn dot_product_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        state: &[f32],
    ) -> [f32; 4] {
        unsafe { dot_product_4x_avx2(w0, w1, w2, w3, state) }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_avx2(weights, state) }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_bf16_fallback(weights, state) }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::tanh_slice_avx2(slice) }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::sigmoid_slice_avx2(slice) }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
        let len = core::cmp::min(src.len(), dst.len());
        for i in 0..len {
            unsafe {
                *dst.get_unchecked_mut(i) = (src.get_unchecked(i).to_bits() >> 16) as u16;
            }
        }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { dot_product_bf16_fallback(a, b) }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe {
            let r0 = dot_product_bf16_fallback(w0, in_frame);
            let r1 = dot_product_bf16_fallback(w1, in_frame);
            let r2 = dot_product_bf16_fallback(w2, in_frame);
            let r3 = dot_product_bf16_fallback(w3, in_frame);
            [r0, r1, r2, r3]
        }
    }

    #[target_feature(enable = "avxvnni")]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { fused_add_gemv_avx2(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[target_feature(enable = "avxvnni")]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_avx2(in_frame, weights, bias, out_frame, do_bias) }
    }
    #[target_feature(enable = "avxvnni")]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[target_feature(enable = "avxvnni")]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { Avx2Math::activation_tanh_block(buf) }
    }

    #[target_feature(enable = "avxvnni")]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { Avx2Math::horizontal_sum::<N>(ptr) }
    }

    #[target_feature(enable = "avxvnni")]
    unsafe fn fused_lstm_gates(
        gf: Self::V,
        gi: Self::V,
        gg: Self::V,
        go: Self::V,
        cs: Self::V,
    ) -> (Self::V, Self::V) {
        unsafe { crate::math::fastmath::fused_lstm_gates_avx2(gf, gi, gg, go, cs) }
    }
}

/// Implementação estática para microarquitetura x86-64-v4 (AVX-512).
pub struct Avx512Math;
impl SimdMath for Avx512Math {
    type V = __m512;
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { dot_product_avx512(a, b) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        state: &[f32],
    ) -> [f32; 4] {
        unsafe { dot_product_4x_avx512(w0, w1, w2, w3, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_avx512(weights, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        let len = core::cmp::min(weights.len(), state.len());
        let mut sum = [0.0f32; 4];
        for i in 0..len {
            let s = f32::from_bits((state[i] as u32) << 16);
            let w = weights[i];
            sum[0] += f32::from_bits((w[0] as u32) << 16) * s;
            sum[1] += f32::from_bits((w[1] as u32) << 16) * s;
            sum[2] += f32::from_bits((w[2] as u32) << 16) * s;
            sum[3] += f32::from_bits((w[3] as u32) << 16) * s;
        }
        sum
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::tanh_slice_avx512(slice) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::sigmoid_slice_avx512(slice) }
    }
    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
        let len = core::cmp::min(src.len(), dst.len());
        for i in 0..len {
            unsafe {
                *dst.get_unchecked_mut(i) = (src.get_unchecked(i).to_bits() >> 16) as u16;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { dot_product_bf16_fallback(a, b) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe {
            let r0 = dot_product_bf16_fallback(w0, in_frame);
            let r1 = dot_product_bf16_fallback(w1, in_frame);
            let r2 = dot_product_bf16_fallback(w2, in_frame);
            let r3 = dot_product_bf16_fallback(w3, in_frame);
            [r0, r1, r2, r3]
        }
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { fused_add_gemv_avx512(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_avx512(in_frame, weights, bias, out_frame, do_bias) }
    }
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        let len = buf.len();
        if len <= 16 {
            let mut tmp = [0.0f32; 16];
            tmp[..len].copy_from_slice(buf);
            unsafe {
                let v = _mm512_loadu_ps(tmp.as_ptr());
                let res = crate::math::fastmath::simd_tanh_avx512(v);
                _mm512_storeu_ps(tmp.as_mut_ptr(), res);
            }
            buf.copy_from_slice(&tmp[..len]);
        } else if len <= 32 {
            let mut tmp = [0.0f32; 32];
            tmp[..len].copy_from_slice(buf);
            unsafe {
                let v0 = _mm512_loadu_ps(tmp.as_ptr());
                let v1 = _mm512_loadu_ps(tmp.as_ptr().add(16));
                let res0 = crate::math::fastmath::simd_tanh_avx512(v0);
                let res1 = crate::math::fastmath::simd_tanh_avx512(v1);
                _mm512_storeu_ps(tmp.as_mut_ptr(), res0);
                _mm512_storeu_ps(tmp.as_mut_ptr().add(16), res1);
            }
            buf.copy_from_slice(&tmp[..len]);
        } else {
            unsafe { Self::tanh_slice(buf) };
        }
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        if N == 0 {
            return 0.0;
        }
        if N == 1 {
            return unsafe { *ptr };
        }

        unsafe {
            if N >= 16 {
                let mut sum_v = _mm512_setzero_ps();
                let mut i = 0;
                while i + 16 <= N {
                    let v = _mm512_loadu_ps(ptr.add(i));
                    sum_v = _mm512_add_ps(sum_v, v);
                    i += 16;
                }
                let mut total = _mm512_reduce_add_ps(sum_v);
                while i < N {
                    total += *ptr.add(i);
                    i += 1;
                }
                total
            } else if N >= 8 {
                let v = if N == 8 {
                    _mm256_loadu_ps(ptr)
                } else {
                    let mut tmp = [0.0f32; 8];
                    core::ptr::copy_nonoverlapping(ptr, tmp.as_mut_ptr(), N);
                    _mm256_loadu_ps(tmp.as_ptr())
                };
                let h1 = _mm256_hadd_ps(v, v);
                let h2 = _mm256_hadd_ps(h1, h1);
                let lo = _mm256_castps256_ps128(h2);
                let hi = _mm256_extractf128_ps::<1>(h2);
                let sum128 = _mm_add_ss(lo, hi);
                let total = _mm_cvtss_f32(sum128);

                // Como N < 16, o tail é pequeno
                let mut final_sum = total;
                for j in 8..N {
                    final_sum += *ptr.add(j);
                }
                final_sum
            } else {
                Avx2Math::horizontal_sum::<N>(ptr)
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn fused_lstm_gates(
        gf: Self::V,
        gi: Self::V,
        gg: Self::V,
        go: Self::V,
        cs: Self::V,
    ) -> (Self::V, Self::V) {
        unsafe { crate::math::fastmath::fused_lstm_gates_avx512(gf, gi, gg, go, cs) }
    }
}

/// Implementação estática para microarquitetura x86-64-v4 com AVX-512 VNNI (Ice Lake+).
pub struct Avx512VnniMath;
impl SimdMath for Avx512VnniMath {
    type V = __m512;
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { dot_product_avx512(a, b) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn dot_product_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        state: &[f32],
    ) -> [f32; 4] {
        unsafe { dot_product_4x_avx512(w0, w1, w2, w3, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_avx512(weights, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        let len = core::cmp::min(weights.len(), state.len());
        let mut sum = [0.0f32; 4];
        for i in 0..len {
            let s = f32::from_bits((state[i] as u32) << 16);
            let w = weights[i];
            sum[0] += f32::from_bits((w[0] as u32) << 16) * s;
            sum[1] += f32::from_bits((w[1] as u32) << 16) * s;
            sum[2] += f32::from_bits((w[2] as u32) << 16) * s;
            sum[3] += f32::from_bits((w[3] as u32) << 16) * s;
        }
        sum
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::tanh_slice_avx512(slice) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::sigmoid_slice_avx512(slice) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
        let len = core::cmp::min(src.len(), dst.len());
        for i in 0..len {
            unsafe {
                *dst.get_unchecked_mut(i) = (src.get_unchecked(i).to_bits() >> 16) as u16;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { dot_product_bf16_fallback(a, b) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe {
            let r0 = dot_product_bf16_fallback(w0, in_frame);
            let r1 = dot_product_bf16_fallback(w1, in_frame);
            let r2 = dot_product_bf16_fallback(w2, in_frame);
            let r3 = dot_product_bf16_fallback(w3, in_frame);
            [r0, r1, r2, r3]
        }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { fused_add_gemv_avx512(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_avx512(in_frame, weights, bias, out_frame, do_bias) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { Avx512Math::activation_tanh_block(buf) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { Avx512Math::horizontal_sum::<N>(ptr) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn fused_lstm_gates(
        gf: Self::V,
        gi: Self::V,
        gg: Self::V,
        go: Self::V,
        cs: Self::V,
    ) -> (Self::V, Self::V) {
        unsafe { crate::math::fastmath::fused_lstm_gates_avx512(gf, gi, gg, go, cs) }
    }
}

/// Implementação estática para microarquitetura com AVX-512 e suporte a BF16 (VNNI).
pub struct Avx512VnniBf16Math;
impl SimdMath for Avx512VnniBf16Math {
    type V = __m512;
    const IS_BF16: bool = true;

    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { dot_product_avx512(a, b) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn dot_product_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        state: &[f32],
    ) -> [f32; 4] {
        unsafe { dot_product_4x_avx512(w0, w1, w2, w3, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_avx512(weights, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::tanh_slice_avx512(slice) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::fastmath::sigmoid_slice_avx512(slice) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
        let len = core::cmp::min(src.len(), dst.len());
        let mut i = 0;
        while i + 32 <= len {
            unsafe {
                let v = _mm512_loadu_ps(src.as_ptr().add(i));
                let vbf16 = _mm512_cvtneps_pbh(v);
                core::arch::x86_64::_mm256_storeu_si256(
                    dst.as_mut_ptr().add(i) as *mut __m256i,
                    core::mem::transmute::<__m256bh, __m256i>(vbf16),
                );
                i += 32;
            }
        }
        while i < len {
            unsafe {
                *dst.get_unchecked_mut(i) = (src.get_unchecked(i).to_bits() >> 16) as u16;
                i += 1;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_bf16_avx512(weights, state) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { dot_product_bf16_native_avx512(a, b) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe { dot_product_bf16_4x_native_avx512(w0, w1, w2, w3, in_frame) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { fused_add_gemv_avx512(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_avx512(in_frame, weights, bias, out_frame, do_bias) }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { Avx512Math::activation_tanh_block(buf) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { Avx512Math::horizontal_sum::<N>(ptr) }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bw,avx512bf16")]
    unsafe fn fused_lstm_gates(
        gf: Self::V,
        gi: Self::V,
        gg: Self::V,
        go: Self::V,
        cs: Self::V,
    ) -> (Self::V, Self::V) {
        unsafe { crate::math::fastmath::fused_lstm_gates_avx512(gf, gi, gg, go, cs) }
    }
}

/// Implementação puramente escalar (fallback universal).
pub struct ScalarMath;
impl SimdMath for ScalarMath {
    type V = f32;
    const IS_BF16: bool = false;

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        let len = core::cmp::min(a.len(), b.len());
        let mut sum = 0.0;
        for i in 0..len {
            sum += a[i] * half::f16::from_bits(b[i]).to_f32();
        }
        sum
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dst: &mut [u16]) {
        let len = core::cmp::min(src.len(), dst.len());
        for i in 0..len {
            dst[i] = (src[i].to_bits() >> 16) as u16;
        }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { dot_product_bf16_fallback(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe {
            [
                dot_product_bf16_fallback(w0, in_frame),
                dot_product_bf16_fallback(w1, in_frame),
                dot_product_bf16_fallback(w2, in_frame),
                dot_product_bf16_fallback(w3, in_frame),
            ]
        }
    }

    #[inline(always)]
    unsafe fn dot_product_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        state: &[f32],
    ) -> [f32; 4] {
        let len = state.len();
        let mut res = [0.0; 4];
        for i in 0..len {
            let s = state[i];
            res[0] += half::f16::from_bits(w0[i]).to_f32() * s;
            res[1] += half::f16::from_bits(w1[i]).to_f32() * s;
            res[2] += half::f16::from_bits(w2[i]).to_f32() * s;
            res[3] += half::f16::from_bits(w3[i]).to_f32() * s;
        }
        res
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        let len = state.len();
        let mut res = [0.0; 4];
        for i in 0..len {
            let s = state[i];
            let w = weights[i];
            res[0] += half::f16::from_bits(w[0]).to_f32() * s;
            res[1] += half::f16::from_bits(w[1]).to_f32() * s;
            res[2] += half::f16::from_bits(w[2]).to_f32() * s;
            res[3] += half::f16::from_bits(w[3]).to_f32() * s;
        }
        res
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_bf16_fallback(weights, state) }
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        for x in slice.iter_mut() {
            *x = x.tanh();
        }
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        for x in slice.iter_mut() {
            *x = 0.5 * (1.0 + (*x * 0.5).tanh());
        }
    }

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        let out_len = out_frame.len();
        let in_len = in_frame.len();
        for out_c in 0..out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w = half::f16::from_bits(weights[in_c * out_len + out_c]).to_f32();
                sum += in_frame[in_c] * w;
            }
            out_frame[out_c] += sum;
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
        let out_len = out_frame.len();
        let in_len = in_frame.len();
        for out_c in 0..out_len {
            let mut sum = if do_bias { bias[out_c] } else { 0.0 };
            for in_c in 0..in_len {
                let w = half::f16::from_bits(weights[in_c * out_len + out_c]).to_f32();
                sum += in_frame[in_c] * w;
            }
            out_frame[out_c] = sum;
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
        unsafe { gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        for x in buf.iter_mut() {
            *x = x.tanh();
        }
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        let mut sum = 0.0;
        for i in 0..N {
            // SAFETY: Caller must ensure ptr is valid for N elements.
            sum += unsafe { *ptr.add(i) };
        }
        sum
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates(gf: f32, gi: f32, gg: f32, go: f32, cs: f32) -> (f32, f32) {
        let f = 0.5 * (1.0 + (gf * 0.5).tanh());
        let i = 0.5 * (1.0 + (gi * 0.5).tanh());
        let g = gg.tanh();
        let o = 0.5 * (1.0 + (go * 0.5).tanh());

        let new_cs = f * cs + i * g;
        let hidden = o * new_cs.tanh();
        (new_cs, hidden)
    }
}

/// Macro de despacho dinâmico no hot-path.
/// Baseia-se no `SimdMathConfig::get().instruction_set` avaliado no startup
/// para monomorfizar o bloco DSP de modelos com a trait mais otimizada disponível,
/// garantindo zero-overhead na thread RT.
#[macro_export]
macro_rules! dispatch_simd {
    // Formato 2: Dispatch explícito (LSTM)
    ($self:ident, $m_bf16:ident, $m_a512v:ident, $m_a512:ident, $m_a2v:ident, $m_a2:ident $(, $arg:expr)*) => {
        match $crate::math::simd::SimdMathConfig::get().instruction_set {
            $crate::math::simd::SimdInstructionSet::Avx512VnniBf16 => $self.$m_bf16($($arg),*),
            $crate::math::simd::SimdInstructionSet::Avx512Vnni => $self.$m_a512v($($arg),*),
            $crate::math::simd::SimdInstructionSet::Avx512 => $self.$m_a512($($arg),*),
            $crate::math::simd::SimdInstructionSet::Avx2Vnni => $self.$m_a2v($($arg),*),
            $crate::math::simd::SimdInstructionSet::Avx2 => $self.$m_a2($($arg),*),
        }
    };
    // Formato 1: Dispatch via monomorfização genérica (WaveNet)
    ($self:ident, $method:ident $(, $arg:expr)*) => {
        match $crate::math::simd::SimdMathConfig::get().instruction_set {
            $crate::math::simd::SimdInstructionSet::Avx512VnniBf16 => {
                $self.$method::<$crate::math::simd::Avx512VnniBf16Math>($($arg),*)
            }
            $crate::math::simd::SimdInstructionSet::Avx512Vnni => {
                $self.$method::<$crate::math::simd::Avx512VnniMath>($($arg),*)
            }
            $crate::math::simd::SimdInstructionSet::Avx512 => {
                $self.$method::<$crate::math::simd::Avx512Math>($($arg),*)
            }
            $crate::math::simd::SimdInstructionSet::Avx2Vnni => {
                $self.$method::<$crate::math::simd::Avx2VnniMath>($($arg),*)
            }
            $crate::math::simd::SimdInstructionSet::Avx2 => {
                $self.$method::<$crate::math::simd::Avx2Math>($($arg),*)
            }
        }
    };
}
pub use dispatch_simd;

/// Calcula o Dot Product de um lote de 4 vetores (h0..h3) com o mesmo vetor de pesos.
/// Otimizado para processamento em lote da cabeça do LSTM.
///
/// # Safety
/// Requer CPU com AVX2, FMA e F16C. Todos os slices devem ser válidos e
/// ter `len() >= weights.len()`.
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
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        while i + 16 <= len {
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h0.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h1.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h2.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h3.as_ptr().add(i + 32) as *const i8);

            let vw_0 = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
            let vh0_0 = _mm256_loadu_ps(h0.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(vw_0, vh0_0, sum0);
            let vh1_0 = _mm256_loadu_ps(h1.as_ptr().add(i));
            sum1 = _mm256_fmadd_ps(vw_0, vh1_0, sum1);
            let vh2_0 = _mm256_loadu_ps(h2.as_ptr().add(i));
            sum2 = _mm256_fmadd_ps(vw_0, vh2_0, sum2);
            let vh3_0 = _mm256_loadu_ps(h3.as_ptr().add(i));
            sum3 = _mm256_fmadd_ps(vw_0, vh3_0, sum3);

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

        #[inline(always)]
        unsafe fn hsum_avx2(v: __m256) -> f32 {
            unsafe {
                let hi = _mm256_extractf128_ps(v, 1);
                let lo = _mm256_castps256_ps128(v);
                let s128 = _mm_add_ps(lo, hi);
                let shuf = _mm_movehdup_ps(s128);
                let sums = _mm_add_ps(s128, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let r = _mm_add_ss(sums, shuf2);
                let mut out = 0.0f32;
                _mm_store_ss(&mut out, r);
                out
            }
        }

        let mut s0 = hsum_avx2(sum0);
        let mut s1 = hsum_avx2(sum1);
        let mut s2 = hsum_avx2(sum2);
        let mut s3 = hsum_avx2(sum3);

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

/// Calcula o Dot Product de um lote de 4 vetores via AVX-512.
///
/// # Safety
/// Requer CPU com AVX-512F e AVX-512VL. Todos os slices devem ser válidos e
/// ter `len() >= weights.len()`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_batch_4x_avx512(
    h0: &[f32],
    h1: &[f32],
    h2: &[f32],
    h3: &[f32],
    weights: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), h0.len());
    let mut i = 0;

    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();

        while i + 16 <= len {
            let vw = _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            let vh0 = _mm512_loadu_ps(h0.as_ptr().add(i));
            sum0 = _mm512_fmadd_ps(vw, vh0, sum0);
            let vh1 = _mm512_loadu_ps(h1.as_ptr().add(i));
            sum1 = _mm512_fmadd_ps(vw, vh1, sum1);
            let vh2 = _mm512_loadu_ps(h2.as_ptr().add(i));
            sum2 = _mm512_fmadd_ps(vw, vh2, sum2);
            let vh3 = _mm512_loadu_ps(h3.as_ptr().add(i));
            sum3 = _mm512_fmadd_ps(vw, vh3, sum3);

            i += 16;
        }

        let mut s0 = _mm512_reduce_add_ps(sum0);
        let mut s1 = _mm512_reduce_add_ps(sum1);
        let mut s2 = _mm512_reduce_add_ps(sum2);
        let mut s3 = _mm512_reduce_add_ps(sum3);

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

/// Realiza a operação fundida Y = X_res + Bias + W * Z (Broadcast GEMV) via AVX2.
///
/// # Safety
/// Requer CPU com AVX2, FMA e F16C. `weights` deve ter `in_len * out_len`
/// elementos no layout `[IN][OUT]`. `bias` deve ter ao menos `out_len` elementos.
/// `out_frame` é lido e escrito (acumulação in-place sobre o residual).
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn fused_add_gemv_avx2(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    use core::arch::x86_64::*;
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 8 <= out_len {
        let mut accum = unsafe { _mm256_loadu_ps(out_frame.as_ptr().add(out_c)) };
        if do_bias {
            accum = unsafe { _mm256_add_ps(accum, _mm256_loadu_ps(bias.as_ptr().add(out_c))) };
        }

        for in_c in 0..in_len {
            let vs = unsafe { _mm256_set1_ps(*in_frame.get_unchecked(in_c)) };
            let weight_ptr = unsafe { weights.as_ptr().add(in_c * out_len + out_c) };
            let vw = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i)) };
            accum = _mm256_fmadd_ps(vs, vw, accum);
        }

        unsafe {
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
        }
        out_c += 8;
    }

    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w = unsafe {
                half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32()
            };
            sum += unsafe { *in_frame.get_unchecked(in_c) } * w;
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) += sum;
        }
        out_c += 1;
    }
}

/// Realiza a projeção linear Y = Bias + W * Z (GEMV) substituindo o conteúdo de out_frame.
/// out_frame: vetor Y (overwrite).
///
/// # Safety
/// Requer CPU com AVX2, FMA e F16C. `weights` deve ter `in_len * out_len`
/// elementos no layout `[IN][OUT]`. `bias` deve ter ao menos `out_len` elementos.
/// `out_frame` é sobrescrito (overwrite).
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn gemv_overwrite_avx2(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    use core::arch::x86_64::*;
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 8 <= out_len {
        let mut accum = if do_bias {
            unsafe { _mm256_loadu_ps(bias.as_ptr().add(out_c)) }
        } else {
            _mm256_setzero_ps()
        };

        for in_c in 0..in_len {
            let vs = unsafe { _mm256_set1_ps(*in_frame.get_unchecked(in_c)) };
            let weight_ptr = unsafe { weights.as_ptr().add(in_c * out_len + out_c) };
            let vw = unsafe { _mm256_cvtph_ps(_mm_loadu_si128(weight_ptr as *const __m128i)) };
            accum = _mm256_fmadd_ps(vs, vw, accum);
        }

        unsafe {
            _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
        }
        out_c += 8;
    }

    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w = unsafe {
                half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32()
            };
            sum += unsafe { *in_frame.get_unchecked(in_c) } * w;
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
        out_c += 1;
    }
}

/// Realiza a operação fundida Y = X_res + Bias + W * Z via AVX-512.
///
/// # Safety
/// Requer CPU com AVX-512F e AVX-512VL. Mesmos invariantes de layout de
/// `fused_add_gemv_avx2`: `weights` em `[IN][OUT]`, `bias` com `out_len` elementos.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_add_gemv_avx512(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    use core::arch::x86_64::*;
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut accum = unsafe { _mm512_loadu_ps(out_frame.as_ptr().add(out_c)) };
        if do_bias {
            accum = unsafe { _mm512_add_ps(accum, _mm512_loadu_ps(bias.as_ptr().add(out_c))) };
        }

        for in_c in 0..in_len {
            let vs = unsafe { _mm512_set1_ps(*in_frame.get_unchecked(in_c)) };
            let weight_ptr = unsafe { weights.as_ptr().add(in_c * out_len + out_c) };
            let vw = unsafe { _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i)) };
            accum = _mm512_fmadd_ps(vs, vw, accum);
        }

        unsafe {
            _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
        }
        out_c += 16;
    }

    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w = unsafe {
                half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32()
            };
            sum += unsafe { *in_frame.get_unchecked(in_c) } * w;
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) += sum;
        }
        out_c += 1;
    }
}

/// Realiza a projeção linear Y = Bias + W * Z via AVX-512.
///
/// # Safety
/// Requer CPU com AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gemv_overwrite_avx512(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    use core::arch::x86_64::*;
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut accum = if do_bias {
            unsafe { _mm512_loadu_ps(bias.as_ptr().add(out_c)) }
        } else {
            _mm512_setzero_ps()
        };

        for in_c in 0..in_len {
            let vs = unsafe { _mm512_set1_ps(*in_frame.get_unchecked(in_c)) };
            let weight_ptr = unsafe { weights.as_ptr().add(in_c * out_len + out_c) };
            let vw = unsafe { _mm512_cvtph_ps(_mm256_loadu_si256(weight_ptr as *const __m256i)) };
            accum = _mm512_fmadd_ps(vs, vw, accum);
        }

        unsafe {
            _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), accum);
        }
        out_c += 16;
    }

    while out_c < out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            let w = unsafe {
                half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32()
            };
            sum += unsafe { *in_frame.get_unchecked(in_c) } * w;
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
        out_c += 1;
    }
}

/// Calcula a energia (Mean Square) de um bloco via AVX2.
/// $E = \frac{1}{N} \sum x_i^2$
///
/// # Safety
/// O slice `data` deve ser válido.
pub unsafe fn compute_energy_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    unsafe {
        // Acumuladores vetoriais (8 floats cada) inicializados com zero.
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();

        // Loop principal desenrolado: processa 16 amostras por iteração (2 vetores AVX).
        // Isso melhora a ocupação das unidades de execução FMA da CPU.
        while i + 16 <= len {
            // Carregamento não-alinhado (unaligned load) de 8 floats por vez.
            let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));

            // Fused Multiply-Add (FMA): sum = (v * v) + sum.
            // Quadrado da amostra acumulado diretamente, reduzindo erros de arredondamento.
            sum0 = _mm256_fmadd_ps(v0, v0, sum0);
            sum1 = _mm256_fmadd_ps(v1, v1, sum1);
            i += 16;
        }

        // Processa blocos remanescentes de 8 amostras.
        while i + 8 <= len {
            let v = _mm256_loadu_ps(data.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(v, v, sum0);
            i += 8;
        }

        // Soma os dois acumuladores vetoriais.
        let sum = _mm256_add_ps(sum0, sum1);

        // --- Redução Horizontal: Somar os 8 floats dentro do registrador AVX ---
        // 1. Extrai a metade alta (128 bits / 4 floats) e soma com a metade baixa.
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let s128 = _mm_add_ps(lo, hi); // [a+e, b+f, c+g, d+h]

        // 2. Embaralha e soma pares internos (Shuffle + Add).
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let r = _mm_add_ss(sums, shuf2);

        // Extrai o resultado final (scalar f32).
        let mut total_sum = 0.0f32;
        _mm_store_ss(&mut total_sum, r);

        // Loop de limpeza (Tail Loop): Processa o que restou (menos de 8 amostras).
        while i < len {
            total_sum += data[i] * data[i];
            i += 1;
        }

        total_sum / (len as f32)
    }
}

/// Fallback escalar para o GEMV BF16.
///
/// # Safety
/// `weights` deve ter `in_len * out_len` elementos no layout `[IN][OUT]`.
pub unsafe fn gemv_overwrite_bf16_fallback(
    in_frame: &[u16],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { *b } else { 0.0 };
        for in_c in 0..in_len {
            let w_bits = unsafe { *weights.get_unchecked(in_c * out_len + out_c) };
            let in_bits = unsafe { *in_frame.get_unchecked(in_c) };
            let w = f32::from_bits((w_bits as u32) << 16);
            let in_val = f32::from_bits((in_bits as u32) << 16);
            sum += in_val * w;
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Calcula a diferença absoluta máxima entre dois blocos via AVX2.
/// $\max(|L_i - R_i|)$
///
/// # Safety
/// Os slices `a` e `b` devem ter o mesmo tamanho.
pub unsafe fn compute_max_diff_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    unsafe {
        // Acumulador do máximo absoluto (8 canais).
        let mut max_v = _mm256_setzero_ps();
        // Máscara para extrair o valor absoluto (limpa o bit de sinal).
        let sign_mask = _mm256_set1_ps(-0.0f32);

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            // Diferença L - R.
            let diff = _mm256_sub_ps(va, vb);

            // Valor absoluto: ANDNOT da máscara de sinal com a diferença.
            // Em IEEE-754, limpar o bit de sinal de um float resulta em abs().
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);

            // Atualiza o vetor de máximos amostra a amostra.
            max_v = _mm256_max_ps(max_v, abs_diff);
            i += 8;
        }

        // --- Redução Horizontal para encontrar o valor máximo global ---
        let hi = _mm256_extractf128_ps(max_v, 1);
        let lo = _mm256_castps256_ps128(max_v);
        let m128 = _mm_max_ps(lo, hi);

        // Shuffle para comparar f32 adjacentes e encontrar o maior.
        let shuf = _mm_shuffle_ps(m128, m128, 0xEE); // [3,2,3,2]
        let m64 = _mm_max_ps(m128, shuf);
        let shuf2 = _mm_shuffle_ps(m64, m64, 0x55); // [1,1,1,1]
        let m32 = _mm_max_ps(m64, shuf2);

        let mut max_diff = 0.0f32;
        _mm_store_ss(&mut max_diff, m32);

        // Tail loop para elementos restantes.
        while i < len {
            let d = (a[i] - b[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
            i += 1;
        }

        max_diff
    }
}

#[cfg(test)]
#[path = "simd_test.rs"]
mod simd_test;
