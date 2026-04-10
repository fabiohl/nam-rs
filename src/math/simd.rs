// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Engenharia de Registradores baseada em instruções explícitas x86_64.
//!
//! Este módulo exporta funções analíticas implementadas com instrinsics de AVX2 e FMA,
//! otimizando os cálculos críticos (como Fused Multiply-Add) limitando os
//! desvios matemáticos inerentes aos loops comuns e reduzindo latência nas CNNs.

#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
/// # Safety
/// O chamador deve assegurar que a CPU suporta os recursos "avx2" e "fma".
/// Verifique o suporte em tempo de inicialização via `is_x86_feature_detected!("avx2")`
/// e `is_x86_feature_detected!("fma")`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;

    unsafe {
        // Inicializa o acumulador de 256 bits (8 floats) com zero
        let mut sum = _mm256_setzero_ps();

        // Processa de 8 em 8 (256 bits = 8 * 32)
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_fmadd_ps(va, vb, sum);
            i += 8;
        }

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum);

        let mut scalar_sum = temp.iter().sum();

        // Loop tail
        while i < len {
            scalar_sum += a[i] * b[i];
            i += 1;
        }

        scalar_sum
    }
}

/// Calcula o Dot Product (Produto Escalar) de duas fatias via AVX-512 (ZMM).
///
/// # Safety
/// O chamador deve assegurar que a CPU suporta os recursos "avx512f" e "avx512vl".
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut i = 0;

    unsafe {
        let mut sum = _mm512_setzero_ps();

        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            sum = _mm512_fmadd_ps(va, vb, sum);
            i += 16;
        }

        let mut scalar_sum = _mm512_reduce_add_ps(sum);

        while i < len {
            scalar_sum += a[i] * b[i];
            i += 1;
        }

        scalar_sum
    }
}

/// Estrutura de Inversão de Controle com Função de Despacho Dinâmico de Ponteiros.
/// Resolve transparentemente o multiversionamento das Redes Inferenciais sem alocações.
#[derive(Clone, Copy)]
pub struct SimdMathConfig {
    /// Função inlined dinamicamente agendada para computar fma vetorial.
    pub dot_product: unsafe fn(&[f32], &[f32]) -> f32,
    /// Loop ativado via fptr para iterar `tanh(x)` na matriz especificada.
    pub tanh_slice: unsafe fn(&mut [f32]),
    /// Loop ativado via fptr para iterar `sigmoid(x)` na matriz especificada.
    pub sigmoid_slice: unsafe fn(&mut [f32]),
}

impl SimdMathConfig {
    /// Inicia e aloca a v-table matemática inspecionando nativamente as capabilities da CPU.
    pub fn current() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            return Self {
                dot_product: dot_product_avx512,
                tanh_slice: crate::math::fastmath::tanh_slice_avx512,
                sigmoid_slice: crate::math::fastmath::sigmoid_slice_avx512,
            };
        }

        Self {
            dot_product: dot_product_avx2,
            tanh_slice: crate::math::fastmath::tanh_slice_avx2,
            sigmoid_slice: crate::math::fastmath::sigmoid_slice_avx2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_avx2_fma() {
        // Ignora no CI se a máquina host não puder rodar avx2
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
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
    }

    #[test]
    fn test_dot_product_avx512() {
        #[cfg(target_arch = "x86_64")]
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
        #[cfg(target_arch = "x86_64")]
        {
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
}
