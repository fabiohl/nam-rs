// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Engenharia de Registradores baseada em instruções explícitas x86_64.
//!
//! Este módulo exporta funções analíticas implementadas com instrinsics de AVX2 e FMA,
//! otimizando os cálculos críticos (como Fused Multiply-Add) limitando os
//! desvios matemáticos inerentes aos loops comuns e reduzindo latência nas CNNs.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
}
