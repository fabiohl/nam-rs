// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de FastMath (Minimax & Pade) para otimização de ativação não linear.
//!
//! Implementa aproximações de alta performance em `core::arch::x86_64` (FMA)
//! mitigando gargalos de ALU ao calcular `tanh` e `sigmoid` na `WaveNet`/`LSTM`.
//! As aproximações são derivadas de polinômios do ecossistema referencial (math_approx).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Aplica aproximação vetorial de `tanh(x)` iterando um polinômio de grau 5.
///
/// Algoritmo otimizado correspondente ao `tanh<5>` usado no NAM core. O polinômio
/// é resolvido usando instruções primitivas de Fused Multiply-Add (FMA) e
/// então dividido pela sua aproximação de raiz quadrada recíproca.
///
/// # Safety
/// O chamador deve assegurar que a máquina host tem as features `avx2` e `fma` ativadas.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh(x: __m256) -> __m256 {
    // Coeficientes do polinômio Minimax/Pade de grau 5
    let c0 = _mm256_set1_ps(0.165_326_98_f32);
    let c1 = _mm256_set1_ps(0.009_702_402_f32);
    let one = _mm256_set1_ps(1.0);

    // x_sq = x * x
    let x_sq = _mm256_mul_ps(x, x);

    // y_3_5 = 0.165326984031 + 0.00970240200826 * x_sq
    let y_3_5 = _mm256_fmadd_ps(c1, x_sq, c0);

    // y_1_3_5 = 1.0 + y_3_5 * x_sq
    let y_1_3_5 = _mm256_fmadd_ps(y_3_5, x_sq, one);

    // p(x) = x * y_1_3_5
    let p_x = _mm256_mul_ps(x, y_1_3_5);

    // Evaluando rsqrt(p(x)^2 + 1)
    let p_x_sq = _mm256_mul_ps(p_x, p_x);
    let radicand = _mm256_add_ps(p_x_sq, one);

    // Instrução HW nativa de rsqrt fornece ~11-14 bits de precisão na base arquitetural
    let mut rr = _mm256_rsqrt_ps(radicand);

    // Refinamento de Newton-Raphson: eleva a precisão das casas decimais (1 iter)
    let three = _mm256_set1_ps(3.0);
    let half = _mm256_set1_ps(0.5);

    let rr_sq = _mm256_mul_ps(rr, rr);
    let rad_rr_sq = _mm256_mul_ps(radicand, rr_sq);

    // diff = 3.0 - (radicand * rr^2)
    let diff = _mm256_sub_ps(three, rad_rr_sq);

    // rr_half = rr * 0.5
    let rr_half = _mm256_mul_ps(rr, half);

    // rr_new = (rr * 0.5) * (3.0 - radicand * rr^2)
    rr = _mm256_mul_ps(rr_half, diff);

    // Retorna tangete final iterada como tanh(x) ~ p(x) * rsqrt(p(x)^2 + 1)
    _mm256_mul_ps(p_x, rr)
}

/// Aplica aproximação vetorial de `sigmoid(x)` através da identidade logarítimica da tanh.
///
/// Baseia-se matematicamente em `sigmoid(x) = 0.5 * (1.0 + tanh(0.5 * x))`.
///
/// # Safety
/// Requer suporte nativo `avx2` e `fma` instanciado pelo chamador dinâmico.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid(x: __m256) -> __m256 {
    unsafe {
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);

        // x * 0.5
        let x_half = _mm256_mul_ps(x, half);

        // tanh(x * 0.5)
        let th = simd_tanh(x_half);

        // 1.0 + tanh(x * 0.5)
        let t_plus_one = _mm256_add_ps(th, one);

        // 0.5 * (1.0 + tanh(x * 0.5))
        _mm256_mul_ps(t_plus_one, half)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_fastmath_tanh_mse() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let input: [f32; 8] = [-4.0, -2.5, -0.75, 0.0, 0.8, 1.2, 3.5, 6.0];
            let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
            let result_vector = unsafe { simd_tanh(vector) };

            let mut result = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

            for i in 0..8 {
                let expected = input[i].tanh();
                let actual = result[i];
                let error = (expected - actual).abs();

                // Tolerância estrita (threshold) ajustada para polinomios < 5 na base original
                assert!(
                    error < 5e-3,
                    "Falha matemática MS/E. Entrada: {}. Esperado: {}, Obtido: {}, Delta (Erro): {}",
                    input[i],
                    expected,
                    actual,
                    error
                );
            }
        }
    }

    #[test]
    fn test_simd_fastmath_sigmoid_mse() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let input: [f32; 8] = [-5.0, -1.0, -0.33, 0.0, 0.22, 1.1, 2.8, 4.0];
            let vector = unsafe { _mm256_loadu_ps(input.as_ptr()) };
            let result_vector = unsafe { simd_sigmoid(vector) };

            let mut result = [0.0f32; 8];
            unsafe { _mm256_storeu_ps(result.as_mut_ptr(), result_vector) };

            let std_sigmoid = |val: f32| -> f32 { 0.5 * (1.0 + (val * 0.5).tanh()) };

            for i in 0..8 {
                let expected = std_sigmoid(input[i]);
                let actual = result[i];
                let error = (expected - actual).abs();

                assert!(
                    error < 5e-3,
                    "Falha ao validar FastMath Sigmoid em {}. Esperado: {}, Obtido: {}, Delta (Erro): {}",
                    input[i],
                    expected,
                    actual,
                    error
                );
            }
        }
    }
}
