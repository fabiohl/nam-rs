// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo para reescalonamento de ganho (Gain Staging) alavancando SIMD.
//!
//! Exerce processamento linear (multiplicação de pacote) de forma atômica
//! e estrita (zero-allocation) para aplicação de parâmetros de predição
//! convolucionais informados pela CLI/PipeWire (ex: `input_level_dbu`).

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Aplica o multiplicador linear de ganho bruto sobre o buffer.
/// Em targets `x86_64`, direciona a instrução intrínseca `AVX2 _mm256_mul_ps`.
/// Aborta rápido para matrizes sem alteração termodinâmica se `gain_linear` ~= 1.0.
pub fn apply_gain_simd(buffer: &mut [f32], gain_linear: f32) {
    // Fast path: bypass paramétrico.
    if (gain_linear - 1.0).abs() < 1e-6 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // Safety: Aplicação central usa apenas _mm256_mul_ps, suportado em AVX2
        // Assumimos x86-64-v3 estrito (validado primariamente pelo startup em main.rs).
        unsafe { apply_gain_avx2(buffer, gain_linear) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for x in buffer.iter_mut() {
            *x *= gain_linear;
        }
    }
}

/// Dispara o pacote multiplicativo em arranjos de 8 Float32 por pulso com resíduo escalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_gain_avx2(buffer: &mut [f32], gain_linear: f32) {
    let mut i = 0;
    let len = buffer.len();

    unsafe {
        let ymm_gain = _mm256_set1_ps(gain_linear);

        while i + 8 <= len {
            let ptr = buffer.as_mut_ptr().add(i);
            let vals = _mm256_loadu_ps(ptr);
            let processed = _mm256_mul_ps(vals, ymm_gain);
            _mm256_storeu_ps(ptr, processed);
            i += 8;
        }

        // Processa o "restolho" em queda escalar se buffer.len() não for múltiplo de 8.
        while i < len {
            *buffer.get_unchecked_mut(i) *= gain_linear;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_gain_simd() {
        let mut buffer = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let original = buffer;

        // Neutro
        apply_gain_simd(&mut buffer, 1.0);
        assert_eq!(buffer, original);

        // Ganho = 2.0
        apply_gain_simd(&mut buffer, 2.0);
        assert_eq!(
            buffer,
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        );

        // Ganho em Zeros
        let mut zeros = [0.0; 25];
        apply_gain_simd(&mut zeros, 5.5);
        for &z in &zeros {
            assert_eq!(z, 0.0);
        }
    }

    /// Valida o cálculo combinado de gain staging (user + model metadata).
    /// Simula a lógica exata de `update_gain_multipliers` de pw_host.rs.
    #[test]
    fn test_combined_gain_staging() {
        // Cenário: usuário define +6dB de input gain,
        // modelo exige ajuste de -3dB (audioInputDBu - modelInputDBu).
        // Total = +3dB → multiplicador linear = 10^(3/20) ≈ 1.4125
        let user_input_db: f32 = 6.0;
        let model_input_adj_db: f32 = -3.0;
        let total_db = user_input_db + model_input_adj_db;
        let gain_linear = 10.0f32.powf(total_db / 20.0);

        let mut buffer = [1.0f32; 16];
        apply_gain_simd(&mut buffer, gain_linear);

        let expected = 10.0f32.powf(3.0 / 20.0);
        for &sample in &buffer {
            assert!(
                (sample - expected).abs() < 1e-5,
                "Esperado ~{expected:.5}, obteve {sample:.5}"
            );
        }
    }

    /// Verifica estabilidade com ganho extremo negativo (-60dB ≈ 0.001)
    /// e positivo (+24dB ≈ 15.85) sem underflow/overflow em Float32.
    #[test]
    fn test_extreme_gain_values() {
        // -60 dB → gain ≈ 0.001
        let gain_neg60 = 10.0f32.powf(-60.0 / 20.0);
        assert!(gain_neg60 > 0.0 && gain_neg60.is_finite());

        let mut buffer = [1.0f32; 20];
        apply_gain_simd(&mut buffer, gain_neg60);
        for &s in &buffer {
            assert!(
                s.is_finite() && s > 0.0 && s < 0.01,
                "Underflow em -60dB: {s}"
            );
        }

        // +24 dB → gain ≈ 15.85
        let gain_pos24 = 10.0f32.powf(24.0 / 20.0);
        assert!(gain_pos24 > 10.0 && gain_pos24.is_finite());

        let mut buffer2 = [0.5f32; 20];
        apply_gain_simd(&mut buffer2, gain_pos24);
        for &s in &buffer2 {
            assert!(
                s.is_finite() && s > 5.0 && s < 10.0,
                "Overflow em +24dB: {s}"
            );
        }
    }
}
