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
}
