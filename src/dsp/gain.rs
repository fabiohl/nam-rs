// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo para reescalonamento de ganho (Gain Staging) alavancando SIMD.
//!
//! Exerce processamento linear (multiplicação de pacote) de forma atômica
//! e estrita (zero-allocation) para aplicação de parâmetros de predição
//! convolucionais informados pela CLI/PipeWire (ex: `input_level_dbu`).

use core::arch::x86_64::*;

/// Aplica o multiplicador linear de ganho bruto sobre o buffer.
/// Direciona a instrução intrínseca `AVX2 _mm256_mul_ps`.
/// Aborta rápido para matrizes sem alteração termodinâmica se `gain_linear` ~= 1.0.
pub fn apply_gain_simd(buffer: &mut [f32], gain_linear: f32) {
    // Fast path: bypass paramétrico.
    if (gain_linear - 1.0).abs() < 1e-6 {
        return;
    }
    unsafe { apply_gain_avx2(buffer, gain_linear) };
}

/// Dispara o pacote multiplicativo em arranjos de 8 Float32 por pulso com resíduo escalar.
///
/// Esta implementação utiliza instruções intrínsecas AVX2 para processar 8 amostras
/// simultaneamente em um único registro YMM de 256 bits, proporcionando um speedup
/// considerável em relação ao loop escalar.
unsafe fn apply_gain_avx2(buffer: &mut [f32], gain_linear: f32) {
    let mut i = 0;
    let len = buffer.len();

    unsafe {
        // Broadcast: Replica o valor de ganho (escalar) em todas as 8 posições (lanes)
        // do registro YMM de 256 bits.
        let ymm_gain = _mm256_set1_ps(gain_linear);

        // Loop principal: Processa blocos de 8 amostras (32 bytes) por vez.
        while i + 8 <= len {
            // Obtém o ponteiro para a posição atual no buffer.
            let ptr = buffer.as_mut_ptr().add(i);

            // Load: Carrega 8 valores f32 do buffer para um registro YMM.
            // 'loadu' permite carregamento não alinhado (unaligned), o que é mais seguro
            // para buffers genéricos, embora ligeiramente menos performático que 'loada'.
            let vals = _mm256_loadu_ps(ptr);

            // Multiply: Realiza a multiplicação ponto flutuante em paralelo nos 8 lanes.
            let processed = _mm256_mul_ps(vals, ymm_gain);

            // Store: Grava os 8 resultados processados de volta para o buffer na memória.
            _mm256_storeu_ps(ptr, processed);

            // Avança o índice em 8 amostras.
            i += 8;
        }

        // Tail Processing: Processa o "restolho" em queda escalar se buffer.len()
        // não for múltiplo de 8. Isso garante que amostras finais não fiquem sem ganho.
        while i < len {
            // Uso de get_unchecked_mut para evitar check de limites (bounds check)
            // já que o loop garante que i < len.
            *buffer.get_unchecked_mut(i) *= gain_linear;
            i += 1;
        }
    }
}

/// Aplica uma rampa linear de ganho sobre o buffer de forma vetorizada.
///
/// Direciona para `apply_ramp_avx2` se o hardware suportar, garantindo que
/// transições de fade-in/fade-out sejam processadas com eficiência máxima.
pub fn apply_ramp_simd(buffer: &mut [f32], start: f32, step: f32) {
    // Fast path: se o incremento for desprezível, aplica ganho constante.
    if step.abs() < 1e-9 {
        apply_gain_simd(buffer, start);
        return;
    }
    unsafe { apply_ramp_avx2(buffer, start, step) };
}

/// Implementação interna AVX2 para rampa linear.
///
/// Mantém um registro YMM com os multiplicadores de rampa e o incrementa
/// a cada iteração de 8 amostras.
unsafe fn apply_ramp_avx2(buffer: &mut [f32], start: f32, step: f32) {
    let mut i = 0;
    let len = buffer.len();

    unsafe {
        // Inicializa a rampa para as primeiras 8 posições: [s, s+1, s+2, s+3, s+4, s+5, s+6, s+7]
        let mut current_ramp = _mm256_set_ps(
            start + 7.0 * step,
            start + 6.0 * step,
            start + 5.0 * step,
            start + 4.0 * step,
            start + 3.0 * step,
            start + 2.0 * step,
            start + 1.0 * step,
            start,
        );
        // Incremento constante para cada salto de 8 amostras.
        let v_step_8 = _mm256_set1_ps(8.0 * step);

        while i + 8 <= len {
            let ptr = buffer.as_mut_ptr().add(i);
            let vals = _mm256_loadu_ps(ptr);

            // Multiplica amostras pela rampa atual.
            let processed = _mm256_mul_ps(vals, current_ramp);
            _mm256_storeu_ps(ptr, processed);

            // Avança a rampa para o próximo bloco de 8.
            current_ramp = _mm256_add_ps(current_ramp, v_step_8);
            i += 8;
        }

        // Tail escalar: processa o restante calculando o multiplicador exato.
        let mut m = start + (i as f32) * step;
        while i < len {
            *buffer.get_unchecked_mut(i) *= m;
            m += step;
            i += 1;
        }
    }
}

#[cfg(test)]
#[path = "gain_test.rs"]
mod gain_test;
