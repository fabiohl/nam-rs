// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo para reescalonamento de ganho (Gain Staging) alavancando SIMD.
//!
//! Exerce processamento linear (multiplicação de pacote) de forma atômica
//! e estrita (zero-allocation) para aplicação de parâmetros de predição
//! convolucionais informados pela CLI/PipeWire (ex: `input_level_dbu`).

use core::arch::x86_64::*;

/// Aplica o multiplicador linear de ganho bruto sobre o buffer.
/// Em targets `x86_64`, direciona a instrução intrínseca `AVX2 _mm256_mul_ps`.
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

/// Detecta clipping estéreo via AVX2 — retorna `true` se qualquer amostra
/// em `left` ou `right` possuir `|x| > 1.0`.
///
/// ## Motivação: Substituição do loop escalar
///
/// O loop escalar original itera até 128 comparações (L+R × 64 samples) com
/// `f32::abs()` + branch por sample. Esta implementação vetorial processa
/// **8 samples por iteração** via `_mm256_andnot_ps` (abs) + `_mm256_cmp_ps`
/// (comparação > 1.0) com acumulação OR — sem branches no loop interno.
/// Para blocos de 64 samples: **8 iterações vetoriais** vs até 128 escalares.
pub fn detect_clipping_stereo_simd(left: &[f32], right: &[f32]) -> bool {
    let n = core::cmp::min(left.len(), right.len());

    unsafe {
        // Define o limite de clipping (1.0). Se |x| > 1.0, há clipping.
        let limit = _mm256_set1_ps(1.0);

        // Máscara de sinal: Representa -0.0f32 em binário (apenas o bit 31 setado).
        // É usada para calcular o valor absoluto via bitwise AND-NOT.
        let sign_mask = _mm256_set1_ps(-0.0f32);

        // Acumulador de clipping: Inicializado com zeros.
        let mut any_clip = _mm256_setzero_ps();
        let mut i = 0;

        // Loop principal: Processa 8 amostras estéreo (16 floats) por iteração.
        while i + 8 <= n {
            // Load: Carrega 8 amostras de cada canal.
            let vl = _mm256_loadu_ps(left.as_ptr().add(i));
            let vr = _mm256_loadu_ps(right.as_ptr().add(i));

            // Absolute Value: Calcula |x| limpando o bit de sinal.
            // andnot(mask, x) faz (~mask & x). Como sign_mask tem apenas o bit 31,
            // isso zera o bit de sinal de cada float no registro.
            let abs_l = _mm256_andnot_ps(sign_mask, vl);
            let abs_r = _mm256_andnot_ps(sign_mask, vr);

            // Compare: Verifica se |x| > 1.0. Retorna uma máscara de bits onde
            // cada lane que satisfaz a condição fica com todos os bits em 1 (NaN binário).
            let cmp_l = _mm256_cmp_ps(abs_l, limit, _CMP_GT_OQ);
            let cmp_r = _mm256_cmp_ps(abs_r, limit, _CMP_GT_OQ);

            // Accumulate: Combina os resultados dos canais L e R com o acumulador.
            // Se qualquer lane em qualquer iteração anterior ou atual detectou clipping,
            // 'any_clip' terá bits setados.
            any_clip = _mm256_or_ps(any_clip, _mm256_or_ps(cmp_l, cmp_r));

            // Early-exit: Como clipping é um evento que deve interromper o processamento
            // ou acionar alertas, verificamos a cada iteração.
            // movemask_ps extrai o bit de sinal de cada um dos 8 lanes e os coloca em um int.
            if _mm256_movemask_ps(any_clip) != 0 {
                return true;
            }

            i += 8;
        }

        // Loop tail escalar: Processa amostras restantes caso o buffer não seja múltiplo de 8.
        while i < n {
            if left[i].abs() > 1.0 || right[i].abs() > 1.0 {
                return true;
            }
            i += 1;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Testa a aplicação básica de ganho.
    /// Verifica bypass (1.0), amplificação (2.0) e comportamento com zeros.
    #[test]
    fn test_apply_gain_simd() {
        let mut buffer = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let original = buffer;

        // Neutro: Ganho 1.0 não deve alterar o buffer.
        apply_gain_simd(&mut buffer, 1.0);
        assert_eq!(buffer, original);

        // Ganho = 2.0: Multiplica todas as amostras por 2.
        apply_gain_simd(&mut buffer, 2.0);
        assert_eq!(
            buffer,
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
        );

        // Ganho em Zeros: Multiplicar zero por qualquer coisa deve resultar em zero.
        let mut zeros = [0.0; 25];
        apply_gain_simd(&mut zeros, 5.5);
        for &z in &zeros {
            assert_eq!(z, 0.0);
        }
    }

    /// Valida o cálculo combinado de gain staging (user + model metadata).
    /// Simula a lógica exata de `update_gain_multipliers` de pw_host.rs.
    /// Garante que a conversão dB -> Linear e a aplicação estão corretas.
    #[test]
    fn test_combined_gain_staging() {
        let lut = crate::math::fastmath::get_gain_lut();
        let user_input_db: f32 = 6.0;
        let model_input_adj_db: f32 = -3.0;

        // Na nova arquitetura, a Main Thread envia multiplicadores lineares.
        let user_input_mult = lut.db_to_linear(user_input_db);
        let model_input_adj_mult = lut.db_to_linear(model_input_adj_db);

        // A thread RT apenas os multiplica (zero powf/LUT no hot-path).
        let gain_linear = user_input_mult * model_input_adj_mult;

        let mut buffer = [1.0f32; 16];
        apply_gain_simd(&mut buffer, gain_linear);

        let expected = lut.db_to_linear(3.0);
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
        let lut = crate::math::fastmath::get_gain_lut();
        // -60 dB → gain ≈ 0.001
        let gain_neg60 = lut.db_to_linear(-60.0);
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
        let gain_pos24 = lut.db_to_linear(24.0);
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

    /// Verifica que `apply_gain_simd` com gain=1.0 é true-bypass bitwise (sem alterar nenhum bit).
    #[test]
    fn test_gain_true_bypass() {
        // Cenário 1: Buffer alinhado (múltiplo de 8)
        let original_aligned: [f32; 16] = [
            0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6,
        ];
        let mut buf_aligned = original_aligned;
        apply_gain_simd(&mut buf_aligned, 1.0);
        assert_eq!(
            buf_aligned, original_aligned,
            "Gain 1.0 deve preservar buffer alinhado bitwise (true-bypass)"
        );

        // Cenário 2: Buffer não-alinhado (tail escalar)
        let original_tail: [f32; 13] = [
            0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10, 0.11, -0.12, 0.13,
        ];
        let mut buf_tail = original_tail;
        apply_gain_simd(&mut buf_tail, 1.0);
        assert_eq!(
            buf_tail, original_tail,
            "Gain 1.0 deve preservar buffer não-alinhado bitwise (true-bypass tail)"
        );

        // Cenário 3: Gain muito próximo de 1.0 (dentro do epsilon de bypass)
        let mut buf_eps = original_aligned;
        apply_gain_simd(&mut buf_eps, 1.0 + 1e-7);
        assert_eq!(
            buf_eps, original_aligned,
            "Gain ≈1.0 (dentro de 1e-6) deve acionar fast-path bypass"
        );
    }

    /// Roundtrip +6dB → -6dB deve preservar o sinal original (MSE < 1e-10).
    #[test]
    fn test_gain_roundtrip_6db() {
        let original: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
            .collect();

        let lut = crate::math::fastmath::get_gain_lut();
        let gain_up = lut.db_to_linear(6.0); // +6 dB
        let gain_down = lut.db_to_linear(-6.0); // -6 dB

        let mut buffer = original.clone();
        apply_gain_simd(&mut buffer, gain_up);
        apply_gain_simd(&mut buffer, gain_down);

        let mse: f64 = original
            .iter()
            .zip(buffer.iter())
            .map(|(a, b)| {
                let d = (*a as f64) - (*b as f64);
                d * d
            })
            .sum::<f64>()
            / (original.len() as f64);

        assert!(
            mse < 1e-10,
            "Roundtrip +6dB/-6dB MSE={mse:.2e} excede 1e-10"
        );
    }
}
