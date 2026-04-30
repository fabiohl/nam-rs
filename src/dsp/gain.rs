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

/// Threshold de silêncio em amplitude linear: −80 dBFS ≈ 1e-4.
///
/// Abaixo desse nível, o sinal é imperceptível por qualquer aparelho humano ou
/// transdutor de áudio. Usamos um valor conservador (−80 dB, não −96 dB) para
/// garantir que ruído de quantização ou dither residual não mantenham o motor
/// neural ativo desnecessariamente.
const SILENCE_THRESHOLD: f32 = 1e-4;

/// Detecta silêncio estéreo via AVX2 — retorna `true` se **todas** as amostras
/// em ambos os canais possuírem `|x| < SILENCE_THRESHOLD`.
///
/// ## Motivação
///
/// Quando nenhuma fonte de áudio está conectada ao Virtual Sink, o PipeWire
/// entrega buffers zerados (ou com ruído de quantização ~−120 dBFS). Processar
/// esses buffers pela rede neural (WaveNet/LSTM) + resampler consome ~85% do
/// budget RT, disparando falsos alarmes de "Sobrecarga de CPU".
///
/// Esta função custa ~10 ns para 128 samples (vs ~500 µs da inferência neural)
/// e permite pular completamente o pipeline DSP pesado em silêncio.
///
/// ## Implementação
///
/// Processa 8 samples stereo por iteração via:
/// - `_mm256_andnot_ps` → abs(x) sem branch
/// - `_mm256_cmp_ps` → compara com threshold
/// - `_mm256_or_ps` → acumula qualquer sample acima do threshold
/// - `_mm256_movemask_ps` → colapsa 8 lanes em bitmask escalar
pub fn is_buffer_silent_stereo_simd(left: &[f32], right: &[f32]) -> bool {
    let n = core::cmp::min(left.len(), right.len());

    unsafe {
        // Define o threshold de silêncio em todas as lanes do registro.
        let threshold = _mm256_set1_ps(SILENCE_THRESHOLD);

        // Máscara de sinal para cálculo de valor absoluto via bitwise AND-NOT.
        let sign_mask = _mm256_set1_ps(-0.0f32);

        // Acumulador: Se qualquer amostra estiver acima do threshold, any_above terá bits setados.
        let mut any_above = _mm256_setzero_ps();
        let mut i = 0;

        // Loop principal: Processa 8 amostras estéreo por iteração.
        while i + 8 <= n {
            // Load: Carrega 8 amostras de cada canal.
            let vl = _mm256_loadu_ps(left.as_ptr().add(i));
            let vr = _mm256_loadu_ps(right.as_ptr().add(i));

            // Absolute Value: abs(x) = x & ~sign_mask (remove bit de sinal).
            let abs_l = _mm256_andnot_ps(sign_mask, vl);
            let abs_r = _mm256_andnot_ps(sign_mask, vr);

            // Compare: Verifica se |x| >= SILENCE_THRESHOLD.
            // Retorna máscara de bits (todos 1 se verdade, todos 0 se falso).
            let cmp_l = _mm256_cmp_ps(abs_l, threshold, _CMP_GE_OQ);
            let cmp_r = _mm256_cmp_ps(abs_r, threshold, _CMP_GE_OQ);

            // Accumulate: Se qualquer canal (L ou R) em qualquer lane tiver sinal, acumula.
            any_above = _mm256_or_ps(any_above, _mm256_or_ps(cmp_l, cmp_r));

            // Early-exit: Se já detectamos qualquer sinal acima do threshold,
            // o buffer não é silencioso. Retornamos 'false' imediatamente.
            if _mm256_movemask_ps(any_above) != 0 {
                return false;
            }

            i += 8;
        }

        // Tail Processing: Verifica as amostras restantes via loop escalar.
        while i < n {
            if left[i].abs() >= SILENCE_THRESHOLD || right[i].abs() >= SILENCE_THRESHOLD {
                return false;
            }
            i += 1;
        }

        // Se percorreu todo o buffer e nada superou o threshold, é silêncio.
        true
    }
}

/// Detecta se o canal direito (right) é puramente zero ou exatamente igual ao esquerdo (left),
/// permitindo bypass no canal direito (processamento mono) economizando 50% de CPU.
///
/// Implementação SIMD:
/// 1. `_mm256_loadu_ps` — carregar 8 samples de L e R
/// 2. `_mm256_cmp_ps(r, zero, _CMP_NEQ_OQ)` — R ≠ 0?
/// 3. `_mm256_cmp_ps(r, l, _CMP_NEQ_OQ)` — R ≠ L?
/// 4. `_mm256_and_ps(cmp_nz, cmp_ne)` — R ≠ 0 e R ≠ L?
/// 5. `_mm256_or_ps(accum, result)` — acumular
/// 6. `_mm256_movemask_ps` — early-exit se não for mono
pub fn is_buffer_mono_simd(left: &[f32], right: &[f32]) -> bool {
    let n = core::cmp::min(left.len(), right.len());

    unsafe {
        // Vetor constante de zeros para comparação.
        let zero = _mm256_setzero_ps();

        // Acumulador: se qualquer amostra quebrar a condição de mono, any_not_mono terá bits setados.
        let mut any_not_mono = _mm256_setzero_ps();
        let mut i = 0;

        // Loop principal: Processa 8 amostras estéreo por iteração.
        while i + 8 <= n {
            // Load: Carrega 8 amostras de cada canal.
            let vl = _mm256_loadu_ps(left.as_ptr().add(i));
            let vr = _mm256_loadu_ps(right.as_ptr().add(i));

            // Comparação 1: Canal direito é diferente de zero?
            let cmp_nz = _mm256_cmp_ps(vr, zero, _CMP_NEQ_OQ);

            // Comparação 2: Canal direito é diferente do canal esquerdo?
            let cmp_ne = _mm256_cmp_ps(vr, vl, _CMP_NEQ_OQ);

            // Lógica: Para ser considerado "não-mono", a amostra R deve ser diferente de 0
            // E diferente da amostra L correspondente.
            let result = _mm256_and_ps(cmp_nz, cmp_ne);

            // Accumulate: Combina com as detecções anteriores.
            any_not_mono = _mm256_or_ps(any_not_mono, result);

            // Early-exit: Se qualquer lane detectou uma amostra não-mono, retornamos false.
            if _mm256_movemask_ps(any_not_mono) != 0 {
                return false;
            }
            i += 8;
        }

        // Tail Processing: Verifica amostras restantes via loop escalar.
        while i < n {
            if right[i] != 0.0 && right[i] != left[i] {
                return false;
            }
            i += 1;
        }

        // Se percorreu tudo e R foi sempre 0 ou igual a L, o buffer é mono.
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::fastmath::{GAIN_MAX_DB, GAIN_MIN_DB};

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
    ///
    /// Testa três cenários:
    /// - Buffer de 16 amostras (alinhado a AVX2 lanes de 8)
    /// - Buffer de 13 amostras (não-múltiplo de 8, exercita o fallback escalar tail)
    /// - Gain muito próximo de 1.0 (dentro do epsilon)
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

    // =========================================================================
    // Testes de Gain Staging Roundtrip (ida-e-volta de ganho)
    // =========================================================================

    /// Roundtrip +6dB → -6dB deve preservar o sinal original (MSE < 1e-10).
    /// Garante que as operações SIMD não introduzem erros de precisão acumulativos.
    #[test]
    fn test_gain_roundtrip_6db() {
        // Gera sinal senoidal de referência
        let original: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
            .collect();

        let lut = crate::math::fastmath::get_gain_lut();
        let gain_up = lut.db_to_linear(6.0); // +6 dB
        let gain_down = lut.db_to_linear(-6.0); // -6 dB

        let mut buffer = original.clone();
        apply_gain_simd(&mut buffer, gain_up);
        apply_gain_simd(&mut buffer, gain_down);

        // MSE entre buffer processado e original
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
            "Roundtrip +6dB/-6dB MSE={mse:.2e} excede 1e-10 — possível acúmulo de erro float"
        );
    }

    /// Aplicar +96dB e -96dB sem gerar NaN/Inf (extremos de ganho em Float32).
    #[test]
    fn test_gain_extreme_values_96db() {
        let lut = crate::math::fastmath::get_gain_lut();
        // +96 dB → gain ≈ 63095.7
        // Nota: A LUT clampa em +24dB, mas para este teste de "estabilidade extrema"
        // usamos o valor manual (powf) para garantir que o kernel SIMD não explode com valores altos.
        let gain_pos96 = 10.0f32.powf(96.0 / 20.0);
        assert!(
            gain_pos96.is_finite(),
            "+96dB gain não é finito: {gain_pos96}"
        );

        let mut buffer = [0.5f32; 32];
        apply_gain_simd(&mut buffer, gain_pos96);
        for &s in &buffer {
            assert!(
                s.is_finite(),
                "Output com +96dB deve ser finito, obteve: {s}"
            );
        }

        // -96 dB → gain ≈ 1.585e-5
        let gain_neg96 = lut.db_to_linear(-96.0);
        assert!(
            gain_neg96.is_finite() && gain_neg96 > 0.0,
            "-96dB gain inválido: {gain_neg96}"
        );

        let mut buffer2 = [1.0f32; 32];
        apply_gain_simd(&mut buffer2, gain_neg96);
        for &s in &buffer2 {
            assert!(
                s.is_finite() && s >= 0.0,
                "Output com -96dB deve ser finito e >= 0, obteve: {s}"
            );
        }
    }

    /// Input com −0.0 (zero negativo IEEE 754) deve produzir output finito sem NaN.
    #[test]
    fn test_gain_negative_zero() {
        let mut buffer = [-0.0f32; 16];
        apply_gain_simd(&mut buffer, 2.5);
        for &s in &buffer {
            assert!(
                s.is_finite(),
                "Gain sobre -0.0 deve ser finito, obteve: {s}"
            );
            // -0.0 * 2.5 = -0.0 (IEEE 754), que é finito
        }
    }

    /// Valida a precisão da GainLUT em relação ao powf original.
    /// O critério de aceite exige erro absoluto de atenuação < 0.001 dB.
    #[test]
    fn test_gain_lut_precision() {
        let lut = crate::math::fastmath::get_gain_lut();

        // Varredura de -96 dB a +24 dB (range nominal da LUT).
        let mut db = -96.0;
        while db <= 24.0 {
            let expected = 10.0f32.powf(db / 20.0);
            let actual = lut.db_to_linear(db);

            // Calculamos a diferença em dB: error_db = |20 * log10(actual/expected)|
            let error_db = (20.0 * (actual / expected).log10()).abs();

            assert!(
                error_db < 0.001,
                "Erro de precisão na LUT excedeu 0.001 dB em {} dB. Erro: {:.6} dB",
                db,
                error_db
            );

            db += 0.1;
        }

        // Verifica clamping nos extremos
        assert_eq!(lut.db_to_linear(-120.0), lut.db_to_linear(GAIN_MIN_DB));
        assert_eq!(lut.db_to_linear(48.0), lut.db_to_linear(GAIN_MAX_DB));
    }

    // =========================================================================
    // Testes de Detecção de Silêncio (Silence Bypass)
    // =========================================================================

    /// Buffer de zeros deve ser detectado como silêncio.
    #[test]
    fn test_silence_zeros() {
        let left = [0.0f32; 128];
        let right = [0.0f32; 128];
        assert!(is_buffer_silent_stereo_simd(&left, &right));
    }

    /// Buffer com valores sub-threshold (−120 dBFS) deve ser silêncio.
    #[test]
    fn test_silence_sub_threshold() {
        let left = [1e-6_f32; 128]; // −120 dBFS
        let right = [1e-6_f32; 128];
        assert!(is_buffer_silent_stereo_simd(&left, &right));
    }

    /// Buffer com um sample acima do threshold deve NÃO ser silêncio.
    #[test]
    fn test_silence_single_loud_sample() {
        let mut left = [0.0f32; 128];
        let right = [0.0f32; 128];
        left[64] = 0.01; // ~−40 dBFS, bem acima do threshold
        assert!(!is_buffer_silent_stereo_simd(&left, &right));
    }

    /// Sample exatamente no threshold (1e-4) deve NÃO ser silêncio (>=).
    #[test]
    fn test_silence_at_threshold() {
        let left = [SILENCE_THRESHOLD; 128];
        let right = [0.0f32; 128];
        assert!(!is_buffer_silent_stereo_simd(&left, &right));
    }

    /// Buffer de −0.0 (zero negativo IEEE 754) deve ser silêncio.
    #[test]
    fn test_silence_negative_zero() {
        let left = [-0.0f32; 128];
        let right = [-0.0f32; 128];
        assert!(is_buffer_silent_stereo_simd(&left, &right));
    }

    /// Buffer não-múltiplo de 8 (exercita tail escalar).
    #[test]
    fn test_silence_non_aligned_buffer() {
        let left = [0.0f32; 13];
        let right = [0.0f32; 13];
        assert!(is_buffer_silent_stereo_simd(&left, &right));

        let mut left_loud = [0.0f32; 13];
        left_loud[12] = 0.5; // Último sample no tail
        assert!(!is_buffer_silent_stereo_simd(&left_loud, &right));
    }

    // =========================================================================
    // Testes de Detecção Mono (Mono Bypass)
    // =========================================================================

    /// Verifica a detecção de áudio mono.
    /// Um sinal é considerado mono se R for zero ou R for igual a L.
    #[test]
    fn test_is_buffer_mono_simd() {
        // Buffer R=zeros -> mono=true
        let l = vec![1.0; 128];
        let r = vec![0.0; 128];
        assert!(is_buffer_mono_simd(&l, &r));

        // Buffer R=L (bitwise) -> mono=true
        let l = vec![0.5; 128];
        let r = vec![0.5; 128];
        assert!(is_buffer_mono_simd(&l, &r));

        // Buffer R!=L em sample 64 -> mono=false
        let l = vec![0.5; 128];
        let mut r = vec![0.5; 128];
        r[64] = 0.6;
        assert!(!is_buffer_mono_simd(&l, &r));

        // Buffer R=zeros exceto último sample (tail escalar) -> mono=false
        let l = vec![1.0; 15]; // length not multiple of 8
        let mut r = vec![0.0; 15];
        r[14] = 0.1;
        assert!(!is_buffer_mono_simd(&l, &r));
    }
}
