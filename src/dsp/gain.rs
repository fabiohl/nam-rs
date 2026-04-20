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
        let limit = _mm256_set1_ps(1.0);
        // Máscara de sinal: bit 31 setado. andnot com ela produz abs(x).
        let sign_mask = _mm256_set1_ps(-0.0f32);
        let mut any_clip = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let vl = _mm256_loadu_ps(left.as_ptr().add(i));
            let vr = _mm256_loadu_ps(right.as_ptr().add(i));
            // abs(x) = x & ~sign_mask (limpa o bit de sinal)
            let abs_l = _mm256_andnot_ps(sign_mask, vl);
            let abs_r = _mm256_andnot_ps(sign_mask, vr);
            // Compara: abs(x) > 1.0? Resultado é máscara all-ones nos lanes que clipam.
            let cmp_l = _mm256_cmp_ps(abs_l, limit, _CMP_GT_OQ);
            let cmp_r = _mm256_cmp_ps(abs_r, limit, _CMP_GT_OQ);
            // Acumula: se qualquer lane clipou, any_clip terá bits setados.
            any_clip = _mm256_or_ps(any_clip, _mm256_or_ps(cmp_l, cmp_r));
            i += 8;
        }

        // Verifica se algum lane do acumulador tem bits setados.
        if _mm256_movemask_ps(any_clip) != 0 {
            return true;
        }

        // Loop tail escalar para remainder
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
        let threshold = _mm256_set1_ps(SILENCE_THRESHOLD);
        // Máscara de sinal: bit 31 setado. andnot com ela produz abs(x).
        let sign_mask = _mm256_set1_ps(-0.0f32);
        let mut any_above = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let vl = _mm256_loadu_ps(left.as_ptr().add(i));
            let vr = _mm256_loadu_ps(right.as_ptr().add(i));
            // abs(x) = x & ~sign_mask
            let abs_l = _mm256_andnot_ps(sign_mask, vl);
            let abs_r = _mm256_andnot_ps(sign_mask, vr);
            // abs(x) >= threshold? → máscara all-ones nos lanes acima.
            let cmp_l = _mm256_cmp_ps(abs_l, threshold, _CMP_GE_OQ);
            let cmp_r = _mm256_cmp_ps(abs_r, threshold, _CMP_GE_OQ);
            any_above = _mm256_or_ps(any_above, _mm256_or_ps(cmp_l, cmp_r));

            // Early-exit: se já encontrou sample acima do threshold, não é silêncio.
            if _mm256_movemask_ps(any_above) != 0 {
                return false;
            }

            i += 8;
        }

        // Tail escalar
        while i < n {
            if left[i].abs() >= SILENCE_THRESHOLD || right[i].abs() >= SILENCE_THRESHOLD {
                return false;
            }
            i += 1;
        }

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
        let zero = _mm256_setzero_ps();
        let mut any_not_mono = _mm256_setzero_ps();
        let mut i = 0;

        while i + 8 <= n {
            let vl = _mm256_loadu_ps(left.as_ptr().add(i));
            let vr = _mm256_loadu_ps(right.as_ptr().add(i));

            let cmp_nz = _mm256_cmp_ps(vr, zero, _CMP_NEQ_OQ);
            let cmp_ne = _mm256_cmp_ps(vr, vl, _CMP_NEQ_OQ);

            let result = _mm256_and_ps(cmp_nz, cmp_ne);
            any_not_mono = _mm256_or_ps(any_not_mono, result);

            if _mm256_movemask_ps(any_not_mono) != 0 {
                return false;
            }
            i += 8;
        }

        while i < n {
            if right[i] != 0.0 && right[i] != left[i] {
                return false;
            }
            i += 1;
        }

        true
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

    /// Verifica que `apply_gain_simd` com gain=1.0 é true-bypass bitwise (sem alterar nenhum bit).
    ///
    /// Testa dois cenários:
    /// - Buffer de 16 amostras (alinhado a AVX2 lanes de 8)
    /// - Buffer de 13 amostras (não-múltiplo de 8, exercita o fallback escalar tail)
    ///
    /// Em ambos os casos, o conteúdo deve ser preservado bit-a-bit.
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
    #[test]
    fn test_gain_roundtrip_6db() {
        // Gera sinal senoidal de referência
        let original: Vec<f32> = (0..256)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
            .collect();

        let gain_up = 10.0f32.powf(6.0 / 20.0); // +6 dB
        let gain_down = 10.0f32.powf(-6.0 / 20.0); // -6 dB

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
        // +96 dB → gain ≈ 63095.7
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
        let gain_neg96 = 10.0f32.powf(-96.0 / 20.0);
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

        // Buffer R!=L in sample 64 -> mono=false
        let l = vec![0.5; 128];
        let mut r = vec![0.5; 128];
        r[64] = 0.6;
        assert!(!is_buffer_mono_simd(&l, &r));

        // Buffer R=zeros except last sample (scalar tail) -> mono=false
        let l = vec![1.0; 15]; // length not multiple of 8
        let mut r = vec![0.0; 15];
        r[14] = 0.1;
        assert!(!is_buffer_mono_simd(&l, &r));
    }
}
