// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Módulo de helpers compartilhados entre os testes de integração do NAM-rs.
//!
//! Centraliza funções de geração de sinais, métricas de erro e validação DSP
//! para evitar duplicação entre os arquivos de teste de integração.

#![allow(dead_code)]

pub mod wav;

use std::fs;
use std::path::{Path, PathBuf};

use nam_rs::models::NamModel;

// =============================================================================
// Constantes
// =============================================================================

/// Número de amostras para testes de golden vectors e auto-consistência.
pub const GOLDEN_NUM_SAMPLES: usize = 2048;

/// Tamanho de bloco para processamento nos testes de validação numérica.
pub const GOLDEN_BLOCK_SIZE: usize = 64;

/// Bloco padrão para testes de estabilidade computacional.
pub const TEST_BLOCK_SIZE: usize = 64;

/// Número de blocos para testes de estabilidade (~5.4 segundos a 48kHz).
pub const TEST_NUM_BLOCKS: usize = 4096;

/// Sample rate padrão usada nos golden vectors (48 kHz).
pub const STRESS_SAMPLE_RATE: u32 = 48000;

// =============================================================================
// Geração de Sinais
// =============================================================================

/// Gera o sinal de stress multi-componente determinístico (2048 amostras @ 48 kHz).
///
/// Componentes:
/// - Harmônicos de guitarra Low-E (82/165/330/659 Hz)
/// - Chirp linear 220 Hz → 3520 Hz
/// - Impulso transiente (+0.9) a 25%
/// - Envelope attack–sustain–release com fade-to-silence
///
/// Totalmente determinístico: mesmos resultados bit-a-bit em Python e Rust.
pub fn generate_stress_signal() -> Vec<f32> {
    let n = GOLDEN_NUM_SAMPLES;
    let sr = STRESS_SAMPLE_RATE as f64;
    let attack_end = (0.002 * sr) as usize; // 96 samples
    let release_beg = n - (0.005 * sr) as usize; // 1808 samples
    let t_total = n as f64 / sr;

    (0..n)
        .map(|i| {
            let t = i as f64 / sr;

            // Envelope (attack 2ms, sustain, release 5ms)
            let env = if i < attack_end {
                i as f64 / attack_end as f64
            } else if i >= release_beg {
                (n - 1 - i) as f64 / (n - release_beg) as f64
            } else {
                1.0
            };

            // Harmônicos de guitarra (Low-E: 82.41 Hz)
            let guitar = 0.40 * (2.0 * std::f64::consts::PI * 82.41 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 164.81 * t).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 329.63 * t).sin()
                + 0.08 * (2.0 * std::f64::consts::PI * 659.25 * t).sin();

            // Chirp linear 220 Hz → 3520 Hz
            let f0: f64 = 220.0;
            let f1: f64 = 3520.0;
            let chirp_phase =
                2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * t_total));
            let chirp = 0.30 * chirp_phase.sin();

            // Impulso transiente a 25%
            let impulse = if i == n / 4 { 0.9 } else { 0.0 };

            let sample = env * (guitar + chirp) + impulse;
            sample.clamp(-1.0, 1.0) as f32
        })
        .collect()
}

/// Gera sinal senoidal determinístico de 440 Hz a 48 kHz (legacy).
///
/// Mantido para retrocompatibilidade com testes de auto-consistência,
/// paridade estático/dinâmico e zero-alloc — estes não dependem do sinal de stress.
pub fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

// =============================================================================
// Métricas de Erro (aritmética f64 para preservar precisão)
// =============================================================================

/// Calcula o Mean Squared Error (MSE) entre dois vetores de amostras.
///
/// Usa aritmética `f64` internamente para evitar perda de precisão no acumulador.
pub fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vetores de tamanhos diferentes para MSE");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum();
    sum / (n as f64)
}

/// Calcula o Max Absolute Error (MAE / L∞) entre dois vetores.
pub fn compute_max_abs_error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vetores de tamanhos diferentes para MaxAbsError"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

// =============================================================================
// Validação DSP — 5 métricas em single-pass
// =============================================================================

/// Valida fidelidade DSP em single-pass, calculando MSE, MAE, SNR, PSNR e bits
/// equivalentes simultaneamente numa única iteração sobre o buffer.
///
/// As 5 métricas derivam dos mesmos acumuladores (`signal_power`, `noise_power`,
/// `max_abs_diff`, `peak_ref`) — zero overhead adicional.
///
/// # Parâmetros
/// - `reference` — vetor de saída de referência (NeuralAmpModelerCore C++)
/// - `test`      — vetor de saída do motor Rust a ser validado
/// - `mse_limit` — threshold máximo permitido de MSE
/// - `min_snr_db` — SNR mínimo em dB que deve ser atingido
/// - `label`     — rótulo para identificação nas mensagens de diagnóstico
///
/// # Formato de output
/// ```text
/// [NeuralAmpModelerCore × NAM-rs — label]
///   MSE     = 3.21e-02      (threshold < 5.0e-02)  ✓
///   MAE     = 2.84e-01
///   SNR     = 10.1 dB       (threshold ≥ 9.0 dB)   ✓
///   PSNR    = 14.9 dB
///   Bits    = 2.5 bits equiv.
///   Samples = 2048 @ 48 kHz (stress signal)
/// ```
#[track_caller]
pub fn report_dsp_fidelity(
    reference: &[f32],
    test: &[f32],
    mse_limit: f64,
    min_snr_db: f64,
    label: &str,
) {
    assert_eq!(
        reference.len(),
        test.len(),
        "[{label}] Vetores de tamanhos diferentes para report_dsp_fidelity"
    );
    let n = reference.len() as f64;
    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    let mut peak_ref = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let r64 = r as f64;
        let t64 = t as f64;
        let diff = r64 - t64;
        signal_power += r64 * r64;
        noise_power += diff * diff;
        let abs_diff = diff.abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        if r64.abs() > peak_ref {
            peak_ref = r64.abs();
        }
    }
    let mse = noise_power / n;
    let mae = max_abs_diff;
    let snr = if noise_power <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    };
    let psnr = if mse <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (peak_ref * peak_ref / mse).log10()
    };
    let signal_avg_power = signal_power / n;
    let bits = if mse <= f64::EPSILON || signal_avg_power <= f64::EPSILON {
        f64::INFINITY
    } else {
        -0.5 * (mse / signal_avg_power).log2()
    };

    println!();
    println!("[NeuralAmpModelerCore × NAM-rs — {label}]");
    println!(
        "  MSE     = {mse:.2e}      (threshold < {mse_limit:.1e})  {}",
        if mse < mse_limit { "✓" } else { "✗" }
    );
    println!("  MAE     = {mae:.2e}");
    if snr.is_finite() {
        println!(
            "  SNR     = {snr:.1} dB       (threshold ≥ {min_snr_db:.1} dB)   {}",
            if snr >= min_snr_db { "✓" } else { "✗" }
        );
    } else {
        println!("  SNR     = ∞ dB");
    }
    if psnr.is_finite() {
        println!("  PSNR    = {psnr:.1} dB");
    } else {
        println!("  PSNR    = ∞ dB");
    }
    if bits.is_finite() {
        println!("  Bits    = {bits:.2} bits equiv.");
    } else {
        println!("  Bits    = ∞ bits equiv.");
    }
    println!("  Samples = {} @ 48 kHz (stress signal)", reference.len());

    assert!(
        mse < mse_limit,
        "[{label}] MSE={mse:.6e} excede limiar {mse_limit:.1e} (MAE={mae:.6e}, SNR={snr:.1} dB)"
    );
    assert!(
        snr >= min_snr_db,
        "[{label}] SNR={snr:.1} dB abaixo do mínimo {min_snr_db:.1} dB (MSE={mse:.6e}, MAE={mae:.6e})"
    );
}

// =============================================================================
// Helpers de I/O
// =============================================================================

/// Lê um arquivo `.golden.bin` no formato binário especificado.
///
/// Retorna `Some((input, expected_output))` ou `None` se o arquivo não existir
/// ou estiver malformado.
///
/// ## Formato
/// ```text
/// [u32 num_samples LE]
/// [f32×N input samples LE]
/// [f32×N expected output LE]
/// ```
pub fn read_golden_bin(path: &Path) -> Option<(Vec<f32>, Vec<f32>)> {
    let data = fs::read(path).ok()?;

    // Mínimo: 4 bytes (u32) + pelo menos 4 bytes de input + 4 bytes de output
    if data.len() < 12 {
        eprintln!(
            "WARN: arquivo golden {path:?} muito pequeno ({} bytes)",
            data.len()
        );
        return None;
    }

    let num_samples = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    let expected_size = 4 + num_samples * 4 * 2; // u32 + N*f32 input + N*f32 output
    if data.len() < expected_size {
        eprintln!(
            "WARN: golden {path:?} declara {num_samples} amostras mas tem {} bytes (esperados {expected_size})",
            data.len()
        );
        return None;
    }

    let input_start = 4;
    let output_start = 4 + num_samples * 4;

    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let offset = input_start + i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    let output: Vec<f32> = (0..num_samples)
        .map(|i| {
            let offset = output_start + i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    Some((input, output))
}

/// Resolve o caminho para um modelo de teste em `tests/fixtures/models/`.
pub fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

/// Processa um bloco de input pelo modelo em chunks de `block_size`.
pub fn process_in_blocks(
    model: &mut nam_rs::models::DynamicModel,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}
