// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de helpers compartilhados entre os testes de integração do NAM-rs.
//!
//! Centraliza funções de geração de sinais, métricas de erro e validação DSP
//! para evitar duplicação entre os arquivos de teste de integração.

use std::fs;
use std::path::{Path, PathBuf};

use nam_rs::models::NamModel;

// =============================================================================
// Constantes
// =============================================================================

/// Número de amostras para testes de golden vectors e auto-consistência.
pub const GOLDEN_NUM_SAMPLES: usize = 512;

/// Tamanho de bloco para processamento nos testes de validação numérica.
pub const GOLDEN_BLOCK_SIZE: usize = 64;

/// Bloco padrão para testes de estabilidade computacional.
pub const TEST_BLOCK_SIZE: usize = 64;

/// Número de blocos para testes de estabilidade (~5.4 segundos a 48kHz).
pub const TEST_NUM_BLOCKS: usize = 4096;

// =============================================================================
// Geração de Sinais
// =============================================================================

/// Gera sinal senoidal determinístico de 440 Hz a 48 kHz.
///
/// O mesmo sinal é usado tanto pelo gerador C++ (`golden_gen.cpp`) quanto pelos
/// testes Rust, garantindo reprodutibilidade cross-platform.
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
// Validação DSP
// =============================================================================

/// Valida fidelidade DSP em single-pass, calculando MSE, MAE e SNR simultaneamente.
///
/// Esta função funde as 3 métricas numa única iteração sobre o buffer, evitando
/// múltiplas passagens sobre os dados (zero overhead adicional vs. validação prévia).
///
/// # Parâmetros
/// - `reference` — vetor de saída de referência (C++ NeuralAudio ou outra implementação)
/// - `test`      — vetor de saída do motor Rust a ser validado
/// - `mse_limit` — threshold máximo permitido de MSE (erro quadrático médio)
/// - `min_snr_db` — SNR mínimo em dB que deve ser atingido
/// - `label`     — rótulo para identificação nas mensagens de diagnóstico
///
/// # Comportamento
/// Imprime `[{label}] MSE=..., MaxAbsErr=..., SNR=... dB` para diagnóstico.
/// Falha com `assert!` se `mse >= mse_limit` **ou** `snr < min_snr_db`.
/// As duas métricas são assertadas independentemente para facilitar diagnóstico.
#[track_caller]
pub fn assert_dsp_fidelity(
    reference: &[f32],
    test: &[f32],
    mse_limit: f64,
    min_snr_db: f64,
    label: &str,
) {
    assert_eq!(
        reference.len(),
        test.len(),
        "[{label}] Vetores de tamanhos diferentes para assert_dsp_fidelity"
    );
    let n = reference.len() as f64;
    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let r64 = r as f64;
        let t64 = t as f64;
        let diff = r64 - t64;
        signal_power += r64 * r64;
        noise_power += diff * diff;
        sum_sq_diff += diff * diff;
        let abs_diff = diff.abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
    }
    let mse = sum_sq_diff / n;
    let snr = if noise_power <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    };
    println!(
        "[{label}] MSE={mse:.2e}, MaxAbsErr={max_abs_diff:.2e}, SNR={snr:.1} dB, amostras={}",
        reference.len()
    );
    assert!(
        mse < mse_limit,
        "[{label}] MSE={mse:.6e} excede limiar {mse_limit:.1e} (MaxAbsErr={max_abs_diff:.6e}, SNR={snr:.1} dB)"
    );
    assert!(
        snr >= min_snr_db,
        "[{label}] SNR={snr:.1} dB abaixo do mínimo {min_snr_db:.1} dB (MSE={mse:.6e}, MaxAbsErr={max_abs_diff:.6e})"
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
