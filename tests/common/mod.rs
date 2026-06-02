// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared helper module for NAM-rs integration tests.
//!
//! Centralizes signal generation functions, error metrics, and DSP validation
//! to avoid duplication across integration test files.

#![allow(dead_code)]

pub mod wav;

use std::fs;
use std::path::{Path, PathBuf};

use nam_rs::models::NamModel;

// =============================================================================
// Constants
// =============================================================================

/// Number of samples for golden vector and self-consistency tests.
pub const GOLDEN_NUM_SAMPLES: usize = 2048;

/// Block size for processing in numerical validation tests.
pub const GOLDEN_BLOCK_SIZE: usize = 64;

/// Default block size for computational stability tests.
pub const TEST_BLOCK_SIZE: usize = 64;

/// Number of blocks for stability tests (~5.4 seconds at 48kHz).
pub const TEST_NUM_BLOCKS: usize = 4096;

/// Default sample rate used in golden vectors (48 kHz).
pub const STRESS_SAMPLE_RATE: u32 = 48000;

// =============================================================================
// Signal Generation
// =============================================================================

/// Generates the deterministic multi-component stress signal (2048 samples @ 48 kHz).
///
/// Components:
/// - Low-E guitar harmonics (82/165/330/659 Hz)
/// - Linear chirp 220 Hz → 3520 Hz
/// - Transient impulse (+0.9) at 25%
/// - Attack–sustain–release envelope with fade-to-silence
///
/// Fully deterministic: bit-for-bit identical results in Python and Rust.
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

            // Low-E guitar harmonics (82.41 Hz)
            let guitar = 0.40 * (2.0 * std::f64::consts::PI * 82.41 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 164.81 * t).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 329.63 * t).sin()
                + 0.08 * (2.0 * std::f64::consts::PI * 659.25 * t).sin();

            // Linear chirp 220 Hz → 3520 Hz
            let f0: f64 = 220.0;
            let f1: f64 = 3520.0;
            let chirp_phase =
                2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * t_total));
            let chirp = 0.30 * chirp_phase.sin();

            // Transient impulse at 25%
            let impulse = if i == n / 4 { 0.9 } else { 0.0 };

            let sample = env * (guitar + chirp) + impulse;
            sample.clamp(-1.0, 1.0) as f32
        })
        .collect()
}

/// Generates a deterministic 440 Hz sine wave signal at 48 kHz (legacy).
///
/// Kept for backward compatibility with self-consistency tests,
/// static/dynamic parity, and zero-alloc — these do not depend on the stress signal.
pub fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}

// =============================================================================
// Error Metrics (f64 arithmetic to preserve precision)
// =============================================================================

/// Computes the Mean Squared Error (MSE) between two sample vectors.
///
/// Uses `f64` arithmetic internally to avoid precision loss in the accumulator.
pub fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors of different sizes for MSE");
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

/// Computes the Max Absolute Error (MAE / L∞) between two vectors.
pub fn compute_max_abs_error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vectors of different sizes for MaxAbsError"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

// =============================================================================
// DSP Validation — 5 metrics in single-pass
// =============================================================================

/// Validates DSP fidelity in a single pass, computing MSE, MAE, SNR, PSNR, and
/// equivalent bits simultaneously in a single iteration over the buffer.
///
/// The 5 metrics derive from the same accumulators (`signal_power`, `noise_power`,
/// `max_abs_diff`, `peak_ref`) — zero additional overhead.
///
/// # Parameters
/// - `reference` — reference output vector (NeuralAmpModelerCore C++)
/// - `test`      — Rust engine output vector to be validated
/// - `mse_limit` — maximum allowed MSE threshold
/// - `min_snr_db` — minimum SNR in dB that must be achieved
/// - `label`     — label for identification in diagnostic messages
///
/// # Output format
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
        "[{label}] Vectors of different sizes for report_dsp_fidelity"
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
        "[{label}] MSE={mse:.6e} exceeds threshold {mse_limit:.1e} (MAE={mae:.6e}, SNR={snr:.1} dB)"
    );
    assert!(
        snr >= min_snr_db,
        "[{label}] SNR={snr:.1} dB below minimum {min_snr_db:.1} dB (MSE={mse:.6e}, MAE={mae:.6e})"
    );
}

// =============================================================================
// I/O Helpers
// =============================================================================

/// Reads a `.golden.bin` file in the specified binary format.
///
/// Returns `Some((input, expected_output))` or `None` if the file does not exist
/// or is malformed.
///
/// ## Format
/// ```text
/// [u32 num_samples LE]
/// [f32×N input samples LE]
/// [f32×N expected output LE]
/// ```
pub fn read_golden_bin(path: &Path) -> Option<(Vec<f32>, Vec<f32>)> {
    let data = fs::read(path).ok()?;

    // Minimum: 4 bytes (u32) + at least 4 bytes of input + 4 bytes of output
    if data.len() < 12 {
        eprintln!(
            "WARN: golden file {path:?} too small ({} bytes)",
            data.len()
        );
        return None;
    }

    let num_samples = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    let expected_size = 4 + num_samples * 4 * 2; // u32 + N*f32 input + N*f32 output
    if data.len() < expected_size {
        eprintln!(
            "WARN: golden {path:?} declares {num_samples} samples but has {} bytes (expected {expected_size})",
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

/// Resolves the path to a test model in `tests/fixtures/models/`.
pub fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

/// Processes an input block through the model in chunks of `block_size`.
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
