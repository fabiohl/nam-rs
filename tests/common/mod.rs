// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared helper module for NAM-rs integration tests.
//!
//! Centralizes signal generation functions, error metrics, and DSP validation
//! to avoid duplication across integration test files.

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod mushra_primitives;
pub mod perceptual;
pub mod wav;

use std::fs;
use std::path::{Path, PathBuf};

use nam_rs::models::NamModel;

// =============================================================================
// Re-exports from library (single source of truth)
// =============================================================================

pub use nam_rs::testing::stress::SUPPORTED_SAMPLE_RATES;
pub use nam_rs::testing::stress::generate_stress_signal_v1;
pub use nam_rs::testing::stress::generate_stress_signal_v2_default as generate_stress_signal_v2;

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
// Legacy alias (deprecated, kept for backward compatibility)
// =============================================================================

/// Generates the deterministic multi-component stress signal (2048 samples @ 48 kHz).
///
/// **Deprecated:** Use `generate_stress_signal_v1()` directly.
/// This alias exists only for backward compatibility with existing test code.
#[deprecated(since = "1.5.0", note = "Use `generate_stress_signal_v1()` directly")]
pub fn generate_stress_signal() -> Vec<f32> {
    generate_stress_signal_v1()
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
// DSP Validation — 7+ metrics in single-pass
// =============================================================================

/// Validates DSP fidelity in a single pass, computing MSE, MAE, SNR, PSNR,
/// equivalent bits, ESR, and LUFS simultaneously.
///
/// # Parameters
/// - `reference` — reference output vector (NeuralAmpModelerCore C++)
/// - `test`      — Rust engine output vector to be validated
/// - `mse_limit` — maximum allowed MSE threshold
/// - `min_snr_db` — minimum SNR in dB that must be achieved
/// - `max_esr`   — optional maximum ESR threshold for regression gating (default `None`)
/// - `label`     — label for identification in diagnostic messages
/// - `sample_rate` — sample rate in Hz (used for LUFS and anchor SNR diagnostics)
///
/// # Output format
/// ```text
/// [NeuralAmpModelerCore × NAM-rs — label]
///   MSE     = 3.21e-02      (threshold < 5.0e-02)  ✓
///   MAE     = 2.84e-01
///   SNR     = 10.1 dB       (threshold ≥ 9.0 dB)   ✓
///   PSNR    = 14.9 dB
///   Bits    = 2.5 bits equiv.
///   ESR     = 1.23e-05       (−49.1 dB)   [baseline A1-Std: 6.23e-03, A2-Full: 3.34e-03]
///   MR-STFT = 0.0042         (relative)
///   LUFS    = −23.4 LUFS
///   Samples = 2048 @ 48 kHz (stress signal)
/// ```
#[track_caller]
pub fn report_dsp_fidelity(
    reference: &[f32],
    test: &[f32],
    mse_limit: f64,
    min_snr_db: f64,
    max_esr: Option<f64>,
    label: &str,
    sample_rate: u32,
) {
    assert_eq!(
        reference.len(),
        test.len(),
        "[{label}] Vectors of different sizes for report_dsp_fidelity"
    );
    let n = reference.len() as f64;
    let sr = sample_rate;

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

    // ESR (linear + dB)
    let esr_linear = if signal_power <= f64::EPSILON {
        if noise_power <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        noise_power / signal_power
    };
    let esr_db = nam_rs::testing::perceptual::esr_to_db(esr_linear);

    // LUFS
    let lufs = nam_rs::testing::perceptual::compute_lufs(test, sr);

    // SNR(reference, anchor) sanity: compute SNR of test against a low-pass 3.5 kHz anchor
    let anchor_snr_db = {
        let anchor = nam_rs::testing::mushra::low_pass_1pole(reference, 3500.0, sr);
        nam_rs::testing::perceptual::compute_snr_db(reference, &anchor)
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
    if esr_linear.is_finite() {
        println!(
            "  ESR     = {esr_linear:.2e}       ({esr_db:.1} dB)   [baseline A1-Std: {a1std:.2e}, A2-Full: {a2full:.2e}]",
            a1std = nam_rs::testing::perceptual::A2ESR_A1_STANDARD_MEDIAN,
            a2full = nam_rs::testing::perceptual::A2ESR_A2_FULL_MEDIAN,
        );
    } else {
        println!("  ESR     = ∞  (identical)");
    }

    // MR-STFT
    let mr_stft = nam_rs::testing::perceptual::compute_mr_stft(reference, test);
    println!("  MR-STFT = {mr_stft:.4e}      (relative)");

    if lufs.is_finite() {
        println!("  LUFS    = {lufs:.1} LUFS");
        if anchor_snr_db.is_finite() {
            println!(
                "  SNR(anchor) = {anchor_snr_db:.1} dB  [threshold > {anchor_min:.1} dB]  {}",
                if anchor_snr_db > 15.0 { "✓" } else { "?" },
                anchor_min = 15.0,
            );
        }
    }
    println!("  Samples = {} @ {sr} Hz (stress signal)", reference.len());

    assert!(
        mse < mse_limit,
        "[{label}] MSE={mse:.6e} exceeds threshold {mse_limit:.1e} (MAE={mae:.6e}, SNR={snr:.1} dB)"
    );
    assert!(
        snr >= min_snr_db,
        "[{label}] SNR={snr:.1} dB below minimum {min_snr_db:.1} dB (MSE={mse:.6e}, MAE={mae:.6e})"
    );
    if let Some(limit) = max_esr {
        assert!(
            esr_linear < limit,
            "[{label}] ESR={esr_linear:.6e} exceeds threshold {limit:.1e} (ESR dB={esr_db:.1})"
        );
    }
}

// =============================================================================
// Adaptive Threshold Calibration
// =============================================================================

/// Computes model-adaptive MSE/SNR test thresholds based on topology.
///
/// More complex models (more channels, more layers) accumulate more
/// quantization noise, so they require more permissive thresholds.
/// Simpler models get tighter thresholds to catch regressions earlier.
///
/// Returns `(mse_limit, min_snr_db)`.
pub fn topology_thresholds(data: &nam_rs::loader::nam_json::NamModelData) -> (f64, f64) {
    match data.architecture.as_str() {
        "WaveNet" => {
            let channels = data
                .config
                .layers
                .first()
                .and_then(|l| l.channels)
                .unwrap_or(16);
            let total_dils: usize = data
                .config
                .layers
                .iter()
                .filter_map(|l| l.dilations.as_ref())
                .map(|d| d.len())
                .sum();
            let noise_factor = (channels + total_dils) as f64;
            let snr_db = (22.0 - noise_factor * 0.35).clamp(9.0, 16.0);
            let mse = 10.0_f64.powf(-snr_db / 10.0) * 0.3;
            (mse.clamp(1e-4, 5e-2), snr_db)
        }
        "LSTM" => {
            let num_layers = data.config.num_layers.unwrap_or(1);
            let hidden_size = data.config.hidden_size.unwrap_or(16);
            let complexity = (num_layers * hidden_size) as f64;
            let snr_db = (28.0 - complexity * 0.65).clamp(10.0, 24.0);
            let mse = 10.0_f64.powf(-snr_db / 10.0) * 0.3;
            (mse.clamp(1e-4, 5e-2), snr_db)
        }
        _ => (5e-2, 9.0),
    }
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

    if data.len() < 12 {
        eprintln!(
            "WARN: golden file {path:?} too small ({} bytes)",
            data.len()
        );
        return None;
    }

    let num_samples = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    let expected_size = 4 + num_samples * 4 * 2;
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
