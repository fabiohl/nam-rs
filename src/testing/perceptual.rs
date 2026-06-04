// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Perceptual metrics for audio quality evaluation.
//!
//! Includes ESR (Error-to-Signal Ratio), LUFS (ITU-R BS.1770-4 simplified),
//! and baseline constants from published A2/Tone3000 data.

#![allow(dead_code)]

use rustfft::{FftPlanner, num_complex::Complex};

// =============================================================================
// Published Baselines (t3k-mushra / A2Esr.tsx:19-38)
// =============================================================================

/// A1-Standard median ESR baseline from t3k-mushra/A2Esr.tsx
pub const A2ESR_A1_STANDARD_MEDIAN: f64 = 0.00623;
/// A1-Standard Q1 ESR baseline
pub const A2ESR_A1_STANDARD_Q1: f64 = 0.00218;
/// A1-Standard Q3 ESR baseline
pub const A2ESR_A1_STANDARD_Q3: f64 = 0.01571;
/// A2-Full median ESR baseline from t3k-mushra/A2Esr.tsx
pub const A2ESR_A2_FULL_MEDIAN: f64 = 0.00334;
/// A2-Full Q1 ESR baseline
pub const A2ESR_A2_FULL_Q1: f64 = 0.00114;
/// A2-Full Q3 ESR baseline
pub const A2ESR_A2_FULL_Q3: f64 = 0.00913;
/// Conservative ESR gate for nam-rs vs C++ implementation parity (no training error).
pub const NAM_RS_CPP_PARITY_ESR_MAX: f64 = 1e-3;

// =============================================================================
// Error-to-Signal Ratio (ESR)
// =============================================================================

/// Computes the Error-to-Signal Ratio (linear scale).
///
/// `ESR = Σ(r - t)² / Σ r²`
///
/// Returns `f64::INFINITY` if the reference signal has zero energy.
pub fn compute_esr(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        test.len(),
        "compute_esr: vectors must have same length"
    );
    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let d = r as f64 - t as f64;
        signal_power += (r as f64) * (r as f64);
        noise_power += d * d;
    }
    if signal_power <= f64::EPSILON {
        if noise_power <= f64::EPSILON {
            return 0.0;
        }
        return f64::INFINITY;
    }
    noise_power / signal_power
}

/// Converts linear ESR to dB: `10 * log10(esr)`.
pub fn esr_to_db(esr: f64) -> f64 {
    if esr <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        10.0 * esr.log10()
    }
}

/// Computes Signal-to-Noise Ratio (SNR) in dB between reference and test.
pub fn compute_snr_db(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        test.len(),
        "compute_snr_db: vectors must have same length"
    );
    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let d = r as f64 - t as f64;
        signal_power += (r as f64) * (r as f64);
        noise_power += d * d;
    }
    if noise_power <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

// =============================================================================
// MR-STFT — Multi-Resolution Short-Time Fourier Transform Loss
// =============================================================================

/// Window sizes (samples) for multi-resolution STFT analysis.
pub const MRSTFT_WINDOW_SIZES: [usize; 3] = [256, 1024, 4096];

/// Recommended weights for each window size from t3k-mushra golden calibration.
pub const MRSTFT_WEIGHTS: [f64; 3] = [0.1, 0.3, 0.5];

/// Computes the Multi-Resolution STFT loss between reference and test signals.
///
/// For each window size in `[256, 1024, 4096]` with hop = window/4:
/// 1. Applies a Hann window
/// 2. Computes STFT via `rustfft::FftPlanner`
/// 3. Calculates L1 and L2 of log-magnitude differences per frame
/// 4. Averages frame losses and weights by window size
///
/// ```text
/// MR-STFT = Σ_w weight[w] · mean_frame( L1_sc + L2_sc )
/// where:
///   L1_sc = (1/F) Σ_f |ln|X_ref[f]| - ln|X_test[f]||
///   L2_sc = sqrt( (1/F) Σ_f (ln|X_ref[f]| - ln|X_test[f]|)² )
/// ```
pub fn compute_mr_stft(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        test.len(),
        "compute_mr_stft: vectors must have same length"
    );

    if reference.is_empty() {
        return 0.0;
    }

    let eps = 1e-8f64;
    let mut planner = FftPlanner::<f64>::new();
    let mut total_loss = 0.0f64;

    for (&ws, &weight) in MRSTFT_WINDOW_SIZES.iter().zip(MRSTFT_WEIGHTS.iter()) {
        let hop = ws / 4;
        if ws > reference.len() {
            continue;
        }

        let fft = planner.plan_fft_forward(ws);

        let window: Vec<f64> = (0..ws)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (ws - 1) as f64).cos()))
            .collect();

        let num_frames = (reference.len() - ws) / hop + 1;
        if num_frames == 0 {
            continue;
        }

        let num_bins = ws / 2 + 1;

        // Reusable FFT scratch buffers
        let mut buf_ref = vec![Complex::new(0.0, 0.0); ws];
        let mut buf_test = vec![Complex::new(0.0, 0.0); ws];
        let mut mag_ref = vec![0.0f64; num_bins];
        let mut mag_test = vec![0.0f64; num_bins];

        let mut window_loss_sum = 0.0f64;

        for frame in 0..num_frames {
            let offset = frame * hop;

            // Fill FFT buffers with windowed samples
            for i in 0..ws {
                let val_ref = reference[offset + i] as f64 * window[i];
                let val_test = test[offset + i] as f64 * window[i];
                buf_ref[i] = Complex::new(val_ref, 0.0);
                buf_test[i] = Complex::new(val_test, 0.0);
            }

            fft.process(&mut buf_ref);
            fft.process(&mut buf_test);

            for i in 0..num_bins {
                mag_ref[i] = (buf_ref[i].re * buf_ref[i].re + buf_ref[i].im * buf_ref[i].im)
                    .sqrt()
                    .max(eps)
                    .ln();
                mag_test[i] = (buf_test[i].re * buf_test[i].re + buf_test[i].im * buf_test[i].im)
                    .sqrt()
                    .max(eps)
                    .ln();
            }

            let mut l1 = 0.0f64;
            let mut l2_sq = 0.0f64;
            for i in 0..num_bins {
                let diff = (mag_ref[i] - mag_test[i]).abs();
                l1 += diff;
                l2_sq += diff * diff;
            }

            let l1_sc = l1 / num_bins as f64;
            let l2_sc = (l2_sq / num_bins as f64).sqrt();
            window_loss_sum += l1_sc + l2_sc;
        }

        let window_loss = window_loss_sum / num_frames as f64;
        total_loss += weight * window_loss;
    }

    total_loss
}

// =============================================================================
// LUFS — ITU-R BS.1770-4 simplified (K-weighting + gating)
// =============================================================================

/// Simplified LUFS (Loudness Units Full Scale) computation per ITU-R BS.1770-4.
///
/// This is a simplified approximation: applies K-weighting (pre-filter + high-shelf)
/// and absolute gating at -70 LUFS below ungated measurement.
/// Full BS.1770-4 uses -70 LUFS relative gating in a second pass; we use a single-pass
/// absolute gate for diagnostic purposes.
pub fn compute_lufs(samples: &[f32], _sample_rate: u32) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }

    // K-weighting pre-filter: 2nd-order high-pass
    // H(z) = (1 - 2z⁻¹ + z⁻²) / (1 - 1.99004745483398z⁻¹ + 0.99007225036621z⁻²)
    let pre_filtered = apply_biquad(samples, 1.0, -2.0, 1.0, 1.99004745483398, -0.99007225036621);

    // K-weighting high-shelf: boosts > 1 kHz
    // H(z) = (1.53512485958697 - 2.69169618940638z⁻¹ + 1.19839281085285z⁻²)
    //      / (1.0 - 1.69065929318241z⁻¹ + 0.73248077421585z⁻²)
    let k_weighted = apply_biquad(
        &pre_filtered,
        1.53512485958697,
        -2.69169618940638,
        1.19839281085285,
        1.69065929318241,
        -0.73248077421585,
    );

    // Channel power (mono)
    let mean_square: f64 =
        k_weighted.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / k_weighted.len() as f64;

    if mean_square <= f64::EPSILON {
        return f64::NEG_INFINITY;
    }

    // Ungated loudness
    let ungated_lk = -0.691 + 10.0 * mean_square.log10();

    // Absolute gate: -70 LUFS below ungated
    let gate_threshold_lk = ungated_lk - 70.0;
    let gate_threshold_linear = 10.0f64.powf((gate_threshold_lk + 0.691) / 10.0);

    let gated_power: f64 = k_weighted
        .iter()
        .map(|&x| {
            let p = (x as f64).powi(2);
            if p > gate_threshold_linear { p } else { 0.0 }
        })
        .sum::<f64>()
        / k_weighted.len() as f64;

    if gated_power <= f64::EPSILON {
        return f64::NEG_INFINITY;
    }

    -0.691 + 10.0 * gated_power.log10()
}

/// Applies a biquad IIR filter.
/// Direct Form II Transposed biquad filter.
///
/// Implements `H(z) = (b0 + b1*z⁻¹ + b2*z⁻²) / (1 - a1*z⁻¹ - a2*z⁻²)`.
fn apply_biquad(samples: &[f32], b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    for &x in samples {
        let xf = x as f64;
        let y = b0 * xf + s1;
        s1 = b1 * xf + a1 * y + s2;
        s2 = b2 * xf + a2 * y;
        out.push(y as f32);
    }
    out
}

#[cfg(test)]
#[path = "perceptual_test.rs"]
mod perceptual_test;
