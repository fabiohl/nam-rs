// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Perceptual metrics for audio quality evaluation.
//!
//! Includes ESR (Error-to-Signal Ratio), LUFS (ITU-R BS.1770-4 simplified),
//! and baseline constants from published A2/Tone3000 data.
// Q1/Q3 constants and NAM_RS_CPP_PARITY_ESR_MAX are reference-only baselines; public API used by integration tests.

#![allow(dead_code)]

use crate::math::dsp::fft::FftPlanner;

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
/// A2-Lite median ESR baseline (preliminary — pending t3k-mushra publication).
/// A2-Lite shares the A2-Full architecture with rank-reduced weights; ESR is
/// expected to be in the same order of magnitude as A2-Full (≤ 0.005).
pub const A2ESR_A2_LITE_MEDIAN: f64 = 0.005;
/// A2-Lite Q1 ESR baseline (preliminary)
pub const A2ESR_A2_LITE_Q1: f64 = 0.0015;
/// A2-Lite Q3 ESR baseline (preliminary)
pub const A2ESR_A2_LITE_Q3: f64 = 0.012;
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
/// 2. Computes STFT via native `crate::math::dsp::fft::FftPlanner` (SoA)
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
    let mut total_loss = 0.0f64;

    for (&ws, &weight) in MRSTFT_WINDOW_SIZES.iter().zip(MRSTFT_WEIGHTS.iter()) {
        let hop = ws / 4;
        if ws > reference.len() {
            continue;
        }

        let fft = FftPlanner::<f64>::new(ws);

        let window: Vec<f64> = (0..ws)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (ws - 1) as f64).cos()))
            .collect();

        let num_frames = (reference.len() - ws) / hop + 1;
        if num_frames == 0 {
            continue;
        }

        let num_bins = ws / 2 + 1;

        // Reusable FFT scratch buffers (SoA)
        let mut buf_ref_re = vec![0.0f64; ws];
        let mut buf_ref_im = vec![0.0f64; ws];
        let mut buf_test_re = vec![0.0f64; ws];
        let mut buf_test_im = vec![0.0f64; ws];
        let mut mag_ref = vec![0.0f64; num_bins];
        let mut mag_test = vec![0.0f64; num_bins];

        let mut window_loss_sum = 0.0f64;

        for frame in 0..num_frames {
            let offset = frame * hop;

            // Fill FFT SoA buffers with windowed samples
            for i in 0..ws {
                buf_ref_re[i] = reference[offset + i] as f64 * window[i];
                buf_test_re[i] = test[offset + i] as f64 * window[i];
                buf_ref_im[i] = 0.0;
                buf_test_im[i] = 0.0;
            }

            fft.process(&mut buf_ref_re, &mut buf_ref_im);
            fft.process(&mut buf_test_re, &mut buf_test_im);

            for i in 0..num_bins {
                mag_ref[i] = (buf_ref_re[i] * buf_ref_re[i] + buf_ref_im[i] * buf_ref_im[i])
                    .sqrt()
                    .max(eps)
                    .ln();
                mag_test[i] = (buf_test_re[i] * buf_test_re[i] + buf_test_im[i] * buf_test_im[i])
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

// =============================================================================
// True-peak (dBTP) — ITU-R BS.1770-4 Annex 2 — 4× oversampling FIR
//
// RT-SAFETY DECISION (Tarefa 2.4, 2026-06-26):
//   - RT hot-path (src/dsp/pipeline/stages/output.rs): keeps sample-peak
//     detection for `RT_STATUS_HAS_CLIPPED`. True-peak with 48-tap FIR × 4×
//     oversampling adds ~48 MAC/sample (12 per phase × 4) — prohibitive in the
//     DSP callback where every μs matters.
//   - Off-RT QA/telemetry: functions below expose full BS.1770-4 dBTP via
//     `compute_true_peak_db()` and `find_true_peak_overs()`. The main-thread
//     telemetry loop (src/standalone/rt_setup/telemetry.rs:81) can optionally
//     run these on buffered audio for comprehensive inter-sample over detection.
//   - Bench number (linked to P-7): to be added in S3 hardware-validation sprint.
// =============================================================================

/// Oversampling factor for BS.1770-4 true-peak measurement.
const TP_OVERSAMPLE: usize = 4;

/// Number of taps in the BS.1770-4 Annex 2 FIR filter (full, 48 taps).
const TP_FIR_LEN: usize = 48;

/// Number of taps per polyphase sub-filter (48 / 4 = 12).
const TP_TAPS: usize = TP_FIR_LEN / TP_OVERSAMPLE;

/// BS.1770-4 Annex 2 polyphase sub-filter coefficients (4 phases × 12 taps each).
///
/// These are the polyphase sub-filters H_p(z) for 4× oversampling, given directly
/// by ITU-R BS.1770-4 Annex 2 Table (p. 17). Each phase sums to ~1.0 (unity DC gain).
///
/// Phase ordering follows the standard: H_0 through H_3 are convolution filters
/// that produce outputs y[4n], y[4n+1], y[4n+2], y[4n+3] respectively.
///
/// Symmetry properties: phase 3 = reversed(phase 0), phase 2 = reversed(phase 1).
#[rustfmt::skip]
const BS1770_PHASES: [[f64; TP_TAPS]; TP_OVERSAMPLE] = [
    // Phase 0: produces y[4n+0]
    [
         0.0017089843750,  0.0109863281250, -0.0196533203125,  0.0332031250000,
        -0.0594482421875,  0.1373291015625,  0.9721679687500, -0.1022949218750,
         0.0476074218750, -0.0266113281250,  0.0148925781250, -0.0083007812500,
    ],
    // Phase 1: produces y[4n+1]
    [
        -0.0291748046875,  0.0292968750000, -0.0517578125000,  0.0891113281250,
        -0.1665039062500,  0.4650878906250,  0.7797851562500, -0.2003173828125,
         0.1015625000000, -0.0582275390625,  0.0330810546875, -0.0189208984375,
    ],
    // Phase 2: produces y[4n+2]
    [
        -0.0189208984375,  0.0330810546875, -0.0582275390625,  0.1015625000000,
        -0.2003173828125,  0.7797851562500,  0.4650878906250, -0.1665039062500,
         0.0891113281250, -0.0517578125000,  0.0292968750000, -0.0291748046875,
    ],
    // Phase 3: produces y[4n+3]
    [
        -0.0083007812500,  0.0148925781250, -0.0266113281250,  0.0476074218750,
        -0.1022949218750,  0.9721679687500,  0.1373291015625, -0.0594482421875,
         0.0332031250000, -0.0196533203125,  0.0109863281250,  0.0017089843750,
    ],
];

/// 4× oversampling via BS.1770-4 Annex 2 polyphase FIR.
///
/// For input sample `x[n]`, produces 4 output samples:
/// ```text
/// y[4n+p] = x[n]*h_p[0] + x[n-1]*h_p[1] + ... + x[n-11]*h_p[11]   (p = 0,1,2,3)
/// ```
/// Uses a sliding window over the input (off-RT — allocates).
fn oversample_4x_bs1770(samples: &[f32]) -> Vec<f64> {
    let in_len = samples.len();
    if in_len == 0 {
        return Vec::new();
    }
    let out_len = in_len * TP_OVERSAMPLE;
    let mut out = vec![0.0f64; out_len];

    for n in 0..in_len {
        let base = n * TP_OVERSAMPLE;
        for p in 0..TP_OVERSAMPLE {
            let phase = &BS1770_PHASES[p];
            let mut acc = 0.0f64;
            for k in 0..TP_TAPS {
                if k > n {
                    break;
                }
                acc += (samples[n - k] as f64) * phase[k];
            }
            out[base + p] = acc;
        }
    }
    out
}

/// Computes the true-peak level in dBTP per ITU-R BS.1770-4 Annex 2.
///
/// Applies 4× oversampling via the standard 48-tap FIR polyphase filter,
/// then measures the absolute peak of the upsampled signal.
///
/// Returns `f64::NEG_INFINITY` for an empty or all-zero input.
pub fn compute_true_peak_db(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }

    let upsampled = oversample_4x_bs1770(samples);
    let peak_abs = upsampled.iter().fold(0.0f64, |max, &x| max.max(x.abs()));

    if peak_abs <= 1e-15 {
        f64::NEG_INFINITY
    } else {
        20.0 * peak_abs.log10()
    }
}

/// A detected inter-sample over.
#[derive(Debug, Clone, PartialEq)]
pub struct TruePeakOver {
    /// Sample index in the original (un-upsampled) signal.
    pub position: usize,
    /// True-peak level in dBTP at this position.
    pub dbtp: f64,
}

/// Finds all inter-sample overs (> 0 dBFS) via BS.1770-4 Annex 2 4× oversampling.
///
/// Scans the 4× upsampled signal and reports each region where `|y[m]| > 1.0`.
/// Consecutive overs within the same original-sample window are merged into a
/// single event with the maximum dBTP of that window.
pub fn find_true_peak_overs(samples: &[f32]) -> Vec<TruePeakOver> {
    let upsampled = oversample_4x_bs1770(samples);
    let mut overs = Vec::new();
    let len = upsampled.len();
    let mut i = 0;

    while i < len {
        if upsampled[i].abs() > 1.0 {
            let start_sample = i / TP_OVERSAMPLE;
            let mut peak = upsampled[i].abs();
            i += 1;
            while i < len && i / TP_OVERSAMPLE == start_sample {
                if upsampled[i].abs() > 1.0 {
                    peak = peak.max(upsampled[i].abs());
                }
                i += 1;
            }
            overs.push(TruePeakOver {
                position: start_sample,
                dbtp: 20.0 * peak.log10(),
            });
        } else {
            i += 1;
        }
    }

    overs
}

/// Returns the full BS.1770-4 Annex 2 4× oversampled signal.
///
/// Output length = `samples.len() * 4`. Useful for detailed analysis and plotting.
pub fn oversample_4x(samples: &[f32]) -> Vec<f64> {
    oversample_4x_bs1770(samples)
}

#[cfg(test)]
#[path = "perceptual_test.rs"]
mod perceptual_test;
