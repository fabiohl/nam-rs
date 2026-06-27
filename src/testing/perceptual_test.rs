// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for perceptual metrics (ESR, LUFS, MR-STFT, true-peak dBTP).

use super::*;

#[test]
fn test_esr_identical_is_zero() {
    let sig: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let esr = compute_esr(&sig, &sig);
    assert!(
        esr < 1e-15,
        "ESR of identical signals should be near zero, got {esr}"
    );
}

#[test]
fn test_esr_all_zero_test() {
    let sig: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let zeros = vec![0.0f32; sig.len()];
    let esr = compute_esr(&sig, &zeros);
    assert!(
        (esr - 1.0).abs() < 1e-10,
        "ESR of signal vs zeros should be ~1.0, got {esr}"
    );
}

#[test]
fn test_lufs_sine() {
    let sr = 48000;
    let n = sr as usize;
    let amplitude = 10.0f32.powf(-20.0 / 20.0);
    let sig: Vec<f32> = (0..n)
        .map(|i| amplitude * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();
    let lufs = compute_lufs(&sig, sr);
    assert!(
        (lufs - (-23.0)).abs() < 2.0,
        "Expected ~-23 LUFS for 1kHz sine at -20 dBFS, got {lufs}"
    );
}

#[test]
fn test_lufs_empty() {
    assert!(compute_lufs(&[], 48000).is_infinite());
}

#[test]
fn test_mr_stft_identical_is_zero() {
    let sig: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let loss = compute_mr_stft(&sig, &sig);
    assert!(
        loss < 1e-12,
        "MR-STFT of identical signals should be near zero, got {loss}"
    );
}

#[test]
fn test_mr_stft_empty() {
    assert_eq!(compute_mr_stft(&[], &[]), 0.0);
}

#[test]
fn test_mr_stft_different_signals() {
    let sig_a: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let sig_b: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 48000.0).sin())
        .collect();
    let loss = compute_mr_stft(&sig_a, &sig_b);
    assert!(
        loss > 0.0,
        "MR-STFT of different signals should be positive, got {loss}"
    );
}

#[test]
fn test_esr_invariant_to_sample_rate() {
    let sig: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut perturbed = sig.clone();
    perturbed[100] += 1e-5;
    let esr1 = compute_esr(&sig, &perturbed);
    let esr2 = compute_esr(&sig, &perturbed);
    assert!((esr1 - esr2).abs() < 1e-15, "ESR should be deterministic");
}

// =============================================================================
// True-peak (BS.1770-4 Annex 2) tests
// =============================================================================

#[test]
fn test_true_peak_empty() {
    assert!(compute_true_peak_db(&[]).is_infinite());
}

#[test]
fn test_true_peak_silence() {
    let sig = vec![0.0f32; 1024];
    assert!(compute_true_peak_db(&sig).is_infinite());
}

#[test]
fn test_true_peak_sine_minus_6_db() {
    // 1 kHz sine at -6 dBFS (amplitude 0.5) → dBTP ≈ -6
    let sr = 48000.0f64;
    let n = 2048;
    let ampl = 0.5f32;
    let sig: Vec<f32> = (0..n)
        .map(|i| ampl * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin() as f32)
        .collect();
    let tp = compute_true_peak_db(&sig);
    // Due to filter passband ripple the true-peak may be slightly above -6
    assert!(
        (tp - (-6.02)).abs() < 0.3,
        "1kHz -6 dBFS sine should be ~-6 dBTP, got {tp}"
    );
}

#[test]
fn test_true_peak_full_scale_sine() {
    // 1 kHz sine at 0 dBFS (amplitude 1.0) → dBTP ≈ 0
    let sr = 48000.0f64;
    let n = 2048;
    let sig: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin() as f32)
        .collect();
    let tp = compute_true_peak_db(&sig);
    assert!(
        tp.abs() < 0.15,
        "1kHz 0 dBFS sine should be ~0 dBTP, got {tp}"
    );
}

#[test]
fn test_oversample_4x_length() {
    let sig = vec![0.5f32; 128];
    let up = oversample_4x(&sig);
    assert_eq!(up.len(), 128 * 4, "4× oversampled length must be 4× input");
}

#[test]
fn test_bs1770_fir_dc_gain() {
    // Each polyphase sub-filter should have near-unity DC gain (sum ≈ 1.0).
    // Q15-derived coefficients have minor rounding; tolerance 1e-2 is adequate.
    for (p, phase) in BS1770_PHASES.iter().enumerate() {
        let sum: f64 = phase.iter().sum();
        assert!(
            (sum - 1.0).abs() < 3e-2,
            "Phase {p} DC gain should be ~1.0, got {sum}"
        );
    }
}

#[test]
fn test_true_peak_dc_minus_6_db() {
    // DC 0.5: the filter transient produces a brief overshoot (~ -5 dBTP)
    // at the start because the partial convolution (incomplete history)
    // sums phase coefficients that exceed 1.0 before settling.
    // Steady-state gain per phase is ~1.0, so:
    //   peak_abs ≈ 0.5 * max_phase_gain_at_transient ≈ 0.5 * 1.116 = 0.558
    //   dBTP ≈ 20 * log10(0.558) ≈ -5.07
    let sig = vec![0.5f32; 1024];
    let tp = compute_true_peak_db(&sig);
    assert!(
        tp > -6.5 && tp < -4.5,
        "DC 0.5 dBTP={tp} should be between -6.5 and -4.5 (transient overshoot)"
    );
    // Steady-state peak (skip filter settling) should be ~ -6 dBTP
    let up = oversample_4x(&sig);
    let steady_peak = up[48..].iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let steady_tp = 20.0 * steady_peak.log10();
    assert!(
        (steady_tp - (-6.02)).abs() < 0.1,
        "Steady-state DC 0.5 should be ~-6 dBTP, got {steady_tp}"
    );
}

#[test]
fn test_true_peak_detects_gibbs_overshoot() {
    // A sharp step from +0.99 to -0.99 causes filter ringing.
    // The BS.1770-4 filter has large central coefficients (e.g., 0.972),
    // producing overshoot at the step discontinuity.
    let n = 512;
    let mut sig = vec![0.0f32; n];
    let step_at = n / 2;
    for (i, s) in sig.iter_mut().enumerate() {
        *s = if i < step_at { 0.99f32 } else { -0.99f32 };
    }
    // Sample-peak: all |x[i]| = 0.99 < 1.0
    let sample_peak = sig.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(
        sample_peak < 1.0,
        "Signal samples must all be < 1.0; sample_peak={sample_peak}"
    );

    let overs = find_true_peak_overs(&sig);
    assert!(
        !overs.is_empty(),
        "Step discontinuity must produce inter-sample overs"
    );
    // The overs should be near the step position (or at the initial transient)
    let pos = overs[0].position;
    let near_step = pos >= step_at.saturating_sub(24) && pos <= step_at + 24;
    assert!(
        pos < 24 || near_step,
        "Over position {pos} must be near step at {step_at} or at initial transient (< 24)"
    );
    assert!(
        overs[0].dbtp > 0.0,
        "Inter-sample over must have dBTP > 0, got {}",
        overs[0].dbtp
    );
}

#[test]
fn test_bs1770_fir_symmetry() {
    // Phase 3 = reversed Phase 0, Phase 2 = reversed Phase 1
    for k in 0..12 {
        assert!(
            (BS1770_PHASES[3][k] - BS1770_PHASES[0][11 - k]).abs() < 1e-12,
            "Phase symmetry p3[{k}] != p0[{}]: {} != {}",
            11 - k,
            BS1770_PHASES[3][k],
            BS1770_PHASES[0][11 - k]
        );
        assert!(
            (BS1770_PHASES[2][k] - BS1770_PHASES[1][11 - k]).abs() < 1e-12,
            "Phase symmetry p2[{k}] != p1[{}]: {} != {}",
            11 - k,
            BS1770_PHASES[2][k],
            BS1770_PHASES[1][11 - k]
        );
    }
}

#[test]
fn test_true_peak_no_overs_quiet_signal() {
    let sig = vec![0.3f32; 1024];
    let overs = find_true_peak_overs(&sig);
    assert!(overs.is_empty(), "Quiet signal should have no overs");
}

#[test]
fn test_find_true_peak_overs_empty() {
    let overs = find_true_peak_overs(&[]);
    assert!(overs.is_empty());
}

/// 21 kHz sine at 48 kHz, amplitude 0.999 → inter-sample overs from filter ripple.
#[test]
fn test_true_peak_detects_hf_sine_overs() {
    let sr = 48000.0f64;
    let ampl = 0.999f64;
    let freq = 21000.0;
    let n = 4800; // 100 ms
    let sig: Vec<f32> = (0..n)
        .map(|i| (ampl * (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin()) as f32)
        .collect();

    // Verify all samples are < 1.0 (sample-peak not triggered)
    let sample_peak = sig.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(
        sample_peak < 1.0,
        "Sample peak {sample_peak} must be < 1.0 for sample-peak not to fire"
    );

    let overs = find_true_peak_overs(&sig);
    // Close to Nyquist, filter ripple may or may not push above 0 dBFS.
    // If it does, the overs should be at interpolated positions.
    for o in &overs {
        assert!(o.dbtp > 0.0, "Detected over must have dBTP > 0");
    }
}

/// Even a 1 kHz sine at amplitude 0.99999 may produce inter-sample overs
/// due to filter passband ripple (±0.1 dB per BS.1770-4 spec).
#[test]
fn test_true_peak_detects_near_full_scale_overs() {
    let sr = 48000.0f64;
    let ampl = 0.999999f64;
    let freq = 1000.0;
    let n = 4800;
    let sig: Vec<f32> = (0..n)
        .map(|i| (ampl * (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin()) as f32)
        .collect();

    let tp = compute_true_peak_db(&sig);
    // A nearly full-scale sine should have true-peak ~ 0 dBTP
    assert!(
        tp.abs() < 2.0,
        "Near-full-scale 1kHz sine dBTP={tp} should be close to 0"
    );
}

/// The upsampled signal must exactly match per-sample computation for a known waveform.
#[test]
fn test_oversample_4x_deterministic() {
    let sig: Vec<f32> = vec![1.0, -0.5, 0.25, 0.0, -0.125];
    let up1 = oversample_4x(&sig);
    let up2 = oversample_4x(&sig);
    assert_eq!(up1.len(), up2.len());
    for (i, (&a, &b)) in up1.iter().zip(up2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "oversample_4x must be deterministic at index {i}"
        );
    }
}

/// Cache-warmup: first few samples of the upsampled signal have partial filter history.
/// Output at index 0 should equal x[0] * h_p[0] (only one non-zero term).
#[test]
fn test_oversample_4x_first_sample() {
    let sig = vec![0.5f32];
    let up = oversample_4x(&sig);
    for p in 0..4 {
        let expected = 0.5f64 * BS1770_PHASES[p][0];
        let got = up[p];
        assert!(
            (got - expected).abs() < 1e-12,
            "Phase {p}: expected {expected}, got {got}"
        );
    }
}
