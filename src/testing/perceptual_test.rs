// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for perceptual metrics (ESR, LUFS, MR-STFT).
//!
//! Extracted from `perceptual.rs` per testing.md convention (files ≥ 300 LoC
//! must keep tests in a separate `_test.rs` file).

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
    // 1 kHz sine at -20 dBFS should be ~-23 LUFS
    let sr = 48000;
    let n = sr as usize; // 1 second
    let amplitude = 10.0f32.powf(-20.0 / 20.0); // -20 dBFS
    let sig: Vec<f32> = (0..n)
        .map(|i| amplitude * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();
    let lufs = compute_lufs(&sig, sr);
    // BS.1770 says 1 kHz sine at -20 dBFS ≈ -23 LUFS (due to K-weighting)
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
