// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Helper function to create a programmatically generated mock Cab IR WAV file for testing.
fn create_temp_cab_ir_wav(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("nam_rs_test_fat_cab_{}.wav", name));

    // Generate some mock IR samples (exponential decay sine wave)
    let samples: Vec<f32> = (0..512)
        .map(|i| {
            let t = i as f32 / 48_000.0;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * (-100.0 * t).exp()
        })
        .collect();

    // Write PCM16 mono WAV (which is standard and supported by CabSimIr::load)
    write_pcm16_wav(&path, &samples, 48_000).expect("failed to write mock IR WAV");
    path
}

/// Loads the mock Cab IR WAV and verifies all metadata.
#[test]
fn test_load_fat_cab_wav() {
    let path = create_temp_cab_ir_wav("fat_cab");
    let ir = CabSimIr::load(&path, 0, false).expect("failed to load mock Cab IR WAV");

    assert_eq!(ir.sample_rate, 48_000, "expected 48 kHz effective rate");
    assert_eq!(ir.original_rate, 48_000, "expected 48 kHz original rate");
    assert!(!ir.samples.is_empty(), "IR samples must not be empty");
    assert!(!ir.normalized, "normalization should be off");
    assert!(
        ir.samples.iter().any(|&s| s.abs() > 0.0),
        "IR must contain non-zero samples"
    );
    std::fs::remove_file(path).ok();
}

/// Loads with resampling to 44.1 kHz.
#[test]
fn test_load_with_resample() {
    let path = create_temp_cab_ir_wav("resample");
    let ir = CabSimIr::load(&path, 44_100, false).expect("failed to load with resample");
    assert_eq!(ir.sample_rate, 44_100, "effective rate should be target");
    assert_eq!(ir.original_rate, 48_000);
    assert!(!ir.samples.is_empty());
    std::fs::remove_file(path).ok();
}

/// Loads with normalization enabled.
#[test]
fn test_load_with_normalize() {
    let path = create_temp_cab_ir_wav("normalize");
    let ir = CabSimIr::load(&path, 0, true).expect("failed to load with normalize");
    assert!(ir.normalized, "normalization flag should be true");
    assert!(!ir.samples.is_empty());

    let peak = ir.samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!(
        (peak - 1.0).abs() < 1e-6,
        "normalized peak should be ~1.0, got {}",
        peak
    );
    std::fs::remove_file(path).ok();
}

/// Loading a non-existent file must return an error, never panic.
#[test]
fn test_load_nonexistent_file() {
    let result = CabSimIr::load(std::path::Path::new("tests/__nonexistent__.wav"), 0, false);
    assert!(result.is_err(), "nonexistent file must error");
}

/// Loading an empty file must return an error.
#[test]
fn test_load_empty_file() {
    let path = std::path::Path::new("/tmp/nam_rs_test_empty.wav");
    std::fs::write(path, []).ok();
    let result = CabSimIr::load(path, 0, false);
    assert!(result.is_err(), "empty file must error");
    std::fs::remove_file(path).ok();
}

/// Loading a file too small to be a valid WAV header must error.
#[test]
fn test_load_too_small() {
    let path = std::path::Path::new("/tmp/nam_rs_test_small.wav");
    std::fs::write(path, [0u8; 30]).ok();
    let result = CabSimIr::load(path, 0, false);
    assert!(result.is_err(), "file too small must error");
    std::fs::remove_file(path).ok();
}

/// PCM16 mono WAV round-trip via the testing wav writer.
#[test]
fn test_load_pcm16_wav() {
    let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let path = std::path::Path::new("/tmp/nam_rs_test_pcm16.wav");

    // Write PCM16 using the format
    write_pcm16_wav(path, &samples, 48_000).expect("failed to write PCM16 WAV");

    let ir = CabSimIr::load(path, 0, false).expect("failed to load PCM16 WAV");
    assert_eq!(ir.sample_rate, 48_000);
    assert!(!ir.samples.is_empty());
    std::fs::remove_file(path).ok();
}

/// Float32 mono WAV via the testing wav writer.
#[test]
fn test_load_float32_wav() {
    let samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.002).sin()).collect();
    let path = std::path::Path::new("/tmp/nam_rs_test_f32.wav");

    crate::testing::wav::write_wav_f32(path, &samples, 48_000).expect("failed to write f32 WAV");

    let ir = CabSimIr::load(path, 0, false).expect("failed to load f32 WAV");
    assert_eq!(ir.sample_rate, 48_000);
    assert!(!ir.samples.is_empty());
    std::fs::remove_file(path).ok();
}

/// Normalization is a no-op when peak is already 1.0.
#[test]
fn test_normalize_noop() {
    let mut samples = vec![1.0f32, -1.0f32, 0.5];
    let applied = CabSimIr::normalize_in_place(&mut samples);
    assert!(!applied, "already normalized — noop expected");
}

/// Normalization scales correctly.
#[test]
fn test_normalize_scale() {
    let mut samples = vec![0.25f32, -0.5, 0.125];
    let applied = CabSimIr::normalize_in_place(&mut samples);
    assert!(applied);
    let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));
    assert!((peak - 1.0).abs() < 1e-6);
}

/// Writes a minimal PCM16 mono WAV file for testing.
fn write_pcm16_wav(
    path: &std::path::Path,
    samples: &[f32],
    sample_rate: u32,
) -> std::io::Result<()> {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 2;
    let file_size: u32 = 36 + data_size;
    let byte_rate = sample_rate * 2;
    let block_align: u16 = 2;
    let bits_per_sample: u16 = 16;

    let mut buf = Vec::with_capacity(44 + data_size as usize);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    for &s in samples {
        let clamped = s.clamp(-1.0, 0.999969);
        let raw = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&raw.to_le_bytes());
    }

    std::fs::write(path, &buf)
}
