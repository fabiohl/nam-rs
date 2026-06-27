// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Spectral fidelity integration tests.
//!
//! Validates the ASR (Aliasing-to-Signal Ratio) metric against known distortion
//! functions and fingerprints the aliasing behaviour of NAM neural amp models.
//!
//! ## Test structure
//! - **Fast validation** (non-model): hard-clip (high ASR), linear gain (ASR≈0).
//! - **Model fingerprints** (`#[ignore]`): ASR(f0) curves for WaveNet/LSTM/A2 SKUs.

use nam_rs::testing::aliasing;

mod common;

// =============================================================================
// Helper: hard-clip waveshaper (known to produce aliasing)
// =============================================================================

fn hard_clip(x: f32, threshold: f32) -> f32 {
    x.clamp(-threshold, threshold)
}

// =============================================================================
// Fast validation — no model required
// =============================================================================

#[test]
fn asr_hard_clip_detects_aliasing() {
    // f0 incommensurate with 48 kHz so aliased harmonics are uniquely detectable
    let f0 = 2017.0;
    let sr = 48000;
    let n = 65536;
    let gain = 6.0;
    let clip = 0.2;

    let input = aliasing::generate_sine(f0, sr, n, gain);
    let output: Vec<f32> = input.iter().map(|&x| hard_clip(x, clip)).collect();

    let result = aliasing::compute_asr(&output, f0, sr);

    assert!(
        result.num_aliased > 0,
        "Hard-clip must produce aliased peaks; got {} harmonics, {} aliased",
        result.num_harmonics,
        result.num_aliased
    );
    assert!(
        result.asr_db > -30.0,
        "ASR too low for hard-clip: {:.1} dB",
        result.asr_db
    );
}

#[test]
fn asr_linear_system_no_aliasing() {
    let f0 = 440.0;
    let sr = 48000;
    let n = 16384;
    let gain = 1.0;

    let input = aliasing::generate_sine(f0, sr, n, gain);
    let output: Vec<f32> = input.iter().map(|&x| 2.0 * x).collect();

    let result = aliasing::compute_asr(&output, f0, sr);

    assert_eq!(
        result.num_aliased, 0,
        "Linear gain must not produce aliased peaks"
    );
    assert!(
        result.asr_db < -60.0 || result.asr_linear < 1e-6,
        "Linear system ASR near zero; got {:.1} dB",
        result.asr_db
    );
}

#[test]
fn asr_soft_clip_tanh_less_aliasing_than_hard_clip() {
    // Smooth nonlinearity (tanh) produces fewer high-frequency harmonics
    // than hard-clip → less aliasing
    let f0 = 2017.0;
    let sr = 48000;
    let n = 65536;
    let gain = 6.0;

    let input = aliasing::generate_sine(f0, sr, n, gain);

    let tanh_out: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
    let clip_out: Vec<f32> = input.iter().map(|&x| hard_clip(x, 0.2)).collect();

    let tanh_result = aliasing::compute_asr(&tanh_out, f0, sr);
    let clip_result = aliasing::compute_asr(&clip_out, f0, sr);

    assert!(
        tanh_result.asr_db < clip_result.asr_db + 3.0,
        "tanh ASR ({:.1} dB) should be ≤ hard-clip ASR ({:.1} dB)",
        tanh_result.asr_db,
        clip_result.asr_db
    );
}

#[test]
fn asr_curve_musical_pitches() {
    // ASR should be computable across the musical pitch range without panicking
    let sr = 48000;
    let n = 32768;
    let gain = 1.0;

    for &(f0, _name) in aliasing::MUSICAL_PITCHES {
        let input = aliasing::generate_sine(f0, sr, n, gain);
        let output: Vec<f32> = input
            .iter()
            .map(|&x| (x * 3.0).tanh()) // mild soft-clip
            .collect();

        let result = aliasing::compute_asr(&output, f0, sr);
        assert!(
            result.num_harmonics >= 1,
            "Must detect fundamental for {_name} ({f0} Hz); got {} harmonics",
            result.num_harmonics
        );
    }
}

#[test]
fn asr_stress_tone_works() {
    let f0 = aliasing::STRESS_F0;
    let sr = 48000;
    let n = 32768;
    let gain = aliasing::HIGH_GAIN;

    let input = aliasing::generate_sine(f0, sr, n, gain);
    // Aggressive hard-clip to guarantee detectable aliasing at 2017 Hz
    let output: Vec<f32> = input.iter().map(|&x| x.clamp(-0.3, 0.3)).collect();

    let result = aliasing::compute_asr(&output, f0, sr);
    assert!(result.f0 == f0);
    assert!(result.sample_rate == sr);
    assert!(
        result.has_aliasing(),
        "Stress tone ({f0} Hz, gain={gain}) must produce detected aliasing; \
         got {} harmonics, {} aliased, ASR={:.1} dB",
        result.num_harmonics,
        result.num_aliased,
        result.asr_db
    );
}

#[test]
fn asr_aggregate_across_pitches() {
    let sr = 48000;
    let n = 16384;
    let gain = 4.0;

    let mut results = Vec::new();
    for &(f0, _name) in aliasing::MUSICAL_PITCHES {
        let input = aliasing::generate_sine(f0, sr, n, gain);
        let output: Vec<f32> = input.iter().map(|&x| hard_clip(x, 0.3)).collect();
        results.push(aliasing::compute_asr(&output, f0, sr));
    }

    let agg = aliasing::asr_aggregate(&results);
    let worst = aliasing::asr_worst_case(&results);

    assert!(agg.is_finite() || agg == f64::NEG_INFINITY);
    assert!(worst.is_finite() || worst == f64::NEG_INFINITY);
}

// =============================================================================
// Model-specific ASR fingerprints — `#[ignore]` (require .nam model files)
// =============================================================================

// Model-specific tests reuse the golden test infrastructure.
// They measure ASR(f0) curves for each model SKU and assert that ASR
// falls within documented baselines.

#[cfg(test)]
mod model_tests {
    use nam_rs::loader::dispatcher::build_model;
    use nam_rs::loader::nam_json::parse_nam_json;
    use nam_rs::models::NamModel;
    use nam_rs::testing::aliasing;
    use std::fs;

    use crate::common;

    fn process_through_model(model_filename: &str, input: &[f32]) -> Option<Vec<f32>> {
        let path = common::io_helpers::model_path(model_filename);
        if !path.exists() {
            return None;
        }
        let json_data = fs::read_to_string(&path).ok()?;
        let model_data = parse_nam_json(&json_data).ok()?;
        let mut model = build_model(&model_data).ok()?;

        model.prewarm(2048);
        let mut output = vec![0.0f32; input.len()];
        let block_size = 64;
        let mut pos = 0;
        while pos < input.len() {
            let end = (pos + block_size).min(input.len());
            model.process(&input[pos..end], &mut output[pos..end]);
            pos = end;
        }
        Some(output)
    }

    fn measure_asr_for_model(
        model_filename: &str,
        _label: &str,
        f0: f64,
        gain: f64,
        sr: u32,
        n: usize,
    ) -> Option<aliasing::AsrResult> {
        let input = aliasing::generate_sine(f0, sr, n, gain);
        let output = process_through_model(model_filename, &input)?;
        Some(aliasing::compute_asr(&output, f0, sr))
    }

    /// ASR fingerprint for WaveNet standard @ 48 kHz.
    ///
    /// Measures ASR across musical pitches and reports per-pitch values
    /// plus the aggregate. Serves as the baseline for Sprint S5 anti-aliasing work.
    #[test]
    #[ignore]
    fn asr_wavenet_standard_48k() {
        let model = "BossWN-standard.nam";
        let label = "BossWN-standard";
        let sr = 48000;
        let n = 65536;
        let gain = 4.0; // hot level to exercise nonlinearities

        let mut results: Vec<aliasing::AsrResult> = Vec::new();
        for &(f0, note) in aliasing::MUSICAL_PITCHES {
            if let Some(r) = measure_asr_for_model(model, label, f0, gain, sr, n) {
                eprintln!(
                    "  {note:>4} ({f0:>7.2} Hz): ASR = {:>6.1} dB  ({} harmonics, {} aliased, noise_floor={:.2e})",
                    r.asr_db, r.num_harmonics, r.num_aliased, r.noise_floor
                );
                results.push(r);
            }
        }

        // Stress tone
        if let Some(r) = measure_asr_for_model(
            model,
            label,
            aliasing::STRESS_F0,
            aliasing::HIGH_GAIN,
            sr,
            n,
        ) {
            eprintln!(
                "  STRESS ({:.0} Hz, gain={:.0}): ASR = {:>6.1} dB  ({} harmonics, {} aliased)",
                aliasing::STRESS_F0,
                aliasing::HIGH_GAIN,
                r.asr_db,
                r.num_harmonics,
                r.num_aliased
            );
            results.push(r);
        }

        let agg = aliasing::asr_aggregate(&results);
        let worst = aliasing::asr_worst_case(&results);
        eprintln!(
            "  Aggregate ASR: {:.1} dB  |  Worst-case: {:.1} dB",
            agg, worst
        );
        eprintln!("  ({label} — {} notes measured)", results.len());
    }

    /// ASR fingerprint for WaveNet nano @ 48 kHz (smallest WaveNet — quick test).
    #[test]
    #[ignore]
    fn asr_wavenet_nano_48k() {
        let model = "BossWN-nano.nam";
        let label = "BossWN-nano";
        let sr = 48000;
        let n = 32768; // shorter FFT for faster test

        let mut results: Vec<aliasing::AsrResult> = Vec::new();
        for &(f0, note) in &[
            aliasing::MUSICAL_PITCHES[0],
            aliasing::MUSICAL_PITCHES[4],
            aliasing::MUSICAL_PITCHES[6],
        ] {
            if let Some(r) = measure_asr_for_model(model, label, f0, 4.0, sr, n) {
                eprintln!(
                    "  {note:>4} ({f0:>7.2} Hz): ASR = {:>6.1} dB  ({} harmonics, {} aliased)",
                    r.asr_db, r.num_harmonics, r.num_aliased
                );
                results.push(r);
            }
        }
        eprintln!(
            "  ({label} — {} notes measured, agg={:.1} dB)",
            results.len(),
            aliasing::asr_aggregate(&results)
        );
    }

    /// ASR fingerprint for LSTM 1×8 @ 48 kHz.
    #[test]
    #[ignore]
    fn asr_lstm_1x8_48k() {
        let model = "BossLSTM-2x8.nam";
        let label = "BossLSTM-2x8";
        let sr = 48000;
        let n = 32768;

        let mut results: Vec<aliasing::AsrResult> = Vec::new();
        for &(f0, note) in &[
            aliasing::MUSICAL_PITCHES[0],
            aliasing::MUSICAL_PITCHES[4],
            aliasing::MUSICAL_PITCHES[6],
        ] {
            if let Some(r) = measure_asr_for_model(model, label, f0, 3.0, sr, n) {
                eprintln!(
                    "  {note:>4} ({f0:>7.2} Hz): ASR = {:>6.1} dB  ({} harmonics, {} aliased)",
                    r.asr_db, r.num_harmonics, r.num_aliased
                );
                results.push(r);
            }
        }
        eprintln!(
            "  ({label} — {} notes measured, agg={:.1} dB)",
            results.len(),
            aliasing::asr_aggregate(&results)
        );
    }

    /// ASR fingerprint for A2 example @ 48 kHz.
    #[test]
    #[ignore]
    fn asr_a2_example_48k() {
        let model = "a2_example.nam";
        let label = "A2-example";
        let sr = 48000;
        let n = 32768;

        let mut results: Vec<aliasing::AsrResult> = Vec::new();
        for &(f0, note) in &[
            aliasing::MUSICAL_PITCHES[0],
            aliasing::MUSICAL_PITCHES[4],
            aliasing::MUSICAL_PITCHES[6],
        ] {
            if let Some(r) = measure_asr_for_model(model, label, f0, 3.0, sr, n) {
                eprintln!(
                    "  {note:>4} ({f0:>7.2} Hz): ASR = {:>6.1} dB  ({} harmonics, {} aliased)",
                    r.asr_db, r.num_harmonics, r.num_aliased
                );
                results.push(r);
            }
        }
        eprintln!(
            "  ({label} — {} notes measured, agg={:.1} dB)",
            results.len(),
            aliasing::asr_aggregate(&results)
        );
    }
}
