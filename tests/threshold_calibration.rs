// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Meta-tests that enforce threshold calibration discipline.
//!
//! Ensures every model with a committed golden `.bin` fixture has a
//! calibrated entry in `get_calibrated_threshold()` — never silently
//! falling back to heuristic thresholds.
//!
//! Part of T3.3: formalizing the calibration infrastructure.

use std::fs;
use std::path::PathBuf;

mod common;
use common::validation::get_calibrated_threshold;

/// Maps a committed golden `.bin` filename to the `model_name` key
/// used in `get_calibrated_threshold()`. Returns `None` for `.bin`
/// files that don't participate in DSP fidelity validation (e.g.
/// cabsim goldens use their own oracles).
fn golden_bin_to_model_name(filename: &str) -> Option<&str> {
    match filename {
        "golden_wavenet_standard.bin" => Some("BossWN-standard"),
        "golden_lstm_1x16.bin" => Some("BossLSTM-1x16"),
        "golden_lstm_2x8.bin" => Some("BossLSTM-2x8"),
        "golden_wavenet_a1_standard.bin" => Some("wavenet_a1_standard"),
        "golden_lstm_official.bin" => Some("lstm (Official)"),
        "golden_wavenet_feather.bin" => Some("BossWN-feather"),
        "golden_wavenet_nano.bin" => Some("BossWN-nano"),
        "golden_wavenet_lite.bin" => Some("BossWN-lite"),
        "golden_wavenet_a2_full.bin" => Some("wavenet_a2_full"),
        "golden_wavenet_a2_lite.bin" => Some("wavenet_a2_lite"),
        // cabsim goldens use their own oracle (convolution / C++ parity),
        // not topology_thresholds.
        _ => None,
    }
}

/// Every model with a committed golden `.bin` fixture MUST have a
/// calibrated entry in `get_calibrated_threshold()`. This test
/// prevents silent fallback to heuristic thresholds.
#[test]
fn test_all_golden_models_have_calibrated_thresholds() {
    let fixtures_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    let mut tested_count = 0;

    for entry in fs::read_dir(&fixtures_dir).expect("Failed to read fixtures directory") {
        let entry = entry.expect("Failed to read fixture entry");
        let filename = entry.file_name();
        let filename_str = filename.to_string_lossy();

        if !filename_str.ends_with(".bin") {
            continue;
        }

        if let Some(model_name) = golden_bin_to_model_name(&filename_str) {
            let threshold = get_calibrated_threshold(model_name);
            assert!(
                threshold.is_some(),
                "Model '{model_name}' has committed golden '{filename_str}' \
                 but NO calibrated entry in get_calibrated_threshold(). \
                 Add an entry with // Measured: SNR=..., ESR=... and recalibrated floors."
            );

            let (_mse, snr_db, esr_opt) = threshold.unwrap();
            // Reject fully-neutralized thresholds (SNR ≤ 0 and no ESR gate)
            if let Some(esr) = esr_opt {
                assert!(
                    snr_db > 0.0 || esr < 1.0,
                    "Model '{model_name}' has both SNR ≤ 0 dB and ESR ≥ 1.0 \
                     — effectively a neutralized/placebo gate. \
                     Thresholds must trace a real measurement."
                );
            } else {
                assert!(
                    snr_db > 0.0,
                    "Model '{model_name}' has SNR ≤ 0 dB with no ESR gate \
                     — effectively a neutralized/placebo gate."
                );
            }

            tested_count += 1;
        }
    }

    assert!(
        tested_count >= 10,
        "Expected ≥ 10 golden models with calibrated thresholds, found {tested_count}"
    );
}

/// Verify that every calibrated entry in `get_calibrated_threshold()`
/// has a `// Measured: SNR=..., ESR=...` comment documenting the
/// real measurement that originated the floors.
///
/// This reads the source file of `validation.rs` and checks for the
/// comment pattern next to each model name pattern.
#[test]
fn test_all_calibrated_entries_have_measurement_comments() {
    let validation_src =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/common/validation.rs");

    let source = fs::read_to_string(&validation_src)
        .expect("Failed to read tests/common/validation.rs");

    let models = [
        "BossWN-standard",
        "BossLSTM-1x16",
        "BossLSTM-2x8",
        "wavenet_a1_standard",
        "lstm (Official)",
        "lstm_official",
        "BossWN-feather",
        "BossWN-nano",
        "BossWN-lite",
        "wavenet_a2_full",
        "wavenet_a2_lite",
    ];

    for &model in &models {
        // Find the line containing the model name pattern (inside a match arm)
        let model_line = source
            .lines()
            .enumerate()
            .find(|(_, l)| {
                l.trim().starts_with('"') && l.contains(model) && l.contains("=>")
            })
            .map(|(i, _)| i)
            .unwrap_or_else(|| {
                panic!(
                    "Could not find match arm for model '{model}' in validation.rs \
                     (expected a line containing '\"{model}\" ... =>')"
                )
            });

        // Check up to 3 lines above the match arm for "// Measured:"
        let mut found_measured = false;
        for offset in 1..=3 {
            if model_line >= offset {
                let prev_line = source.lines().nth(model_line - offset).unwrap_or("");
                if prev_line.contains("// Measured: SNR") && prev_line.contains("ESR") {
                    found_measured = true;
                    break;
                }
            }
        }

        assert!(
            found_measured,
            "Calibrated entry for model '{model}' at line {} is missing \
             '// Measured: SNR=..., ESR=...' comment within 3 lines above. \
             Add the measurement documentation.",
            model_line + 1,
        );
    }
}
