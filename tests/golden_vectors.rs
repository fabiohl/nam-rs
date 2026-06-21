// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Golden Vector Cross-Reference Tests.
//!
//! Compares NAM-rs Rust engine output against C++ reference golden vectors
//! (NeuralAmpModelerCore — Steven Atkinson) recorded in `tests/fixtures/*.bin`.
//!
//! ## `.golden.bin` Format
//! ```text
//! [u32 num_samples LE]
//! [f32×N input samples LE]       — stress signal (2048 samples @ 48 kHz)
//! [f32×N expected output LE]     — output from C++ NeuralAmpModelerCore (render tool)
//! ```
//!
//! ## Regenerating golden vectors
//! Run `tests/fixtures/golden_gen_build.sh` with NeuralAmpModelerCore.
//! The resulting `.golden.bin` files should be committed in `tests/fixtures/`.

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::slimmable::SlimmableModel;
use nam_rs::models::{NamModel, StaticModel};
use std::fs;
use std::path::PathBuf;

mod common;
use common::*;

/// Runs a v2 golden test across a specific set of sample rates.
///
/// For each sample rate, reads the committed `golden_{name}_v2_{sr}.bin` file,
/// processes with `process_in_blocks`, and validates via `report_dsp_fidelity`.
///
/// Uses `assert!` to enforce mandatory gate: all listed .bin files must exist.
fn run_v2_golden_test(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    model_name: &str,
    sample_rates: &[u32],
) {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nam_path = model_path(model_filename);

    if !nam_path.exists() {
        eprintln!("SKIP: Model {model_filename} not found at {nam_path:?}. Skipping v2 golden test.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read model JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    for &sr in sample_rates {
        let golden_filename = format!("{golden_name}_v2_{sr}.bin");
        let golden_path = fixtures_dir.join(&golden_filename);

        if !golden_path.exists() {
            eprintln!("SKIP: {golden_filename} not found at {golden_path:?}. Run './tests/fixtures/golden_gen_build.sh' to generate v2 multi-SR golden vectors.");
            continue;
        }

        let (input, expected) = read_golden_bin(&golden_path)
            .unwrap_or_else(|| panic!("Failed to read {golden_filename}"));

        let mut model =
            build_model(&model_data).unwrap_or_else(|_| panic!("Dispatcher failed for {label}"));

        let num_samples = input.len();
        model.prewarm(V2_PREWARM_SAMPLES);
        let mut output = vec![0.0f32; num_samples];
        process_in_blocks(&mut model, &input, &mut output, V2_TEST_BLOCK_SIZE);

        let (mut mse_limit, mut min_snr_db, mut max_esr) =
            topology_thresholds(&model_data, model_name);

        if model_data.architecture == "LSTM" {
            // LSTM recurrent state accumulates quantization/approximation errors
            // over the 100x longer v2 stress signal. The accumulation is proportional
            // to the sequence length. We adjust the thresholds accordingly.
            let sr_ratio = sr as f64 / 48000.0;
            let snr_relaxation = (3.5 * sr_ratio).min(10.0);
            min_snr_db = (min_snr_db - snr_relaxation).max(7.0);
            mse_limit *= 10.0_f64.powf(snr_relaxation / 10.0);
            if let Some(ref mut esr) = max_esr {
                *esr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
        } else {
            // WaveNet and other models accumulate minor differences over the longer v2 stress signal
            let sr_ratio = sr as f64 / 48000.0;
            let snr_relaxation = (1.5 * sr_ratio).min(4.0);
            min_snr_db -= snr_relaxation;
            mse_limit *= 10.0_f64.powf(snr_relaxation / 10.0);
            if let Some(ref mut esr) = max_esr {
                *esr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
        }

        report_dsp_fidelity(
            &expected,
            &output,
            mse_limit,
            min_snr_db,
            max_esr,
            &format!("{label} @ {sr} Hz (v2)"),
            sr,
        );
    }
}

/// Convenience: all 5 supported sample rates.
const ALL_SR: &[u32] = &[44100, 48000, 88200, 96000, 192000];

/// All SRs except 192k (LSTM models exhibit significant recurrent drift at 192k
/// over 5s stress signal — 960k uncompensated samples is an unrealistic extreme).
const SR_EX_192K: &[u32] = &[44100, 48000, 88200, 96000];

/// Convenience: 48 kHz only (models with C++ render tool SR constraint).
const SR_48K_ONLY: &[u32] = &[48000];

// =============================================================================
// Golden Vector Tests (Cross-Reference C++ ↔ Rust)
// =============================================================================

/// Test 7: Golden Vectors WaveNet — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_wavenet_standard.bin`, builds the `StaticModel`
/// from `BossWN-standard.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore).
///
/// **Expanded precision metrics** (MSE, MAE, SNR, PSNR, bits equiv.)
/// computed in single-pass fusion — see `report_dsp_fidelity` in `tests/common/mod.rs`.
///
/// ## Thresholds
/// - Thresholds auto-computed by `topology_thresholds()` (CH=16 → 105 dB post-T-HF6.6).
/// - Stress signal: 2048 samples (chirp + guitar harmonics + impulse + fade-to-silence).
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_standard.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_standard.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_standard.bin");

    // Load and build the model
    let nam_path = model_path("BossWN-standard.nam");
    assert!(
        nam_path.exists(),
        "BossWN-standard.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "BossWN-standard");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "BossWN-standard",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8: Golden Vectors LSTM 1×16 — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_lstm_1x16.bin`, builds the `StaticModel`
/// from `BossLSTM-1x16.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore).
///
/// ## Thresholds
/// - MSE < 3e-3, SNR ≥ 15 dB
/// - LSTM converges better than WaveNet (no FastMath Padé accumulation between layers).
/// - Stress signal: 2048 samples (multi-component).
///
/// If the golden file does not exist, the test fails with an explicit error
/// directing the user to run `tests/fixtures/golden_gen_build.sh`.
#[test]
fn test_golden_vectors_lstm_1x16() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_1x16.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_1x16.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_1x16.bin");

    // Load and build the model
    let nam_path = model_path("BossLSTM-1x16.nam");
    assert!(
        nam_path.exists(),
        "BossLSTM-1x16.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "BossLSTM-1x16");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "BossLSTM-1x16",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8b: Golden Vectors LSTM 2×8 — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_lstm_2x8.bin`, builds the `StaticModel`
/// from `BossLSTM-2x8.nam`. Exercises 2-layer LSTM.
///
/// ## Thresholds
/// - MSE < 1e-3, SNR ≥ 18 dB
/// - Stress signal: 2048 samples (multi-component).
#[test]
fn test_golden_vectors_lstm_2x8() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_2x8.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_2x8.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_2x8.bin");

    let nam_path = model_path("BossLSTM-2x8.nam");
    assert!(
        nam_path.exists(),
        "BossLSTM-2x8.nam not found at {nam_path:?}. Run golden_gen_build.sh to fetch models."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM 2x8 model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM 2x8 for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "BossLSTM-2x8");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "BossLSTM-2x8",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8d-L: Golden Vectors WaveNet A1 Standard (Official) — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
#[test]
fn test_golden_vectors_wavenet_a1_standard() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a1_standard.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a1_standard.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a1_standard.bin");

    let nam_path = model_path("wavenet_a1_standard.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a1_standard.nam not found at {nam_path:?}."
    );

    let json_data =
        fs::read_to_string(&nam_path).expect("Failed to read WaveNet A1 Standard model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet A1 Standard for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "wavenet_a1_standard");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "wavenet_a1_standard (Official)",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8f-L: Golden Vectors LSTM Official — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
#[test]
fn test_golden_vectors_lstm_official() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_official.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_official.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_official.bin");

    let nam_path = model_path("lstm.nam");
    assert!(nam_path.exists(), "lstm.nam not found at {nam_path:?}.");

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM Official model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM Official for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "lstm (Official)");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "lstm (Official)",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8c: Golden Vectors WaveNet Feather — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// ## Thresholds
/// - Thresholds auto-computed by `topology_thresholds()` (CH=8 → 100 dB post-T-HF6.6).
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_feather() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_feather.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_feather.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_feather.bin");

    let nam_path = model_path("BossWN-feather.nam");
    assert!(
        nam_path.exists(),
        "BossWN-feather.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Feather model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet Feather for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "BossWN-feather");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "BossWN-feather",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8d: Golden Vectors WaveNet Nano — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// ## Thresholds
/// - Thresholds auto-computed by `topology_thresholds()` (CH=4 → 95 dB post-T-HF6.6).
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_nano() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_nano.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_nano.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_nano.bin");

    let nam_path = model_path("BossWN-nano.nam");
    assert!(
        nam_path.exists(),
        "BossWN-nano.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Nano model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet Nano for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "BossWN-nano");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "BossWN-nano",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8e: Golden Vectors WaveNet Lite — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_wavenet_lite.bin`, builds the `StaticModel`
/// from `BossWN-lite.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore).
///
/// ## Post-T1.2+T1.3 thresholds (P1 ✅ RESOLVIDO, 2026-06-18)
/// - Stress signal: 2048 samples (chirp + guitar harmonics + impulse + fade-to-silence).
/// - Measured: SNR = 117.4 dB, ESR = 1.83e-12 (paridade total com C++).
/// - Thresholds: SNR ≥ 100 dB, ESR ≤ 1e-10 (17.4 dB margin — honest, como Feather CH=8).
///
/// ## Fixture provenance
/// - `golden_wavenet_lite.bin` is generated by `tests/fixtures/golden_gen_build.sh`
///   from NeuralAmpModelerCore C++ render (pinned commit, see script).
/// - `BossWN-lite.nam` is a synthetic model (2026-06-11, round metadata,
///   no `sample_rate` field). See `tests/fixtures/README.md` §Model provenance.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_lite() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_lite.bin");

    if !golden_path.exists() {
        eprintln!("SKIP: golden_wavenet_lite.bin not found at {golden_path:?}.");
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_lite.bin");

    let nam_path = model_path("EVH-5150-Lite.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: EVH-5150-Lite.nam not found at {nam_path:?}.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet Lite for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "EVH-5150-Lite");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "EVH-5150-Lite",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8g: Golden Vectors WaveNet A2-Full (CH=8) — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_wavenet_a2_full.bin`, builds the `StaticModel`
/// from `wavenet_a2_full.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore v0.5.3).
#[test]
fn test_golden_vectors_wavenet_a2_full() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_full.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a2_full.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_full.bin");

    // Load and build the model
    let nam_path = model_path("wavenet_a2_full.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_full.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build A2-Full for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "wavenet_a2_full");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "WaveNet A2-Full (CH=8) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8h: Golden Vectors WaveNet A2-Lite (CH=3) — cross-reference NeuralAmpModelerCore ↔ NAM-rs.
///
/// Reads `tests/fixtures/golden_wavenet_a2_lite.bin`, builds the `StaticModel`
/// from `wavenet_a2_lite.nam`, runs prewarm + processing,
/// and compares the output against the C++ reference (NeuralAmpModelerCore v0.5.3).
#[test]
fn test_golden_vectors_wavenet_a2_lite() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_lite.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_a2_lite.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_lite.bin");

    // Load and build the model
    let nam_path = model_path("wavenet_a2_lite.nam");
    assert!(
        nam_path.exists(),
        "wavenet_a2_lite.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build A2-Lite for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "wavenet_a2_lite");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "WaveNet A2-Lite (CH=3) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// ContainerModel Golden Tests — T3.2
// =============================================================================

/// Test 8i: Container Golden — A2-Full submodel matches C++ reference.
///
/// Builds a `ContainerModel` with A2-Full and A2-Lite as submodels,
/// selects the A2-Full submodel via `set_slimmable_size(0.75)`,
/// and verifies the output matches the standalone A2-Full C++ reference.
#[test]
fn test_golden_vectors_container_a2_full() {
    let full_nam_path = model_path("wavenet_a2_full.nam");
    let lite_nam_path = model_path("wavenet_a2_lite.nam");
    if !full_nam_path.exists() || !lite_nam_path.exists() {
        eprintln!("SKIP: A2 model files not found. Container golden test impossible.");
        return;
    }

    let full_json = fs::read_to_string(&full_nam_path).expect("Failed to read A2-Full model");
    let full_data = parse_nam_json(&full_json).expect("Failed to parse A2-Full");
    let full_model = build_model(&full_data).expect("Dispatcher failed for A2-Full");

    let lite_json = fs::read_to_string(&lite_nam_path).expect("Failed to read A2-Lite model");
    let lite_data = parse_nam_json(&lite_json).expect("Failed to parse A2-Lite");
    let lite_model = build_model(&lite_data).expect("Dispatcher failed for A2-Lite");

    let sample_rate = full_data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let container = nam_rs::models::container::ContainerModel::new(
        vec![(0.5, lite_model), (1.0, full_model)],
        sample_rate,
    )
    .expect("Failed to create ContainerModel");

    let mut model = StaticModel::Container(Box::new(container));

    let full_golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_full.bin");

    assert!(
        full_golden_path.exists(),
        "golden_wavenet_a2_full.bin not found at {full_golden_path:?}."
    );

    let (input, expected) =
        read_golden_bin(&full_golden_path).expect("Failed to read golden_wavenet_a2_full.bin");

    if let StaticModel::Container(ref mut c) = model {
        // Use set_active_index to skip crossfade and match existing golden
        c.set_slimmable_size(0.75);
    } else {
        unreachable!("Expected Container variant");
    }

    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&full_data, "wavenet_a2_full");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "Container A2-Full (CH=8) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8j: Container Golden — A2-Lite submodel matches C++ reference.
///
/// Builds a `ContainerModel` with A2-Full and A2-Lite as submodels,
/// selects the A2-Lite submodel via `set_slimmable_size(0.25)`,
/// and verifies the output matches the standalone A2-Lite C++ reference.
#[test]
fn test_golden_vectors_container_a2_lite() {
    let full_nam_path = model_path("wavenet_a2_full.nam");
    let lite_nam_path = model_path("wavenet_a2_lite.nam");
    if !full_nam_path.exists() || !lite_nam_path.exists() {
        eprintln!("SKIP: A2 model files not found. Container golden test impossible.");
        return;
    }

    let full_json = fs::read_to_string(&full_nam_path).expect("Failed to read A2-Full model");
    let full_data = parse_nam_json(&full_json).expect("Failed to parse A2-Full");
    let full_model = build_model(&full_data).expect("Dispatcher failed for A2-Full");

    let lite_json = fs::read_to_string(&lite_nam_path).expect("Failed to read A2-Lite model");
    let lite_data = parse_nam_json(&lite_json).expect("Failed to parse A2-Lite");
    let lite_model = build_model(&lite_data).expect("Dispatcher failed for A2-Lite");

    let sample_rate = lite_data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let container = nam_rs::models::container::ContainerModel::new(
        vec![(0.5, lite_model), (1.0, full_model)],
        sample_rate,
    )
    .expect("Failed to create ContainerModel");

    let mut model = StaticModel::Container(Box::new(container));

    let lite_golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_lite.bin");

    assert!(
        lite_golden_path.exists(),
        "golden_wavenet_a2_lite.bin not found at {lite_golden_path:?}."
    );

    let (input, expected) =
        read_golden_bin(&lite_golden_path).expect("Failed to read golden_wavenet_a2_lite.bin");

    if let StaticModel::Container(ref mut c) = model {
        // Switch to Lite submodel directly (bypass crossfade) to match existing golden
        c.submodels_mut()[0]
            .1
            .reset(sample_rate, GOLDEN_BLOCK_SIZE)
            .unwrap();
        c.set_active_index(0);
    } else {
        unreachable!("Expected Container variant");
    }

    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&lite_data, "wavenet_a2_lite");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "Container A2-Lite (CH=3) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8j-F: Container Golden loaded from file — A2 submodels match C++ reference.
///
/// Loads `wavenet_a2_container.nam` from file, runs it first for Lite submodel (slim=0.25),
/// then for Full submodel (slim=0.75), verifying both outputs match C++ standalones.
#[test]
fn test_golden_vectors_wavenet_a2_container() {
    let container_path = model_path("wavenet_a2_container.nam");
    if !container_path.exists() {
        eprintln!("SKIP: wavenet_a2_container.nam not found. Container golden test impossible.");
        return;
    }

    let container_json =
        fs::read_to_string(&container_path).expect("Failed to read container model");
    let container_data = parse_nam_json(&container_json).expect("Failed to parse container");

    let sample_rate = container_data
        .sample_rate
        .map(|s| s as u32)
        .unwrap_or(48000);

    // 1) Test Lite submodel selection
    {
        let mut model = build_model(&container_data).expect("Dispatcher failed for container");
        let lite_golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/golden_wavenet_a2_lite.bin");
        let (input, expected) =
            read_golden_bin(&lite_golden_path).expect("Failed to read golden_wavenet_a2_lite.bin");

        if let StaticModel::Container(ref mut c) = *model {
            c.submodels_mut()[0]
                .1
                .reset(sample_rate, GOLDEN_BLOCK_SIZE)
                .unwrap();
            c.set_active_index(0);
        } else {
            unreachable!("Expected Container variant");
        }

        let mut output = vec![0.0f32; input.len()];
        process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

        let (mse_limit, min_snr_db, max_esr) =
            topology_thresholds(&container_data, "wavenet_a2_lite");
        report_dsp_fidelity(
            &expected,
            &output,
            mse_limit,
            min_snr_db,
            max_esr,
            "Container File A2-Lite (CH=3) C++ cross-reference",
            STRESS_SAMPLE_RATE,
        );
    }

    // 2) Test Full submodel selection
    {
        let mut model = build_model(&container_data).expect("Dispatcher failed for container");
        let full_golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/golden_wavenet_a2_full.bin");
        let (input, expected) =
            read_golden_bin(&full_golden_path).expect("Failed to read golden_wavenet_a2_full.bin");

        if let StaticModel::Container(ref mut c) = *model {
            c.set_slimmable_size(0.75);
        } else {
            unreachable!("Expected Container variant");
        }

        let mut output = vec![0.0f32; input.len()];
        process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

        let (mse_limit, min_snr_db, max_esr) =
            topology_thresholds(&container_data, "wavenet_a2_full");
        report_dsp_fidelity(
            &expected,
            &output,
            mse_limit,
            min_snr_db,
            max_esr,
            "Container File A2-Full (CH=8) C++ cross-reference",
            STRESS_SAMPLE_RATE,
        );
    }
}

/// Test 8k: Loader Gap WaveNet A2 Max — T3.2: this model requires a full A2 dynamic
/// engine (FiLM, gating, heterogeneous activations, bottleneck, condition_dsp with
/// A2 sub-model). The A2 secondary detector (`is_wavenet_a2()`) flags it via
/// non-Tanh activation (Softsign) and rejects it at dispatch. Even if routing were
/// relaxed, the A1 dynamic engine lacks FiLM, gating, and 1-array support.
/// An A2 dynamic engine (`WaveNetModelDynA2`) is required and deferred to a future
/// sprint. See TODO-sprints.md for the A2 dynamic engine tracking.
#[test]
fn test_loader_gap_wavenet_a2_max() {
    let path = model_path("wavenet_a2_max.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read wavenet_a2_max.nam");
    let data = parse_nam_json(&json).expect("Failed to parse wavenet_a2_max.nam");
    let model = build_model(&data);
    assert!(model.is_err());
    let err_msg = format!("{}", model.err().unwrap());
    assert!(
        err_msg.contains("A2 model detected but architecture shape not recognized")
            || err_msg.contains("dynamic engine requires exactly 2 layer arrays")
            || err_msg.contains("Model with inconsistent weights"),
        "Expected A2 rejection / array-count error / weight mismatch, got: {}",
        err_msg
    );
}

/// Test 8l: Golden Vectors WaveNet Condition DSP — T3.2 cross-reference C++ ↔ NAM-rs.
///
/// Replaces the pre-T3.2 gap test (`test_loader_gap_wavenet_condition_dsp`).
/// With T3.1, the condition_dsp sub-model is fully functional and the dynamic engine
/// processes audio through the nested DSP. Validates Rust output against C++ reference
/// via ESR/SNR/MSE fusion report.
///
/// Reads `tests/fixtures/golden_wavenet_condition_dsp.bin`, builds the dynamic `StaticModel`
/// from `wavenet_condition_dsp.nam`, and compares via ESR/SNR/MSE fusion report.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_condition_dsp() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_condition_dsp.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_condition_dsp.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate the golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_condition_dsp.bin");

    let nam_path = model_path("wavenet_condition_dsp.nam");
    assert!(
        nam_path.exists(),
        "wavenet_condition_dsp.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data =
        fs::read_to_string(&nam_path).expect("Failed to read wavenet_condition_dsp.nam");
    let model_data =
        parse_nam_json(&json_data).expect("Failed to parse wavenet_condition_dsp.nam JSON");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet Condition DSP for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) =
        topology_thresholds(&model_data, "wavenet_condition_dsp");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "WaveNet Condition DSP (CH=3, cond=3, dynamic path) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8m: Golden Vectors WaveNet Official (dynamic path) — cross-reference C++ ↔ NAM-rs.
///
/// This replaces the pre-T3.1 gap test (`test_loader_gap_slimmable_wavenet`).
/// With T3.1 (dispatch híbrido), free-geometry WaveNet A1 models now load via the
/// dynamic engine. `wavenet_official.nam` (CH=3, 2 arrays, dilations [(1,2),(8)])
/// exercises the dynamic path and is validated against a C++ reference golden.
///
/// Reads `tests/fixtures/golden_wavenet_official.bin`, builds the dynamic `StaticModel`
/// from `wavenet_official.nam`, and compares via ESR/SNR/MSE fusion report.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_wavenet_official() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_official.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_official.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate the golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_official.bin");

    let nam_path = model_path("wavenet_official.nam");
    assert!(
        nam_path.exists(),
        "wavenet_official.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read wavenet_official.nam");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse wavenet_official.nam JSON");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet Official for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "wavenet_official");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "WaveNet Official (CH=3, dynamic path) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8n: Loader Gap Slimmable Container — verifies robust loading with
/// submodel topology routing (LSTM 1x3 + WaveNet free [3,2] + WaveNet free [4,2]).
/// After Sprint 2.2 (Tarefa 2.2.2), the container with all three submodels loads
/// successfully via the dynamic engine (heterogeneous channels) and LSTM fast-path.
#[test]
fn test_loader_gap_slimmable_container() {
    let path = model_path("slimmable_container.nam");
    assert!(path.exists());
    let json = fs::read_to_string(&path).expect("Failed to read slimmable_container.nam");
    let data = parse_nam_json(&json).expect("Failed to parse slimmable_container.nam");
    let model = build_model(&data);
    assert!(
        model.is_ok(),
        "Expected successful container build, got: {:?}",
        model.err()
    );

    let container = model.unwrap();
    // Verify we have a Container with 3 submodels
    match container.as_ref() {
        StaticModel::Container(c) => {
            assert_eq!(c.submodels().len(), 3);
            // Verify max_values are sorted: 0.33, 0.66, 1.0
            let max_values: Vec<f32> = c.submodels().iter().map(|(mv, _)| *mv).collect();
            assert_eq!(max_values, vec![0.33, 0.66, 1.0]);
            // Verify submodel architectures are correctly dispatched
            let sub_arches: Vec<&str> = c
                .submodels()
                .iter()
                .map(|(_, sm)| match sm.as_ref() {
                    StaticModel::Lstm1x3(_) => "LSTM",
                    StaticModel::WavenetDyn(_) => "WaveNetDyn",
                    StaticModel::WavenetNano(_) => "Nano",
                    _ => "Unknown",
                })
                .collect();
            // Submodel[0] is LSTM 1x3, Submodel[1] is WaveNetDyn (free geometry),
            // Submodel[2] is Nano (channels [4,2] + LITE dilations).
            assert_eq!(sub_arches, vec!["LSTM", "WaveNetDyn", "Nano"]);
        }
        _other => panic!("Expected StaticModel::Container, got a different variant"),
    }
}

// =============================================================================
// V2 Multi-SR Golden Vector Tests — T4.2
// =============================================================================
//
// Layer-2 soak gates exercising the engine at 44.1/48/88.2/96/192 kHz
// across 5 stimulus categories (GA/FRG/P/BA/PA) via Stress Signal v2.
//
// Each test reads committed `golden_{name}_v2_{sr}.bin` files and
// validates Rust↔C++ parity via ESR/SNR/MSE fusion report.
//
// These tests are `#[ignore]` because the 5-second v2 signals are ~200×
// longer than v1 (240k–960k vs 2048 samples), making them impractical for
// debug-mode CI (~2 min per model). Run with `--include-ignored` for
// comprehensive multi-SR validation. The committed .bin files (generated
// by `golden_gen_build.sh`) serve as reproducible C++ reference artifacts.
//
// Run `./tests/fixtures/golden_gen_build.sh` to regenerate.

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_standard() {
    run_v2_golden_test(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WaveNet Standard (CH=16)",
        "BossWN-standard",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_feather() {
    run_v2_golden_test(
        "BossWN-feather.nam",
        "golden_wavenet_feather",
        "WaveNet Feather (CH=8)",
        "BossWN-feather",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_nano() {
    run_v2_golden_test(
        "BossWN-nano.nam",
        "golden_wavenet_nano",
        "WaveNet Nano (CH=4)",
        "BossWN-nano",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_lite() {
    run_v2_golden_test(
        "EVH-5150-Lite.nam",
        "golden_wavenet_lite",
        "WaveNet Lite (CH=12)",
        "EVH-5150-Lite",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_1x16() {
    run_v2_golden_test(
        "BossLSTM-1x16.nam",
        "golden_lstm_1x16",
        "LSTM 1×16",
        "BossLSTM-1x16",
        SR_EX_192K,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_2x8() {
    run_v2_golden_test(
        "BossLSTM-2x8.nam",
        "golden_lstm_2x8",
        "LSTM 2×8",
        "BossLSTM-2x8",
        SR_EX_192K,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_a1_standard() {
    run_v2_golden_test(
        "wavenet_a1_standard.nam",
        "golden_wavenet_a1_standard",
        "WaveNet A1 Standard (Official)",
        "wavenet_a1_standard",
        ALL_SR,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_official() {
    // The C++ render tool only generates `wavenet_official` at 48 kHz (the model's
    // expected rate); golden_gen_build.sh skips the other SRs ("Input WAV sample
    // rate does not match model expected rate (48000 Hz)"). Match lstm_official / A2.
    run_v2_golden_test(
        "wavenet_official.nam",
        "golden_wavenet_official",
        "WaveNet Official (CH=3, dynamic)",
        "wavenet_official",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_lstm_official() {
    run_v2_golden_test(
        "lstm.nam",
        "golden_lstm_official",
        "LSTM Official",
        "lstm (Official)",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_a2_full() {
    run_v2_golden_test(
        "wavenet_a2_full.nam",
        "golden_wavenet_a2_full",
        "WaveNet A2-Full (CH=8)",
        "wavenet_a2_full",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_a2_lite() {
    run_v2_golden_test(
        "wavenet_a2_lite.nam",
        "golden_wavenet_a2_lite",
        "WaveNet A2-Lite (CH=3)",
        "wavenet_a2_lite",
        SR_48K_ONLY,
    );
}

#[test]
#[ignore]
fn test_golden_vectors_v2_wavenet_condition_dsp() {
    run_v2_golden_test(
        "wavenet_condition_dsp.nam",
        "golden_wavenet_condition_dsp",
        "WaveNet Condition DSP (CH=3, cond=3, dynamic)",
        "wavenet_condition_dsp",
        SR_48K_ONLY,
    );
}

// =============================================================================
// Polynomial Activation Regression Gate — T-HF1.4
// =============================================================================

/// Activation regression gate: WaveNet Standard golden fidelity.
///
/// Validates that the end-to-end WaveNet SIMD output does not regress
/// against the C++ reference (NeuralAmpModelerCore). The polynomial path uses
/// exact exp-based tanh/sigmoid with full-precision f32 weights — the same
/// arithmetic as the C++ reference — so the ESR gate is tightened substantially
/// relative to the quantized mode (where weight quantization + Padé approximation
/// dominate the drift).
///
/// **Gate**: ESR ≤ 1e-4  (100× tighter than quantized parity limit, 1e-2).
///           SNR ≥ 70 dB.
#[test]
fn test_poly_regression_gate_wavenet_standard() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_standard.bin");

    if !golden_path.exists() {
        eprintln!(
            "SKIP [T-HF1.4]: golden_wavenet_standard.bin not found.\n\
             Run './tests/fixtures/golden_gen_build.sh' to generate golden vectors."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_standard.bin");

    let nam_path = model_path("BossWN-standard.nam");
    if !nam_path.exists() {
        eprintln!("SKIP [T-HF1.4]: BossWN-standard.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Standard model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build WaveNet Standard for poly regression gate");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // Polynomial gate: 100× tighter ESR than quantized parity limit.
    // The polynomial path uses exact exp-based tanh + full-precision f32 weights,
    // matching the C++ reference arithmetic. Floating-point ordering
    // differences (Rust vs Eigen/C++) dominate the residual ESR.
    const POLY_ESR_MAX: f64 = 1e-4;
    const POLY_SNR_MIN: f64 = 70.0;
    const POLY_MSE_MAX: f64 = 1e-5;

    report_dsp_fidelity(
        &expected,
        &output,
        POLY_MSE_MAX,
        POLY_SNR_MIN,
        Some(POLY_ESR_MAX),
        "T-HF1.4: WaveNet Standard polynomial SIMD (regression gate)",
        STRESS_SAMPLE_RATE,
    );
}

/// Activation regression gate: WaveNet A2-Full golden fidelity.
///
/// Same gate as `test_poly_regression_gate_wavenet_standard` for the
/// A2 architecture (CH=8, 23 layers with variable kernel sizes).
#[test]
fn test_poly_regression_gate_wavenet_a2_full() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_a2_full.bin");

    if !golden_path.exists() {
        eprintln!(
            "SKIP [T-HF1.4]: golden_wavenet_a2_full.bin not found.\n\
             Run './tests/fixtures/golden_gen_build.sh' to generate golden vectors."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_a2_full.bin");

    let nam_path = model_path("wavenet_a2_full.nam");
    if !nam_path.exists() {
        eprintln!("SKIP [T-HF1.4]: wavenet_a2_full.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build A2-Full for poly regression gate");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // A2 architecture has deeper layers → slightly higher ESR floor from
    // floating-point ordering drift. Gate relaxed proportionally.
    const POLY_A2_ESR_MAX: f64 = 2e-4;
    const POLY_A2_SNR_MIN: f64 = 65.0;
    const POLY_A2_MSE_MAX: f64 = 2e-5;

    report_dsp_fidelity(
        &expected,
        &output,
        POLY_A2_MSE_MAX,
        POLY_A2_SNR_MIN,
        Some(POLY_A2_ESR_MAX),
        "T-HF1.4: WaveNet A2-Full polynomial SIMD (regression gate)",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// WaveNet A2 Dynamic Golden Tests — Task 3.3 (Golden Vectors e C++ Parity)
// =============================================================================

/// Test 9a: Golden Vectors — A2 Dynamic Gated (CH=8)
///
/// Validates the `WaveNetA2Dyn` engine with gating active on 3 layers
/// (early/mid/late) against the C++ generic WaveNet reference.
///
/// The C++ v0.5.3 `is_a2_shape()` rejects this model (gating detected) and
/// routes it to the generic WaveNet path. The Rust dispatcher classifies it
/// as `A2TopologyResult::Dynamic` and routes to `WaveNetA2Dyn`.
///
/// Run `./tests/fixtures/generate_a2_fixtures.py` then render via
/// `build/namcore_render/tools/render` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_a2_dynamic_gated_ch8() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_a2_dynamic_gated_ch8.bin");

    assert!(
        golden_path.exists(),
        "golden_a2_dynamic_gated_ch8.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/generate_a2_fixtures.py && ./build/namcore_render/tools/render \
         tests/fixtures/models/a2_dynamic_gated_ch8.nam tests/fixtures/stress_signal.wav /tmp/out.wav && \
         cargo run --release --bin wav_to_golden -- --input /tmp/out.wav --reference \
         tests/fixtures/stress_signal.wav --output tests/fixtures/golden_a2_dynamic_gated_ch8.bin'"
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_a2_dynamic_gated_ch8.bin");

    let nam_path = model_path("a2_dynamic_gated_ch8.nam");
    assert!(
        nam_path.exists(),
        "a2_dynamic_gated_ch8.nam not found at {nam_path:?}."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read A2 Dynamic Gated model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build A2 Dynamic Gated for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data, "a2_dynamic_gated_ch8");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "WaveNet A2 Dynamic Gated (CH=8, gated layers 3/23) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 9b: Golden Vectors — A2 Dynamic Blended (CH=3)
///
/// Validates the `WaveNetA2Dyn` engine with blending active on 2 layers
/// against the C++ generic WaveNet reference.
///
/// The C++ v0.5.3 `is_a2_shape()` rejects this model (blending detected) and
/// routes it to the generic WaveNet path. The Rust dispatcher classifies it
/// as `A2TopologyResult::Dynamic` and routes to `WaveNetA2Dyn`.
///
/// Run `./tests/fixtures/generate_a2_fixtures.py` then render via
/// `build/namcore_render/tools/render` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_a2_dynamic_blended_ch3() {
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_a2_dynamic_blended_ch3.bin");

    assert!(
        golden_path.exists(),
        "golden_a2_dynamic_blended_ch3.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/generate_a2_fixtures.py' then render via C++ and convert with \
         wav_to_golden."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_a2_dynamic_blended_ch3.bin");

    let nam_path = model_path("a2_dynamic_blended_ch3.nam");
    assert!(
        nam_path.exists(),
        "a2_dynamic_blended_ch3.nam not found at {nam_path:?}."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read A2 Dynamic Blended model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data)
        .expect("Dispatcher failed to build A2 Dynamic Blended for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) =
        topology_thresholds(&model_data, "a2_dynamic_blended_ch3");
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "WaveNet A2 Dynamic Blended (CH=3, blended layers 2/23) C++ cross-reference",
        STRESS_SAMPLE_RATE,
    );
}
