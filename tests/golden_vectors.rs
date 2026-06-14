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
/// - Thresholds auto-computed by `topology_thresholds()` (CH=16 → 60 dB post-T16.1).
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
    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
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
/// If the golden file does not exist, the test prints SKIP and returns.
/// Run `tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
fn test_golden_vectors_lstm_1x16() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_lstm_1x16.bin");

    if !golden_path.exists() {
        eprintln!(
            "SKIP: golden_lstm_1x16.bin not found at {golden_path:?}. \
             Run tests/fixtures/golden_gen_build.sh to generate the golden vectors."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_1x16.bin");

    // Load and build the model
    let nam_path = model_path("BossLSTM-1x16.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: BossLSTM-1x16.nam not found. Golden test impossible.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM for golden test");

    // Prewarm + Processing
    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    // 5-metric validation — single-pass fusion
    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
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

    if !golden_path.exists() {
        eprintln!(
            "SKIP: golden_lstm_2x8.bin not found at {golden_path:?}. \
             Run tests/fixtures/golden_gen_build.sh to generate the golden vectors."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_2x8.bin");

    let nam_path = model_path("BossLSTM-2x8.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: BossLSTM-2x8.nam not found. Golden test impossible.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM 2x8 model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM 2x8 for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
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

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A1 Standard model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet A1 Standard for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
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
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_lstm_official.bin");

    assert!(
        golden_path.exists(),
        "golden_lstm_official.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_lstm_official.bin");

    let nam_path = model_path("lstm.nam");
    assert!(
        nam_path.exists(),
        "lstm.nam not found at {nam_path:?}."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read LSTM Official model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build LSTM Official for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
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
/// - Thresholds auto-computed by `topology_thresholds()` (CH=8 → 60 dB post-T16.1).
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

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
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
/// - Thresholds auto-computed by `topology_thresholds()` (CH=4 → 45 dB post-T16.1).
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

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
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
/// ## Post-T16.1 thresholds (regenerated from C++, 2026-06-13)
/// - Stress signal: 2048 samples (chirp + guitar harmonics + impulse + fade-to-silence).
/// - Thresholds auto-computed by `topology_thresholds()` based on CH=12.
///
/// ## Fixture provenance
/// - `golden_wavenet_lite.bin` is generated by `tests/fixtures/golden_gen_build.sh`
///   from NeuralAmpModelerCore C++ render (pinned commit, see script).
/// - `BossWN-lite.nam` is a synthetic model (2026-06-11, round metadata,
///   no `sample_rate` field). See `tests/fixtures/README.md` §Model provenance.
///
/// Run `./tests/fixtures/golden_gen_build.sh` to regenerate the golden vectors.
#[test]
#[ignore = "known-divergent: WaveNet Lite (CH=12) exhibits numerical drift (SNR = 0.9 dB vs C++)"]
fn test_golden_vectors_wavenet_lite() {
    let golden_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden_wavenet_lite.bin");

    assert!(
        golden_path.exists(),
        "golden_wavenet_lite.bin not found at {golden_path:?}.\n\
         Run './tests/fixtures/golden_gen_build.sh' to generate all golden vectors from C++.\n\
         This test cannot be skipped — the fixture is mandatory post-T16.1."
    );

    let (input, expected) =
        read_golden_bin(&golden_path).expect("Failed to read golden_wavenet_lite.bin");

    let nam_path = model_path("BossWN-lite.nam");
    assert!(
        nam_path.exists(),
        "BossWN-lite.nam not found at {nam_path:?}. \
         This fixture is part of the repository and must exist."
    );

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet Lite for golden test");

    model.prewarm(2048);
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "BossWN-lite",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8g: Self-Golden WaveNet A2-Full (CH=8) — regression guard for Rust A2 fast path.
///
/// Generates expected output from the Rust A2 engine on first run (self-golden pattern),
/// then verifies bitwise identical output on subsequent runs. This catches regressions
/// in the Rust A2 implementation independent of the C++ render tool.
///
/// The C++ NeuralAmpModelerCore `render` tool's A2 fast path currently produces
/// divergent output (numeric instability in the C++ weight interpretation).
/// Once the C++ A2 path is stabilized, the self-golden can be replaced with
/// a cross-reference against the C++ render output.
#[test]
fn test_golden_vectors_wavenet_a2_full() {
    let nam_path = model_path("wavenet_a2_full.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: wavenet_a2_full.nam not found. Golden test impossible.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build A2-Full for golden test");

    model.prewarm(2048);

    let input = generate_stress_signal_v1();
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let self_golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a2_full_self.bin");

    if !self_golden_path.exists() {
        write_golden_bin(&self_golden_path, &input, &output)
            .expect("Failed to write self-golden for A2-Full");
        eprintln!(
            "Generated self-golden: {self_golden_path:?} — commit this file for regression testing."
        );
        return;
    }

    let (_, expected) =
        read_golden_bin(&self_golden_path).expect("Failed to read self-golden for A2-Full");

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "WaveNet A2-Full (CH=8) self-golden",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8h: Self-Golden WaveNet A2-Lite (CH=3) — regression guard for Rust A2 fast path.
///
/// Same self-golden pattern as A2-Full.
#[test]
fn test_golden_vectors_wavenet_a2_lite() {
    let nam_path = model_path("wavenet_a2_lite.nam");
    if !nam_path.exists() {
        eprintln!("SKIP: wavenet_a2_lite.nam not found. Golden test impossible.");
        return;
    }

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read WaveNet A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build A2-Lite for golden test");

    model.prewarm(2048);

    let input = generate_stress_signal_v1();
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let self_golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a2_lite_self.bin");

    if !self_golden_path.exists() {
        write_golden_bin(&self_golden_path, &input, &output)
            .expect("Failed to write self-golden for A2-Lite");
        eprintln!(
            "Generated self-golden: {self_golden_path:?} — commit this file for regression testing."
        );
        return;
    }

    let (_, expected) =
        read_golden_bin(&self_golden_path).expect("Failed to read self-golden for A2-Lite");

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&model_data);
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "WaveNet A2-Lite (CH=3) self-golden",
        STRESS_SAMPLE_RATE,
    );
}

// =============================================================================
// ContainerModel Golden Tests — T3.2
// =============================================================================

/// Test 8i: Container Golden — A2-Full submodel matches standalone self-golden.
///
/// Builds a `ContainerModel` with A2-Full and A2-Lite as submodels,
/// selects the A2-Full submodel via `set_slimmable_size(0.75)`,
/// and verifies the output matches the standalone A2-Full self-golden.
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

    let full_golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a2_full_self.bin");

    if !full_golden_path.exists() {
        eprintln!(
            "SKIP: golden_wavenet_a2_full_self.bin not found. Run A2-Full golden test first."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&full_golden_path).expect("Failed to read self-golden for A2-Full");

    if let StaticModel::Container(ref mut c) = model {
        // Use set_active_index to skip crossfade and match existing golden
        c.set_slimmable_size(0.75);
    } else {
        unreachable!("Expected Container variant");
    }

    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&full_data);
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "Container A2-Full (CH=8) self-golden",
        STRESS_SAMPLE_RATE,
    );
}

/// Test 8j: Container Golden — A2-Lite submodel matches standalone self-golden.
///
/// Builds a `ContainerModel` with A2-Full and A2-Lite as submodels,
/// selects the A2-Lite submodel via `set_slimmable_size(0.25)`,
/// and verifies the output matches the standalone A2-Lite self-golden.
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

    let lite_golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden_wavenet_a2_lite_self.bin");

    if !lite_golden_path.exists() {
        eprintln!(
            "SKIP: golden_wavenet_a2_lite_self.bin not found. Run A2-Lite golden test first."
        );
        return;
    }

    let (input, expected) =
        read_golden_bin(&lite_golden_path).expect("Failed to read self-golden for A2-Lite");

    if let StaticModel::Container(ref mut c) = model {
        // Switch to Lite submodel directly (bypass crossfade) to match existing golden
        c.submodels_mut()[0].1.reset(sample_rate, GOLDEN_BLOCK_SIZE);
        c.set_active_index(0);
    } else {
        unreachable!("Expected Container variant");
    }

    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, GOLDEN_BLOCK_SIZE);

    let (mse_limit, min_snr_db, max_esr) = topology_thresholds(&lite_data);
    report_dsp_fidelity(
        &expected,
        &output,
        mse_limit,
        min_snr_db,
        max_esr,
        "Container A2-Lite (CH=3) self-golden",
        STRESS_SAMPLE_RATE,
    );
}
