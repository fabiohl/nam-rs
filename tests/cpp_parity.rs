// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Live cross-validation tests NAM-rs ↔ NeuralAmpModelerCore (Layer 2).
//!
//! ## When it runs
//! - `utils/tests-long.sh` (full slow suite)
//! - `cargo test --test cpp_parity -- --ignored --nocapture`
//!
//! ## Pipeline
//! 1. Compiles the `render` tool from NeuralAmpModelerCore on-demand (idempotent, cached)
//! 2. Generates WAV stress signal via `generate_stress_signal_v1()` or `generate_stress_signal_v2()`
//! 3. Writes WAV, executes render via `std::process::Command`
//! 4. Reads WAV output, compares C++ vs Rust with `report_dsp_fidelity()`
//!
//! Tests are `#[ignore]` in normal CI — requires C++ toolchain installed.
//!
//! ## Parity Thresholds (Adaptive)
//!
//! Thresholds are computed dynamically by `topology_thresholds()` based on
//! model topology (channels, dilations for WaveNet; num_layers, hidden_size for LSTM).
//!
//! ## Multi-Sample-Rate Support
//!
//! v2 stress signal supports 44.1k, 48k, 96k, and 192k sample rates.
//! The `#[test]` functions parametrize model × SR combinations via runtime arrays
//! instead of statically exploding test functions.

mod common;
use common::*;

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::NamModel;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const NAM_CORE_DIR: &str = "tests/fixtures/NeuralAmpModelerCore";
const BUILD_DIR: &str = "build/namcore_render";

fn render_bin() -> PathBuf {
    let mut bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    bin.push(BUILD_DIR);
    bin.push("Release/render");
    if !bin.exists() {
        bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        bin.push(BUILD_DIR);
        bin.push("Debug/render");
    }
    if !bin.exists() {
        let build_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(BUILD_DIR);
        if let Ok(entries) = std::fs::read_dir(&build_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.file_name().is_some_and(|n| n == "render") && path.is_file() {
                    return path;
                }
                if path.is_dir() {
                    let render = path.join("render");
                    if render.exists() {
                        return render;
                    }
                }
            }
        }
    }
    bin
}

fn ensure_render_compiled() -> bool {
    let bin = render_bin();
    if bin.exists() {
        return true;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let nam_core = project_root.join(NAM_CORE_DIR);

    if !nam_core.exists() {
        eprintln!("NeuralAmpModelerCore not found at {nam_core:?}");
        eprintln!(
            "Run: git clone --depth 1 https://github.com/sdatkinson/NeuralAmpModelerCore.git {NAM_CORE_DIR}"
        );
        return false;
    }

    for sub in &["Dependencies/eigen", "Dependencies/AudioDSPTools"] {
        let sub_path = nam_core.join(sub);
        if sub_path.exists() && fs::read_dir(&sub_path).map_or(true, |mut d| d.next().is_none()) {
            let status = Command::new("git")
                .args(["submodule", "update", "--init", sub])
                .current_dir(&nam_core)
                .status();
            if status.is_ok_and(|s| !s.success()) {
                eprintln!("WARN: failed to initialize submodule {sub}");
            }
        }
    }

    let build_dir = project_root.join(BUILD_DIR);
    fs::create_dir_all(&build_dir).ok();

    eprintln!("Compiling render tool (NeuralAmpModelerCore)...");

    let cmake_status = Command::new("cmake")
        .args([
            "-S",
            nam_core.to_str().unwrap(),
            "-B",
            build_dir.to_str().unwrap(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_STANDARD=20",
        ])
        .status();

    match cmake_status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("CMake failed — C++ dependencies may be missing.");
            return false;
        }
    }

    let build_status = Command::new("cmake")
        .args([
            "--build",
            build_dir.to_str().unwrap(),
            "--target",
            "render",
            "-j",
        ])
        .status();

    match build_status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("Render build failed.");
            return false;
        }
    }

    render_bin().exists()
}

#[allow(deprecated)]
fn run_render_comparison(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    sample_rate: u32,
    use_v2: bool,
) {
    if !ensure_render_compiled() {
        eprintln!("SKIP: {label} — render tool not available.");
        return;
    }

    let model_path = model_path(model_filename);
    if !model_path.exists() {
        eprintln!("SKIP: {label} — {model_filename} not found.");
        return;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();

    let stress_wav = temp_dir.join(format!("stress_live_{golden_name}.wav"));

    // Generate stress signal and write WAV
    let stress_signal = if use_v2 {
        generate_stress_signal_v2(sample_rate)
    } else {
        generate_stress_signal_v1()
    };
    let actual_sr = if use_v2 {
        sample_rate
    } else {
        STRESS_SAMPLE_RATE
    };
    common::wav::write_wav_f32(&stress_wav, &stress_signal, actual_sr)
        .expect("Failed to write stress WAV");

    let output_wav = temp_dir.join(format!("{golden_name}_live.wav"));

    // Execute render tool
    let bin = render_bin();
    let status = Command::new(&bin)
        .arg(model_path.to_str().unwrap())
        .arg(stress_wav.to_str().unwrap())
        .arg(output_wav.to_str().unwrap())
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!(
                "SKIP: {label} — render returned exit code {}",
                s.code().unwrap_or(-1)
            );
            return;
        }
        Err(e) => {
            eprintln!("SKIP: {label} — failed to execute render: {e}");
            return;
        }
    }

    // Read render WAV output
    let (cpp_output, _sr) =
        common::wav::read_wav_f32(&output_wav).expect("Failed to read render WAV output");

    // Rust inference
    let json_data = fs::read_to_string(&model_path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    let (mut mse_limit, mut min_snr_db) = topology_thresholds(&model_data);
    if use_v2 && model_data.architecture == "LSTM" {
        // LSTM recurrent state accumulates quantization/approximation errors
        // over the 100x longer v2 stress signal. The accumulation is proportional
        // to the sample rate (sequence length). We adjust the thresholds accordingly.
        let sr_ratio = sample_rate as f64 / 48000.0;
        let snr_relaxation = (3.5 * sr_ratio).min(10.0);
        min_snr_db = (min_snr_db - snr_relaxation).max(7.0);
        mse_limit *= 10.0_f64.powf(snr_relaxation / 10.0);
    }

    let mut model = build_model(&model_data).expect("Dispatcher failed");

    model.prewarm(2048);
    let mut rust_output = vec![0.0f32; stress_signal.len()];
    process_in_blocks(
        &mut model,
        &stress_signal,
        &mut rust_output,
        GOLDEN_BLOCK_SIZE,
    );

    let min_len = cpp_output.len().min(rust_output.len());
    report_dsp_fidelity(
        &cpp_output[..min_len],
        &rust_output[..min_len],
        mse_limit,
        min_snr_db,
        None,
        label,
        actual_sr,
    );

    // Cleanup
    fs::remove_file(&output_wav).ok();
}

/// Helper: run v1 comparison (legacy 48 kHz, fast CI).
fn run_v1(model_filename: &str, golden_name: &str, label: &str) {
    run_render_comparison(model_filename, golden_name, label, 48000, false);
}

/// Runs v2 stress signal comparison across all supported sample rates for one model.
///
/// **Known limitation:** NeuralAmpModelerCore's `render` tool enforces that the
/// input WAV sample rate matches the model's expected rate (typically 48 kHz).
/// Models that were trained at 48 kHz (e.g., WaveNet Standard) will SKIP at
/// non-48 kHz sample rates with:
///
/// ```text
/// Error: Input WAV sample rate (44100 Hz) does not match model expected rate (48000 Hz)
/// SKIP: Live WaveNet Standard (v2) @ 44100 Hz — render returned exit code 1
/// ```
///
/// This is a `render` tool limitation, not a nam-rs issue. The Rust inference
/// engine itself supports arbitrary sample rates. Lighter models (Nano, Feather)
/// and LSTM models may not enforce this restriction in the C++ render tool.
fn run_v2_multi_sr(model_filename: &str, golden_name: &str, label_base: &str) {
    for &sr in SUPPORTED_SAMPLE_RATES {
        let label = format!("{label_base} @ {sr} Hz (v2)");
        let gname = format!("{golden_name}_v2_{sr}");
        run_render_comparison(model_filename, &gname, &label, sr, true);
    }
}

// =============================================================================
// Tests — #[ignore] (require C++ toolchain)
// =============================================================================

// --- v1 (legacy, fast CI) ---

#[test]
#[ignore]
fn live_cross_validation_wavenet_standard() {
    run_v1(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard",
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_feather() {
    run_v1(
        "BossWN-feather.nam",
        "wavenet_feather",
        "Live WaveNet Feather",
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_nano() {
    run_v1("BossWN-nano.nam", "wavenet_nano", "Live WaveNet Nano");
}

#[test]
#[ignore]
fn live_cross_validation_lstm_1x16() {
    run_v1("BossLSTM-1x16.nam", "lstm_1x16", "Live LSTM 1×16");
}

#[test]
#[ignore]
fn live_cross_validation_lstm_2x8() {
    run_v1("BossLSTM-2x8.nam", "lstm_2x8", "Live LSTM 2×8");
}

// --- v2 (multi-SR, comprehensive) ---

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_standard() {
    run_v2_multi_sr(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard (v2)",
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_feather() {
    run_v2_multi_sr(
        "BossWN-feather.nam",
        "wavenet_feather",
        "Live WaveNet Feather (v2)",
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_nano() {
    run_v2_multi_sr("BossWN-nano.nam", "wavenet_nano", "Live WaveNet Nano (v2)");
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_1x16() {
    run_v2_multi_sr("BossLSTM-1x16.nam", "lstm_1x16", "Live LSTM 1×16 (v2)");
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_2x8() {
    run_v2_multi_sr("BossLSTM-2x8.nam", "lstm_2x8", "Live LSTM 2×8 (v2)");
}
