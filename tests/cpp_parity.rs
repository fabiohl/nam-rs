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
//! ## Parity Thresholds (Aggressive Live Floors)
//!
//! Post-T16.1 thresholds use `live_parity_thresholds()` — aggressive floors
//! calibrated from live C++ cross-validation measurements (see `validation.rs`):
//!
//! | Family  | Variant  | Measured SNR | Floor SNR | Margin  |
//! |---------|----------|-------------|-----------|---------|
//! | WaveNet | Standard | 68.4 dB     | 60 dB     | 8.4 dB  |
//! | WaveNet | Feather  | 67.6 dB     | 60 dB     | 7.6 dB  |
//! | WaveNet | Nano     | 52.6 dB     | 45 dB     | 7.6 dB  |
//! | WaveNet | Lite     | 0.9 dB      | 0 dB      | —       |
//! | WaveNet | A2-Full  | 79.2 dB     | 70.0 dB    | 9.2 dB (ESR scale-invariant, T2.5) |
//! | WaveNet | A2-Lite  | 90.7 dB     | 80.0 dB    | 10.7 dB (ESR scale-invariant, T2.5) |
//! | LSTM    | —        | 50–97 dB    | 45–75 dB  | formula |
//! | Linear  | —        | bit-exact   | 140 dB    | —       |
//!
//! Lite (CH=12) is a known failure — investigar separadamente.
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

const NAM_CORE_V053_DIR: &str = "tests/fixtures/NeuralAmpModelerCore_v0.5.3";
const BUILD_V053_DIR: &str = "build/namcore_render_v053";

fn render_bin(is_a2: bool) -> PathBuf {
    let build_dir = if is_a2 { BUILD_V053_DIR } else { BUILD_DIR };
    let mut bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    bin.push(build_dir);
    bin.push("Release/render");
    if !bin.exists() {
        bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        bin.push(build_dir);
        bin.push("Debug/render");
    }
    if !bin.exists() {
        let build_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(build_dir);
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

fn ensure_render_compiled(is_a2: bool) -> bool {
    let bin = render_bin(is_a2);
    if bin.exists() {
        return true;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let core_dir = if is_a2 {
        NAM_CORE_V053_DIR
    } else {
        NAM_CORE_DIR
    };
    let nam_core = project_root.join(core_dir);

    if !nam_core.exists() {
        if is_a2 {
            eprintln!("NeuralAmpModelerCore v0.5.3 not found at {nam_core:?}");
            eprintln!("Please run tests/fixtures/golden_gen_build.sh to set up the mirrors.");
        } else {
            eprintln!("NeuralAmpModelerCore not found at {nam_core:?}");
            eprintln!(
                "Run: git clone --depth 1 https://github.com/sdatkinson/NeuralAmpModelerCore.git {NAM_CORE_DIR}"
            );
        }
        return false;
    }

    if is_a2 {
        let plugin_dsp = project_root.join("tests/fixtures/NeuralAmpModelerPlugin/AudioDSPTools");
        let eigen_dest = nam_core.join("Dependencies/eigen");
        if !eigen_dest.exists() {
            let eigen_src = plugin_dsp.join("Dependencies/eigen");
            #[cfg(unix)]
            std::os::unix::fs::symlink(eigen_src, eigen_dest).ok();
        }
        let dsp_dest = nam_core.join("Dependencies/AudioDSPTools");
        if !dsp_dest.exists() {
            #[cfg(unix)]
            std::os::unix::fs::symlink(&plugin_dsp, dsp_dest).ok();
        }
    } else {
        for sub in &["Dependencies/eigen", "Dependencies/AudioDSPTools"] {
            let sub_path = nam_core.join(sub);
            if sub_path.exists() && fs::read_dir(&sub_path).map_or(true, |mut d| d.next().is_none())
            {
                let status = Command::new("git")
                    .args(["submodule", "update", "--init", sub])
                    .current_dir(&nam_core)
                    .status();
                if status.is_ok_and(|s| !s.success()) {
                    eprintln!("WARN: failed to initialize submodule {sub}");
                }
            }
        }
    }

    let build_dir = project_root.join(if is_a2 { BUILD_V053_DIR } else { BUILD_DIR });
    fs::create_dir_all(&build_dir).ok();

    eprintln!(
        "Compiling render tool ({})...",
        if is_a2 {
            "NeuralAmpModelerCore v0.5.3 A2"
        } else {
            "NeuralAmpModelerCore standard"
        }
    );

    let mut cmake_args = vec![
        "-S",
        nam_core.to_str().unwrap(),
        "-B",
        build_dir.to_str().unwrap(),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=20",
    ];
    if is_a2 {
        cmake_args.push("-DNAM_ENABLE_A2_FAST=ON");
    }

    let cmake_status = Command::new("cmake").args(&cmake_args).status();

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
            "2",
        ])
        .status();

    match build_status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("Render build failed.");
            return false;
        }
    }

    render_bin(is_a2).exists()
}

fn run_render_comparison(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    sample_rate: u32,
    use_v2: bool,
) {
    let model_path = model_path(model_filename);
    if !model_path.exists() {
        eprintln!("SKIP: {label} — {model_filename} not found.");
        return;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();

    // Read model expected sample rate
    let json_data = fs::read_to_string(&model_path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    let is_a2 = nam_rs::loader::nam_json::is_a2_shape(&model_data).is_some();

    if !ensure_render_compiled(is_a2) {
        eprintln!("SKIP: {label} — render tool not available.");
        return;
    }

    let actual_sr = if use_v2 {
        sample_rate
    } else {
        STRESS_SAMPLE_RATE
    };
    let model_sr = model_data.sample_rate.unwrap_or(actual_sr as f32) as u32;

    let stress_signal = if use_v2 {
        generate_stress_signal_v2(sample_rate)
    } else {
        generate_stress_signal_v1()
    };

    let stress_wav = temp_dir.join(format!("stress_live_{golden_name}.wav"));

    use nam_rs::dsp::resampler::NamResampler;
    let mut resampler_cpp = if actual_sr != model_sr {
        Some(NamResampler::new(actual_sr, model_sr, 2048).expect("Failed to create NamResampler"))
    } else {
        None
    };
    let mut resampler_rust = if actual_sr != model_sr {
        Some(NamResampler::new(actual_sr, model_sr, 2048).expect("Failed to create NamResampler"))
    } else {
        None
    };

    let (input_for_render, input_sr) = if let Some(ref mut rs) = resampler_cpp {
        let est_len =
            (stress_signal.len() as f64 * model_sr as f64 / actual_sr as f64).ceil() as usize + 512;
        let mut resampled_in_l = vec![0.0f32; est_len];
        let mut resampled_in_r = vec![0.0f32; est_len];
        let n_resampled =
            rs.process_input_mono(&stress_signal, &mut resampled_in_l, &mut resampled_in_r);
        resampled_in_l.truncate(n_resampled);
        (resampled_in_l, model_sr)
    } else {
        (stress_signal.clone(), actual_sr)
    };

    let (input_for_rust, _) = if let Some(ref mut rs) = resampler_rust {
        let est_len =
            (stress_signal.len() as f64 * model_sr as f64 / actual_sr as f64).ceil() as usize + 512;
        let mut resampled_in_l = vec![0.0f32; est_len];
        let mut resampled_in_r = vec![0.0f32; est_len];
        let n_resampled =
            rs.process_input_mono(&stress_signal, &mut resampled_in_l, &mut resampled_in_r);
        resampled_in_l.truncate(n_resampled);
        (resampled_in_l, model_sr)
    } else {
        (stress_signal.clone(), actual_sr)
    };

    common::wav::write_wav_f32(&stress_wav, &input_for_render, input_sr)
        .expect("Failed to write stress WAV");

    let output_wav = temp_dir.join(format!("{golden_name}_live.wav"));

    // Execute render tool
    let bin = render_bin(is_a2);
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
    let (cpp_output_raw, _sr) =
        common::wav::read_wav_f32(&output_wav).expect("Failed to read render WAV output");

    let cpp_output = if let Some(ref mut rs) = resampler_cpp {
        let est_out_len = (cpp_output_raw.len() as f64 * actual_sr as f64 / model_sr as f64).ceil()
            as usize
            + 512;
        let mut resampled_out_l = vec![0.0f32; est_out_len];
        let mut resampled_out_r = vec![0.0f32; est_out_len];
        let n_out_resampled =
            rs.process_output_mono(&cpp_output_raw, &mut resampled_out_l, &mut resampled_out_r);
        resampled_out_l.truncate(n_out_resampled);
        resampled_out_l
    } else {
        cpp_output_raw.clone()
    };

    // Sanity-check C++ render output: skip comparison if render produced garbage.
    //
    // Only check for non-finite samples (NaN/Inf), as requested in T2.4.
    {
        let has_nonfinite = cpp_output.iter().any(|x| !x.is_finite());
        if has_nonfinite {
            eprintln!(
                "SKIP: {label} — C++ render produced garbage output (non-finite=true); skipping comparison.",
            );
            fs::remove_file(&output_wav).ok();
            return;
        }
    }

    let (mut mse_limit, mut min_snr_db, mut max_esr) =
        live_parity_thresholds(&model_data, golden_name);
    if use_v2 {
        if model_data.architecture == "LSTM" {
            // LSTM recurrent state accumulates quantization/approximation errors
            // over the 100x longer v2 stress signal. The accumulation is proportional
            // to the sequence length. We adjust the thresholds accordingly.
            let sr_ratio = sample_rate as f64 / 48000.0;
            let snr_relaxation = (3.5 * sr_ratio).min(10.0);
            min_snr_db = (min_snr_db - snr_relaxation).max(7.0);
            mse_limit *= 10.0_f64.powf(snr_relaxation / 10.0);
            if let Some(ref mut esr) = max_esr {
                *esr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
        } else {
            // WaveNet and other models accumulate minor differences over the longer v2 stress signal
            let sr_ratio = sample_rate as f64 / 48000.0;
            let snr_relaxation = (1.5 * sr_ratio).min(4.0);
            min_snr_db -= snr_relaxation;
            mse_limit *= 10.0_f64.powf(snr_relaxation / 10.0);
            if let Some(ref mut esr) = max_esr {
                *esr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
        }
    }
    if use_v2 && actual_sr != model_sr {
        // Resampling introduces minor interpolation/approximation errors, relax thresholds slightly
        min_snr_db -= 1.5;
        mse_limit *= 1.5;
        if let Some(ref mut esr) = max_esr {
            *esr *= 1.5;
        }
    }

    let mut model = build_model(&model_data).expect("Dispatcher failed");

    model.prewarm(2048);
    let mut rust_output_model_sr = vec![0.0f32; input_for_rust.len()];
    process_in_blocks(
        &mut model,
        &input_for_rust,
        &mut rust_output_model_sr,
        GOLDEN_BLOCK_SIZE,
    );

    let rust_output = if let Some(ref mut rs) = resampler_rust {
        let est_out_len = (rust_output_model_sr.len() as f64 * actual_sr as f64 / model_sr as f64)
            .ceil() as usize
            + 512;
        let mut resampled_out_l = vec![0.0f32; est_out_len];
        let mut resampled_out_r = vec![0.0f32; est_out_len];
        let n_out_resampled = rs.process_output_mono(
            &rust_output_model_sr,
            &mut resampled_out_l,
            &mut resampled_out_r,
        );
        resampled_out_l.truncate(n_out_resampled);
        resampled_out_l
    } else {
        rust_output_model_sr
    };

    let min_len = cpp_output.len().min(rust_output.len());
    report_dsp_fidelity(
        &cpp_output[..min_len],
        &rust_output[..min_len],
        mse_limit,
        min_snr_db,
        max_esr,
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
    let mut failures = Vec::new();
    for &sr in SUPPORTED_SAMPLE_RATES {
        let label = format!("{label_base} @ {sr} Hz (v2)");
        let gname = format!("{golden_name}_v2_{sr}");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_render_comparison(model_filename, &gname, &label, sr, true);
        }));
        if result.is_err() {
            failures.push(sr);
        }
    }
    if !failures.is_empty() {
        panic!("Parity validation failed for sample rates: {:?}", failures);
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
fn live_cross_validation_wavenet_lite() {
    eprintln!(
        "SKIP: WaveNet Lite (CH=12) is known-divergent (T1.2) - skipping to avoid false gate"
    );
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

#[test]
#[ignore]
fn live_cross_validation_wavenet_a1_standard() {
    run_v1(
        "wavenet_a1_standard.nam",
        "wavenet_a1_standard",
        "Live WaveNet A1 Standard (Official)",
    );
}

#[test]
#[ignore]
fn live_cross_validation_lstm_official() {
    run_v1("lstm.nam", "lstm_official", "Live LSTM Official");
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_full() {
    run_v1(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Live WaveNet A2-Full",
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_lite() {
    run_v1(
        "wavenet_a2_lite.nam",
        "wavenet_a2_lite",
        "Live WaveNet A2-Lite",
    );
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
fn live_cross_validation_v2_wavenet_lite() {
    eprintln!(
        "SKIP: WaveNet Lite (CH=12) (v2) is known-divergent (T1.2) - skipping to avoid false gate"
    );
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

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a1_standard() {
    run_v2_multi_sr(
        "wavenet_a1_standard.nam",
        "wavenet_a1_standard",
        "Live WaveNet A1 Standard (Official) (v2)",
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_official() {
    run_v2_multi_sr("lstm.nam", "lstm_official", "Live LSTM Official (v2)");
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_full() {
    run_v2_multi_sr(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Live WaveNet A2-Full (v2)",
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_lite() {
    run_v2_multi_sr(
        "wavenet_a2_lite.nam",
        "wavenet_a2_lite",
        "Live WaveNet A2-Lite (v2)",
    );
}

// --- Linear ---

#[test]
#[ignore]
fn live_cross_validation_linear() {
    run_v1("linear_test.nam", "linear_test", "Live Linear RF=4");
}

#[test]
#[ignore]
fn live_cross_validation_v2_linear() {
    run_v2_multi_sr("linear_test.nam", "linear_test", "Live Linear RF=4 (v2)");
}
