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
//! Post-T-HF6.6 thresholds use `live_parity_thresholds()` — aggressive floors
//! calibrated from live C++ cross-validation measurements (f32-exact, 2026-06-18):
//!
//! | Family  | Variant  | Measured SNR | Floor SNR | Margin  |
//! |---------|----------|-------------|-----------|---------|
//! | WaveNet | Standard | 134.6 dB    | 105 dB    | 29.6 dB |
//! | WaveNet | Feather  | 133.1 dB    | 100 dB    | 33.1 dB |
//! | WaveNet | Nano     | 132.0 dB    | 95 dB     | 37.0 dB |
//! | WaveNet | Lite     | 117.4 dB    | 100 dB    | 17.4 dB |
//! | WaveNet | A2-Full  | 79.2 dB     | 70.0 dB   | 9.2 dB  |
//! | WaveNet | A2-Lite  | 90.7 dB     | 80.0 dB   | 10.7 dB |
//! | LSTM    | —        | 50–97 dB    | 45–75 dB  | formula |
//! | Linear  | —        | bit-exact   | 140 dB    | —       |
//!
//! Lite (CH=12) P1 resolved (T1.2 ringbuffer alignment fix) — now matches C++.
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
    let build_dir = BUILD_DIR;
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

fn ensure_render_compiled() -> bool {
    // Serialize the C++ render build across parallel test threads. All
    // `live_cross_validation_*` tests share the same CMake build directory
    // (`build/namcore_render`); on a cold run they would otherwise race to
    // invoke `cmake` concurrently, corrupting the CMake cache and producing
    // spurious "CMAKE_CXX_COMPILER not set" failures. The first thread builds;
    // the rest wait and then find the binary already present.
    use std::sync::Mutex;
    static BUILD_LOCK: Mutex<()> = Mutex::new(());
    let _guard = BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let bin = render_bin();
    if bin.exists() {
        return true;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let nam_core = project_root.join(NAM_CORE_DIR);

    if !nam_core.exists() {
        eprintln!(
            "SKIP: NeuralAmpModelerCore not found at {nam_core:?}.\n\
             Run './tests/fixtures/golden_gen_build.sh' to set up mirrors and generate golden vectors."
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

    eprintln!("Compiling render tool (v0.5.3 + A2-fast)...");

    let cmake_args = vec![
        "-S",
        nam_core.to_str().unwrap(),
        "-B",
        build_dir.to_str().unwrap(),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=20",
        "-DNAM_ENABLE_A2_FAST=ON",
    ];

    let cmake_status = Command::new("cmake").args(&cmake_args).status();

    match cmake_status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!(
                "SKIP: CMake failed — C++ build dependencies missing.\n\
                 Install cmake and a C++20 compiler (g++ or clang++), then re-run."
            );
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
            eprintln!("SKIP: Render build failed — check build logs in {build_dir:?}.");
            return false;
        }
    }

    if !render_bin().exists() {
        eprintln!("SKIP: Render binary not found after build — expected at {bin:?}");
        return false;
    }

    true
}

fn run_render_comparison(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    sample_rate: u32,
    use_v2: bool,
    check_lufs_gate: bool,
) {
    let model_path = model_path(model_filename);
    if !model_path.exists() {
        eprintln!("SKIP: {label} — model file {model_filename} not found at {model_path:?}.");
        return;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();

    // Read model expected sample rate
    let json_data = fs::read_to_string(&model_path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    if !ensure_render_compiled() {
        eprintln!("SKIP: {label} — C++ render tool not available, skipping cross-validation.");
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

    // Execute render tool — capture stdout/stderr to prevent interleaving
    // with the Rust test harness output (F-1 fix).
    let bin = render_bin();
    let output = Command::new(&bin)
        .arg(model_path.to_str().unwrap())
        .arg(stress_wav.to_str().unwrap())
        .arg(output_wav.to_str().unwrap())
        .output();

    match output {
        Ok(o) if o.status.success() => { /* silence on success */ }
        Ok(o) => {
            let stderr_msg = String::from_utf8_lossy(&o.stderr);
            let stdout_msg = if o.stdout.is_empty() {
                String::new()
            } else {
                format!(
                    "--- render stdout ---\n{}\n",
                    String::from_utf8_lossy(&o.stdout)
                )
            };
            eprintln!(
                "SKIP: {label} — render returned exit code {}\n\
                 --- render stderr ---\n{stderr_msg}{stdout_msg}",
                o.status.code().unwrap_or(-1)
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

    let (mut mse_limit, mut min_snr_db, mut max_esr, mrstft_max) =
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

    if use_v2 {
        // Tarefa 3.2 (F-2): Impor teto absoluto à relaxação.
        // Após toda a relaxação (v2 + resampling), os gates hard não podem
        // afrouxar além de limites absolutos, para que "passar" continue
        // significando paridade, não apenas "não totalmente quebrado".
        // - ESR nunca acima do baseline A1-Std (6.23e-3)
        // - SNR nunca abaixo de 5.0 dB (piso absoluto)
        // Casos de 88.2/96/192 kHz que falharem sob este teto viram achados
        // encaminhados à Tarefa 3.3 — não mascarar.
        const ABSOLUTE_ESR_CAP: f64 = nam_rs::testing::perceptual::A2ESR_A1_STANDARD_MEDIAN;
        const ABSOLUTE_SNR_FLOOR: f64 = 5.0;

        min_snr_db = min_snr_db.max(ABSOLUTE_SNR_FLOOR);
        if let Some(ref mut esr) = max_esr
            && *esr > ABSOLUTE_ESR_CAP
        {
            let scale_back = ABSOLUTE_ESR_CAP / *esr;
            *esr = ABSOLUTE_ESR_CAP;
            mse_limit *= scale_back;
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
    let cpp_slice = &cpp_output[..min_len];
    let rust_slice = &rust_output[..min_len];
    if check_lufs_gate {
        report_dsp_fidelity(
            cpp_slice, rust_slice, mse_limit, min_snr_db, max_esr, mrstft_max, label, actual_sr,
        );
    } else {
        report_dsp_fidelity_no_lufs(
            cpp_slice, rust_slice, mse_limit, min_snr_db, max_esr, mrstft_max, label, actual_sr,
        );
    }

    // Cleanup
    fs::remove_file(&output_wav).ok();
}

/// Helper: run v1 comparison (legacy 48 kHz, fast CI).
fn run_v1(model_filename: &str, golden_name: &str, label: &str, check_lufs_gate: bool) {
    run_render_comparison(
        model_filename,
        golden_name,
        label,
        48000,
        false,
        check_lufs_gate,
    );
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
fn run_v2_multi_sr(
    model_filename: &str,
    golden_name: &str,
    label_base: &str,
    check_lufs_gate: bool,
) {
    let mut failures = Vec::new();
    for &sr in SUPPORTED_SAMPLE_RATES {
        let label = format!("{label_base} @ {sr} Hz (v2)");
        let gname = format!("{golden_name}_v2_{sr}");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_render_comparison(model_filename, &gname, &label, sr, true, check_lufs_gate);
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
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_feather() {
    run_v1(
        "BossWN-feather.nam",
        "wavenet_feather",
        "Live WaveNet Feather",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_nano() {
    run_v1("BossWN-nano.nam", "wavenet_nano", "Live WaveNet Nano", true);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_lite() {
    run_v1(
        "EVH-5150-Lite.nam",
        "wavenet_lite",
        "Live WaveNet Lite",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_lstm_1x16() {
    run_v1("BossLSTM-1x16.nam", "lstm_1x16", "Live LSTM 1×16", true);
}

#[test]
#[ignore]
fn live_cross_validation_lstm_2x8() {
    run_v1("BossLSTM-2x8.nam", "lstm_2x8", "Live LSTM 2×8", true);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a1_standard() {
    run_v1(
        "wavenet_a1_standard.nam",
        "wavenet_a1_standard",
        "Live WaveNet A1 Standard (Official)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_lstm_official() {
    run_v1("lstm.nam", "lstm_official", "Live LSTM Official", true);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_full() {
    run_v1(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Live WaveNet A2-Full",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_lite() {
    run_v1(
        "wavenet_a2_lite.nam",
        "wavenet_a2_lite",
        "Live WaveNet A2-Lite",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_condition_dsp() {
    run_v1(
        "wavenet_condition_dsp.nam",
        "wavenet_condition_dsp",
        "Live WaveNet Condition DSP",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_a2_example_slimmable() {
    run_v1(
        "a2_example.nam",
        "a2_example",
        "Live SlimmableContainer A2 Example (CH=3→6)",
        true,
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
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_feather() {
    run_v2_multi_sr(
        "BossWN-feather.nam",
        "wavenet_feather",
        "Live WaveNet Feather (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_nano() {
    run_v2_multi_sr(
        "BossWN-nano.nam",
        "wavenet_nano",
        "Live WaveNet Nano (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_lite() {
    run_v2_multi_sr(
        "EVH-5150-Lite.nam",
        "wavenet_lite",
        "Live WaveNet Lite (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_1x16() {
    run_v2_multi_sr(
        "BossLSTM-1x16.nam",
        "lstm_1x16",
        "Live LSTM 1×16 (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_2x8() {
    run_v2_multi_sr("BossLSTM-2x8.nam", "lstm_2x8", "Live LSTM 2×8 (v2)", true);
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a1_standard() {
    run_v2_multi_sr(
        "wavenet_a1_standard.nam",
        "wavenet_a1_standard",
        "Live WaveNet A1 Standard (Official) (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_official() {
    run_v2_multi_sr("lstm.nam", "lstm_official", "Live LSTM Official (v2)", true);
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_full() {
    run_v2_multi_sr(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Live WaveNet A2-Full (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_app_evh() {
    run_v2_multi_sr(
        "APP-EVH-Stealth100-Dialled-xSTD.nam",
        "wavenet_app_evh",
        "Live APP EVH Stealth 100 (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_boss_bd2() {
    run_v2_multi_sr(
        "Boss BD-2 H2O Mod T-12_00 G-12_00.nam",
        "wavenet_boss_bd2",
        "Live Boss BD-2 H2O Mod (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_slammin_marshall() {
    run_v2_multi_sr(
        "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam",
        "wavenet_slammin_marshall",
        "Live SLAMMIN MARSHALL JTM 45 (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_lite() {
    run_v2_multi_sr(
        "wavenet_a2_lite.nam",
        "wavenet_a2_lite",
        "Live WaveNet A2-Lite (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_condition_dsp() {
    run_v2_multi_sr(
        "wavenet_condition_dsp.nam",
        "wavenet_condition_dsp",
        "Live WaveNet Condition DSP (v2)",
        true,
    );
}

// --- Dynamic Models (Sprint B.2.2) ---

#[test]
#[ignore]
fn live_cross_validation_wavenet_dyn() {
    run_v1(
        "wavenet_dyn_free.nam",
        "wavenet_dyn_free",
        "Live WaveNetDyn Free-Shape",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_lstm_dyn() {
    run_v1(
        "lstm_dyn_test.nam",
        "lstm_dyn_test",
        "Live LSTM-Dyn 1×7",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_dyn() {
    run_v2_multi_sr(
        "wavenet_dyn_free.nam",
        "wavenet_dyn_free",
        "Live WaveNetDyn Free-Shape (v2)",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_dyn() {
    run_v2_multi_sr(
        "lstm_dyn_test.nam",
        "lstm_dyn_test",
        "Live LSTM-Dyn 1×7 (v2)",
        false,
    );
}

// --- Linear ---

#[test]
#[ignore]
fn live_cross_validation_linear() {
    run_v1("linear_test.nam", "linear_test", "Live Linear RF=4", true);
}

#[test]
#[ignore]
fn live_cross_validation_v2_linear() {
    run_v2_multi_sr(
        "linear_test.nam",
        "linear_test",
        "Live Linear RF=4 (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_nondist_models() {
    let mut nondist_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    nondist_path.push("tests/fixtures/models-nondist");

    if !nondist_path.exists() {
        println!(
            "SKIP: Non-distributable models directory {:?} not found.",
            nondist_path
        );
        return;
    }

    let models: Vec<_> = discovery::find_models_in_dir(&nondist_path)
        .into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n != "manifest.json"))
        .collect();

    if models.is_empty() {
        println!("SKIP: No models found in {:?}", nondist_path);
        return;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();
    if !ensure_render_compiled() {
        println!("SKIP: C++ render tool not available, skipping nondist cross-validation.");
        return;
    }

    for model_path in models {
        let filename = model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        println!("Live C++ cross-validating nondist model: {}", filename);

        let mut dest_path = project_root.join("tests/fixtures/models");
        dest_path.push(&filename);

        let copied = if !dest_path.exists() {
            fs::copy(&model_path, &dest_path).is_ok()
        } else {
            false
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_v1(
                &filename,
                &format!("nondist_{}", filename.replace('.', "_")),
                &format!("Live Nondist {}", filename),
                true,
            );
        }));

        if copied {
            fs::remove_file(&dest_path).ok();
        }

        if let Err(e) = result {
            std::panic::resume_unwind(e);
        }
    }
}
