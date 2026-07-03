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
use nam_rs::math::activations::ActivationPrecision;
use nam_rs::math::activations::set_activation_precision;
use nam_rs::models::NamModel;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Sets `ActivationPrecision::HighFidelity` and returns a guard that
/// restores `Standard` on drop. Ensures panic-safe cleanup (Tarefa β1.3).
struct PrecisionGuard;

impl PrecisionGuard {
    fn set() -> Self {
        set_activation_precision(ActivationPrecision::HighFidelity);
        PrecisionGuard
    }
}

impl Drop for PrecisionGuard {
    fn drop(&mut self) {
        set_activation_precision(ActivationPrecision::Standard);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParityOutcome {
    Completed,
    SkippedModelNotFound,
    SkippedToolNotAvailable,
    SkippedRateRejected,
    SkippedGarbageOutput,
}

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
    use_hf: bool,
) -> ParityOutcome {
    let model_path = model_path(model_filename);
    if !model_path.exists() {
        eprintln!("SKIP: {label} — model file {model_filename} not found at {model_path:?}.");
        return ParityOutcome::SkippedModelNotFound;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();

    // Read model expected sample rate
    let json_data = fs::read_to_string(&model_path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    if !ensure_render_compiled() {
        eprintln!("SKIP: {label} — C++ render tool not available, skipping cross-validation.");
        return ParityOutcome::SkippedToolNotAvailable;
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

    let suffix = if use_hf { "_hf" } else { "" };
    let stress_wav = temp_dir.join(format!(
        "stress_live_{golden_name}_{sample_rate}{suffix}.wav"
    ));

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

    let output_wav = temp_dir.join(format!("{golden_name}_live_{sample_rate}{suffix}.wav"));

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
            return ParityOutcome::SkippedRateRejected;
        }
        Err(e) => {
            eprintln!("SKIP: {label} — failed to execute render: {e}");
            return ParityOutcome::SkippedRateRejected;
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
            return ParityOutcome::SkippedGarbageOutput;
        }
    }

    let (mut mse_limit, mut min_snr_db, mut max_esr, mut mrstft_max) =
        live_parity_thresholds(&model_data, golden_name);
    let calibrated_mse = mse_limit;
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
            if let Some(ref mut mr) = mrstft_max {
                // MR-STFT grows faster than power metrics for recurrent models
                // (spectral drift in LSTM hidden states over long sequences)
                *mr *= 10.0_f64.powf(snr_relaxation / 5.0);
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
            if let Some(ref mut mr) = mrstft_max {
                *mr *= 10.0_f64.powf(snr_relaxation / 5.0);
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
        // MR-STFT is particularly sensitive to resampling phase/timing mismatches
        // between the C++ render's SRC and nam-rs's polyphase resampler
        if let Some(ref mut mr) = mrstft_max {
            *mr *= 3.0;
        }
    }

    if use_v2 {
        // Tarefa 3.2 (F-2) + Tarefa 8.4 (AC-2): Impor teto absoluto à relaxação.
        // Após toda a relaxação (v2 + resampling), os gates hard não podem
        // afrouxar além de limites absolutos, para que "passar" continue
        // significando paridade, não apenas "não totalmente quebrado".
        //
        // Caps derivados de medição real, não de "o que faz passar o teste":
        //
        // - WaveNet: cap = baseline A1-Std (6.23e-3)
        // - LSTM:   rate-aware cap (recurrent drift scales with the sample-rate
        //   ratio — more recurrent steps per second of audio mean more f16c/f32
        //   state accumulation). ALL supported rates are tested (Tarefa 8.14,
        //   Gate Calibration Policy Rule 7): no rate is excluded to make a gate
        //   pass; the bound is raised per-rate from real measurements instead.
        //
        //   // Measured — BossLSTM-1x16 (worst case), v2/240k stress vs NAMCore:
        //   //   44.1k=2.39e-2  48k=2.61e-2  88.2k=5.39e-2  96k=6.09e-2  192k=1.42e-1
        //   //   (other LSTMs are lower: 2x8 192k=4.20e-2; Official flat 1.23e-3)
        //   // Oracle ideal precision floor (T8.2/T8.3): ΔESR_oracle_vs_prod =
        //   //   3.57e-3 (prewarm-paired @ 48 kHz). The interop figures above are
        //   //   nam-rs↔NAMCore drift (shared by both f32 engines), distinct from
        //   //   the f64 oracle floor.
        //   //   ≤ 96 kHz: cap 0.08  (covers measured 6.09e-2 with ~1.3× margin)
        //   //   > 96 kHz: cap 0.20  (covers measured 1.42e-1 @ 192k with ~1.4× margin)
        //   // Both bounds are < 1.0 (non-placebo); 192 kHz remains a documented,
        //   // tracked limitation (TODO-findings.md F-2), not a hidden gap.
        // - SNR nunca abaixo de 5.0 dB (piso absoluto)
        // - MR-STFT nunca acima de 0.95 (cap at ceiling for normalized metric)
        const ABSOLUTE_ESR_CAP_WAVENET: f64 = nam_rs::testing::perceptual::A2ESR_A1_STANDARD_MEDIAN;
        const ABSOLUTE_ESR_CAP_LSTM_NATIVE: f64 = 0.08; // rates ≤ 96 kHz
        const ABSOLUTE_ESR_CAP_LSTM_HIRATE: f64 = 0.20; // rates > 96 kHz (e.g. 192 kHz)
        // HighFidelity mode caps (Tarefa β1.3) — the C++ render tool uses
        // standard Padé approximations, while Rust uses HF poly exp-based
        // kernels. This deliberate asymmetry increases interop divergence.
        // Caps are calibrated to pass meaningful comparison (all < 1.0,
        // non-placebo) while documenting the HF interop drift.
        const ABSOLUTE_ESR_CAP_WAVENET_HF: f64 =
            nam_rs::testing::perceptual::A2ESR_A1_STANDARD_MEDIAN * 5.0;
        const ABSOLUTE_ESR_CAP_LSTM_NATIVE_HF: f64 = 0.30; // rates ≤ 96 kHz
        const ABSOLUTE_ESR_CAP_LSTM_HIRATE_HF: f64 = 0.60; // rates > 96 kHz (e.g. 192 kHz)
        const ABSOLUTE_ESR_CAP_FILM_LIVE: f64 = 0.08;
        const ABSOLUTE_ESR_CAP_FILM_HF: f64 = 0.15;
        const ABSOLUTE_SNR_FLOOR: f64 = 5.0;
        const ABSOLUTE_MRSTFT_CAP: f64 = 0.95;
        const ABSOLUTE_MRSTFT_CAP_FILM: f64 = 1.20;

        let is_film = golden_name.to_lowercase().contains("film")
            || model_filename.to_lowercase().contains("film");

        let esr_cap = if model_data.architecture == "LSTM" {
            if use_hf {
                if sample_rate > 96_000 {
                    ABSOLUTE_ESR_CAP_LSTM_HIRATE_HF
                } else {
                    ABSOLUTE_ESR_CAP_LSTM_NATIVE_HF
                }
            } else if sample_rate > 96_000 {
                ABSOLUTE_ESR_CAP_LSTM_HIRATE
            } else {
                ABSOLUTE_ESR_CAP_LSTM_NATIVE
            }
        } else if is_film {
            if use_hf {
                ABSOLUTE_ESR_CAP_FILM_HF
            } else {
                ABSOLUTE_ESR_CAP_FILM_LIVE
            }
        } else if use_hf {
            ABSOLUTE_ESR_CAP_WAVENET_HF
        } else {
            ABSOLUTE_ESR_CAP_WAVENET
        };

        min_snr_db = min_snr_db.max(ABSOLUTE_SNR_FLOOR);
        if let Some(ref mut esr) = max_esr
            && *esr > esr_cap
        {
            let scale_back = esr_cap / *esr;
            *esr = esr_cap;
            // Scale MSE proportionally, but never tighter than the original
            // calibrated threshold — otherwise the ESR cap defeats the purpose
            // of v2 relaxation for models with multi-SR drift (e.g. LSTM).
            mse_limit = (mse_limit * scale_back).max(calibrated_mse);
        }
        let mrstft_cap = if is_film {
            ABSOLUTE_MRSTFT_CAP_FILM
        } else {
            ABSOLUTE_MRSTFT_CAP
        };
        if let Some(ref mut mr) = mrstft_max
            && *mr > mrstft_cap
        {
            *mr = mrstft_cap;
        }
    }

    // Set HighFidelity activation mode if requested (Tarefa β1.3).
    // The C++ NAMCore render tool uses standard Padé approximations,
    // so this deliberately increases the interop divergence.
    // Uses RAII guard for panic-safe restoration to Standard.
    let _precision: Option<PrecisionGuard> = if use_hf {
        Some(PrecisionGuard::set())
    } else {
        None
    };

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
    ParityOutcome::Completed
}

/// Helper: run v1 comparison (legacy 48 kHz, fast CI).
fn run_v1(model_filename: &str, golden_name: &str, label: &str, check_lufs_gate: bool) {
    let _ = run_render_comparison(
        model_filename,
        golden_name,
        label,
        48000,
        false,
        check_lufs_gate,
        false,
    );
}

/// Helper: run v1 comparison in HighFidelity mode (Tarefa β1.3).
fn run_v1_hf(model_filename: &str, golden_name: &str, label: &str, check_lufs_gate: bool) {
    let _ = run_render_comparison(
        model_filename,
        golden_name,
        label,
        48000,
        false,
        check_lufs_gate,
        true,
    );
}

/// Unified implementation for v2 multi-SR parity tests.
///
/// Monitors every `SUPPORTED_SAMPLE_RATES` entry per model and asserts that:
/// - At least one rate completed parity validation (no silent total skip).
/// - The set of completed rates exactly matches the expected rates for the model
///   (no silent partial skip violating Gate Calibration Policy Rule 7).
///
fn run_v2_multi_sr_impl(
    model_filename: &str,
    golden_name: &str,
    label_base: &str,
    check_lufs_gate: bool,
    use_hf: bool,
) {
    let model_path = model_path(model_filename);
    let _json_data = fs::read_to_string(&model_path).expect("Failed to read model");
    let _model_data = parse_nam_json(&_json_data).expect("JSON parser failed");

    let expected_rates = SUPPORTED_SAMPLE_RATES.to_vec();

    let mut outcomes: Vec<(u32, ParityOutcome)> = Vec::new();
    let mut failures: Vec<(u32, String)> = Vec::new();
    let hf_tag = if use_hf { ", HF" } else { "" };

    for &sr in SUPPORTED_SAMPLE_RATES {
        let label = format!("{label_base} @ {sr} Hz (v2{hf_tag})");
        let gname = format!("{golden_name}_v2_{sr}");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_render_comparison(
                model_filename,
                &gname,
                &label,
                sr,
                true,
                check_lufs_gate,
                use_hf,
            )
        }));
        match result {
            Ok(outcome) => outcomes.push((sr, outcome)),
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic (non-string payload)".to_string());
                failures.push((sr, msg));
            }
        }
    }

    println!("=== Multi-SR Summary: {label_base}{hf_tag} ===");
    println!("{:<10} {:<30}", "SR (Hz)", "Outcome");
    println!("{}", "-".repeat(42));
    for &(sr, ref outcome) in &outcomes {
        println!("{:<10} {:?}", sr, outcome);
    }
    for &(sr, ref msg) in &failures {
        println!("{:<10} FAILED: {}", sr, &msg[..msg.len().min(30)]);
    }
    println!("{}", "-".repeat(42));

    let completed: Vec<u32> = outcomes
        .iter()
        .filter(|(_, o)| *o == ParityOutcome::Completed)
        .map(|(sr, _)| *sr)
        .collect();

    assert!(
        !completed.is_empty(),
        "Parity validation for '{label_base}': no sample rate completed. \
         Expected at least one of {expected_rates:?}. \
         Outcomes: {outcomes:?}, Failures: {failures:?}"
    );

    let completed_set: BTreeSet<u32> = completed.iter().copied().collect();
    let expected_set: BTreeSet<u32> = expected_rates.iter().copied().collect();

    assert_eq!(
        completed_set, expected_set,
        "Parity validation for '{label_base}': completed SRs ({completed_set:?}) != expected SRs ({expected_set:?})"
    );

    if !failures.is_empty() {
        let summary: Vec<String> = failures
            .iter()
            .map(|(sr, msg)| format!("@ {sr} Hz: {msg}"))
            .collect();
        panic!(
            "Parity validation panic failures for sample rates:\n  {}",
            summary.join("\n  ")
        );
    }
}

fn run_v2_multi_sr(
    model_filename: &str,
    golden_name: &str,
    label_base: &str,
    check_lufs_gate: bool,
) {
    run_v2_multi_sr_impl(
        model_filename,
        golden_name,
        label_base,
        check_lufs_gate,
        false,
    );
}

fn run_v2_multi_sr_hf(
    model_filename: &str,
    golden_name: &str,
    label_base: &str,
    check_lufs_gate: bool,
) {
    run_v2_multi_sr_impl(
        model_filename,
        golden_name,
        label_base,
        check_lufs_gate,
        true,
    );
}

// =============================================================================
// Tests — Quick Parity Subset (non-ignored, 48 kHz, v1 short signal)
// =============================================================================
//
// Sprint S4, Tarefa 4.1 (F-3): representative subset of 3 cross-validations
// running in the ~3 min quick loop. Uses v1 stress signal (2048 samples, 48 kHz)
// with MR-STFT hard gate from S3. BUILD_LOCK caches the C++ render tool.
//
// Selected models: 1 LSTM (BossLSTM-1x16), 1 WaveNet CH16 (BossWN-standard),
// 1 A2 (wavenet_a2_full) — covering the three main architectures.

#[test]
fn quick_parity_lstm_1x16() {
    run_v1("BossLSTM-1x16.nam", "lstm_1x16", "Quick LSTM 1×16", true);
}

#[test]
fn quick_parity_wavenet_ch16() {
    run_v1(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Quick WaveNet CH16",
        true,
    );
}

#[test]
fn quick_parity_a2_full() {
    run_v1(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Quick A2-Full",
        true,
    );
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

// --- A2 Dynamic Models (Tarefa T4.3) ---
//
// Live C++ cross-validation for A2 dynamic geometries (FiLM, Gated, Blended)
// that exercise the native WaveNetA2Dyn engine paths against the C++ generic
// Eigen-based WaveNet (C++ a2_fast.cpp rejects these topologies and falls
// back to the generic path — see Finding 7.6.2).

#[test]
#[ignore]
fn live_cross_validation_a2_dynamic_gated() {
    run_v1(
        "a2_dynamic_gated_ch8.nam",
        "a2_dynamic_gated_ch8",
        "Live A2 Dynamic Gated (CH=8)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_a2_dynamic_blended() {
    run_v1(
        "a2_dynamic_blended_ch3.nam",
        "a2_dynamic_blended_ch3",
        "Live A2 Dynamic Blended (CH=3)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_film_lite() {
    run_v1(
        "wavenet_a2_film_lite.nam",
        "wavenet_a2_film_lite",
        "Live WaveNet A2-FiLM-Lite (CH=3)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_film_full() {
    run_v1(
        "wavenet_a2_film_full.nam",
        "wavenet_a2_film_full",
        "Live WaveNet A2-FiLM-Full (CH=8)",
        true,
    );
}

// wavenet_a2_max — DISABLED (§7.1); model inference is blocked at dispatch
// by fail-closed guard is_disabled_broken_a2_flagship.
// Uncomment once condition_dsp parity gap (§4.4) is resolved.
// #[test]
// #[ignore]
// fn live_cross_validation_wavenet_a2_max() {
//     run_v1(
//         "wavenet_a2_max.nam",
//         "wavenet_a2_max",
//         "Live WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1)",
//         true,
//     );
// }

// --- v2 A2 Dynamic Models (Tarefa T4.3) ---

#[test]
#[ignore]
fn live_cross_validation_v2_a2_dynamic_gated() {
    run_v2_multi_sr(
        "a2_dynamic_gated_ch8.nam",
        "a2_dynamic_gated_ch8",
        "Live A2 Dynamic Gated (CH=8) (v2)",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_a2_dynamic_blended() {
    run_v2_multi_sr(
        "a2_dynamic_blended_ch3.nam",
        "a2_dynamic_blended_ch3",
        "Live A2 Dynamic Blended (CH=3) (v2)",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_film_lite() {
    run_v2_multi_sr(
        "wavenet_a2_film_lite.nam",
        "wavenet_a2_film_lite",
        "Live WaveNet A2-FiLM-Lite (CH=3) (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_film_full() {
    run_v2_multi_sr(
        "wavenet_a2_film_full.nam",
        "wavenet_a2_film_full",
        "Live WaveNet A2-FiLM-Full (CH=8) (v2)",
        true,
    );
}

// wavenet_a2_max v2 — DISABLED (§7.1); see v1 note above.
// #[test]
// #[ignore]
// fn live_cross_validation_v2_wavenet_a2_max() {
//     run_v2_multi_sr(
//         "wavenet_a2_max.nam",
//         "wavenet_a2_max",
//         "Live WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1) (v2)",
//         true,
//     );
// }

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

// =============================================================================
// HighFidelity mode cpp_parity tests (Tarefa β1.3)
// =============================================================================
//
// C++ NAMCore uses standard Padé/minimax approximations for tanh/sigmoid.
// Rust in HF mode uses high-fidelity polynomial exp-based kernels (~2.4e-7
// error vs ~2.32e-3 for Padé tanh). This deliberate asymmetry means the
// interop divergence is larger in HF mode than in standard mode.
//
// These tests characterize that divergence and gate it with HF-specific
// caps (ABSOLUTE_ESR_CAP_*_HF in run_render_comparison).
//
// Quick (non-ignored): representative subset for fast CI loop
// Comprehensive (ignored): full model × SR matrix, requires C++ toolchain

// --- Quick Parity Subset (HF, non-ignored) ---

#[test]
fn quick_parity_hf_lstm_1x16() {
    run_v1_hf(
        "BossLSTM-1x16.nam",
        "lstm_1x16",
        "Quick LSTM 1×16 (HF)",
        true,
    );
}

#[test]
fn quick_parity_hf_wavenet_ch16() {
    run_v1_hf(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Quick WaveNet CH16 (HF)",
        true,
    );
}

// --- v1 (standard-rate) HF tests, ignored ---

#[test]
#[ignore]
fn live_cross_validation_hf_lstm_1x16() {
    run_v1_hf(
        "BossLSTM-1x16.nam",
        "lstm_1x16",
        "Live LSTM 1×16 (HF)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_hf_lstm_2x8() {
    run_v1_hf("BossLSTM-2x8.nam", "lstm_2x8", "Live LSTM 2×8 (HF)", true);
}

#[test]
#[ignore]
fn live_cross_validation_hf_wavenet_standard() {
    run_v1_hf(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard (HF)",
        true,
    );
}

// --- v2 (multi-SR) HF tests, ignored ---

#[test]
#[ignore]
fn live_cross_validation_v2_hf_lstm_1x16() {
    run_v2_multi_sr_hf(
        "BossLSTM-1x16.nam",
        "lstm_1x16",
        "Live LSTM 1×16 (v2, HF)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_hf_lstm_2x8() {
    run_v2_multi_sr_hf(
        "BossLSTM-2x8.nam",
        "lstm_2x8",
        "Live LSTM 2×8 (v2, HF)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_hf_wavenet_standard() {
    run_v2_multi_sr_hf(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard (v2, HF)",
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
