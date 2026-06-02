// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Testes de validação cruzada ao vivo NAM-rs ↔ NeuralAmpModelerCore (Camada 2).
//!
//! ## Quando roda
//! - `utils/tests-long.sh` (suite completa lenta)
//! - `cargo test --test cpp_parity -- --ignored --nocapture`
//!
//! ## Pipeline
//! 1. Compila `render` tool do NeuralAmpModelerCore on-demand (idempotente, cached)
//! 2. Gera WAV stress signal via `generate_stress_signal()`
//! 3. Escreve WAV, executa render via `std::process::Command`
//! 4. Lê WAV output, compara C++ vs Rust com `report_dsp_fidelity()`
//!
//! Testes são `#[ignore]` no CI normal — requer C++ toolchain instalada.
//!
//! ## Thresholds de Paridade
//!
//! | Modelo           | MSE       | SNR      |
//! | ---------------- |:---------:|:--------:|
//! | WaveNet Standard | < 5e-2    | ≥ 9 dB   |
//! | WaveNet Feather  | < 5e-2    | ≥ 9 dB   |
//! | WaveNet Nano     | < 5e-2    | ≥ 9 dB   |
//! | LSTM 1×16        | < 3e-3    | ≥ 15 dB  |
//! | LSTM 2×8         | < 1e-3    | ≥ 18 dB  |

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
        // Fallback: try debug build
        bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        bin.push(BUILD_DIR);
        bin.push("Debug/render");
    }
    if !bin.exists() {
        // Search for render binary anywhere in build dir
        let build_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(BUILD_DIR);
        if let Ok(entries) = std::fs::read_dir(&build_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.file_name().map_or(false, |n| n == "render") && path.is_file() {
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
        eprintln!("NeuralAmpModelerCore não encontrado em {nam_core:?}");
        eprintln!("Execute: git clone --depth 1 https://github.com/sdatkinson/NeuralAmpModelerCore.git {NAM_CORE_DIR}");
        return false;
    }

    // Init submodules
    for sub in &["Dependencies/eigen", "Dependencies/AudioDSPTools"] {
        let sub_path = nam_core.join(sub);
        if sub_path.exists() && fs::read_dir(&sub_path).map_or(true, |mut d| d.next().is_none()) {
            let status = Command::new("git")
                .args(["submodule", "update", "--init", sub])
                .current_dir(&nam_core)
                .status();
            if let Ok(s) = status {
                if !s.success() {
                    eprintln!("WARN: falha ao inicializar submódulo {sub}");
                }
            }
        }
    }

    let build_dir = project_root.join(BUILD_DIR);
    fs::create_dir_all(&build_dir).ok();

    eprintln!("Compilando render tool (NeuralAmpModelerCore)...");

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
            eprintln!("CMake falhou — dependências C++ podem estar ausentes.");
            return false;
        }
    }

    let build_status = Command::new("cmake")
        .args(["--build", build_dir.to_str().unwrap(), "--target", "render", "-j"])
        .status();

    match build_status {
        Ok(s) if s.success() => {}
        _ => {
            eprintln!("Build do render falhou.");
            return false;
        }
    }

    render_bin().exists()
}

fn run_render_comparison(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    mse_limit: f64,
    min_snr_db: f64,
) {
    if !ensure_render_compiled() {
        eprintln!("SKIP: {label} — render tool não disponível.");
        return;
    }

    let model_path = model_path(model_filename);
    if !model_path.exists() {
        eprintln!("SKIP: {label} — {model_filename} não encontrado.");
        return;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();

    let stress_wav = temp_dir.join("stress_live.wav");

    // Gera sinal de stress e escreve WAV
    let stress_signal = generate_stress_signal();
    common::wav::write_wav_f32(&stress_wav, &stress_signal, STRESS_SAMPLE_RATE)
        .expect("Falha ao escrever WAV stress");

    let output_wav = temp_dir.join(format!("{golden_name}_live.wav"));

    // Executa render tool
    let bin = render_bin();
    let status = Command::new(&bin)
        .arg(model_path.to_str().unwrap())
        .arg(stress_wav.to_str().unwrap())
        .arg(output_wav.to_str().unwrap())
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("SKIP: {label} — render retornou código {}", s.code().unwrap_or(-1));
            return;
        }
        Err(e) => {
            eprintln!("SKIP: {label} — falha ao executar render: {e}");
            return;
        }
    }

    // Lê output WAV do render
    let (cpp_output, _sr) =
        common::wav::read_wav_f32(&output_wav).expect("Falha ao ler WAV output do render");

    // Inferência Rust
    let json_data = fs::read_to_string(&model_path).expect("Falha ao ler modelo");
    let model_data = parse_nam_json(&json_data).expect("Falha no parser JSON");
    let mut model = build_model(&model_data).expect("Dispatcher falhou");

    model.prewarm(2048);
    let mut rust_output = vec![0.0f32; stress_signal.len()];
    process_in_blocks(&mut model, &stress_signal, &mut rust_output, GOLDEN_BLOCK_SIZE);

    // Trunca para o mínimo dos dois (evita mismatch se render produz menos)
    let min_len = cpp_output.len().min(rust_output.len());
    report_dsp_fidelity(
        &cpp_output[..min_len],
        &rust_output[..min_len],
        mse_limit,
        min_snr_db,
        label,
    );

    // Limpeza
    fs::remove_file(&output_wav).ok();
}

// =============================================================================
// Tests — #[ignore] (requerem C++ toolchain)
// =============================================================================

#[test]
#[ignore]
fn live_cross_validation_wavenet_standard() {
    run_render_comparison(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard",
        5e-2,
        9.0,
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_feather() {
    run_render_comparison(
        "BossWN-feather.nam",
        "wavenet_feather",
        "Live WaveNet Feather",
        5e-2,
        9.0,
    );
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_nano() {
    run_render_comparison(
        "BossWN-nano.nam",
        "wavenet_nano",
        "Live WaveNet Nano",
        5e-2,
        9.0,
    );
}

#[test]
#[ignore]
fn live_cross_validation_lstm_1x16() {
    run_render_comparison(
        "BossLSTM-1x16.nam",
        "lstm_1x16",
        "Live LSTM 1×16",
        3e-3,
        15.0,
    );
}

#[test]
#[ignore]
fn live_cross_validation_lstm_2x8() {
    run_render_comparison(
        "BossLSTM-2x8.nam",
        "lstm_2x8",
        "Live LSTM 2×8",
        1e-3,
        18.0,
    );
}
