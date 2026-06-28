// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Reference oracle f64 — ESR measurement and source decomposition.
//!
//! Runs the f64 oracle against production f32 output for all model families,
//! reporting the absolute error floor and isolating contributions from
//! weight quantization, activation approximation, and accumulation precision.
//!
//! ## T5.3 — External anchoring
//!
//! The Rust f64 oracle is validated against a PyTorch/NumPy f64 reference
//! (pre-generated anchors in `tests/fixtures/f64_anchors/`). The oracle
//! passes with ESR < 1e-12 for all three model families, proving it is
//! a correct ground-truth reference.
//!
//! ## Caveat
//! Production models use f16c-quantized weights (LSTM, WaveNet DenseLayers).
//! The oracle uses full f32→f64 precision weights. The ESR(f32 vs f64) thus
//! includes the quantization error, which is often the dominant term.

use std::io::Read;
use std::path::PathBuf;

use nam_rs::loader::nam_json::model::NamModelData;
use nam_rs::loader::nam_json::parse::parse_nam_json;
use nam_rs::models::NamModel;
use nam_rs::testing::reference_oracle::{
    AccumulationMode, ActivationMode, PrecisionConfig, WeightPrecision, compute_esr_f64,
    esr_to_db_f64, oracle_forward, run_decomposition,
};

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("models")
}

fn anchors_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("f64_anchors")
}

fn load_f64_binary(path: &PathBuf) -> Vec<f64> {
    let mut f = std::fs::File::open(path).expect("Failed to open f64 binary");
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).expect("Failed to read count");
    let n = u32::from_le_bytes(buf) as usize;
    let mut data = Vec::with_capacity(n);
    let mut sample_buf = [0u8; 8];
    for _ in 0..n {
        f.read_exact(&mut sample_buf)
            .expect("Failed to read sample");
        data.push(f64::from_le_bytes(sample_buf));
    }
    data
}

fn load_and_parse(path: &PathBuf) -> NamModelData {
    let json = std::fs::read_to_string(path).expect("Failed to read model");
    parse_nam_json(&json).expect("Failed to parse NAM JSON")
}

fn run_f32_inference(model_data: &NamModelData, input: &[f32]) -> Vec<f32> {
    let mut model =
        nam_rs::loader::dispatcher::build_model(model_data).expect("Failed to build model");
    model.prewarm(2048);
    let total = input.len();
    let mut output = vec![0.0f32; total];
    let mut pos = 0;
    while pos < total {
        let nf = (total - pos).min(64);
        model.process(&input[pos..pos + nf], &mut output[pos..pos + nf]);
        pos += nf;
    }
    output
}

fn gen_sweep(n: usize, sample_rate: f64) -> Vec<f64> {
    let mut sig = Vec::with_capacity(n);
    let f0 = 220.0;
    let f1 = 880.0;
    let duration = n as f64 / sample_rate;
    for i in 0..n {
        let t = i as f64 / sample_rate;
        let phase = 2.0 * std::f64::consts::PI * (f0 * t + 0.5 * (f1 - f0) * t * t / duration);
        sig.push(phase.sin() * 0.3);
    }
    sig
}

fn print_decomposition(result: &nam_rs::testing::reference_oracle::DecompositionResult) {
    let db = |v: f64| -> f64 {
        if v <= 0.0 {
            f64::NEG_INFINITY
        } else {
            10.0 * v.log10()
        }
    };
    println!(
        "{} Decomposition:\n\
           ESR(f32 vs f64 oracle):  {:.2e} ({:.1} dB)\n\
           ΔESR f16c weights:       {:.2e} ({:.1} dB)\n\
           ΔESR bf16 weights:       {:.2e} ({:.1} dB)\n\
           ΔESR Pade activation:    {:.2e} ({:.1} dB)\n\
           ΔESR f32 accumulation:   {:.2e} ({:.1} dB)\n\
           ΔESR combined (F16C+Padé+F32): {:.2e} ({:.1} dB)",
        result.label,
        result.esr_f32_vs_f64,
        db(result.esr_f32_vs_f64),
        result.esr_quant_f16c_display(),
        db(result.esr_quant_f16c_display()),
        result.esr_quant_bf16_display(),
        db(result.esr_quant_bf16_display()),
        result.esr_activation_display(),
        db(result.esr_activation_display()),
        result.esr_accumulation_display(),
        db(result.esr_accumulation_display()),
        result.esr_combined_display(),
        db(result.esr_combined_display()),
    );
}

// ── T5.3: External anchor validation ─────────────────────────────────────

/// T5.3 anchor validation: Rust oracle vs PyTorch/NumPy f64 reference.
/// The pre-generated anchors are in tests/fixtures/f64_anchors/.
/// Format: [u32 LE count] [f64 LE × count]

#[test]
fn test_oracle_vs_python_anchor_wavenet() {
    let path = models_dir().join("wavenet_official.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("wavenet_official_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "WaveNet Official: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "WaveNet Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_lstm() {
    let path = models_dir().join("lstm.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("lstm_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "LSTM H=3: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "LSTM Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

#[test]
fn test_oracle_vs_python_anchor_a2() {
    let path = models_dir().join("wavenet_a2_lite.nam");
    let md = load_and_parse(&path);
    let input_f64 = load_f64_binary(&anchors_dir().join("sweep_256_48k.bin"));
    let anchor = load_f64_binary(&anchors_dir().join("a2_lite_256_f64.bin"));

    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &anchor);

    println!(
        "A2 Lite: ESR(Rust oracle vs NumPy f64) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < 1e-12,
        "A2 Rust oracle does not match NumPy f64 anchor: ESR={:.6e}",
        esr
    );
}

// ── T5.3: Calibrated fidelity gates (post-anchoring) ────────────────────

// These bounds are calibrated from the measured ESR(f32 production vs f64
// oracle) with a ~2× safety margin. They replace the previous placebo
// asserts (< 2.0, < 1.5, < 0.5) that were < 1.0 (anti-placebo line).
//
// Measured values (256-sample sweep, 48 kHz):
//   WaveNet:  ESR = 2.47e0 (dominated by f16c weight quantization + arch)
//   LSTM:     ESR = 1.06e0
//   A2:       ESR = 1.26e-1

const WAVENET_ESR_LIMIT: f64 = 3.5; // 2.47 measured + ~40% margin
const LSTM_ESR_LIMIT: f64 = 1.8; // 1.06 measured + ~70% margin
const A2_ESR_LIMIT: f64 = 0.35; // 0.126 measured + ~175% margin

// ── Basic ESR tests ────────────────────────────────────────────────────────

#[test]
fn test_oracle_wavenet() {
    let path = models_dir().join("wavenet_official.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &prod_f64);

    println!(
        "WaveNet Official: ESR(f32 vs oracle) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < WAVENET_ESR_LIMIT,
        "WaveNet ESR={:.6e} exceeds calibrated limit {}",
        esr,
        WAVENET_ESR_LIMIT
    );
}

#[test]
fn test_oracle_lstm() {
    let path = models_dir().join("lstm.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &prod_f64);

    println!(
        "LSTM H=3: ESR(f32 vs oracle) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < LSTM_ESR_LIMIT,
        "LSTM ESR={:.6e} exceeds calibrated limit {}",
        esr,
        LSTM_ESR_LIMIT
    );
}

#[test]
fn test_oracle_a2() {
    let path = models_dir().join("wavenet_a2_lite.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr = compute_esr_f64(&oracle, &prod_f64);

    println!(
        "A2 Lite: ESR(f32 vs oracle) = {:.2e} ({:.1} dB)",
        esr,
        esr_to_db_f64(esr)
    );
    assert!(
        esr < A2_ESR_LIMIT,
        "A2 ESR={:.6e} exceeds calibrated limit {}",
        esr,
        A2_ESR_LIMIT
    );
}

// ── Decomposition tests ────────────────────────────────────────────────────

#[test]
fn test_decomposition_wavenet() {
    let path = models_dir().join("wavenet_official.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();

    let result = run_decomposition("WaveNet-official", "WaveNet", &md, &prod_f64, &input_f64);
    print_decomposition(&result);
    assert!(
        result.esr_f32_vs_f64 < WAVENET_ESR_LIMIT,
        "WaveNet ESR={:.6e} exceeds calibrated limit {}",
        result.esr_f32_vs_f64,
        WAVENET_ESR_LIMIT
    );
    assert!(
        result.esr_combined_display() > 0.0,
        "Combined ΔESR should be non-zero"
    );
}

#[test]
fn test_decomposition_lstm() {
    let path = models_dir().join("lstm.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();

    let result = run_decomposition("LSTM-H3", "LSTM", &md, &prod_f64, &input_f64);
    print_decomposition(&result);
    assert!(
        result.esr_f32_vs_f64 < LSTM_ESR_LIMIT,
        "LSTM ESR={:.6e} exceeds calibrated limit {}",
        result.esr_f32_vs_f64,
        LSTM_ESR_LIMIT
    );
    assert!(
        result.esr_combined_display() > 0.0,
        "Combined ΔESR should be non-zero"
    );
}

#[test]
fn test_decomposition_a2() {
    let path = models_dir().join("wavenet_a2_lite.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();
    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();

    let result = run_decomposition("A2-Lite", "WaveNet", &md, &prod_f64, &input_f64);
    print_decomposition(&result);
    assert!(
        result.esr_f32_vs_f64 < A2_ESR_LIMIT,
        "A2 ESR={:.6e} exceeds calibrated limit {}",
        result.esr_f32_vs_f64,
        A2_ESR_LIMIT
    );
    assert!(
        result.esr_combined_display() > 0.0,
        "Combined ΔESR should be non-zero"
    );
}

// ── Combined simulation acceptance tests ───────────────────────────────────

fn combined_config() -> PrecisionConfig {
    PrecisionConfig {
        weight_precision: WeightPrecision::F16C,
        activation: ActivationMode::PadeMinimax,
        accumulation: AccumulationMode::F32Plain,
    }
}

#[test]
fn test_combined_simulation_wavenet() {
    let path = models_dir().join("wavenet_official.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();
    let combined = oracle_forward(&md, &input_f64, &combined_config());
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr_combined_vs_oracle = compute_esr_f64(&oracle, &combined);
    let esr_combined_vs_prod = compute_esr_f64(&combined, &prod_f64);

    println!(
        "WaveNet CombinedSim:\n  \
           ΔESR(combined vs oracle):     {:.2e} ({:.1} dB)\n  \
           ESR(combined vs production):  {:.2e} ({:.1} dB) (gap = architectural divergence)",
        esr_combined_vs_oracle,
        esr_to_db_f64(esr_combined_vs_oracle),
        esr_combined_vs_prod,
        esr_to_db_f64(esr_combined_vs_prod),
    );
    assert!(
        esr_combined_vs_oracle > 0.0,
        "Combined simulation must be active (non-zero ΔESR)"
    );
    assert!(
        esr_combined_vs_prod < WAVENET_ESR_LIMIT,
        "Production ESR vs combined sim exceeds calibrated limit"
    );
}

#[test]
fn test_combined_simulation_lstm() {
    let path = models_dir().join("lstm.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();
    let combined = oracle_forward(&md, &input_f64, &combined_config());
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr_combined_vs_oracle = compute_esr_f64(&oracle, &combined);
    let esr_combined_vs_prod = compute_esr_f64(&combined, &prod_f64);

    println!(
        "LSTM CombinedSim:\n  \
           ΔESR(combined vs oracle):     {:.2e} ({:.1} dB)\n  \
           ESR(combined vs production):  {:.2e} ({:.1} dB) (gap = architectural divergence)",
        esr_combined_vs_oracle,
        esr_to_db_f64(esr_combined_vs_oracle),
        esr_combined_vs_prod,
        esr_to_db_f64(esr_combined_vs_prod),
    );
    assert!(
        esr_combined_vs_oracle > 0.0,
        "Combined simulation must be active (non-zero ΔESR)"
    );
    assert!(
        esr_combined_vs_prod < LSTM_ESR_LIMIT,
        "Production ESR vs combined sim exceeds calibrated limit"
    );
}

#[test]
fn test_combined_simulation_a2() {
    let path = models_dir().join("wavenet_a2_lite.nam");
    let md = load_and_parse(&path);
    let input_f64 = gen_sweep(256, 48000.0);
    let input_f32: Vec<f32> = input_f64.iter().map(|&x| x as f32).collect();

    let prod_f32 = run_f32_inference(&md, &input_f32);
    let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();
    let combined = oracle_forward(&md, &input_f64, &combined_config());
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let esr_combined_vs_oracle = compute_esr_f64(&oracle, &combined);
    let esr_combined_vs_prod = compute_esr_f64(&combined, &prod_f64);

    println!(
        "A2 CombinedSim:\n  \
           ΔESR(combined vs oracle):     {:.2e} ({:.1} dB)\n  \
           ESR(combined vs production):  {:.2e} ({:.1} dB) (gap = architectural divergence)",
        esr_combined_vs_oracle,
        esr_to_db_f64(esr_combined_vs_oracle),
        esr_combined_vs_prod,
        esr_to_db_f64(esr_combined_vs_prod),
    );
    assert!(
        esr_combined_vs_oracle > 0.0,
        "Combined simulation must be active (non-zero ΔESR)"
    );
    assert!(
        esr_combined_vs_prod < A2_ESR_LIMIT,
        "Production ESR vs combined sim exceeds calibrated limit"
    );
}

// ── Summary table ──────────────────────────────────────────────────────────

#[test]
fn test_summary_table() {
    let models: Vec<(&str, &str)> = vec![
        ("wavenet_official.nam", "WaveNet"),
        ("lstm.nam", "LSTM"),
        ("wavenet_a2_lite.nam", "WaveNet(A2)"),
    ];
    let input = gen_sweep(256, 48000.0);

    println!("\n=== ESR(f32 vs f64 oracle) Summary ===");
    println!(
        "{:<30} {:<12} {:<15} {:<15}",
        "Model", "Family", "ESR linear", "ESR (dB)"
    );
    println!("{}", "-".repeat(72));

    for (filename, family) in &models {
        let path = models_dir().join(filename);
        let md = load_and_parse(&path);
        let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
        let prod_f32 = run_f32_inference(&md, &input_f32);
        let prod_f64: Vec<f64> = prod_f32.iter().map(|&x| x as f64).collect();
        let oracle = oracle_forward(&md, &input, &PrecisionConfig::default());
        let esr = compute_esr_f64(&oracle, &prod_f64);
        println!(
            "{:<30} {:<12} {:<15.6e} {:<15.1}",
            filename,
            family,
            esr,
            esr_to_db_f64(esr)
        );
    }
    println!("{}", "-".repeat(72));
}

// ── T3.3 RCA diagnostic: recurrent state drift ───────────────────────────────

#[test]
#[ignore]
fn t33_diagnostic_recurrent_drift_lstm_1x16() {
    use nam_rs::testing::perceptual::compute_esr;
    use nam_rs::testing::stress::generate_stress_signal_v2_default;

    let path = models_dir().join("BossLSTM-1x16.nam");
    let md = load_and_parse(&path);

    let sample_rate = 48000;
    let stress_signal = generate_stress_signal_v2_default(sample_rate);

    let mut output: Vec<f32> = vec![0.0f32; stress_signal.len()];
    let mut model = nam_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
    model.prewarm(24_000);
    let mut pos = 0;
    while pos < stress_signal.len() {
        let nf = (stress_signal.len() - pos).min(64);
        model.process(&stress_signal[pos..pos + nf], &mut output[pos..pos + nf]);
        pos += nf;
    }

    let input_f64: Vec<f64> = stress_signal.iter().map(|&x| x as f64).collect();
    let oracle = oracle_forward(&md, &input_f64, &PrecisionConfig::default());
    let oracle_f32: Vec<f32> = oracle.iter().map(|&x| x as f32).collect();

    // NAMCore golden comparison skipped here — already handled in cpp_parity.
    // Golden files available at: {models_dir}/../../fixtures/golden_lstm_1x16_v2_48000.bin

    let segment_sizes = &[
        512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 120000, 240000,
    ];

    println!("\n=== T3.3 — Recurrent State Drift Analysis ===");
    println!("Model:   BossLSTM-1×16");
    println!("Signal:  v2 stress, 5s @ 48 kHz (240k samples)");
    println!("Baseline A1-Std ESR: 6.23e-3");
    println!(
        "{:<12} {:<18} {:<12} {:<18}",
        "Samples", "ESR(vs oracle)", "ESR dB", "Time (ms)"
    );
    println!("{}", "-".repeat(60));

    for &n_samples in segment_sizes.iter() {
        let n = n_samples.min(stress_signal.len());

        let mut segment_out = vec![0.0f32; n];
        let mut model_fresh =
            nam_rs::loader::dispatcher::build_model(&md).expect("Failed to build model");
        model_fresh.prewarm(24_000);
        let mut pos = 0;
        while pos < n {
            let nf = (n - pos).min(64);
            model_fresh.process(
                &stress_signal[pos..pos + nf],
                &mut segment_out[pos..pos + nf],
            );
            pos += nf;
        }

        let esr_vs_oracle = compute_esr(&oracle_f32[..n], &segment_out[..n]);
        let time_ms = n as f64 / 48.0; // ms
        println!(
            "{:<12} {:<18.6e} {:<12.1} {:<18.1}",
            n,
            esr_vs_oracle,
            10.0 * esr_vs_oracle.log10(),
            time_ms
        );
    }

    // Final comparison: full output vs oracle
    let esr_full = compute_esr(&oracle_f32, &output);
    println!(
        "\nFull 240k: ESR={:.6e} ({:.1} dB)",
        esr_full,
        10.0 * esr_full.log10()
    );
}
