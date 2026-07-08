// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Investigation [T5.5]: LSTM activation precision — SNR gain analysis.
// 
//  Measures the SNR gain from using exact `f32::tanh` (libm) vs the production
//  Padé [5,4] rational approximant in LSTM fused gates.  Runs the same golden
//  vector input through both paths and reports the SNR delta.

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::NamModel;
use nam_rs::testing::perceptual::compute_snr_db;
use std::fs;
use std::path::PathBuf;

use super::common;
use common::*;

const GOLDEN_BLOCK_SIZE: usize = 64;

#[derive(Debug)]
struct SnrResult {
    model_name: String,
    gain_db: f64,
}

impl SnrResult {
    fn verdict(&self) -> &'static str {
        if self.gain_db < 2.0 {
            "NEGLIGIBLE — FastMath is adequate"
        } else if self.gain_db < 6.0 {
            "MODEST — consider documenting tradeoff"
        } else {
            "SIGNIFICANT — higher precision warranted"
        }
    }
}

fn process_in_blocks_scalar(
    model: &mut nam_rs::models::StaticModel,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.process_scalar(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}

fn measure_lstm_snr(golden_path: &str, model_filename: &str, label: &str) -> (f64, f64) {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let golden_full = fixtures_dir.join(golden_path);

    if !golden_full.exists() {
        eprintln!("SKIP: {} not found at {:?}.", golden_path, golden_full);
        return (f64::NAN, f64::NAN);
    }

    let nam_path = model_path(model_filename);
    if !nam_path.exists() {
        eprintln!("SKIP: {} not found.", model_filename);
        return (f64::NAN, f64::NAN);
    }

    let (input, expected) =
        read_golden_bin(&golden_full).unwrap_or_else(|| panic!("Failed to read {}", golden_path));

    let json_data = fs::read_to_string(&nam_path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    // --- FastMath path (SIMD Padé tanh + minimax sigmoid) ---
    let mut model_fast = build_model(&model_data).expect("Dispatcher failed");
    model_fast.prewarm(2048);
    let mut output_fast = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_fast, &input, &mut output_fast, GOLDEN_BLOCK_SIZE);
    let snr_fast = compute_snr_db(&expected, &output_fast);

    // --- Exact tanh path (f32::tanh via libm, scalar minimax sigmoid) ---
    let mut model_exact = build_model(&model_data).expect("Dispatcher failed");
    model_exact.prewarm(2048);
    let mut output_exact = vec![0.0f32; input.len()];
    process_in_blocks_scalar(
        &mut model_exact,
        &input,
        &mut output_exact,
        GOLDEN_BLOCK_SIZE,
    );
    let snr_exact = compute_snr_db(&expected, &output_exact);

    println!(
        "{label:>22}  FastMath(Padé): {snr_fast:6.1} dB  |  Exact(tanh): {snr_exact:6.1} dB  |  Δ={gain:+.1} dB",
        gain = snr_exact - snr_fast,
    );

    (snr_fast, snr_exact)
}

/// [T5.5] Precision investigation: measure SNR gain of exact tanh vs FastMath Padé.
///
/// Runs the 3 LSTM golden vectors through both SIMD (FastMath) and scalar
/// (exact `f32::tanh`) paths, computing SNR against the C++ reference.
/// Reports per-model SNR gain and overall verdict.
#[test]
fn test_lstm_activation_precision_gain() {
    eprintln!();
    eprintln!("══════════════════════════════════════════════════════════════════");
    eprintln!("  [T5.5] LSTM Activation Precision Investigation");
    eprintln!("  Comparing FastMath (Padé tanh) vs Exact tanh (libm f32::tanh)");
    eprintln!("  SNR vs C++ golden vectors (NeuralAmpModelerCore)");
    eprintln!("══════════════════════════════════════════════════════════════════");

    let results = [
        measure_lstm_snr("golden_lstm_1x16.bin", "BossLSTM-1x16.nam", "LSTM 1×16"),
        measure_lstm_snr("golden_lstm_2x8.bin", "BossLSTM-2x8.nam", "LSTM 2×8"),
        measure_lstm_snr("golden_lstm_official.bin", "lstm.nam", "LSTM Official"),
    ];

    let valid: Vec<_> = results
        .iter()
        .filter(|(fast, exact)| fast.is_finite() && exact.is_finite())
        .collect();

    if valid.is_empty() {
        eprintln!("\n  All tests SKIPPED (golden files or .nam models missing).");
        return;
    }

    let parsed: Vec<SnrResult> = results
        .iter()
        .enumerate()
        .filter(|(_, (f, e))| f.is_finite() && e.is_finite())
        .map(|(i, &(snr_fastmath_db, snr_exact_tanh_db))| SnrResult {
            model_name: ["LSTM 1×16", "LSTM 2×8", "LSTM Official"][i].to_string(),
            gain_db: snr_exact_tanh_db - snr_fastmath_db,
        })
        .collect();

    let min_gain = parsed
        .iter()
        .map(|r| r.gain_db)
        .fold(f64::INFINITY, f64::min);
    let max_gain = parsed
        .iter()
        .map(|r| r.gain_db)
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_gain = parsed.iter().map(|r| r.gain_db).sum::<f64>() / parsed.len() as f64;

    eprintln!();
    eprintln!("  ────────────────────────────────────────────────────────────────");
    eprintln!("  Summary:");
    for r in &parsed {
        eprintln!(
            "    {:<16}  SNR gain: {:+5.1} dB  →  {}",
            r.model_name,
            r.gain_db,
            r.verdict()
        );
    }
    let separator = "-".repeat(16);
    eprintln!(
        "    {separator}  SNR gain range: [{:+.1}, {:.1}] dB  avg: {:.1} dB",
        min_gain, max_gain, avg_gain,
    );
    eprintln!("  ────────────────────────────────────────────────────────────────");

    if avg_gain < 3.0 {
        eprintln!(
            "  VERDICT: FastMath Padé [5,4] is adequate for LSTM goldens.\n\
               The SNR gain from exact tanh is negligible (< 3 dB avg).\n\
               Keeping FastMath as the default production path."
        );
    } else if avg_gain < 6.0 {
        eprintln!(
            "  VERDICT: Modest SNR gain detected ({} dB avg).\n\
               Consider an optional high-precision LSTM feature flag\n\
               for users who prioritize accuracy over speed.",
            avg_gain
        );
    } else {
        eprintln!(
            "  VERDICT: Significant SNR gain ({} dB avg).\n\
               A higher-precision LSTM tanh kernel is warranted.",
            avg_gain
        );
    }
    eprintln!();
}
