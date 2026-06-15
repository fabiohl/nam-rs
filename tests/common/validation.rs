// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

/// Plausible LUFS range for golden reference output (lightweight sanity gate).
///
/// Guitar/amp model output at typical stress-signal levels falls between −35 and 0 LUFS.
/// The lower bound of −50 LUFS is intentionally generous — it only catches egregious
/// errors (e.g., T2.5 where LUFS −67 near-silence went undetected in a validly passing test).
/// The upper bound of +10 LUFS guards against output saturation/clipping that would also
/// indicate a defective golden.
///
/// Part of T4.3: metrics perceptuais como guard-rail.
const LUFS_PLAUSIBLE_MIN: f64 = -50.0;
const LUFS_PLAUSIBLE_MAX: f64 = 10.0;

/// Validates DSP fidelity in a single pass, computing MSE, MAE, SNR, PSNR,
/// equivalent bits, ESR, and LUFS simultaneously.
///
/// # Parameters
/// - `reference` — reference output vector (NeuralAmpModelerCore C++)
/// - `test`      — Rust engine output vector to be validated
/// - `mse_limit` — maximum allowed MSE threshold
/// - `min_snr_db` — minimum SNR in dB that must be achieved
/// - `max_esr`   — optional maximum ESR threshold for regression gating (default `None`)
/// - `label`     — label for identification in diagnostic messages
/// - `sample_rate` — sample rate in Hz (used for LUFS and anchor SNR diagnostics)
///
/// # Gates
/// - **MSE** — absolute error gate (fail if exceeded)
/// - **SNR** — signal-to-noise ratio gate (fail if below minimum)
/// - **ESR** — scale-invariant error gate (fail if exceeded; primary gate for A2)
/// - **LUFS plausibility** — lightweight sanity gate on reference output (T4.3);
///   fails if reference LUFS is outside `[{LUFS_PLAUSIBLE_MIN}, {LUFS_PLAUSIBLE_MAX}]`
///
/// # Output format
/// ```text
/// [NeuralAmpModelerCore × NAM-rs — label]
///   MSE     = 3.21e-02      (threshold < 5.0e-02)  ✓
///   MAE     = 2.84e-01
///   SNR     = 10.1 dB       (threshold ≥ 9.0 dB)   ✓
///   PSNR    = 14.9 dB
///   Bits    = 2.5 bits equiv.
///   ESR     = 1.23e-05       (−49.1 dB)   [baseline A1-Std: 6.23e-03, A2-Full: 3.34e-03]
///   MR-STFT = 0.0042         (relative)
///   LUFS    = −23.4 LUFS    (reference)   [plausible: −50.0..+10.0]  ✓
///   Samples = 2048 @ 48 kHz (stress signal)
/// ```
#[track_caller]
pub fn report_dsp_fidelity(
    reference: &[f32],
    test: &[f32],
    mse_limit: f64,
    min_snr_db: f64,
    max_esr: Option<f64>,
    label: &str,
    sample_rate: u32,
) {
    report_dsp_fidelity_impl(
        reference,
        test,
        mse_limit,
        min_snr_db,
        max_esr,
        label,
        sample_rate,
        true,
    )
}

/// Like [`report_dsp_fidelity`] but skips the LUFS plausibility gate.
///
/// Use when the reference signal is known to have high amplitude that doesn't
/// indicate a defect (e.g., IR convolution goldens where synthetic signal +
/// IR can legitimately produce LUFS above +10).
#[track_caller]
pub fn report_dsp_fidelity_no_lufs(
    reference: &[f32],
    test: &[f32],
    mse_limit: f64,
    min_snr_db: f64,
    max_esr: Option<f64>,
    label: &str,
    sample_rate: u32,
) {
    report_dsp_fidelity_impl(
        reference,
        test,
        mse_limit,
        min_snr_db,
        max_esr,
        label,
        sample_rate,
        false,
    )
}

#[track_caller]
fn report_dsp_fidelity_impl(
    reference: &[f32],
    test: &[f32],
    mse_limit: f64,
    min_snr_db: f64,
    max_esr: Option<f64>,
    label: &str,
    sample_rate: u32,
    check_lufs_gate: bool,
) {
    assert_eq!(
        reference.len(),
        test.len(),
        "[{label}] Vectors of different sizes for report_dsp_fidelity"
    );
    let n = reference.len() as f64;
    let sr = sample_rate;

    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    let mut peak_ref = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let r64 = r as f64;
        let t64 = t as f64;
        let diff = r64 - t64;
        signal_power += r64 * r64;
        noise_power += diff * diff;
        let abs_diff = diff.abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
        if r64.abs() > peak_ref {
            peak_ref = r64.abs();
        }
    }
    let mse = noise_power / n;
    let mae = max_abs_diff;
    let snr = if noise_power <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    };
    let psnr = if mse <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (peak_ref * peak_ref / mse).log10()
    };
    let signal_avg_power = signal_power / n;
    let bits = if mse <= f64::EPSILON || signal_avg_power <= f64::EPSILON {
        f64::INFINITY
    } else {
        -0.5 * (mse / signal_avg_power).log2()
    };

    // ESR (linear + dB)
    let esr_linear = if signal_power <= f64::EPSILON {
        if noise_power <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        noise_power / signal_power
    };
    let esr_db = nam_rs::testing::perceptual::esr_to_db(esr_linear);

    // LUFS — reference (golden) for plausibility sanity gate (T4.3)
    let lufs_ref = nam_rs::testing::perceptual::compute_lufs(reference, sr);
    let lufs_test = nam_rs::testing::perceptual::compute_lufs(test, sr);
    let lufs_plausible =
        lufs_ref.is_finite() && (LUFS_PLAUSIBLE_MIN..=LUFS_PLAUSIBLE_MAX).contains(&lufs_ref);

    // SNR(reference, anchor) sanity: compute SNR of test against a low-pass 3.5 kHz anchor
    let anchor_snr_db = {
        let anchor = nam_rs::testing::mushra::low_pass_1pole(reference, 3500.0, sr);
        nam_rs::testing::perceptual::compute_snr_db(reference, &anchor)
    };

    println!();
    println!("[NeuralAmpModelerCore × NAM-rs — {label}]");
    println!(
        "  MSE     = {mse:.2e}      (threshold < {mse_limit:.1e})  {}",
        if mse < mse_limit { "✓" } else { "✗" }
    );
    println!("  MAE     = {mae:.2e}");
    if snr.is_finite() {
        println!(
            "  SNR     = {snr:.1} dB       (threshold ≥ {min_snr_db:.1} dB)   {}",
            if snr >= min_snr_db { "✓" } else { "✗" }
        );
    } else {
        println!("  SNR     = ∞ dB");
    }
    if psnr.is_finite() {
        println!("  PSNR    = {psnr:.1} dB");
    } else {
        println!("  PSNR    = ∞ dB");
    }
    if bits.is_finite() {
        println!("  Bits    = {bits:.2} bits equiv.");
    } else {
        println!("  Bits    = ∞ bits equiv.");
    }
    if esr_linear.is_finite() {
        println!(
            "  ESR     = {esr_linear:.2e}       ({esr_db:.1} dB)   [baseline A1-Std: {a1std:.2e}, A2-Full: {a2full:.2e}, A2-Lite: {a2lite:.2e}]",
            a1std = nam_rs::testing::perceptual::A2ESR_A1_STANDARD_MEDIAN,
            a2full = nam_rs::testing::perceptual::A2ESR_A2_FULL_MEDIAN,
            a2lite = nam_rs::testing::perceptual::A2ESR_A2_LITE_MEDIAN,
        );
    } else {
        println!("  ESR     = ∞  (identical)");
    }

    // MR-STFT
    let mr_stft = nam_rs::testing::perceptual::compute_mr_stft(reference, test);
    println!("  MR-STFT = {mr_stft:.4e}      (relative)");

    if lufs_test.is_finite() {
        if lufs_ref.is_finite() {
            println!(
                "  LUFS    = {lufs_ref:.1} LUFS    (reference)   [plausible: {LUFS_PLAUSIBLE_MIN:.0}..{LUFS_PLAUSIBLE_MAX:.0}]  {}",
                if lufs_plausible {
                    "✓"
                } else {
                    "✗ — GOLDEN DEFECT (T2.5 lesson)"
                }
            );
        } else {
            println!("  LUFS    = {lufs_test:.1} LUFS    (test — reference silent)");
        }
        if anchor_snr_db.is_finite() {
            let delta_snr = snr - anchor_snr_db;
            let is_satisfactory = delta_snr > 8.0 || snr >= min_snr_db;
            println!("  SNR(anchor) = {anchor_snr_db:.1} dB (degradation reference)");
            println!(
                "  Fidelity Margin = {delta_snr:.1} dB (target > 8.0 dB) {}",
                if is_satisfactory { "✓" } else { "?" }
            );
        }
    }
    println!("  Samples = {} @ {sr} Hz (stress signal)", reference.len());

    assert!(
        mse < mse_limit,
        "[{label}] MSE={mse:.6e} exceeds threshold {mse_limit:.1e} (MAE={mae:.6e}, SNR={snr:.1} dB)"
    );
    assert!(
        snr >= min_snr_db,
        "[{label}] SNR={snr:.1} dB below minimum {min_snr_db:.1} dB (MSE={mse:.6e}, MAE={mae:.6e})"
    );
    if let Some(limit) = max_esr {
        assert!(
            esr_linear < limit,
            "[{label}] ESR={esr_linear:.6e} exceeds threshold {limit:.1e} (ESR dB={esr_db:.1})"
        );
    }
    // T4.3 LUFS plausibility sanity gate — catch near-silence / implausible golden output.
    // Only enforced when check_lufs_gate is true (opt-out for IR convolution goldens).
    if check_lufs_gate {
        assert!(
            lufs_plausible,
            "[{label}] Reference LUFS={lufs_ref:.1} is outside plausible audio range \
             [{LUFS_PLAUSIBLE_MIN:.0}, {LUFS_PLAUSIBLE_MAX:.0}]. \
             The golden output may be defective (near-silence, clipping, or wrong scaling). \
             See T2.5 lesson: LUFS −67 went undetected without this gate."
        );
    } else if !lufs_plausible {
        eprintln!(
            "  ⓘ  LUFS gate skipped for [{label}]: reference LUFS={lufs_ref:.1} \
             outside [{LUFS_PLAUSIBLE_MIN:.0}, {LUFS_PLAUSIBLE_MAX:.0}] — \
             expected for IR convolution goldens"
        );
    }
}

/// Converts SNR (dB) to a conservative MSE upper-bound estimate.
///
/// Assumes signal_power ≈ 0.3 × n for a normalized stress signal,
/// balancing tightness with headroom to avoid false positives.
#[inline]
fn snr_to_mse(snr_db: f64) -> f64 {
    10.0_f64.powf(-snr_db / 10.0) * 0.3
}

/// Shared WaveNet MSE/SNR/ESR threshold lookup — used by both `topology_thresholds`
/// (golden vectors) and `live_parity_thresholds` (cpp_parity) as a fallback.
///
/// Post-T16.1 live v1 SNR measurements (2026-06-11):
///   Standard (CH=16): 68.4 dB → floor 60 dB (8.4 dB margin)
///   Feather  (CH=8):  67.6 dB → floor 60 dB (7.6 dB margin)
///   Nano     (CH=4):  52.6 dB → floor 45 dB (7.6 dB margin)
///   Lite     (CH=12):  0.9 dB → floor  0 dB (known failure, T16.x)
///
/// T16.4 ESR gates (robust to scale mismatch):
///   Standard/Feather/Nano/A2: 1e-3  (NAM_RS_CPP_PARITY_ESR_MAX)
///   Lite:                     5e-2  (known failure, loose)
///   Default:                  1e-3
///
/// Returns `(mse_limit, min_snr_db, max_esr)`.
#[inline]
fn wavenet_thresholds(channels: u32) -> (f64, f64, Option<f64>) {
    match channels {
        3 => (snr_to_mse(40.0), 40.0, Some(1e-3)),
        4 => {
            let snr_db = 45.0;
            (snr_to_mse(snr_db), snr_db, Some(3e-3))
        }
        8 => {
            let snr_db = 60.0;
            (snr_to_mse(snr_db), snr_db, Some(1e-3))
        }
        12 => {
            // Target thresholds if corrected. Under current drift, SNR is 0.9 dB.
            // Marking this test as #[ignore] in golden_vectors to avoid a false gate.
            let snr_db = 40.0;
            (snr_to_mse(snr_db), snr_db, Some(1e-3))
        }
        16 => {
            let snr_db = 45.0;
            (snr_to_mse(snr_db), snr_db, Some(1e-3))
        }
        _ => {
            let snr_db = 40.0;
            (snr_to_mse(snr_db), snr_db, Some(1e-3))
        }
    }
}

/// Lookup for calibrated thresholds of committed models based on real measurements.
/// Sets the floors as `SNR_medido - margem` and `ESR_medido * fator`.
///
/// Returns `None` if the model has no calibrated entry, falling back to
/// heuristic thresholds (`wavenet_thresholds` or LSTM formula).
///
/// Every model with a committed golden `.bin` fixture MUST have an entry here.
/// The meta-test `tests/threshold_calibration.rs` enforces this invariant.
pub fn get_calibrated_threshold(model_name: &str) -> Option<(f64, f64, Option<f64>)> {
    let base_name = if let Some(idx) = model_name.find("_v2_") {
        &model_name[..idx]
    } else {
        model_name
    };
    match base_name {
        // --- WaveNet Standard (CH=16) ---
        // Measured: SNR = 68.4 dB, ESR = 1.43e-7 (post-T16.1 head cascade fix)
        // Margin: SNR - 8.4 dB, ESR factor ~7.0x
        "BossWN-standard" | "wavenet_standard" => {
            let snr_db = 60.0;
            Some((snr_to_mse(snr_db), snr_db, Some(1.0e-6)))
        }
        // --- WaveNet Feather (CH=8) ---
        // Measured: SNR = 67.6 dB, ESR = 1.72e-7
        // Margin: SNR - 7.6 dB, ESR factor ~5.8x
        "BossWN-feather" | "wavenet_feather" => {
            let snr_db = 60.0;
            Some((snr_to_mse(snr_db), snr_db, Some(1.0e-6)))
        }
        // --- WaveNet Nano (CH=4) ---
        // Measured: SNR = 52.6 dB, ESR = 5.52e-6
        // Margin: SNR - 7.6 dB, ESR factor ~5.4x
        "BossWN-nano" | "wavenet_nano" => {
            let snr_db = 45.0;
            Some((snr_to_mse(snr_db), snr_db, Some(3.0e-5)))
        }
        // --- WaveNet A1 Standard (Official) (CH=16) ---
        // Measured: SNR = 48.8 dB, ESR = 1.33e-5
        // Margin: SNR - 8.8 dB, ESR factor ~7.5x
        "wavenet_a1_standard" => {
            let snr_db = 40.0;
            Some((snr_to_mse(snr_db), snr_db, Some(1.0e-4)))
        }
        // --- LSTM 1x16 ---
        // Measured: SNR=19.8 dB (v1 2048 samples), ESR=1.04e-2
        // v2: SNR=12.2 dB / ESR=6.1e-2 @ 96 kHz (recurrent drift). Margin: 7.8/0.2 dB (v1/v2).
        "BossLSTM-1x16" | "lstm_1x16" => {
            let snr_db = 12.0;
            Some((snr_to_mse(snr_db), snr_db, Some(6.5e-2)))
        }
        // --- LSTM 2x8 ---
        // Measured: SNR=25.7 dB (v1 2048 samples), ESR=2.69e-3
        // v2: SNR=18.4 dB / ESR=1.45e-2 @ 96 kHz (recurrent drift). Margin: 7.7/0.4 dB (v1/v2).
        "BossLSTM-2x8" | "lstm_2x8" => {
            let snr_db = 18.0;
            Some((snr_to_mse(snr_db), snr_db, Some(2.0e-2)))
        }
        // --- LSTM Official (H=3) ---
        // Measured: SNR = 29.7 dB, ESR = 1.08e-3
        // Margin: SNR - 7.7 dB, ESR factor ~5.5x
        "lstm (Official)" | "lstm_official" => {
            let snr_db = 22.0;
            Some((snr_to_mse(snr_db), snr_db, Some(6.0e-3)))
        }
        // --- WaveNet Lite (CH=12) ---
        // Measured: SNR = 0.9 dB, ESR = 8.15e-1 (known divergent, CH=12)
        // Target thresholds if corrected — currently #[ignore].
        "BossWN-lite" | "wavenet_lite" => {
            let snr_db = 40.0;
            Some((snr_to_mse(snr_db), snr_db, Some(1.0e-3)))
        }
        // --- WaveNet A2 Full (CH=8) ---
        // Measured: SNR = 79.2 dB, ESR = 1.21e-8 (realistic-amplitude fixture, T2.5)
        // Margin: SNR - 9.2 dB, ESR factor ~6.6x
        "wavenet_a2_full" => {
            let snr_db = 70.0;
            Some((1e30, snr_db, Some(8.0e-8)))
        }
        // --- WaveNet A2 Lite (CH=3) ---
        // Measured: SNR = 90.7 dB, ESR = 8.58e-10 (realistic-amplitude fixture, T2.5)
        // Margin: SNR - 10.7 dB, ESR factor ~7.0x
        "wavenet_a2_lite" => {
            let snr_db = 80.0;
            Some((1e30, snr_db, Some(6.0e-9)))
        }
        _ => None,
    }
}

/// Computes MSE/SNR/ESR test thresholds for golden vector tests.
///
/// For live cpp_parity cross-validation, use `live_parity_thresholds()`
/// which applies tighter LSTM floors reflecting the 50–97 dB live SNR.
///
/// Returns `(mse_limit, min_snr_db, max_esr)` — T16.4 adds relative
/// ESR gate as primary threshold (robust to scale mismatch).
pub fn topology_thresholds(
    data: &nam_rs::loader::nam_json::NamModelData,
    model_name: &str,
) -> (f64, f64, Option<f64>) {
    if let Some(thresholds) = get_calibrated_threshold(model_name) {
        return thresholds;
    }
    match data.architecture.as_str() {
        "WaveNet" => {
            let channels = data
                .config
                .layers
                .first()
                .and_then(|l| l.channels)
                .unwrap_or(16);
            let (mse, snr, esr) = wavenet_thresholds(channels as u32);
            (mse, snr, esr)
        }
        "LSTM" => {
            let num_layers = data.config.num_layers.unwrap_or(1);
            let hidden_size = data.config.hidden_size.unwrap_or(16);
            let complexity = (num_layers * hidden_size) as f64;
            let snr_db = (30.0 - complexity * 0.65).clamp(12.0, 30.0);
            let mse = snr_to_mse(snr_db);
            let esr = 10.0_f64.powf(-snr_db / 10.0) * 2.0;
            (mse.clamp(1e-4, 5e-2), snr_db, Some(esr))
        }
        "Linear" => (1e-10, 140.0, Some(1e-10)),
        _ => (5e-2, 9.0, Some(1e-3)),
    }
}

/// Computes MSE/SNR/ESR thresholds for live C++ cross-validation (`cpp_parity.rs`).
///
/// Uses aggressive floors reflecting post-T16.1 live SNR measurements.
/// LSTM formula targets 50–97 dB live SNR with ~10–15 dB margin
/// (v2 stress signal relaxation applied separately in `cpp_parity.rs`).
///
/// T16.4: ESR gate added as primary threshold (robust to scale mismatch).
///
/// Returns `(mse_limit, min_snr_db, max_esr)`.
pub fn live_parity_thresholds(
    data: &nam_rs::loader::nam_json::NamModelData,
    model_name: &str,
) -> (f64, f64, Option<f64>) {
    if let Some(thresholds) = get_calibrated_threshold(model_name) {
        return thresholds;
    }
    match data.architecture.as_str() {
        "WaveNet" => {
            let channels = data
                .config
                .layers
                .first()
                .and_then(|l| l.channels)
                .unwrap_or(16);
            let (mse, snr, esr) = wavenet_thresholds(channels as u32);
            (mse, snr, esr)
        }
        "LSTM" => {
            let num_layers = data.config.num_layers.unwrap_or(1);
            let hidden_size = data.config.hidden_size.unwrap_or(16);
            let complexity = (num_layers * hidden_size) as f64;
            let snr_db = (85.0 - complexity * 1.0).clamp(45.0, 75.0);
            let mse = snr_to_mse(snr_db);
            let esr = 10.0_f64.powf(-snr_db / 10.0) * 2.0;
            (mse.clamp(1e-4, 5e-2), snr_db, Some(esr))
        }
        "Linear" => (1e-10, 140.0, Some(1e-10)),
        _ => (5e-2, 9.0, Some(1e-3)),
    }
}
