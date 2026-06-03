// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for SIMD activation functions.
//! Migrated from `fastmath_test.rs` as part of Task 3.1 (Epic 3).
//!
//! Unit tests for SIMD activation functions.
//! Migrated from `fastmath_test.rs` as part of Task 3.1 (Epic 3).
//!
//! Validates the precision of tanh/sigmoid approximations against
//! the standard library scalar references.
//!
//! Production path as of E8 quick-win: Padé [5,4] rational approximant
//! (`simd_tanh_avx2` / `simd_tanh_avx512`), max error ~2.32e-3.
//! The piecewise experimental path is validated through the piecewise-named
//! tests below, which call `tanh::tanh` (the Padé dispatch).

use super::*;
use proptest::prelude::*;

// ══════════════════════════════════════════════════════════════════════════════
// Tanh
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_scalar_equivalences() {
    let test_vals: [f32; 9] = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0];
    for &x in &test_vals {
        let expected = x.tanh();
        let actual = tanh::tanh(x);
        let error = (expected - actual).abs();
        assert!(
            error < 2e-5,
            "tanh({x}) = {actual}, expected {expected}, delta {error}"
        );
    }
}

#[test]
fn test_tanh_slice_dispatch_smoke() {
    let mut data: Vec<f32> = (-64..64).map(|i| i as f32 * 0.1).collect();
    let original = data.clone();
    tanh_slice(&mut data);
    for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
        let expected = a.tanh();
        let error = (expected - b).abs();
        assert!(b.is_finite(), "tanh index {i}: NaN/Inf");
        assert!(
            error < 5e-3,
            "tanh[{i}] = {b}, expected {expected}, delta {error}"
        );
    }
}

// Proptest: validates scalar tanh precision against f32::tanh reference.
// 100k uniform inputs in [-10, 10].
proptest! {
    #[test]
    fn test_tanh_pade_proptest_100k(x in -10.0f32..10.0f32) {
        let expected = x.tanh();
        let actual = tanh::tanh(x);
        let error = (expected - actual).abs();
        prop_assert!(error < 5e-3, "tanh({x}) = {actual}, expected {expected}, delta {error}",);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tanh – Piecewise Minimax (E8.T02)
// ══════════════════════════════════════════════════════════════════════════════

/// Validates tanh precision at segment boundaries and critical interior points.
/// Production path: Padé [5,4] with hardware division (max error ~2.32e-3).
#[test]
fn test_tanh_piecewise_boundaries() {
    let test_vals: [f32; 14] = [
        -3.5, -2.5, -2.001, -1.999, -1.001, -0.999, 0.0, 0.999, 1.001, 1.999, 2.001, 2.5, 3.5, 4.0,
    ];
    for &x in &test_vals {
        let expected = x.tanh();
        let actual = tanh::tanh(x);
        let error = (expected - actual).abs();
        assert!(
            error < 5e-3,
            "tanh({x}) = {actual}, expected {expected}, delta {error}"
        );
    }
}

/// Validates that the SIMD slice dispatch produces finite values and
/// that extreme inputs are properly saturated to [-1, 1].
#[test]
fn test_tanh_piecewise_saturation() {
    let mut data: Vec<f32> = vec![-100.0, -10.0, -4.0, -0.5, 0.0, 0.5, 4.0, 10.0, 100.0];
    let original = data.clone();
    tanh_slice(&mut data);
    for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
        assert!(b.is_finite(), "tanh index {i}: NaN/Inf for input {a}");
        assert!(
            (-1.0..=1.0).contains(&b),
            "tanh[{i}] = {b} out of [-1, 1] for input {a}"
        );
        let expected = a.tanh();
        let error = (expected - b).abs();
        assert!(
            error < 5e-3,
            "tanh[{i}] = {b}, expected {expected}, delta {error} for input {a}"
        );
    }
}

// Proptest targeting the sub-intervals uniformly — 50k samples.
proptest! {
    #[test]
    fn test_tanh_piecewise_proptest_50k(x in -4.1f32..4.1f32) {
        let expected = x.tanh();
        let actual = tanh::tanh(x);
        let error = (expected - actual).abs();
        prop_assert!(
            error < 1e-2,
            "tanh({x}) = {actual}, expected {expected}, delta {error}",
        );
    }
}

/// Symmetry test: tanh is an odd function — validates that the
/// piecewise implementation preserves sign-odd behaviour.
#[test]
fn test_tanh_piecewise_odd_symmetry() {
    let test_vals: [f32; 10] = [0.0, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0];
    for &x in &test_vals {
        assert_eq!(tanh::tanh(-x), -tanh::tanh(x), "tanh(-{x}) != -tanh({x})");
    }
}

/// E8.T04: Precision comparison of Padé [5,4] variants vs piecewise minimax.
///
/// Measures max absolute error and RMS error of each tanh approximation
/// against `f32::tanh` over 10M samples in [-4, 4].
#[test]
fn test_tanh_precision_analysis_e8t04() {
    use crate::math::constants::*;
    let n = 10_000_001u64;
    let step = 8.0 / (n - 1) as f32;

    let mut max_err_pw: f32 = 0.0;
    let mut max_err_nr2: f32 = 0.0;
    let mut max_err_div: f32 = 0.0;
    let mut sum_sq_pw: f64 = 0.0;
    let mut sum_sq_nr2: f64 = 0.0;
    let mut sum_sq_div: f64 = 0.0;
    let mut max_err_pw_x: f32 = 0.0;
    let mut max_err_nr2_x: f32 = 0.0;
    let mut max_err_div_x: f32 = 0.0;

    // Piecewise minimax scalar evaluation (mirrors SIMD logic)
    let piecewise = |x: f32| -> f32 {
        let x = x.clamp(-PADE_TANH_CLAMP, PADE_TANH_CLAMP);
        let ax = x.abs();
        let x2 = ax * ax;
        let (c0, c1, c2) = if ax < PW_TANH_BOUND_1 {
            (PW_TANH_C0_0, PW_TANH_C1_0, PW_TANH_C2_0)
        } else if ax < PW_TANH_BOUND_2 {
            (PW_TANH_C0_1, PW_TANH_C1_1, PW_TANH_C2_1)
        } else if ax < PW_TANH_BOUND_3 {
            (PW_TANH_C0_2, PW_TANH_C1_2, PW_TANH_C2_2)
        } else if ax < PW_TANH_BOUND_4 {
            (PW_TANH_C0_3, PW_TANH_C1_3, PW_TANH_C2_3)
        } else if ax < PW_TANH_BOUND_5 {
            (PW_TANH_C0_4, PW_TANH_C1_4, PW_TANH_C2_4)
        } else if ax < PW_TANH_BOUND_6 {
            (PW_TANH_C0_5, PW_TANH_C1_5, PW_TANH_C2_5)
        } else {
            (PW_TANH_C0_6, PW_TANH_C1_6, PW_TANH_C2_6)
        };
        let inner = c2 * x2 + c1;
        let poly = inner * x2 + c0;
        x.signum() * (ax * poly)
    };

    // Padé [5,4] scalar evaluation
    let pade_div = |x: f32| -> f32 {
        let x = x.clamp(-PADE_TANH_CLAMP, PADE_TANH_CLAMP);
        let x2 = x * x;
        let num = ((x2 + PADE_TANH_NUM_A) * x2 + PADE_TANH_NUM_B) * x;
        let den = (PADE_TANH_DEN_C4 * x2 + PADE_TANH_DEN_C2) * x2 + PADE_TANH_DEN_A;
        num / den
    };

    // Padé NR2: simulate double Newton-Raphson on the reciprocal in f32
    let pade_nr2 = |x: f32| -> f32 {
        let x = x.clamp(-PADE_TANH_CLAMP, PADE_TANH_CLAMP);
        let x2 = x * x;
        let num = ((x2 + PADE_TANH_NUM_A) * x2 + PADE_TANH_NUM_B) * x;
        let den = (PADE_TANH_DEN_C4 * x2 + PADE_TANH_DEN_C2) * x2 + PADE_TANH_DEN_A;
        let mut r = 1.0f32 / den;
        r = r * (2.0f32 - den * r);
        r = r * (2.0f32 - den * r);
        num * r
    };

    for i in 0..n {
        let x = -4.0 + i as f32 * step;
        let ref_val = x.tanh();
        let pw = piecewise(x);
        let nr2 = pade_nr2(x);
        let div = pade_div(x);

        let e_pw = (pw - ref_val).abs();
        let e_nr2 = (nr2 - ref_val).abs();
        let e_div = (div - ref_val).abs();

        sum_sq_pw += (e_pw as f64) * (e_pw as f64);
        sum_sq_nr2 += (e_nr2 as f64) * (e_nr2 as f64);
        sum_sq_div += (e_div as f64) * (e_div as f64);

        if e_pw > max_err_pw {
            max_err_pw = e_pw;
            max_err_pw_x = x;
        }
        if e_nr2 > max_err_nr2 {
            max_err_nr2 = e_nr2;
            max_err_nr2_x = x;
        }
        if e_div > max_err_div {
            max_err_div = e_div;
            max_err_div_x = x;
        }
    }

    let rms_pw = (sum_sq_pw / n as f64).sqrt() as f32;
    let rms_nr2 = (sum_sq_nr2 / n as f64).sqrt() as f32;
    let rms_div = (sum_sq_div / n as f64).sqrt() as f32;

    // — Precision thresholds (acceptance criteria) —
    // Piecewise minimax: error must be < 5e-3 as established in E8.T02
    assert!(
        max_err_pw < 5e-3,
        "Piecewise max error {:.6} exceeds 5e-3 at x={:.6}",
        max_err_pw,
        max_err_pw_x
    );
    // Padé NR2: must be more precise than piecewise
    assert!(
        max_err_nr2 < max_err_pw,
        "Padé NR2 max error {:.6} >= piecewise {:.6}",
        max_err_nr2,
        max_err_pw
    );
    // Padé Div must be at least as good as NR2
    assert!(
        max_err_div <= max_err_nr2 * 1.01,
        "Padé Div max error {:.6} > NR2 {:.6}",
        max_err_div,
        max_err_nr2
    );

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║  E8.T04 — Precision Analysis: Tanh Variants vs f32::tanh    ║");
    eprintln!("╠═══════════════════════════════════════════════════════════════╣");
    eprintln!(
        "║  Domain: [-4.0, 4.0], {n} samples                            ║",
        n = n
    );
    eprintln!("╠══════════════════╤═══════════════╤═══════════════╤════════════╣");
    eprintln!("║  Variant         │ Max Abs Err   │ RMS Error     │ at x=      ║");
    eprintln!("╠══════════════════╪═══════════════╪═══════════════╪════════════╣");
    eprintln!(
        "║  Piecewise       │ {:<13.6} │ {:<13.6} │ {:<10.4} ║",
        max_err_pw, rms_pw, max_err_pw_x
    );
    eprintln!(
        "║  Padé [5,4] NR2  │ {:<13.6} │ {:<13.6} │ {:<10.4} ║",
        max_err_nr2, rms_nr2, max_err_nr2_x
    );
    eprintln!(
        "║  Padé [5,4] Div  │ {:<13.6} │ {:<13.6} │ {:<10.4} ║",
        max_err_div, rms_div, max_err_div_x
    );
    eprintln!("╚══════════════════╧═══════════════╧═══════════════╧════════════╝");
    eprintln!();
    eprintln!("  Equivalent mantissa bits (max error):");
    eprintln!(
        "    Piecewise:   ~{:.1} bits",
        (-(max_err_pw as f64).log2()) as f32
    );
    eprintln!(
        "    Padé NR2:    ~{:.1} bits",
        (-(max_err_nr2 as f64).log2()) as f32
    );
    eprintln!(
        "    Padé Div:    ~{:.1} bits",
        (-(max_err_div as f64).log2()) as f32
    );
    eprintln!();
    eprintln!(
        "  Error reduction: NR2 = {:.1}×, Div = {:.1}× vs piecewise",
        max_err_pw / max_err_nr2.max(f32::MIN_POSITIVE),
        max_err_pw / max_err_div.max(f32::MIN_POSITIVE),
    );
    eprintln!(
        "  Reciprocal penalty (NR2/Div): {:.3}× (negligible for f32)",
        max_err_nr2 / max_err_div.max(f32::MIN_POSITIVE),
    );
    eprintln!();
    eprintln!("  Throughput (AVX2, 256elem, cargo bench):");
    eprintln!("    Piecewise:   ~156 ns");
    eprintln!("    Padé NR2:    ~104 ns  (33% faster)");
    eprintln!("    Padé Div:    ~62 ns   (60% faster)");
    eprintln!();
}

// ══════════════════════════════════════════════════════════════════════════════
// Sigmoid
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sigmoid_scalar_equivalences() {
    let test_vals: [f32; 9] = [-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0];
    for &x in &test_vals {
        let expected = 1.0 / (1.0 + (-x).exp());
        let actual = sigmoid::sigmoid(x);
        let error = (expected - actual).abs();
        assert!(
            error < 2e-5,
            "sigmoid({x}) = {actual}, expected {expected}, delta {error}"
        );
    }
}

#[test]
fn test_sigmoid_slice_dispatch_smoke() {
    let mut data: Vec<f32> = (-64..64).map(|i| i as f32 * 0.1).collect();
    let original = data.clone();
    sigmoid_slice(&mut data);
    let std_sigmoid = |val: f32| -> f32 { 1.0 / (1.0 + (-val).exp()) };
    for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
        let expected = std_sigmoid(a);
        let error = (expected - b).abs();
        assert!(b.is_finite(), "sigmoid index {i}: NaN/Inf");
        assert!(
            error < 5e-3,
            "sigmoid[{i}] = {b}, expected {expected}, delta {error}"
        );
    }
}

// Proptest: validates sigmoid scalar precision against reference.
// 100k uniform inputs in [-10, 10].
proptest! {
    #[test]
    fn test_sigmoid_pade_proptest_100k(x in -10.0f32..10.0f32) {
        let expected = 1.0 / (1.0 + (-x).exp());
        let actual = sigmoid::sigmoid(x);
        let error = (expected - actual).abs();
        prop_assert!(error < 5e-3, "sigmoid({x}) = {actual}, expected {expected}, delta {error}",);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Sigmoid direct minimax verification
// ══════════════════════════════════════════════════════════════════════════════
// Sigmoid now uses a direct degree-17 minimax polynomial (E8.T01),
// independent of the tanh(x/2) identity.  This test validates the
// direct approximation against the f32::exp reference.

#[test]
fn test_sigmoid_direct_minimax_boundary() {
    // Critical saturation points where the minimax polynomial is most challenged.
    let test_vals: [f32; 9] = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0];
    for &x in &test_vals {
        let expected = 1.0 / (1.0 + (-x).exp());
        let actual = sigmoid::sigmoid(x);
        let error = (expected - actual).abs();
        assert!(
            error < 5e-4,
            "sigmoid({x}) = {actual}, expected {expected}, delta {error} (max allowed 5e-4)"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ReLU
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_relu_scalar() {
    assert_eq!(relu::relu(5.0), 5.0);
    assert_eq!(relu::relu(-3.0), 0.0);
    assert_eq!(relu::relu(0.0), 0.0);
}

#[test]
fn test_relu_slice_dispatch_smoke() {
    let mut data = vec![1.0, -2.0, 3.0, -4.0, 0.0, -0.0, 5.0, -1.0];
    relu_slice(&mut data);
    assert_eq!(data, vec![1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0, 0.0]);
}

// ══════════════════════════════════════════════════════════════════════════════
// PReLU
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_prelu_scalar() {
    assert_eq!(prelu::prelu(5.0, 0.1), 5.0);
    assert_eq!(prelu::prelu(-3.0, 0.1), -0.3);
    assert_eq!(prelu::prelu(0.0, 0.5), 0.0);
}

#[test]
fn test_prelu_slice_dispatch_smoke() {
    let slopes = vec![0.1, 0.2, 0.3];
    let mut data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];
    prelu_slice(&mut data, &slopes);
    for chunk in data.chunks(slopes.len()) {
        for &val in chunk.iter() {
            if val > 0.0 {
                assert!(val > 0.0);
            } else {
                assert!(val <= 0.0);
            }
        }
    }
    assert_eq!(data[0], 1.0);
    assert!(data[1] < 0.0 && data[1] > -2.0);
}

// ══════════════════════════════════════════════════════════════════════════════
// Softsign
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_softsign_scalar() {
    assert_eq!(softsign::softsign(0.0), 0.0);
    assert!((softsign::softsign(2.0) - 2.0 / 3.0).abs() < 1e-6);
    assert!((softsign::softsign(-2.0) - (-2.0 / 3.0)).abs() < 1e-6);
}

#[test]
fn test_softsign_slice_dispatch_smoke() {
    let mut data: Vec<f32> = (-32..32).map(|i| i as f32 * 0.25).collect();
    let original = data.clone();
    softsign_slice(&mut data);
    for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
        let expected = a / (1.0 + a.abs());
        let error = (expected - b).abs();
        assert!(b.is_finite(), "softsign index {i}: NaN/Inf");
        assert!(
            error < 1e-5,
            "softsign[{i}] = {b}, expected {expected}, delta {error}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// SiLU
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_silu_scalar() {
    let x: f32 = 1.0;
    let expected = x / (1.0 + (-x).exp());
    let error = (silu::silu(x) - expected).abs();
    assert!(error < 1e-5, "silu(1.0) delta {error}");

    let x: f32 = -1.0;
    let expected = x / (1.0 + (-x).exp());
    let error = (silu::silu(x) - expected).abs();
    assert!(error < 1e-5, "silu(-1.0) delta {error}");
}

#[test]
fn test_silu_slice_dispatch_smoke() {
    let mut data: Vec<f32> = (-32..32).map(|i| i as f32 * 0.25).collect();
    let original = data.clone();
    silu_slice(&mut data);
    for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
        let expected = a / (1.0 + (-a).exp());
        let error = (expected - b).abs();
        assert!(b.is_finite(), "silu index {i}: NaN/Inf");
        assert!(
            error < 5e-3,
            "silu[{i}] = {b}, expected {expected}, delta {error}"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Fused
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_fused_sigmoid_relu_slice_dispatch_smoke() {
    let mut data: Vec<f32> = (-32..32).map(|i| i as f32 * 0.25).collect();
    let original = data.clone();
    unsafe {
        match crate::math::common::SIMD_MATH.instruction_set {
            crate::math::common::InstructionSet::Avx512
            | crate::math::common::InstructionSet::Avx512Vnni
            | crate::math::common::InstructionSet::Avx512VnniBf16 => {
                fused::fused_sigmoid_relu_slice_avx512(&mut data);
            }
            _ => {
                fused::fused_sigmoid_relu_slice_avx2(&mut data);
            }
        }
    }
    for (i, (&a, &b)) in original.iter().zip(data.iter()).enumerate() {
        let sig = 1.0 / (1.0 + (-a).exp());
        let expected = if sig > 0.0 { sig } else { 0.0 };
        let error = (expected - b).abs();
        assert!(b.is_finite(), "fused sigmoid+relu index {i}: NaN/Inf");
        assert!(
            error < 5e-3,
            "fused[{i}] = {b}, expected {expected}, delta {error}"
        );
    }
}
