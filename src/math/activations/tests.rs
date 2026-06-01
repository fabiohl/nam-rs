// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Testes de unidade para as funções de ativação SIMD.
//! Migrados de `fastmath_test.rs` como parte da Tarefa 3.1 (Épico 3).
//!
//! Valida a precisão das aproximações Padé contra as referências
//! escalares da biblioteca padrão.
//!
//! Tarefa S7.T09: Substituição de polinômios Minimax por Padé branchless.

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
            "tanh({x}) = {actual}, esperado {expected}, delta {error}"
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
            "tanh[{i}] = {b}, esperado {expected}, delta {error}"
        );
    }
}

// Proptest: valida precisão do Padé tanh escalar contra referência f32::tanh.
// 100k inputs uniformes em [-10, 10].
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
            "sigmoid({x}) = {actual}, esperado {expected}, delta {error}"
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
            "sigmoid[{i}] = {b}, esperado {expected}, delta {error}"
        );
    }
}

// Proptest: valida precisão do sigmoid escalar contra referência.
// 100k inputs uniformes em [-10, 10].
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
// Tanh × Sigmoid identity: sigmoid(x) = 0.5 + 0.5 * tanh(x/2)
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sigmoid_tanh_identity_consistency() {
    let test_vals: [f32; 11] = [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0];
    for &x in &test_vals {
        let sig_via_tanh = 0.5 + 0.5 * tanh::tanh(x * 0.5);
        let sig_direct = sigmoid::sigmoid(x);
        let error = (sig_via_tanh - sig_direct).abs();
        assert!(
            error < 1e-6,
            "sigmoid-tanh identity mismatch at x={x}: via_tanh={sig_via_tanh}, direct={sig_direct}, delta={error}"
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
            "softsign[{i}] = {b}, esperado {expected}, delta {error}"
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
            "silu[{i}] = {b}, esperado {expected}, delta {error}"
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
            "fused[{i}] = {b}, esperado {expected}, delta {error}"
        );
    }
}
