// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Exhaustive validation tests for `LstmModelDyn` (Task 2.2.4).
//!
//! Covers arbitrary (num_layers, hidden_size) topologies not matched by the
//! 10 static const-generic profiles.

use nam_rs::math::common::half::f32_to_f16_bits;
use nam_rs::models::lstm::LstmModelDyn;

// =============================================================================
// Helper: deterministic weight filling for a LstmModelDyn
// =============================================================================

/// Fills a `LstmModelDyn` with deterministic (non-zero) weights, biases, and
/// head values so that numeric parity tests are stable and reproducible.
///
/// Uses the same pattern as `fill_dyn_layer` in `src/models/lstm/tests.rs`.
fn fill_model_dyn(model: &mut LstmModelDyn) {
    let hidden_size = model.layers[0].hidden_size;
    let num_layers = model.layers.len();

    for (li, layer) in model.layers.iter_mut().enumerate() {
        let h = layer.hidden_size;
        let input_size = layer.input_size;
        let ih = input_size + h;

        for i in 0..(4 * h) {
            layer.bias[i] = 0.1 * (1 + ((i + li * 4 * h) % 4)) as f32;
        }
        for k in 0..4 {
            let w_start = k * ih * h;
            for j in 0..ih {
                for hi in 0..h {
                    layer.input_hidden_weights[w_start + j * h + hi] =
                        f32_to_f16_bits(0.01 * (j + hi + k + li + 1) as f32);
                }
            }
        }
    }

    for i in 0..hidden_size {
        let w = f32_to_f16_bits(0.1 * (i + num_layers) as f32);
        model.head_weights[i] = w;
        model.head_weights_f32[i] = 0.1 * (i + num_layers) as f32;
    }
    model.head_bias = 0.25;
    model.use_f32_head = true;
}

/// Runs the model on a sine-wave input and asserts SIMD vs scalar parity.
fn assert_model_dyn_parity(num_layers: usize, hidden_size: usize) {
    let mut model_simd = LstmModelDyn::new(num_layers, hidden_size);
    let mut model_scalar = LstmModelDyn::new(num_layers, hidden_size);

    model_simd.use_f32_head = true;
    model_scalar.use_f32_head = true;

    // Fill both models with identical weights
    fill_model_dyn(&mut model_simd);
    // Copy weights from simd to scalar
    for li in 0..num_layers {
        model_scalar.layers[li]
            .input_hidden_weights
            .copy_from_slice(&model_simd.layers[li].input_hidden_weights);
        model_scalar.layers[li]
            .bias
            .copy_from_slice(&model_simd.layers[li].bias);
        model_scalar.layers[li]
            .state
            .copy_from_slice(&model_simd.layers[li].state);
        model_scalar.layers[li]
            .cell_state
            .copy_from_slice(&model_simd.layers[li].cell_state);
    }
    model_scalar
        .head_weights
        .copy_from_slice(&model_simd.head_weights);
    model_scalar
        .head_weights_f32
        .copy_from_slice(&model_simd.head_weights_f32);
    model_scalar.head_bias = model_simd.head_bias;

    let n_samples = 64;
    let input: Vec<f32> = (0..n_samples).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut out_simd = vec![0.0f32; n_samples];
    let mut out_scalar = vec![0.0f32; n_samples];

    model_simd.process(&input, &mut out_simd);
    model_scalar.process_scalar(&input, &mut out_scalar);

    for i in 0..n_samples {
        let diff = (out_simd[i] - out_scalar[i]).abs();
        let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
        let rel_diff = diff / max_val;
        assert!(
            rel_diff < 5e-3
                || (out_simd[i].is_nan() && out_scalar[i].is_nan())
                || (out_simd[i].is_infinite()
                    && out_scalar[i].is_infinite()
                    && out_simd[i].to_bits() == out_scalar[i].to_bits()),
            "ModelDyn {num_layers}x{hidden_size}: SIMD vs scalar parity failed at [{i}]: \
             SIMD={}, Scalar={}, Delta={}, Rel={}",
            out_simd[i],
            out_scalar[i],
            diff,
            rel_diff,
        );
    }
}

// =============================================================================
// A. Unit tests — fast, non-ignored
// =============================================================================

#[test]
fn test_model_dyn_parity_1x7() {
    assert_model_dyn_parity(1, 7);
}

#[test]
fn test_model_dyn_parity_2x5() {
    assert_model_dyn_parity(2, 5);
}

#[test]
fn test_model_dyn_parity_3x10() {
    assert_model_dyn_parity(3, 10);
}

#[test]
fn test_model_dyn_parity_2x13() {
    assert_model_dyn_parity(2, 13);
}

#[test]
fn test_model_dyn_parity_4x6() {
    assert_model_dyn_parity(4, 6);
}

#[test]
fn test_model_dyn_parity_1x1() {
    assert_model_dyn_parity(1, 1);
}

#[test]
fn test_model_dyn_parity_5x4() {
    assert_model_dyn_parity(5, 4);
}

#[test]
fn test_model_dyn_parity_1x20() {
    assert_model_dyn_parity(1, 20);
}

#[test]
fn test_model_dyn_parity_3x8() {
    assert_model_dyn_parity(3, 8);
}

// Determinism: two identically-initialized LstmModelDyn produce identical output.
#[test]
fn test_model_dyn_determinism() {
    let mut model_a = LstmModelDyn::new(2, 10);
    let mut model_b = LstmModelDyn::new(2, 10);

    fill_model_dyn(&mut model_a);
    // Copy to model_b
    for li in 0..2 {
        model_b.layers[li]
            .input_hidden_weights
            .copy_from_slice(&model_a.layers[li].input_hidden_weights);
        model_b.layers[li]
            .bias
            .copy_from_slice(&model_a.layers[li].bias);
        model_b.layers[li]
            .state
            .copy_from_slice(&model_a.layers[li].state);
        model_b.layers[li]
            .cell_state
            .copy_from_slice(&model_a.layers[li].cell_state);
    }
    model_b.head_weights.copy_from_slice(&model_a.head_weights);
    model_b
        .head_weights_f32
        .copy_from_slice(&model_a.head_weights_f32);
    model_b.head_bias = model_a.head_bias;
    model_b.use_f32_head = true;

    let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.13).sin()).collect();
    let mut out_a = vec![0.0f32; 64];
    let mut out_b = vec![0.0f32; 64];

    model_a.process(&input, &mut out_a);
    model_b.process(&input, &mut out_b);

    for i in 0..64 {
        assert!(
            (out_a[i] - out_b[i]).abs() < 1e-6,
            "ModelDyn determinism failed at [{i}]: {} vs {}",
            out_a[i],
            out_b[i],
        );
    }
}

// Block-size invariance: processing in chunks of different sizes yields identical result.
#[test]
fn test_model_dyn_block_size_invariance() {
    let n_samples = 64;
    let input: Vec<f32> = (0..n_samples).map(|i| (i as f32 * 0.1).sin()).collect();
    let block_sizes = [1usize, 8, 16, 32, 64];

    let mut reference_out = [0.0f32; 64];

    for (idx, &block_size) in block_sizes.iter().enumerate() {
        let mut model = LstmModelDyn::new(3, 8);
        model.use_f32_head = true;
        fill_model_dyn(&mut model);

        let mut out = [0.0f32; 64];
        for chunk in 0..(n_samples / block_size) {
            let start = chunk * block_size;
            let end = start + block_size;
            model.process(&input[start..end], &mut out[start..end]);
        }

        if idx == 0 {
            reference_out.copy_from_slice(&out);
        } else {
            for i in 0..n_samples {
                assert!(
                    (out[i] - reference_out[i]).abs() < 1e-6,
                    "ModelDyn block-size invariance failed: block_size={block_size} sample[{i}]: {} vs {}",
                    out[i],
                    reference_out[i],
                );
            }
        }
    }
}

// No-panic for extreme/edge topologies — output must be finite.
#[test]
fn test_model_dyn_no_panic_edge() {
    let topologies: &[(usize, usize)] = &[
        (1, 1),   // absolute minimum
        (1, 2),   // tiny
        (1, 40),  // large single layer
        (1, 64),  // large single layer
        (2, 3),   // small stacked
        (3, 5),   // shallow stack
        (4, 6),   // medium
        (6, 4),   // deeper stack, small H
        (8, 4),   // many layers
        (2, 20),  // medium H
        (1, 128), // large H
        (4, 12),  // medium-deep
    ];

    let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

    for &(num_layers, hidden_size) in topologies {
        let mut model = LstmModelDyn::new(num_layers, hidden_size);
        model.use_f32_head = true;
        fill_model_dyn(&mut model);

        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "ModelDyn {num_layers}x{hidden_size}: output[{i}] is NaN/Inf: {v}",
            );
        }
    }
}

// Zero input produces finite output (no NaN/Inf when processing silence).
#[test]
fn test_model_dyn_zero_input() {
    let topologies: &[(usize, usize)] = &[(1, 7), (2, 5), (3, 10), (4, 6)];

    for &(num_layers, hidden_size) in topologies {
        let mut model = LstmModelDyn::new(num_layers, hidden_size);
        model.use_f32_head = true;
        fill_model_dyn(&mut model);

        let input = vec![0.0f32; 64];
        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "ModelDyn {num_layers}x{hidden_size}: zero-input output[{i}] is NaN/Inf: {v}",
            );
        }
    }
}

// reset_states() zeros hidden, cell, and gates for all layers.
#[test]
fn test_model_dyn_reset_states() {
    let mut model = LstmModelDyn::new(3, 8);
    model.use_f32_head = true;
    fill_model_dyn(&mut model);

    // Process to push states away from zero
    let input = vec![1.0f32; 32];
    let mut out = vec![0.0f32; 32];
    model.process(&input, &mut out);

    // Artificially perturb
    for layer in &mut model.layers {
        layer.cell_state.iter_mut().for_each(|c| *c = 5.0);
        layer.state.iter_mut().for_each(|s| *s = 5.0);
    }

    // Verify non-zero
    assert_ne!(model.layers[0].cell_state[0], 0.0);

    model.reset_states();

    for (li, layer) in model.layers.iter().enumerate() {
        for (j, &v) in layer.state.iter().enumerate() {
            assert_eq!(
                v, 0.0,
                "ModelDyn reset: layer[{li}] state[{j}] non-zero: {v}"
            );
        }
        for (j, &v) in layer.cell_state.iter().enumerate() {
            assert_eq!(
                v, 0.0,
                "ModelDyn reset: layer[{li}] cell_state[{j}] non-zero: {v}"
            );
        }
        for (j, &v) in layer.gates.iter().enumerate() {
            assert_eq!(
                v, 0.0,
                "ModelDyn reset: layer[{li}] gates[{j}] non-zero: {v}"
            );
        }
    }
}

// reset_input_slots() zeros only input slot, preserving hidden and cell state.
#[test]
fn test_model_dyn_reset_input_slots() {
    let mut model = LstmModelDyn::new(3, 8);
    model.use_f32_head = true;
    fill_model_dyn(&mut model);

    // Process to push hidden/cell away from zero
    let input = vec![1.0f32; 32];
    let mut out = vec![0.0f32; 32];
    model.process(&input, &mut out);

    // Capture hidden/cell state snapshot
    let saved_hidden: Vec<Vec<f32>> = model
        .layers
        .iter()
        .map(|l| l.get_hidden_state().to_vec())
        .collect();
    let saved_cell: Vec<Vec<f32>> = model.layers.iter().map(|l| l.cell_state.to_vec()).collect();

    // Verify hidden is non-zero
    assert!(
        saved_hidden[0].iter().any(|&v| v != 0.0),
        "Expected non-zero hidden state",
    );

    model.reset_input_slots();

    for (li, layer) in model.layers.iter().enumerate() {
        // Input slot (first input_size entries of state) should be zero
        for j in 0..layer.input_size {
            assert_eq!(
                layer.state[j], 0.0,
                "ModelDyn reset_input_slots: layer[{li}] input_slot[{j}] non-zero: {}",
                layer.state[j],
            );
        }
        // Hidden state (after input_size) should be preserved
        let hidden = layer.get_hidden_state();
        for (j, &v) in hidden.iter().enumerate() {
            assert!(
                (v - saved_hidden[li][j]).abs() < 1e-6,
                "ModelDyn reset_input_slots: layer[{li}] hidden[{j}] changed: {} -> {v}",
                saved_hidden[li][j],
            );
        }
        // Cell state should be preserved
        for (j, &v) in layer.cell_state.iter().enumerate() {
            assert!(
                (v - saved_cell[li][j]).abs() < 1e-6,
                "ModelDyn reset_input_slots: layer[{li}] cell[{j}] changed: {} -> {v}",
                saved_cell[li][j],
            );
        }
    }
}

// State evolution: hidden state changes across multiple processing steps.
#[test]
fn test_model_dyn_state_evolution() {
    let mut model = LstmModelDyn::new(3, 8);
    model.use_f32_head = true;
    fill_model_dyn(&mut model);

    let input = vec![0.7f32; 1];
    let mut out = [0.0f32; 1];

    model.process(&input, &mut out);
    let hidden_1: Vec<f32> = model.layers[0].get_hidden_state().to_vec();

    for _ in 0..9 {
        model.process(&input, &mut out);
    }
    let hidden_10: Vec<f32> = model.layers[0].get_hidden_state().to_vec();

    // State must have evolved
    for j in 0..8 {
        assert!(
            (hidden_1[j] - hidden_10[j]).abs() > 1e-4,
            "ModelDyn state_evolution: hidden[{j}] unchanged: {} vs {}",
            hidden_1[j],
            hidden_10[j],
        );
    }
}

// Quantized head path: SIMD vs scalar parity with use_f32_head=false.
#[test]
fn test_model_dyn_parity_quantized_head() {
    let (num_layers, hidden_size) = (2, 10);
    let mut model_simd = LstmModelDyn::new(num_layers, hidden_size);
    let mut model_scalar = LstmModelDyn::new(num_layers, hidden_size);

    fill_model_dyn(&mut model_simd);
    // Switch to quantized head after filling (fill_model_dyn sets use_f32_head=true)
    model_simd.use_f32_head = false;
    model_scalar.use_f32_head = false;

    // Override head with quantized u16 values (not f32)
    for i in 0..hidden_size {
        let w = f32_to_f16_bits(0.1 * i as f32);
        model_simd.head_weights[i] = w;
        model_scalar.head_weights[i] = w;
    }
    model_simd.head_weights_f32.fill(0.0);
    model_scalar.head_weights_f32.fill(0.0);

    // Copy layer weights
    for li in 0..num_layers {
        model_scalar.layers[li]
            .input_hidden_weights
            .copy_from_slice(&model_simd.layers[li].input_hidden_weights);
        model_scalar.layers[li]
            .bias
            .copy_from_slice(&model_simd.layers[li].bias);
        model_scalar.layers[li]
            .state
            .copy_from_slice(&model_simd.layers[li].state);
        model_scalar.layers[li]
            .cell_state
            .copy_from_slice(&model_simd.layers[li].cell_state);
    }
    model_scalar.head_bias = model_simd.head_bias;

    let n_samples = 64;
    let input: Vec<f32> = (0..n_samples).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut out_simd = vec![0.0f32; n_samples];
    let mut out_scalar = vec![0.0f32; n_samples];

    model_simd.process(&input, &mut out_simd);
    model_scalar.process_scalar(&input, &mut out_scalar);

    for i in 0..n_samples {
        let diff = (out_simd[i] - out_scalar[i]).abs();
        let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
        let rel_diff = diff / max_val;
        assert!(
            rel_diff < 5e-3
                || (out_simd[i].is_nan() && out_scalar[i].is_nan())
                || (out_simd[i].is_infinite()
                    && out_scalar[i].is_infinite()
                    && out_simd[i].to_bits() == out_scalar[i].to_bits()),
            "ModelDyn quantized-head parity failed at [{i}]: SIMD={}, Scalar={}, Rel={}",
            out_simd[i],
            out_scalar[i],
            rel_diff,
        );
    }
}

// =============================================================================
// B. Proptest — ignored (slow), exhaustive random-weight parity
// =============================================================================

#[cfg(test)]
mod proptest_tests {
    use nam_rs::math::common::half::f32_to_f16_bits;
    use nam_rs::models::lstm::LstmModelDyn;
    use proptest::prelude::*;

    fn quantize_weight(f: f32, is_bf16: bool) -> u16 {
        if is_bf16 {
            (f.to_bits() >> 16) as u16
        } else {
            f32_to_f16_bits(f)
        }
    }

    proptest! {
            #![proptest_config(ProptestConfig {
                failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
                .. ProptestConfig::with_cases(200)
            })]

    /// Proptest: random (num_layers, hidden_size, weights, input) —
        /// ensures scalar vs SIMD parity across arbitrary dynamic topologies.
        #[test]
        #[ignore]
        fn test_model_dyn_proptest_scalar_simd_parity(
            num_layers in 1usize..=5usize,
            hidden_size in 2usize..=32usize,
        ) {
            let is_bf16 = nam_rs::math::common::SimdMathConfig::get().instruction_set
                == nam_rs::math::common::InstructionSet::Avx512VnniBf16;

            let n_samples = 32;
            let mut rng = proptest::test_runner::TestRng::deterministic_rng(
                proptest::test_runner::RngAlgorithm::ChaCha,
            );

            // Generate random input (bounded for stability)
            let input: Vec<f32> = (0..n_samples)
                .map(|_| -1.0f32 + rng.random::<f32>() * 2.0)
                .collect();

            // Build model with random weights
            let mut model_simd = LstmModelDyn::new(num_layers, hidden_size);
            let mut model_scalar = LstmModelDyn::new(num_layers, hidden_size);

            // Use f32 head for cleaner comparison
            model_simd.use_f32_head = true;
            model_scalar.use_f32_head = true;

            // Fill layers with random quantized weights (scaled down for stability)
            for li in 0..num_layers {
                let h = hidden_size;
                let input_size = model_simd.layers[li].input_size;
                let ih = input_size + h;
                let h4 = 4 * h;
                let w_total = 4 * ih * h;

                for i in 0..w_total {
                    let w = (-1.5f32 + rng.random::<f32>() * 3.0) / 8.0;
                    model_simd.layers[li].input_hidden_weights[i] =
                        quantize_weight(w, is_bf16);
                    model_scalar.layers[li].input_hidden_weights[i] =
                        model_simd.layers[li].input_hidden_weights[i];
                }
                for i in 0..h4 {
                    let b = (-1.5f32 + rng.random::<f32>() * 3.0) / 8.0;
                    model_simd.layers[li].bias[i] = b;
                    model_scalar.layers[li].bias[i] = b;
                }
            }

            // Random head weights (f32 path)
            for i in 0..hidden_size {
                let w = -1.0f32 + rng.random::<f32>() * 2.0;
                model_simd.head_weights_f32[i] = w;
                model_scalar.head_weights_f32[i] = w;
            }

            let mut out_simd = vec![0.0f32; n_samples];
            let mut out_scalar = vec![0.0f32; n_samples];

            model_simd.reset_states();
            model_scalar.reset_states();

            model_simd.process(&input, &mut out_simd);
            model_scalar.process_scalar(&input, &mut out_scalar);

            for i in 0..n_samples {
                let diff = (out_simd[i] - out_scalar[i]).abs();
                let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
                let rel_diff = diff / max_val;
                assert!(
                    rel_diff < 5e-3
                        || (out_simd[i].is_nan() && out_scalar[i].is_nan())
                        || (out_simd[i].is_infinite()
                            && out_scalar[i].is_infinite()
                            && out_simd[i].to_bits() == out_scalar[i].to_bits()),
                    "ModelDyn proptest {num_layers}x{hidden_size} parity failed at [{i}]: \
                     SIMD={}, Scalar={}, Delta={}, Rel={}",
                    out_simd[i],
                    out_scalar[i],
                    diff,
                    rel_diff,
                );
            }
        }

        /// Proptest: random weights with quantized head (use_f32_head=false) —
        /// scalar vs SIMD parity.
        #[test]
        #[ignore]
        fn test_model_dyn_proptest_quantized_head_parity(
            num_layers in 1usize..=4usize,
            hidden_size in 2usize..=24usize,
        ) {
            let is_bf16 = nam_rs::math::common::SimdMathConfig::get().instruction_set
                == nam_rs::math::common::InstructionSet::Avx512VnniBf16;

            let n_samples = 32;
            let mut rng = proptest::test_runner::TestRng::deterministic_rng(
                proptest::test_runner::RngAlgorithm::ChaCha,
            );

            let input: Vec<f32> = (0..n_samples)
                .map(|_| -1.0f32 + rng.random::<f32>() * 2.0)
                .collect();

            let mut model_simd = LstmModelDyn::new(num_layers, hidden_size);
            let mut model_scalar = LstmModelDyn::new(num_layers, hidden_size);

            // Quantized head path
            model_simd.use_f32_head = false;
            model_scalar.use_f32_head = false;

            // Fill layers with random quantized weights
            for li in 0..num_layers {
                let h = hidden_size;
                let input_size = model_simd.layers[li].input_size;
                let ih = input_size + h;
                let h4 = 4 * h;
                let w_total = 4 * ih * h;

                for i in 0..w_total {
                    let w = (-1.5f32 + rng.random::<f32>() * 3.0) / 8.0;
                    model_simd.layers[li].input_hidden_weights[i] =
                        quantize_weight(w, is_bf16);
                    model_scalar.layers[li].input_hidden_weights[i] =
                        model_simd.layers[li].input_hidden_weights[i];
                }
                for i in 0..h4 {
                    let b = (-1.5f32 + rng.random::<f32>() * 3.0) / 8.0;
                    model_simd.layers[li].bias[i] = b;
                    model_scalar.layers[li].bias[i] = b;
                }
            }

            // Random quantized head weights
            for i in 0..hidden_size {
                let w = (-1.0f32 + rng.random::<f32>() * 2.0) / 8.0;
                let q = quantize_weight(w, is_bf16);
                model_simd.head_weights[i] = q;
                model_scalar.head_weights[i] = q;
            }

            let mut out_simd = vec![0.0f32; n_samples];
            let mut out_scalar = vec![0.0f32; n_samples];

            model_simd.reset_states();
            model_scalar.reset_states();

            model_simd.process(&input, &mut out_simd);
            model_scalar.process_scalar(&input, &mut out_scalar);

            for i in 0..n_samples {
                let diff = (out_simd[i] - out_scalar[i]).abs();
                let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
                let rel_diff = diff / max_val;
                assert!(
                    rel_diff < 5e-3
                        || (out_simd[i].is_nan() && out_scalar[i].is_nan())
                        || (out_simd[i].is_infinite()
                            && out_scalar[i].is_infinite()
                            && out_simd[i].to_bits() == out_scalar[i].to_bits()),
                    "ModelDyn qhead {num_layers}x{hidden_size} parity failed at [{i}]: \
                     SIMD={}, Scalar={}, Delta={}, Rel={}",
                    out_simd[i],
                    out_scalar[i],
                    diff,
                    rel_diff,
                );
            }
        }
        }
}
