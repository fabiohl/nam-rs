// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stress tests for WaveNet prewarm backfill underflow prevention (S4.T01).
//!
//! Validates that models with large receptive fields (RF up to 2048) execute
//! `prewarm()` without panics, segfaults, or underflow warnings in the backfill path.
//!
//! ## Test Scenarios
//! 1. Custom static model with RF=2046 (10 layers, K=3, dilations up to 512).
//! 2. Deterministic output equality across two identically-constructed models with large RF.
//! 3. Dynamic model with RF=1024 (stress test with larger CH values).
//! 4. Model with K=5 kernel, dilation=512, large total RF=4092.

use nam_rs::math::common::{AlignedVec, SimdMathConfig};
use nam_rs::models::NamModel;
use nam_rs::models::a2::{A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, WaveNetA2};
use nam_rs::models::wavenet::{
    Conv1d, DenseLayer, WAVENET_MAX_NUM_FRAMES, WaveNetLayer, WaveNetLayerArray, WaveNetLayerState,
    WaveNetModel,
};

// =============================================================================
// Helper: builds a WaveNetModel<4, 3, 2> with RF~=2046 (10 layers)
// =============================================================================

fn build_large_rf_wavenet() -> WaveNetModel<4, 3, 2> {
    let is_bf16 = SimdMathConfig::get().instruction_set
        == nam_rs::math::common::InstructionSet::Avx512VnniBf16;

    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 4, 3> {
        let raw_weights = vec![0.01f32; 4 * 3 * 4];
        let mut weights = AlignedVec::new(48, 0u16);
        nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            4,
            4,
            3,
            is_bf16,
        );
        WaveNetLayer {
            conv1d: Conv1d {
                weights,
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    nam_rs::math::common::prefetch_strategy_2stage
                } else {
                    nam_rs::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
        }
    };

    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 2, 3> {
        let raw_weights = vec![0.01f32; 2 * 3 * 2];
        let mut weights = AlignedVec::new(24, 0u16);
        nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            2,
            2,
            3,
            is_bf16,
        );
        WaveNetLayer {
            conv1d: Conv1d {
                weights,
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    nam_rs::math::common::prefetch_strategy_2stage
                } else {
                    nam_rs::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
        }
    };

    // 10 dilations: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    // RF = sum((K-1) * d) = 2 * 1023 = 2046
    let dilations_1 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let dilations_2 = [1, 2, 4, 8];

    let rf1: usize = dilations_1.iter().map(|&d| (3 - 1) * d).sum();
    let rf2: usize = dilations_2.iter().map(|&d| (3 - 1) * d).sum();

    let layers_1: Vec<WaveNetLayer<1, 4, 3>> =
        dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(4, rf1, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array1 = WaveNetLayerArray::<1, 1, 4, 3, 2> {
        layers: layers_1,
        states: states_1,
        rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
            bias: AlignedVec::from_vec(vec![0.0; 4]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 4]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf1,
        block_size: 4,
        block_buffer: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: dilations_1.len(),
    };

    let layers_2: Vec<WaveNetLayer<1, 2, 3>> =
        dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(2, rf2, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array2 = WaveNetLayerArray::<4, 1, 2, 3, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 2]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf2,
        block_size: 2,
        block_buffer: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: dilations_2.len(),
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: rf1.max(rf2),
    }
}

// =============================================================================
// Tests
// =============================================================================

/// Stress test: prewarm with RF=2046 must not cause underflow, segfault, or NaN.
#[test]
fn test_prewarm_large_rf_no_undeflow() {
    let mut model = build_large_rf_wavenet();

    // Prewarm must complete without panic.
    model.prewarm();

    // Verify no NaN/Inf in internal buffers after prewarm.
    for state in &model.array1.states {
        for &v in state.layer_buffer.iter() {
            assert!(
                v.is_finite(),
                "NaN/Inf in array1 after prewarm (RF={})",
                model.receptive_field_size
            );
        }
    }
    for state in &model.array2.states {
        for &v in state.layer_buffer.iter() {
            assert!(
                v.is_finite(),
                "NaN/Inf in array2 after prewarm (RF={})",
                model.receptive_field_size
            );
        }
    }
}

/// Determinism: two identical large-RF models produce the same output after prewarm.
#[test]
fn test_prewarm_large_rf_deterministic() {
    let mut model_a = build_large_rf_wavenet();
    let mut model_b = build_large_rf_wavenet();

    model_a.prewarm();
    model_b.prewarm();

    // Process a block of silence to verify identical output.
    let input = [0.0f32; 16];
    let mut output_a = [0.0f32; 16];
    let mut output_b = [0.0f32; 16];

    model_a.process(&input, &mut output_a);
    model_b.process(&input, &mut output_b);

    for (i, (&a, &b)) in output_a.iter().zip(output_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "Deterministic mismatch at sample {}: a={}, b={}",
            i,
            a,
            b
        );
    }

    // Process a sine wave block for non-trivial signal path exercise.
    let sine: Vec<f32> = (0..64)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let mut out_a = vec![0.0f32; 64];
    let mut out_b = vec![0.0f32; 64];

    model_a.process(&sine, &mut out_a);
    model_b.process(&sine, &mut out_b);

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "Sine determinism mismatch at sample {}: a={}, b={}",
            i,
            a,
            b
        );
        assert!(
            a.is_finite(),
            "Non-finite sine output at sample {}: {}",
            i,
            a
        );
    }
}

/// Stress test: process multiple blocks after prewarm to ensure state machine stability.
#[test]
fn test_prewarm_large_rf_multiblock() {
    let mut model = build_large_rf_wavenet();
    model.prewarm();

    // Process 16 blocks of 64 samples each (covering all jitter positions).
    for block_idx in 0..16 {
        let sine: Vec<f32> = (0..64)
            .map(|i| {
                let t = (block_idx * 64 + i) as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin()
            })
            .collect();
        let mut output = vec![0.0f32; 64];
        model.process(&sine, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Non-finite output at block {}, sample {}: {}",
                block_idx,
                i,
                v
            );
        }
    }
}

/// RF=0 edge case: model with no dilation layers (RF=0) — prewarm should be a no-op.
#[test]
fn test_prewarm_zero_rf() {
    let is_bf16 = SimdMathConfig::get().instruction_set
        == nam_rs::math::common::InstructionSet::Avx512VnniBf16;

    let make_layer = |dilation: usize| -> WaveNetLayer<1, 1, 3> {
        let raw_weights = vec![0.01f32; 3];
        let mut weights = AlignedVec::new(1usize.div_ceil(4) * 3 * 4, 0u16);
        nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            1,
            1,
            3,
            is_bf16,
        );
        WaveNetLayer {
            conv1d: Conv1d {
                weights,
                bias: AlignedVec::from_vec(vec![0.0; 1]),
                do_bias: false,
                dilation,
                prefetch_fn: nam_rs::math::common::prefetch_strategy_simple,
            },
            input_mixin: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 1]),
                bias: AlignedVec::from_vec(vec![0.0; 1]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 1]),
                bias: AlignedVec::from_vec(vec![0.0; 1]),
                do_bias: false,
            },
        }
    };

    let rf = 0;
    let layers1: Vec<WaveNetLayer<1, 1, 3>> = vec![make_layer(1)];
    let states1: Vec<WaveNetLayerState> = (0..layers1.len())
        .map(|i| WaveNetLayerState::new(1, rf, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array1 = WaveNetLayerArray::<1, 1, 1, 3, 1> {
        layers: layers1,
        states: states1,
        rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 1]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 1]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf,
        block_size: 1,
        block_buffer: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: 1,
    };

    let layers2: Vec<WaveNetLayer<1, 1, 3>> = vec![make_layer(1)];
    let states2: Vec<WaveNetLayerState> = (0..layers2.len())
        .map(|i| WaveNetLayerState::new(1, rf, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array2 = WaveNetLayerArray::<1, 1, 1, 3, 1> {
        layers: layers2,
        states: states2,
        rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 1]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 1]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf,
        block_size: 1,
        block_buffer: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: 1,
    };

    let mut model = WaveNetModel::<1, 3, 1> {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: rf,
    };

    model.prewarm();

    let input = [0.0f32; 16];
    let mut output = [0.0f32; 16];
    model.process(&input, &mut output);
    for &v in &output {
        assert!(v.is_finite(), "Non-finite output with RF=0");
    }
}

// =============================================================================
// Helper: builds a WaveNetModel<4, 5, 2> with K=5 and max dilation 512
// =============================================================================

fn build_k5_large_rf_wavenet() -> WaveNetModel<4, 5, 2> {
    let is_bf16 = SimdMathConfig::get().instruction_set
        == nam_rs::math::common::InstructionSet::Avx512VnniBf16;

    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 4, 5> {
        let raw_weights = vec![0.01f32; 4 * 5 * 4];
        let mut weights = AlignedVec::new(5 * 4 * 4, 0u16);
        nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            4,
            4,
            5,
            is_bf16,
        );
        WaveNetLayer {
            conv1d: Conv1d {
                weights,
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    nam_rs::math::common::prefetch_strategy_2stage
                } else {
                    nam_rs::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
        }
    };

    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 2, 5> {
        let raw_weights = vec![0.01f32; 2 * 5 * 2];
        let mut weights = AlignedVec::new(5 * 2 * 4, 0u16);
        nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            2,
            2,
            5,
            is_bf16,
        );
        WaveNetLayer {
            conv1d: Conv1d {
                weights,
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    nam_rs::math::common::prefetch_strategy_2stage
                } else {
                    nam_rs::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
        }
    };

    // 10 dilations: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    // RF per array = sum((K-1) * d) = 4 * 1023 = 4092
    let dilations_1 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let dilations_2 = [1, 2, 4, 8];

    let rf1: usize = dilations_1.iter().map(|&d| (5 - 1) * d).sum();
    let rf2: usize = dilations_2.iter().map(|&d| (5 - 1) * d).sum();

    let layers_1: Vec<WaveNetLayer<1, 4, 5>> =
        dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(4, rf1, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array1 = WaveNetLayerArray::<1, 1, 4, 5, 2> {
        layers: layers_1,
        states: states_1,
        rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
            bias: AlignedVec::from_vec(vec![0.0; 4]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 4]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf1,
        block_size: 4,
        block_buffer: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: dilations_1.len(),
    };

    let layers_2: Vec<WaveNetLayer<1, 2, 5>> =
        dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(2, rf2, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array2 = WaveNetLayerArray::<4, 1, 2, 5, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 2]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf2,
        block_size: 2,
        block_buffer: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: dilations_2.len(),
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: rf1.max(rf2),
    }
}

// =============================================================================
// Tests K=5 (Kernel Size 5) — Edge: large RF, non-standard kernel
// =============================================================================

/// Stress test: prewarm with K=5, dilation=512 and large RF (4092) must not cause
/// underflow, segfault, or NaN.
#[test]
fn test_prewarm_k5_large_rf_no_undeflow() {
    let mut model = build_k5_large_rf_wavenet();

    model.prewarm();

    for state in &model.array1.states {
        for &v in state.layer_buffer.iter() {
            assert!(
                v.is_finite(),
                "NaN/Inf in array1 after K=5 prewarm (RF={})",
                model.receptive_field_size
            );
        }
    }
    for state in &model.array2.states {
        for &v in state.layer_buffer.iter() {
            assert!(
                v.is_finite(),
                "NaN/Inf in array2 after K=5 prewarm (RF={})",
                model.receptive_field_size
            );
        }
    }
}

/// Tests prewarm via trait NamModel::prewarm(num_samples=2048) with K=5 model.
/// WaveNet ignores num_samples internally, but StaticModel dispatch passes it through.
#[test]
fn test_prewarm_k5_large_rf_trait_num_samples() {
    use nam_rs::models::NamModel;

    let mut model = build_k5_large_rf_wavenet();

    NamModel::prewarm(&mut model, 2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Non-finite output at sample {} after K=5 prewarm(2048)",
            i
        );
    }
}

/// K=5 determinism: two identical models produce the same output after prewarm.
#[test]
fn test_prewarm_k5_large_rf_deterministic() {
    let mut model_a = build_k5_large_rf_wavenet();
    let mut model_b = build_k5_large_rf_wavenet();

    model_a.prewarm();
    model_b.prewarm();

    let input = [0.0f32; 16];
    let mut output_a = [0.0f32; 16];
    let mut output_b = [0.0f32; 16];

    model_a.process(&input, &mut output_a);
    model_b.process(&input, &mut output_b);

    for (i, (&a, &b)) in output_a.iter().zip(output_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "K=5 determinism mismatch at sample {}: a={}, b={}",
            i,
            a,
            b
        );
    }

    let sine: Vec<f32> = (0..64)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let mut out_a = vec![0.0f32; 64];
    let mut out_b = vec![0.0f32; 64];

    model_a.process(&sine, &mut out_a);
    model_b.process(&sine, &mut out_b);

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "K=5 sine determinism mismatch at sample {}: a={}, b={}",
            i,
            a,
            b
        );
        assert!(
            a.is_finite(),
            "Non-finite K=5 sine output at sample {}: {}",
            i,
            a
        );
    }
}

/// K=5 multiblock: processes multiple blocks after prewarm to ensure stability.
#[test]
fn test_prewarm_k5_large_rf_multiblock() {
    let mut model = build_k5_large_rf_wavenet();
    model.prewarm();

    for block_idx in 0..16 {
        let sine: Vec<f32> = (0..64)
            .map(|i| {
                let t = (block_idx * 64 + i) as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin()
            })
            .collect();
        let mut output = vec![0.0f32; 64];
        model.process(&sine, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Non-finite K=5 output at block {}, sample {}: {}",
                block_idx,
                i,
                v
            );
        }
    }
}

// =============================================================================
// A2 Architecture — WaveNetA2 prewarm edge tests
// =============================================================================
// A2 has 23 layers with fixed dilations (up to 239) and kernels (6 or 15).
// Total RF ~6331 — the largest in the codebase.

/// A2 receptive field from canonical constants (matching `a2_receptive_field()` in model.rs).
fn a2_rf() -> usize {
    let mut rf = 0usize;
    for i in 0..23 {
        rf += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
    }
    rf + (A2_HEAD_KERNEL_SIZE - 1)
}

/// Computes total A2 weight count for a given CH (3 = Lite, 8 = Full).
const fn a2_total_weights<const CH: usize>() -> usize {
    let mut total = CH;
    let mut layer_idx = 0;
    while layer_idx < 23 {
        let k = A2_KERNEL_SIZES[layer_idx];
        total += CH * CH * k + CH + CH + CH * CH + CH;
        layer_idx += 1;
    }
    total += 16 * CH + 2;
    total
}

/// Assembles synthetic A2 weight stream with exact count.
fn a2_synth_weights<const CH: usize>(weight_val: f32) -> Vec<f32> {
    let num_weights = a2_total_weights::<CH>();
    let mut w = Vec::with_capacity(num_weights);

    w.extend(std::iter::repeat_n(weight_val, CH));

    for &k in &A2_KERNEL_SIZES {
        w.extend(std::iter::repeat_n(weight_val, CH * CH * k));
        w.extend(std::iter::repeat_n(0.0f32, CH));
        w.extend(std::iter::repeat_n(weight_val, CH));
        w.extend(std::iter::repeat_n(weight_val, CH * CH));
        w.extend(std::iter::repeat_n(0.0f32, CH));
    }

    w.extend(std::iter::repeat_n(weight_val, 16 * CH));
    w.push(0.0);
    w.push(0.02);

    assert_eq!(w.len(), num_weights, "A2 weight count mismatch");
    w
}

/// Builds a WaveNetA2<CH> with all weights set to `weight_val` and zero biases.
/// Uses the exact weight count expected by `set_weights`.
fn build_synth_a2<const CH: usize>(weight_val: f32) -> WaveNetA2<CH> {
    let weights = a2_synth_weights::<CH>(weight_val);
    let mut model = WaveNetA2::<CH>::new();
    model
        .set_weights(&weights)
        .expect("Failed to set A2 synthetic weights");
    model
}

/// A2-Full (CH=8): prewarm with RF~6331 must not cause underflow, segfault, or NaN.
#[test]
fn test_a2_full_prewarm_no_undeflow() {
    let mut model = build_synth_a2::<8>(0.01);
    model.prewarm();

    let rf = a2_rf();
    // Verify no NaN/Inf in internal buffers after prewarm.
    for buf in &model.layer_buffers {
        let len = buf.size();
        for &v in buf[..len].iter() {
            assert!(
                v.is_finite(),
                "NaN/Inf in A2-Full layer_buffer after prewarm (RF={})",
                rf
            );
        }
    }
    for &v in model.head_accum.iter() {
        assert!(
            v.is_finite(),
            "NaN/Inf in A2-Full head_accum after prewarm (RF={})",
            rf
        );
    }
}

/// A2-Lite (CH=3): prewarm with RF~6331 must not cause underflow, segfault, or NaN.
#[test]
fn test_a2_lite_prewarm_no_undeflow() {
    let mut model = build_synth_a2::<3>(0.01);
    model.prewarm();

    let rf = a2_rf();
    for buf in &model.layer_buffers {
        let len = buf.size();
        for &v in buf[..len].iter() {
            assert!(
                v.is_finite(),
                "NaN/Inf in A2-Lite layer_buffer after prewarm (RF={})",
                rf
            );
        }
    }
    for &v in model.head_accum.iter() {
        assert!(
            v.is_finite(),
            "NaN/Inf in A2-Lite head_accum after prewarm (RF={})",
            rf
        );
    }
}

/// A2-Full determinism: two identical models produce the same output after prewarm.
#[test]
fn test_a2_full_prewarm_deterministic() {
    let mut model_a = build_synth_a2::<8>(0.01);
    let mut model_b = build_synth_a2::<8>(0.01);

    model_a.prewarm();
    model_b.prewarm();

    let input = [0.0f32; 16];
    let mut output_a = [0.0f32; 16];
    let mut output_b = [0.0f32; 16];

    model_a.process(&input, &mut output_a);
    model_b.process(&input, &mut output_b);

    for (i, (&a, &b)) in output_a.iter().zip(output_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "A2-Full determinism mismatch at sample {}: a={}, b={}",
            i,
            a,
            b
        );
    }

    let sine: Vec<f32> = (0..64)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let mut out_a = vec![0.0f32; 64];
    let mut out_b = vec![0.0f32; 64];

    model_a.process(&sine, &mut out_a);
    model_b.process(&sine, &mut out_b);

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "A2-Full sine determinism mismatch at sample {}: a={}, b={}",
            i,
            a,
            b
        );
        assert!(
            a.is_finite(),
            "Non-finite A2-Full sine output at sample {}: {}",
            i,
            a
        );
    }
}

/// A2-Lite determinism: two identical models produce the same output after prewarm.
#[test]
fn test_a2_lite_prewarm_deterministic() {
    let mut model_a = build_synth_a2::<3>(0.01);
    let mut model_b = build_synth_a2::<3>(0.01);

    model_a.prewarm();
    model_b.prewarm();

    let input = [0.0f32; 16];
    let mut output_a = [0.0f32; 16];
    let mut output_b = [0.0f32; 16];

    model_a.process(&input, &mut output_a);
    model_b.process(&input, &mut output_b);

    for (i, (&a, &b)) in output_a.iter().zip(output_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "A2-Lite determinism mismatch at sample {}: a={}, b={}",
            i,
            a,
            b
        );
    }

    let sine: Vec<f32> = (0..64)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let mut out_a = vec![0.0f32; 64];
    let mut out_b = vec![0.0f32; 64];

    model_a.process(&sine, &mut out_a);
    model_b.process(&sine, &mut out_b);

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "A2-Lite sine determinism mismatch at sample {}: a={}, b={}",
            i,
            a,
            b
        );
        assert!(
            a.is_finite(),
            "Non-finite A2-Lite sine output at sample {}: {}",
            i,
            a
        );
    }
}

/// A2-Full multiblock: processes multiple blocks of variable sizes after prewarm.
#[test]
fn test_a2_full_prewarm_multiblock() {
    let mut model = build_synth_a2::<8>(0.01);
    model.prewarm();

    let block_sizes = [1usize, 8, 16, 32, 48, 64, 63, 17];
    let mut sample_offset = 0usize;

    for &block_size in &block_sizes {
        let sine: Vec<f32> = (0..block_size)
            .map(|i| {
                let t = (sample_offset + i) as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin()
            })
            .collect();
        let mut output = vec![0.0f32; block_size];
        model.process(&sine, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Non-finite A2-Full output at block_size={}, sample {}: {}",
                block_size,
                i,
                v
            );
        }
        sample_offset += block_size;
    }
}

/// A2-Lite multiblock: processes multiple blocks of variable sizes after prewarm.
#[test]
fn test_a2_lite_prewarm_multiblock() {
    let mut model = build_synth_a2::<3>(0.01);
    model.prewarm();

    let block_sizes = [1usize, 8, 16, 32, 48, 64, 63, 17];
    let mut sample_offset = 0usize;

    for &block_size in &block_sizes {
        let sine: Vec<f32> = (0..block_size)
            .map(|i| {
                let t = (sample_offset + i) as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin()
            })
            .collect();
        let mut output = vec![0.0f32; block_size];
        model.process(&sine, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Non-finite A2-Lite output at block_size={}, sample {}: {}",
                block_size,
                i,
                v
            );
        }
        sample_offset += block_size;
    }
}

/// A2-Full: prewarm via NamModel trait (num_samples ignored but dispatch passes through).
#[test]
fn test_a2_full_prewarm_trait_num_samples() {
    let mut model = build_synth_a2::<8>(0.01);
    NamModel::prewarm(&mut model, 2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for (i, &v) in output.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Non-finite A2-Full output at sample {} after trait prewarm(2048)",
            i
        );
    }
}

/// A2-Full receptive field verification: the model reports the correct canonical RF.
#[test]
fn test_a2_full_receptive_field_canonical() {
    let model = build_synth_a2::<8>(0.01);
    let expected_rf = a2_rf();
    assert_eq!(
        model.receptive_field_size, expected_rf,
        "A2-Full receptive field mismatch"
    );
    assert!(
        model.receptive_field_size > 6000,
        "A2-Full RF too small: {}",
        model.receptive_field_size
    );
}

/// A2-Lite receptive field verification.
#[test]
fn test_a2_lite_receptive_field_canonical() {
    let model = build_synth_a2::<3>(0.01);
    let expected_rf = a2_rf();
    assert_eq!(
        model.receptive_field_size, expected_rf,
        "A2-Lite receptive field mismatch"
    );
    assert!(
        model.receptive_field_size > 6000,
        "A2-Lite RF too small: {}",
        model.receptive_field_size
    );
}
