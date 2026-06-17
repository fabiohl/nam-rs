// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Numerical parity tests: dynamic (runtime-dimensional) ↔ const-generic WaveNet.
//!
//! Validates that `WaveNetModelDyn` produces identical outputs to the
//! corresponding const-generic `WaveNetModel<CH, K, HEAD>` for the four
//! catalog geometries (Standard, Lite, Feather, Nano).
//!
//! Each test:
//! 1. Builds a const-generic model with deterministic synthetic weights (0.01)
//! 2. Builds an equivalent dynamic model using the same weights
//! 3. Prewarms both models
//! 4. Feeds identical random input through both
//! 5. Asserts bit-exact output match

use crate::math::common::AlignedVec;
use crate::models::wavenet::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use crate::models::wavenet::conv1d::Conv1d;
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;
use crate::models::wavenet::dense::DenseLayer;
use crate::models::wavenet::dense_dyn::DenseLayerDyn;
use crate::models::wavenet::layer::WaveNetLayer;
use crate::models::wavenet::layer_array::WaveNetLayerArray;
use crate::models::wavenet::layer_array_dyn::WaveNetLayerArrayDyn;
use crate::models::wavenet::layer_dyn::WaveNetLayerDyn;
use crate::models::wavenet::model::WaveNetModel;
use crate::models::wavenet::model_dyn::WaveNetModelDyn;

/// Weight value used throughout for deterministic synthetic model construction.
const SYNTHETIC_WEIGHT: f32 = 0.01;

/// Dilations for the test models (3 layers).
const TEST_DILATIONS: [usize; 3] = [1, 2, 4];

/// Helper: interleave f32 weights into the `[OUT/4][K][IN][4]` layout used by Conv1d and Conv1dDyn.
fn make_conv1d_weights(in_ch: usize, out_ch: usize, k: usize) -> AlignedVec<u16> {
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;
    let raw_weights = vec![SYNTHETIC_WEIGHT; out_ch * k * in_ch];
    let num_blocks = out_ch.div_ceil(4);
    let interleaved_len = num_blocks * k * in_ch * 4;
    let mut weights = AlignedVec::new(interleaved_len, 0u16);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
        &raw_weights,
        &mut weights,
        in_ch,
        out_ch,
        k,
        is_bf16,
    );
    weights
}

/// Helper: create f32-native interleaved Conv1D weights for high-fidelity mode.
#[cfg(feature = "high-fidelity")]
fn make_conv1d_f32_weights(in_ch: usize, out_ch: usize, k: usize) -> AlignedVec<f32> {
    let raw_weights = vec![SYNTHETIC_WEIGHT; out_ch * k * in_ch];
    let num_blocks = out_ch.div_ceil(4);
    let interleaved_len = num_blocks * k * in_ch * 4;
    let mut weights = AlignedVec::new(interleaved_len, 0.0f32);
    for b in 0..num_blocks {
        for ki in 0..k {
            for in_c in 0..in_ch {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    let target_idx = b * (k * in_ch * 4) + ki * (in_ch * 4) + in_c * 4 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * k + ki;
                        weights[target_idx] = raw_weights[raw_idx];
                    }
                }
            }
        }
    }
    weights
}

/// Helper: replicate a u16-quantized weight value into a dense layer weight matrix.
fn make_dense_weights(in_ch: usize, out_ch: usize) -> AlignedVec<u16> {
    AlignedVec::from_vec(vec![
        half::f16::from_f32(SYNTHETIC_WEIGHT).to_bits();
        out_ch * in_ch
    ])
}

/// Helper: create f32-native dense weights for high-fidelity mode.
#[cfg(feature = "high-fidelity")]
fn make_dense_f32_weights(in_ch: usize, out_ch: usize) -> AlignedVec<f32> {
    AlignedVec::from_vec(vec![SYNTHETIC_WEIGHT; out_ch * in_ch])
}

/// Helper: create a bias AlignedVec of given size filled with zeros.
fn make_bias(len: usize) -> AlignedVec<f32> {
    AlignedVec::from_vec(vec![0.0; len])
}

/// Helper: prefetch function based on dilation.
fn prefetch_for(dilation: usize) -> crate::math::common::PrefetchFn {
    if dilation >= 128 {
        crate::math::common::prefetch_strategy_2stage
    } else {
        crate::math::common::prefetch_strategy_simple
    }
}

/// Builds a const-generic `WaveNetModel<CH, K, HEAD>` with synthetic weights.
fn build_const_generic_model<const CH: usize, const K: usize, const HEAD: usize>()
-> WaveNetModel<CH, K, HEAD> {
    let rf1 = *TEST_DILATIONS.iter().max().unwrap_or(&1) * (K - 1);
    let rf2 = rf1;

    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, CH, K> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: make_conv1d_weights(CH, CH, K),
                #[cfg(feature = "high-fidelity")]
                f32_weights: make_conv1d_f32_weights(CH, CH, K),
                bias: make_bias(CH),
                do_bias: false,
                dilation,
                prefetch_fn: prefetch_for(dilation),
            },
            input_mixin: DenseLayer {
                #[cfg(not(feature = "high-fidelity"))]
                f32_weights: None,
                #[cfg(feature = "high-fidelity")]
                f32_weights: Some(make_dense_f32_weights(1, CH)),
                weights: make_dense_weights(1, CH),
                bias: make_bias(CH),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                #[cfg(not(feature = "high-fidelity"))]
                f32_weights: None,
                #[cfg(feature = "high-fidelity")]
                f32_weights: Some(make_dense_f32_weights(CH, CH)),
                weights: make_dense_weights(CH, CH),
                bias: make_bias(CH),
                do_bias: false,
            },
        }
    };

    let layers_1: Vec<WaveNetLayer<1, CH, K>> =
        TEST_DILATIONS.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(CH, rf1, i).expect("Failed to create WaveNetLayerState"))
        .collect();
    let num_layers_1 = layers_1.len();

    let array1 = WaveNetLayerArray::<1, 1, CH, K, HEAD> {
        layers: layers_1,
        states: states_1,
        effective_layers: num_layers_1,
        rechannel: DenseLayer {
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(make_dense_f32_weights(1, CH)),
            weights: make_dense_weights(1, CH),
            bias: make_bias(CH),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(make_dense_f32_weights(CH, HEAD)),
            weights: make_dense_weights(CH, HEAD),
            bias: make_bias(HEAD),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf1,
        block_size: CH,
        block_buffer: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
    };

    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, HEAD, K> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: make_conv1d_weights(HEAD, HEAD, K),
                #[cfg(feature = "high-fidelity")]
                f32_weights: make_conv1d_f32_weights(HEAD, HEAD, K),
                bias: make_bias(HEAD),
                do_bias: false,
                dilation,
                prefetch_fn: prefetch_for(dilation),
            },
            input_mixin: DenseLayer {
                #[cfg(not(feature = "high-fidelity"))]
                f32_weights: None,
                #[cfg(feature = "high-fidelity")]
                f32_weights: Some(make_dense_f32_weights(1, HEAD)),
                weights: make_dense_weights(1, HEAD),
                bias: make_bias(HEAD),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                #[cfg(not(feature = "high-fidelity"))]
                f32_weights: None,
                #[cfg(feature = "high-fidelity")]
                f32_weights: Some(make_dense_f32_weights(HEAD, HEAD)),
                weights: make_dense_weights(HEAD, HEAD),
                bias: make_bias(HEAD),
                do_bias: false,
            },
        }
    };

    let layers_2: Vec<WaveNetLayer<1, HEAD, K>> =
        TEST_DILATIONS.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(HEAD, rf2, i).expect("Failed to create WaveNetLayerState"))
        .collect();
    let num_layers_2 = layers_2.len();

    let array2 = WaveNetLayerArray::<CH, 1, HEAD, K, 1> {
        layers: layers_2,
        states: states_2,
        effective_layers: num_layers_2,
        rechannel: DenseLayer {
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(make_dense_f32_weights(CH, HEAD)),
            weights: make_dense_weights(CH, HEAD),
            bias: make_bias(HEAD),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(make_dense_f32_weights(HEAD, 1)),
            weights: make_dense_weights(HEAD, 1),
            bias: make_bias(1),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf2,
        block_size: HEAD,
        block_buffer: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: rf1.max(rf2),
    }
}

/// Builds a dynamic `WaveNetModelDyn` equivalent to `WaveNetModel<CH, K, HEAD>`
/// using the same synthetic weights.
fn build_dynamic_model(ch: usize, k: usize, head: usize) -> WaveNetModelDyn {
    let rf = TEST_DILATIONS.iter().max().unwrap_or(&1) * (k - 1);

    let make_conv1d_dyn = |in_ch: usize, out_ch: usize, dilation: usize| -> Conv1dDyn {
        Conv1dDyn {
            weights: make_conv1d_weights(in_ch, out_ch, k),
            #[cfg(feature = "high-fidelity")]
            f32_weights: make_conv1d_f32_weights(in_ch, out_ch, k),
            bias: make_bias(out_ch),
            do_bias: false,
            dilation,
            in_ch,
            out_ch,
            num_blocks: out_ch.div_ceil(4),
            kernel: k,
            prefetch_fn: prefetch_for(dilation),
        }
    };

    let make_dense_dyn = |in_ch: usize, out_ch: usize, do_bias: bool| -> DenseLayerDyn {
        DenseLayerDyn {
            in_ch,
            out_ch,
            weights: make_dense_weights(in_ch, out_ch),
            bias: make_bias(out_ch),
            do_bias,
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(make_dense_f32_weights(in_ch, out_ch)),
        }
    };

    // Array 1 layers: COND=1, CH=ch, K=k
    let layers_1: Vec<WaveNetLayerDyn> = TEST_DILATIONS
        .iter()
        .map(|&d| {
            WaveNetLayerDyn::new(
                ch,
                make_conv1d_dyn(ch, ch, d),
                make_dense_dyn(1, ch, false),
                make_dense_dyn(ch, ch, false),
            )
        })
        .collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(ch, rf, i).expect("Failed to create WaveNetLayerState"))
        .collect();
    let num_layers_1 = layers_1.len();

    let array1 = WaveNetLayerArrayDyn {
        in_ch: 1,
        cond: 1,
        ch,
        k,
        head,
        layers: layers_1,
        states: states_1,
        effective_layers: num_layers_1,
        rechannel: make_dense_dyn(1, ch, false),
        head_rechannel: make_dense_dyn(ch, head, false),
        array_outputs: AlignedVec::from_vec(vec![0.0; ch * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; ch * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; head * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf,
        block_size: ch,
        block_buffer: AlignedVec::from_vec(vec![0.0; ch * WAVENET_MAX_NUM_FRAMES]),
        last_condition: AlignedVec::from_vec(vec![0.0; 1]),
        last_condition_bf16: AlignedVec::from_vec(vec![0; 1]),
        condition_init: false,
    };

    // Array 2 layers: COND=1, CH=head, K=k
    let layers_2: Vec<WaveNetLayerDyn> = TEST_DILATIONS
        .iter()
        .map(|&d| {
            WaveNetLayerDyn::new(
                head,
                make_conv1d_dyn(head, head, d),
                make_dense_dyn(1, head, false),
                make_dense_dyn(head, head, false),
            )
        })
        .collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(head, rf, i).expect("Failed to create WaveNetLayerState"))
        .collect();
    let num_layers_2 = layers_2.len();

    let array2 = WaveNetLayerArrayDyn {
        in_ch: ch,
        cond: 1,
        ch: head,
        k,
        head: 1,
        layers: layers_2,
        states: states_2,
        effective_layers: num_layers_2,
        rechannel: make_dense_dyn(ch, head, false),
        head_rechannel: make_dense_dyn(head, 1, true),
        array_outputs: AlignedVec::from_vec(vec![0.0; head * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; head * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf,
        block_size: head,
        block_buffer: AlignedVec::from_vec(vec![0.0; head * WAVENET_MAX_NUM_FRAMES]),
        last_condition: AlignedVec::from_vec(vec![0.0; 1]),
        last_condition_bf16: AlignedVec::from_vec(vec![0; 1]),
        condition_init: false,
    };

    WaveNetModelDyn {
        ch,
        k,
        head,
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: rf,
    }
}

/// Magnitude comparison of two model outputs using max absolute error.
/// For bit-exact parity, max_error should be 0.0.
fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

macro_rules! parity_test {
    ($name:ident, $ch:literal, $k:literal, $head:literal) => {
        #[test]
        fn $name() {
            let mut const_model = build_const_generic_model::<$ch, $k, $head>();
            let mut dyn_model = build_dynamic_model($ch, $k, $head);

            const_model.prewarm();
            dyn_model.prewarm();

            let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();

            let mut const_out = vec![0.0f32; input.len()];
            let mut dyn_out = vec![0.0f32; input.len()];

            const_model.process(&input, &mut const_out);
            dyn_model.process(&input, &mut dyn_out);

            let err = max_abs_error(&const_out, &dyn_out);
            assert!(
                err < 1e-7,
                "Parity failed for CH={}, K={}, HEAD={}: max_abs_error = {:.3e}",
                $ch,
                $k,
                $head,
                err
            );
        }
    };
}

// Catalog geometries: Standard, Lite, Feather, Nano
parity_test!(test_dynamic_parity_standard, 16, 3, 8);
parity_test!(test_dynamic_parity_lite, 12, 3, 6);
parity_test!(test_dynamic_parity_feather, 8, 3, 4);
parity_test!(test_dynamic_parity_nano, 4, 3, 2);

/// Determinism: two identically built and prewarmed dynamic models must produce
/// identical output for the same input.
#[test]
fn test_dynamic_determinism() {
    let mut model_a = build_dynamic_model(4, 3, 2);
    let mut model_b = build_dynamic_model(4, 3, 2);
    model_a.prewarm();
    model_b.prewarm();

    let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();

    let mut out_a = vec![0.0f32; input.len()];
    let mut out_b = vec![0.0f32; input.len()];

    model_a.process(&input, &mut out_a);
    model_b.process(&input, &mut out_b);

    let err = max_abs_error(&out_a, &out_b);
    assert!(
        err < 1e-7,
        "Dynamic model is not deterministic across instances: max_abs_error = {:.3e}",
        err
    );
}

/// All buffers remain finite after prewarm and processing.
#[test]
fn test_dynamic_no_nan() {
    let mut model = build_dynamic_model(4, 3, 2);
    model.prewarm();

    let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
    let mut output = vec![0.0f32; input.len()];
    model.process(&input, &mut output);

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Non-finite output at sample {}: {}", i, v);
    }
}
