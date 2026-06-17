// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::AlignedVec;
use crate::models::wavenet::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use crate::models::wavenet::conv1d::Conv1d;
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;
use crate::models::wavenet::dense::DenseLayer;
use crate::models::wavenet::layer::WaveNetLayer;
use crate::models::wavenet::layer_array::WaveNetLayerArray;
use crate::models::wavenet::model::*;

/// Helper: create f32 synthetic dense weights for high-fidelity test models.
#[cfg(feature = "high-fidelity")]
fn test_dense_f32(in_ch: usize, out_ch: usize) -> AlignedVec<f32> {
    AlignedVec::from_vec(vec![0.01f32; out_ch * in_ch])
}

/// Helper: create f32 synthetic interleaved Conv1D weights for high-fidelity test models.
#[cfg(feature = "high-fidelity")]
fn test_conv1d_f32(raw: &[f32], in_ch: usize, out_ch: usize, k: usize) -> AlignedVec<f32> {
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
                        weights[target_idx] = raw[raw_idx];
                    }
                }
            }
        }
    }
    weights
}

/// Builds a minimal WaveNetModel<4, 3, 2> for tests with static, controlled data.
/// This function serves as a "mock" (simulated model) for unit tests.
fn build_tiny_wavenet() -> WaveNetModel<4, 3, 2> {
    // Layer factory for Array 1 (Main Array).
    // Generics: <COND=1, CH=4, K=3>.
    // In WaveNet, each layer is a functional unit that processes the dilated signal.
    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 4, 3> {
        let raw_weights = vec![0.01f32; 4 * 3 * 4];
        let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
            == crate::math::common::InstructionSet::Avx512VnniBf16;
        let mut weights = AlignedVec::new(48, 0u16);
        crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            4, // IN
            4, // OUT
            3, // K
            is_bf16,
        );
        WaveNetLayer {
            // Dilated Causal Convolution enables capturing long temporal dependencies
            // without linearly increasing the number of parameters.
            conv1d: Conv1d {
                // Dimensions: OUT * K * IN = 4 * 3 * 4.
                // Here, IN=CH because the layer receives the signal from previous layers.
                weights,
                #[cfg(feature = "high-fidelity")]
                f32_weights: test_conv1d_f32(&raw_weights, 4, 4, 3),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    crate::math::common::prefetch_strategy_2stage
                } else {
                    crate::math::common::prefetch_strategy_simple
                },
            },
            // input_mixin injects conditioning (e.g., timbre metadata) into the signal.
            // Dimensions: OUT * IN = 4 * 1.
            input_mixin: DenseLayer {
                #[cfg(feature = "high-fidelity")]
                f32_weights: Some(test_dense_f32(1, 4)),
                #[cfg(not(feature = "high-fidelity"))]
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
            // The 1x1 projection (Dense) finalizes the cell, preparing the signal for the residual.
            // Dimensions: OUT * IN = 4 * 4.
            one_by_one: DenseLayer {
                #[cfg(feature = "high-fidelity")]
                f32_weights: Some(test_dense_f32(4, 4)),
                #[cfg(not(feature = "high-fidelity"))]
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
        }
    };

    // Array2: CH=2 (=HEAD), layers with COND=1, CH=2
    // Layer factory for Array 2 (Head Array).
    // Generics: <COND=1, CH=2, K=3>.
    // This array usually has fewer channels and focuses on final audio refinement.
    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 2, 3> {
        let raw_weights = vec![0.01f32; 2 * 3 * 2];
        let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
            == crate::math::common::InstructionSet::Avx512VnniBf16;
        let mut weights = AlignedVec::new(24, 0u16);
        crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            2, // IN
            2, // OUT
            3, // K
            is_bf16,
        );
        WaveNetLayer {
            conv1d: Conv1d {
                weights,
                #[cfg(feature = "high-fidelity")]
                f32_weights: test_conv1d_f32(&raw_weights, 2, 2, 3),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    crate::math::common::prefetch_strategy_2stage
                } else {
                    crate::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                #[cfg(feature = "high-fidelity")]
                f32_weights: Some(test_dense_f32(1, 2)),
                #[cfg(not(feature = "high-fidelity"))]
                f32_weights: None,
                // Dimensions: OUT * IN = 2 * 1.
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                #[cfg(feature = "high-fidelity")]
                f32_weights: Some(test_dense_f32(2, 2)),
                #[cfg(not(feature = "high-fidelity"))]
                f32_weights: None,
                // Dimensions: OUT * IN = 2 * 2.
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
        }
    };

    // Define the dilation pattern. Exponential growth (1, 2, 4...)
    // is what allows WaveNet to have a vast receptive field with few layers.
    let dilations_1 = [1, 2, 4];
    let dilations_2 = [1, 2, 4];

    // Receptive Field (RF) calculation: determines how many past samples influence the present.
    // Simplified formula: max_dilation * (kernel_size - 1).
    let rf1 = *dilations_1.iter().max().unwrap_or(&1) * (3 - 1);
    let rf2 = *dilations_2.iter().max().unwrap_or(&1) * (3 - 1);

    // Manual array construction with explicit const generics.
    // Array1 (Main Receptive Field): Ensures primary feature extraction.
    // For each dilation, we build a layer and allocate its internal state (historical buffer).
    let layers_1: Vec<WaveNetLayer<1, 4, 3>> =
        dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(4, rf1, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let num_layers_1 = layers_1.len();
    let array1 = WaveNetLayerArray::<1, 1, 4, 3, 2> {
        layers: layers_1,
        states: states_1,
        effective_layers: num_layers_1,
        // Rechannel: Projects raw input (Mono/Stereo) to the internal dimension (Channels).
        rechannel: DenseLayer {
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(test_dense_f32(1, 4)),
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
            bias: AlignedVec::from_vec(vec![0.0; 4]),
            do_bias: false,
        },
        // Head Rechannel: Aggregates "skip connections" from all layers for the array's output.
        head_rechannel: DenseLayer {
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(test_dense_f32(4, 2)),
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 4]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        // Pre-allocated output buffers to ensure RT-Safety (Zero Alloc in the loop).
        array_outputs: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf1,
        block_size: 4,
        block_buffer: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
    };

    // Array2 (Head Definition): The secondary array acts on refined final predictions.
    // IN=4(=CH of array1), COND=1, CH=2(=HEAD1), K=3, HEAD2=1
    // `head_rechannel` defines the final data transition.
    let layers_2: Vec<WaveNetLayer<1, 2, 3>> =
        dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(2, rf2, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let num_layers_2 = layers_2.len();
    let array2 = WaveNetLayerArray::<4, 1, 2, 3, 1> {
        layers: layers_2,
        states: states_2,
        effective_layers: num_layers_2,
        // Projects Array 1 output (HEAD1=2) to Array 2 dimension (CH2=2).
        rechannel: DenseLayer {
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(test_dense_f32(4, 2)),
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 2]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        // The final NAM model projection reduces everything to 1 channel (mono audio).
        head_rechannel: DenseLayer {
            #[cfg(feature = "high-fidelity")]
            f32_weights: Some(test_dense_f32(2, 1)),
            #[cfg(not(feature = "high-fidelity"))]
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: true, // Enable bias for final DC offset correction.
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
    };

    // WaveNetModel orchestrates the array cascade and applies the final gain.
    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        // The global RF is the largest among the arrays (usually Array1's RF dominates).
        receptive_field_size: rf1.max(rf2),
    }
}

#[path = "test_files/conv1d_dyn_tests.rs"]
mod conv1d_dyn_tests;
#[path = "test_files/conv1d_tests.rs"]
mod conv1d_tests;
#[path = "test_files/dense_tests.rs"]
mod dense_tests;
#[path = "test_files/dynamic_parity.rs"]
mod dynamic_parity;
#[path = "test_files/wavenet_tests.rs"]
mod wavenet_tests;
