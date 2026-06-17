// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Verifies that a 1D Convolution acting as an "Identity Kernel"
/// exactly reproduces the input at the output, without any modification.
///
/// The pedagogical importance of this is enormous: we use it to ensure that
/// the lowest-level SIMD primitives are not transposing, corrupting,
/// or dropping channels when accessing and storing values in memory (SoA/AoS).
#[test]
fn test_conv1d_identity_kernel() {
    // Create a 4x4 weight matrix (flattened to 16 floats).
    // By filling only the main diagonal with 1.0, we create an "Identity Kernel".
    let mut raw_weights = vec![0.0f32; 16];
    for i in 0..4 {
        raw_weights[i * 4 + i] = 1.0;
    }
    let mut weights = AlignedVec::new(16, 0.0f32);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide_f32(
        &raw_weights,
        &mut weights,
        4, // IN
        4, // OUT
        1, // K
    );

    // Instantiate 1D Convolution without bias and with dilation 1 (linear processing).
    let conv = Conv1d::<4, 4, 1> {
        weights,
        bias: AlignedVec::from_vec(vec![0.0; 4]),
        do_bias: false,
        dilation: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    // Simulate a layer buffer (input) with sequential values.
    let layer_buffer = vec![1.0, 2.0, 3.0, 4.0];
    let mut block = vec![0.0; 4];

    // Manually invoke the SIMD routine optimized for AVX2.
    // The unsafe block is necessary because we access hardware intrinsic primitives.
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 0, 1);
    }

    // Since the kernel is identity, the output should be a bit-perfect copy of the input.
    assert_eq!(block, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Verifies the *Bias* addition functionality in Conv1D.
///
/// A layer equipped with Identity matrix weights plus a constant bias
/// should merely translate the numeric axis of the processed values. This attests
/// to the correct use of FMA (Fused Multiply-Add) with the `do_bias` flag.
#[test]
fn test_conv1d_with_bias() {
    // Again, we use an identity kernel to isolate the Bias effect.
    let mut raw_weights = vec![0.0f32; 16];
    for i in 0..4 {
        raw_weights[i * 4 + i] = 1.0;
    }
    let mut weights = AlignedVec::new(16, 0.0f32);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide_f32(
        &raw_weights,
        &mut weights,
        4, // IN
        4, // OUT
        1, // K
    );

    // Configure the layer with Bias of 0.5 on all output channels.
    let conv = Conv1d::<4, 4, 1> {
        weights,
        bias: AlignedVec::from_vec(vec![0.5; 4]),
        do_bias: true, // Enable bias vector addition.
        dilation: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    let layer_buffer = vec![1.0, 2.0, 3.0, 4.0];
    let mut block = vec![0.0; 4];

    // The SIMD engine executes Fused Multiply-Add (FMA): (input * 1.0) + 0.5.
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 0, 1);
    }

    // Verify that each element was correctly translated by the bias value.
    assert_eq!(block, vec![1.5, 2.5, 3.5, 4.5]);
}

/// Tests the central concept of WaveNet: Dilated Convolutions.
///
/// Dilation inserts deterministic spacing in the historical data sampling.
/// Here we validate whether the mathematical offset in `Conv1d` correctly accesses
/// the delayed frames (with `dilation: 2`). This means skipping adjacent samples
/// and jumping windows within the Ring Buffer. The success of this test ensures the
/// correct construction of the architecture's overall *Receptive Field*.
#[test]
fn test_conv1d_dilation() {
    // Configure unit weights (1.0) to sum all inputs directly.
    let raw_weights = vec![1.0f32; 2 * 3 * 2];
    let mut weights = AlignedVec::new(24, 0.0f32);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide_f32(
        &raw_weights,
        &mut weights,
        2, // IN
        2, // OUT
        3, // K
    );

    // Define dilation: 2. This will make the kernel "skip" one frame per tap.
    let conv = Conv1d::<2, 2, 3> {
        weights,
        bias: AlignedVec::from_vec(vec![0.0; 2]),
        do_bias: false,
        dilation: 2,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    // Create a history of 6 frames (12 floats).
    // Layout: [F0, F1, F2, F3, F4, F5] -> [ (1,2), (10,20), (3,4), (30,40), (5,6), (0,0) ]
    let mut layer_buffer = vec![0.0; 6 * 2];
    layer_buffer[0] = 1.0;
    layer_buffer[1] = 2.0; // F0
    layer_buffer[2] = 10.0;
    layer_buffer[3] = 20.0; // F1 (Will be ignored by dilation 2)
    layer_buffer[4] = 3.0;
    layer_buffer[5] = 4.0; // F2
    layer_buffer[6] = 30.0;
    layer_buffer[7] = 40.0; // F3 (Will be ignored by dilation 2)
    layer_buffer[8] = 5.0;
    layer_buffer[9] = 6.0; // F4

    let mut block = vec![0.0; 2];

    // Process at frame index 4 (F4).
    // With K=3 and Dilation=2, the taps will be:
    // Tap 0: frame 4 - (2 * 2) = frame 0 -> [1.0, 2.0]
    // Tap 1: frame 4 - (2 * 1) = frame 2 -> [3.0, 4.0]
    // Tap 2: frame 4 - (2 * 0) = frame 4 -> [5.0, 6.0]
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 4, 1);
    }

    // Expected total sum: 1+2 + 3+4 + 5+6 = 21.0.
    assert_eq!(block[0], 21.0);
    assert_eq!(block[1], 21.0);
}

/// Ensures that giant weights (100.0) multiplied by silence inputs (0.0)
/// will stay absolutely at zero, proving that the multiplication loop does not
/// introduce digital noise (*spontaneous DC offset*).
/// Then, activates *bias* mid-execution to prove that the runtime
/// switch is respected by the DSP primitive.
#[test]
fn test_conv1d_zero_input() {
    // Extremely high weights to test if any residual noise is amplified.
    let raw_weights = vec![100.0f32; 2 * 3 * 2];
    let mut weights = AlignedVec::new(24, 0.0f32);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide_f32(
        &raw_weights,
        &mut weights,
        2, // IN
        2, // OUT
        3, // K
    );

    let mut conv = Conv1d::<2, 2, 3> {
        weights: weights.clone(),
        bias: AlignedVec::from_vec(vec![0.0; 2]),
        do_bias: false,
        dilation: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    // Buffer filled with zeros.
    let layer_buffer = vec![0.0; 4 * 2];
    let mut block = vec![0.0; 2];

    // First pass: Without bias, output should be absolute 0.0.
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 2, 1);
    }

    assert_eq!(block, vec![0.0, 0.0]);

    // Second pass: Activate bias. The output should reflect exactly the injected bias.
    conv.do_bias = true;
    conv.bias = AlignedVec::from_vec(vec![7.5, 8.5]);

    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 2, 1);
    }

    assert_eq!(block, vec![7.5, 8.5]);
}

/// Manual matrix cross-product test (Dot Product)
/// against an expected result pre-calculated by a scientist.
///
/// Provides absolute guarantee that the `process_block` routine optimized for
/// x86-64 hardware handles perfectly summations containing concurrent positive
/// and negative values and weights, validating the correctness of mathematical calculations.
#[test]
fn test_conv1d_known_output() {
    // Heterogeneous weight matrix to validate channel cross-product (Dot Product).
    // Structure: OUT=2, K=2, IN=2. Total 8 weights.
    let raw_weights = vec![
        0.5, 1.5, // out0, in0, k0 and k1
        1.0, 2.0, // out0, in1, k0 and k1
        -0.5, -1.5, // out1, in0, k0 and k1
        -1.0, -2.0, // out1, in1, k0 and k1
    ];
    let mut weights = AlignedVec::new(16, 0.0f32);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide_f32(
        &raw_weights,
        &mut weights,
        2, // IN
        2, // OUT
        2, // K
    );

    let conv = Conv1d::<2, 2, 2> {
        weights,
        bias: AlignedVec::from_vec(vec![1.0, -1.0]),
        do_bias: true,
        dilation: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    // Layer buffer with 2 frames: F0=(2.0, 3.0), F1=(4.0, 5.0).
    let layer_buffer = vec![2.0, 3.0, 4.0, 5.0];
    let mut block = vec![0.0; 2];

    // Process at frame index 1 (F1).
    // out0 = bias[0] + dot(F0, w[out0,k0]) + dot(F1, w[out0,k1])
    //      = 1.0 + (2*0.5 + 3*1.0) + (4*1.5 + 5*2.0) = 1.0 + 4.0 + 16.0 = 21.0
    // out1 = bias[1] + dot(F0, w[out1,k0]) + dot(F1, w[out1,k1])
    //      = -1.0 + (2*-0.5 + 3*-1.0) + (4*-1.5 + 5*-2.0) = -1.0 - 4.0 - 16.0 = -21.0
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 1, 1);
    }

    assert_eq!(block[0], 21.0);
    assert_eq!(block[1], -21.0);
}
