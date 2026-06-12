// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Verifies the basic "Identity" functionality of a Dense layer
/// (Fully Connected). This is used extensively in the *1x1* connections and
/// WaveNet output channel aggregations (*Skip Connections*).
#[test]
fn test_dense_layer_identity() {
    // 4x4 Identity weight matrix.
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 16]); // OUT=4 * IN=4
    for out_c in 0..4 {
        weights[out_c * 4 + out_c] = half::f16::from_f32(1.0).to_bits();
    }

    let dense = DenseLayer::<4, 4> {
        f32_weights: None,
        weights,
        bias: AlignedVec::from_vec(vec![0.0; 4]),
        do_bias: false,
    };

    let input = vec![1.5, 2.5, 3.5, 4.5];
    let mut output = vec![0.0; 4];

    // 1x1 dense layers are fundamental for mixing channels without looking at time.
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    assert_eq!(output, vec![1.5, 2.5, 3.5, 4.5]);
}

/// Verifies the correct injection of *Bias* tensors in Dense Layers
/// via SIMD pointers, ensuring the final Output's linear alteration.
#[test]
fn test_dense_layer_with_bias() {
    // Identity weights + Bias of 1.0.
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 16]);
    for out_c in 0..4 {
        weights[out_c * 4 + out_c] = half::f16::from_f32(1.0).to_bits();
    }

    let dense = DenseLayer::<4, 4> {
        f32_weights: None,
        weights,
        bias: AlignedVec::from_vec(vec![1.0; 4]),
        do_bias: true,
    };

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];

    // The result should be translated by the bias across all 4 channels.
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    assert_eq!(output, vec![2.0, 3.0, 4.0, 5.0]);
}

/// Runs a Dense Layer with non-square dimensionality (IN=8, OUT=4).
///
/// Since WaveNet constantly changes matrices (from CH to HEAD and vice-versa),
/// SIMD engines must never assume perfectly symmetric matrices (NxN).
/// This test injects heterogeneous values to ensure that
/// nested FMA loop stops compute correctly up to the exact allocation limit.
#[test]
fn test_dense_layer_rectangular() {
    // Asymmetric Matrix: IN=8, OUT=4.
    // In real models, this happens when projecting CH (e.g., 16) to HEAD (e.g., 8).
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 32]); // 4 * 8
    // out_c = 0: Soma ponderada de in[0] e in[1]
    // [IN][OUT] -> in_c * OUT + out_c
    weights[0] = half::f16::from_f32(1.0).to_bits(); // in0, out0
    weights[4] = half::f16::from_f32(2.0).to_bits(); // in1, out0

    // out_c = 1: Soma ponderada de in[2] e in[3]
    weights[9] = half::f16::from_f32(3.0).to_bits(); // in2, out1
    weights[13] = half::f16::from_f32(4.0).to_bits(); // in3, out1

    // out_c = 2: Escala simples de in[4]
    weights[18] = half::f16::from_f32(0.5).to_bits(); // in4, out2

    // out_c = 3: Phase inversion of in[7]
    weights[31] = half::f16::from_f32(-1.0).to_bits(); // in7, out3

    let dense = DenseLayer::<8, 4> {
        f32_weights: None,
        weights,
        bias: AlignedVec::from_vec(vec![0.5, -0.5, 1.0, -1.0]),
        do_bias: true,
    };

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0; 4];

    // Validate that the SIMD loop correctly handles the matrix row end (stride).
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    // Manual calculation trace:
    // out[0] = (1.0 * 1.0) + (2.0 * 2.0) + 0.5 = 1.0 + 4.0 + 0.5 = 5.5
    // out[1] = (3.0 * 3.0) + (4.0 * 4.0) - 0.5 = 9.0 + 16.0 - 0.5 = 24.5
    // out[2] = (5.0 * 0.5) + 1.0 = 2.5 + 1.0 = 3.5
    // out[3] = (8.0 * -1.0) - 1.0 = -8.0 - 1.0 = -9.0

    assert_eq!(output[0], 5.5);
    assert_eq!(output[1], 24.5);
    assert_eq!(output[2], 3.5);
    assert_eq!(output[3], -9.0);
}
