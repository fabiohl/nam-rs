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
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
            // The 1x1 projection (Dense) finalizes the cell, preparing the signal for the residual.
            // Dimensions: OUT * IN = 4 * 4.
            one_by_one: DenseLayer {
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
                f32_weights: None,
                // Dimensions: OUT * IN = 2 * 1.
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
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
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
            bias: AlignedVec::from_vec(vec![0.0; 4]),
            do_bias: false,
        },
        // Head Rechannel: Aggregates "skip connections" from all layers for the array's output.
        head_rechannel: DenseLayer {
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
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 2]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        // The final NAM model projection reduces everything to 1 channel (mono audio).
        head_rechannel: DenseLayer {
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

/// Tests that the stack/heap allocated sizes of WaveNet structures
/// match the mathematical specifications for channels and frames.
///
/// Since WaveNet uses fixed vectors and arrays, it is crucial to ensure
/// that `head_outputs` has exact space (e.g., 2 channels * 64 frames),
/// preventing segfaults and "out of bounds" during real-time audio processing.
#[test]
fn test_wavenet_model_allocation() {
    let model = build_tiny_wavenet();
    assert_eq!(model.array1.layers.len(), 3);
    assert_eq!(model.array2.layers.len(), 3);
    assert_eq!(model.array1.head_outputs.len(), 2 * WAVENET_MAX_NUM_FRAMES); // HEAD1=2
    assert_eq!(model.array2.head_outputs.len(), WAVENET_MAX_NUM_FRAMES); // HEAD2=1 (always fixed)
    assert!((model.head_scale - 0.02).abs() < 1e-6);
}

/// Verifies that "hot" initialization (prewarm) of the model does not generate
/// numerical instability like NaN (Not a Number) or Inf (Infinity).
///
/// *Prewarm* serves to stabilize the network (especially delay buffers)
/// by iterating repeatedly with zeros before starting playback. An
/// uninitialized memory bug would rapidly appear here.
#[test]
fn test_wavenet_prewarm_no_nan() {
    let mut model = build_tiny_wavenet();
    model.prewarm();

    // Verify that internal buffers contain no NaN/Inf after prewarm
    for state in &model.array1.states {
        for &v in state.layer_buffer.iter() {
            assert!(v.is_finite(), "NaN/Inf detected in array1 after prewarm");
        }
    }
    for state in &model.array2.states {
        for &v in state.layer_buffer.iter() {
            assert!(v.is_finite(), "NaN/Inf detected in array2 after prewarm");
        }
    }
}

/// Processes an entire block of absolute silence (zeros) and ensures
/// that the model reacts in a linear and predictable fashion.
///
/// Silence at the input should, at most, extract the model's *bias*
/// in a stable manner. Any divergence to NaN or Inf means that
/// the model suffered division by zero or pointer corruption in the SIMD flow.
#[test]
fn test_wavenet_process_zeros() {
    // Instantiate the "tiny" model (mock) and stabilize the historical Ring Buffers.
    let mut model = build_tiny_wavenet();
    model.prewarm();

    // Prepare a block of 16 samples of absolute silence.
    let input = [0.0f32; 16];
    let mut output = [0.0f32; 16];

    // The forward pass should convert silence into stable values (usually small DC offsets).
    model.process(&input, &mut output);

    // Validate that the output signal is finite. If there's division by zero or overflow
    // in any SIMD layer, the result will be NaN or Inf, which is unacceptable in DSP.
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Output sample [{}] is NaN/Inf: {}", i, v);
    }
}

/// Deterministic integrity test.
///
/// Runs two identical model instances processing the same stimulus.
/// The DSP engine (in this case, the AVX2 SIMD routines) must be mathematically
/// deterministic. Point differences (diverging floats) would indicate
/// state leakage from previous processing (*state bleeding*),
/// which would ruin the phase quality of the generated audio.
#[test]
fn test_wavenet_process_deterministic() {
    // Create two isolated and identical instances to ensure internal state
    // is not shared unsafely (memory leak or thread).
    let mut model_a = build_tiny_wavenet();
    let mut model_b = build_tiny_wavenet();

    model_a.prewarm();
    model_b.prewarm();

    // Apply a constant stimulus of 0.1 (DC impulse).
    let input = [0.1f32; 8];
    let mut out_a = [0.0f32; 8];
    let mut out_b = [0.0f32; 8];

    // Process the same signal in both models.
    model_a.process(&input, &mut out_a);
    model_b.process(&input, &mut out_b);

    // The NAM architecture must be deterministic: the same input must generate the same output
    // bit-for-bit or within hardware rounding tolerance (1e-6).
    for i in 0..8 {
        assert!(
            (out_a[i] - out_b[i]).abs() < 1e-6,
            "Non-deterministic result at sample [{}]: {} vs {}",
            i,
            out_a[i],
            out_b[i]
        );
    }
}

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
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;
    let mut weights = AlignedVec::new(16, 0u16);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
        &raw_weights,
        &mut weights,
        4, // IN
        4, // OUT
        1, // K
        is_bf16,
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
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;
    let mut weights = AlignedVec::new(16, 0u16);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
        &raw_weights,
        &mut weights,
        4, // IN
        4, // OUT
        1, // K
        is_bf16,
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
    let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
        == crate::math::common::InstructionSet::Avx512VnniBf16;
    let mut weights = AlignedVec::new(16, 0u16);
    crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
        &raw_weights,
        &mut weights,
        2, // IN
        2, // OUT
        2, // K
        is_bf16,
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

#[test]
fn test_conv1d_dyn_padding_non_multiple_of_4() {
    let in_ch = 2;
    let out_ch: usize = 6;
    let kernel = 3;
    let dilation = 1;

    let num_blocks = out_ch.div_ceil(4);
    let total_padded = num_blocks * 4 * in_ch * kernel;

    let mut raw_weights = vec![0.0f32; out_ch * kernel * in_ch];
    for out_c in 0..out_ch {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                let idx = (out_c * in_ch + in_c) * kernel + k;
                raw_weights[idx] = (out_c + 1) as f32;
            }
        }
    }

    let mut weights = AlignedVec::new(total_padded, 0u16);
    for b in 0..num_blocks {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    let target_idx = b * (kernel * in_ch * 4) + k * (in_ch * 4) + in_c * 4 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * kernel + k;
                        weights[target_idx] = half::f16::from_f32(raw_weights[raw_idx]).to_bits();
                    } else {
                        weights[target_idx] = 0;
                    }
                }
            }
        }
    }

    let bias = AlignedVec::from_vec(vec![0.5f32; out_ch]);

    let conv = Conv1dDyn {
        weights,
        bias,
        do_bias: true,
        dilation,
        in_ch,
        out_ch,
        num_blocks: out_ch.div_ceil(4),
        kernel,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    let layer_buffer = vec![1.0f32; 5 * in_ch];
    let mut block = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<crate::math::common::Avx2Math>(
            &layer_buffer,
            &mut block,
            4,
            None,
        );
    }

    let expected = vec![6.5, 12.5, 18.5, 24.5, 30.5, 36.5];
    assert_eq!(block, expected);
}

#[test]
fn test_conv1d_dyn_large_kernel_no_segfault() {
    let in_ch = 2;
    let out_ch: usize = 4;
    let kernel = 10;
    let dilation = 1;

    let num_blocks = out_ch.div_ceil(4);
    let total_padded = num_blocks * 4 * in_ch * kernel;

    let mut weights = AlignedVec::new(total_padded, 0u16);
    for i in 0..total_padded {
        weights[i] = half::f16::from_f32(1.0).to_bits();
    }

    let bias = AlignedVec::from_vec(vec![0.5f32; out_ch]);

    let conv = Conv1dDyn {
        weights,
        bias,
        do_bias: true,
        dilation,
        in_ch,
        out_ch,
        num_blocks: out_ch.div_ceil(4),
        kernel,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    let layer_buffer = vec![1.0f32; 24];
    let mut out_f0 = vec![0.0f32; out_ch];
    let mut out_f1 = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<crate::math::common::Avx2Math>(
            &layer_buffer,
            &mut out_f0,
            9,
            None,
        );
        conv.process_dual_frame::<crate::math::common::Avx2Math>(
            &layer_buffer,
            &mut out_f0,
            &mut out_f1,
            9,
            10,
            None,
            None,
        );
    }

    // Single frame calculation: bias (0.5) + 10 (taps) * 2 (channels) * 1.0 (input) * 1.0 (weight) = 20.5
    for val in out_f0 {
        assert!((val - 20.5).abs() < 1e-4);
    }
    for val in out_f1 {
        assert!((val - 20.5).abs() < 1e-4);
    }
}
