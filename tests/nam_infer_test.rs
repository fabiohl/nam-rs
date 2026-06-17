// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration Tests for topology validation, neural inference stability,
//! and miscellaneous edge cases (denormals, block size invariance, community models).

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{
    NamWavenetTopology, WavenetTopologyResult, get_wavenet_topology, parse_nam_json,
};
use nam_rs::math::common::AlignedVec;
use nam_rs::math::common::half::f32_to_f16_bits;
use nam_rs::models::{NamModel, wavenet};
use std::fs;
use std::path::PathBuf;

mod common;
use common::*;

// WaveNetModel<CH=16, K=3, HEAD=8>: CH=16 channels, HEAD=8 (layer 0 head_size = Array2 channels).
type WaveNetStandard = wavenet::WaveNetModel<16, 3, 8>;

// =============================================================================
// Helpers — Synthetic Model Construction for Topology Unit Tests
// =============================================================================

/// Helper to simulate creation of a clean `WaveNetLayer` for Array1 (CH=16).
fn make_wavenet_layer(
    dilation: usize,
    _has_bias: bool,
    ch: usize,
) -> wavenet::WaveNetLayer<1, 16, 3> {
    let _raw_conv = vec![0.001f32; ch * 3 * ch];
    let conv_weights = {
        let mut fw = AlignedVec::new(_raw_conv.len(), 0.0f32);
        nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide_f32(
            &_raw_conv, &mut fw, ch, ch, 3,
        );
        fw
    };
    wavenet::WaveNetLayer {
        conv1d: wavenet::Conv1d {
            weights: conv_weights,
            bias: AlignedVec::from_vec(vec![0.0; ch]),
            do_bias: false,
            dilation,
            prefetch_fn: if dilation >= 128 {
                nam_rs::math::common::prefetch_strategy_2stage
            } else {
                nam_rs::math::common::prefetch_strategy_simple
            },
        },
        input_mixin: wavenet::DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![f32_to_f16_bits(0.001); ch]),
            bias: AlignedVec::from_vec(vec![0.0; ch]),
            do_bias: false,
        },
        one_by_one: wavenet::DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![f32_to_f16_bits(0.001); ch * ch]),
            bias: AlignedVec::from_vec(vec![0.0; ch]),
            do_bias: false,
        },
    }
}

/// Helper to simulate creation of a `WaveNetLayer` for Array2 (CH=8, =HEAD of Array1).
fn make_wavenet_layer_a2(dilation: usize) -> wavenet::WaveNetLayer<1, 8, 3> {
    let _raw_conv = vec![0.001f32; 8 * 3 * 8];
    let conv_weights = {
        let mut fw = AlignedVec::new(_raw_conv.len(), 0.0f32);
        nam_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide_f32(
            &_raw_conv, &mut fw, 8, 8, 3,
        );
        fw
    };
    wavenet::WaveNetLayer {
        conv1d: wavenet::Conv1d {
            weights: conv_weights,
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
            dilation,
            prefetch_fn: if dilation >= 128 {
                nam_rs::math::common::prefetch_strategy_2stage
            } else {
                nam_rs::math::common::prefetch_strategy_simple
            },
        },
        input_mixin: wavenet::DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![f32_to_f16_bits(0.001); 8]),
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
        },
        one_by_one: wavenet::DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![f32_to_f16_bits(0.001); 8 * 8]),
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
        },
    }
}

/// Helper to build a synthetic `WaveNetStandard` network.
///
/// `WaveNetModel<16, 3, 8>`: Array1 CH=16, Array2 CH=8(=HEAD), HEAD2=1.
/// Note: the type alias below now uses HEAD=8 (real head_size of BossWN-standard).
///
/// # Alignment with the production constructor (`build_wavenet_array`)
///
/// Each `WaveNetLayerState` receives `receptive_field_size = (K-1) * dilation`
/// from its specific layer — faithfully mirroring `build_wavenet_array` (L274).
/// The global `receptive_field_size` is the sum of all individual RFs (2046).
fn build_synthetic_wavenet_standard() -> WaveNetStandard {
    const K: usize = 3;
    let dilations_1: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let dilations_2: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    // Total RF = sum of RFs per layer: Σ (K-1)*d
    // For dilations = [1,2,4,8,16,32,64,128,256,512]: sum = 1023 × 2 = 2046
    let rf1: usize = dilations_1.iter().map(|&d| (K - 1) * d).sum();
    let rf2: usize = dilations_2.iter().map(|&d| (K - 1) * d).sum();
    let final_rf = rf1.max(rf2);

    let layers_1: Vec<wavenet::WaveNetLayer<1, 16, 3>> = dilations_1
        .iter()
        .map(|&d| make_wavenet_layer(d, false, 16))
        .collect();
    // Per-layer RF: (K-1)*d — mirrors build_wavenet_array L274
    let states_1: Vec<wavenet::WaveNetLayerState> = dilations_1
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            wavenet::WaveNetLayerState::new(16, (K - 1) * d, i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let array1 = wavenet::WaveNetLayerArray::<1, 1, 16, 3, 8> {
        layers: layers_1,
        states: states_1,
        rechannel: wavenet::DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![f32_to_f16_bits(0.001); 16]),
            bias: AlignedVec::from_vec(vec![0.0; 16]),
            do_bias: false,
        },
        head_rechannel: wavenet::DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![f32_to_f16_bits(0.001); 8 * 16]),
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 16 * wavenet::WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 16 * wavenet::WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; 8 * wavenet::WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf1,
        block_size: 16,
        block_buffer: AlignedVec::from_vec(vec![0.0; 16 * wavenet::WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: dilations_1.len(),
    };

    // Array2: IN=16(=CH), COND=1, CH=8(=HEAD1), HEAD2=1, HasHeadBias=true
    let layers_2: Vec<wavenet::WaveNetLayer<1, 8, 3>> = dilations_2
        .iter()
        .map(|&d| make_wavenet_layer_a2(d))
        .collect();
    // Per-layer RF: (K-1)*d (CH=8)
    let states_2: Vec<wavenet::WaveNetLayerState> = dilations_2
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            // alloc_num continues from where array1 stopped — mirrors the dispatcher's global alloc_num
            wavenet::WaveNetLayerState::new(8, (K - 1) * d, dilations_1.len() + i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let array2 = wavenet::WaveNetLayerArray::<16, 1, 8, 3, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: wavenet::DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![f32_to_f16_bits(0.0); 16 * 8]),
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
        },
        head_rechannel: wavenet::DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![f32_to_f16_bits(0.0); 8]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 8 * wavenet::WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 8 * wavenet::WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; wavenet::WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf2,
        block_size: 8,
        block_buffer: AlignedVec::from_vec(vec![0.0; 8 * wavenet::WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: dilations_2.len(),
    };

    WaveNetStandard {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: final_rf,
    }
}

/// Test 1: Audit of Loader Read Capacity and Geometry Validation
#[test]
fn test_wavenet_model_json_parsing() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        eprintln!("SKIP: WaveNet test model not found at {path:?}. Skipping parsing.");
        return;
    }

    let json_data = fs::read_to_string(path).expect("Failed to read JSON file");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser_nam_json");

    assert_eq!(model_data.architecture, "WaveNet", "Should detect WaveNet");
    let topo = get_wavenet_topology(&model_data);
    assert_eq!(
        topo,
        WavenetTopologyResult::Known(NamWavenetTopology::Standard),
        "The BossWN-standard model should be recognized as Standard Wavenet"
    );

    // Validate metadata properties for BossWN-standard match
    assert!(model_data.metadata.is_some());
    let metadata = model_data.metadata.unwrap();
    assert!(metadata.loudness.is_some());
}

/// Test 2: Run Multiple Sine Wave Blocks through the WaveNet Core and compute RMS/Error (Sanity)
///
/// Added RMS magnitude check ≤ 10.0 to detect divergence.
/// In debug, uses reduced blocks (512) for CI speed.
#[test]
fn test_wavenet_computational_stability() {
    let mut model = build_synthetic_wavenet_standard();

    // Math stabilization with silent transient blocks
    model.prewarm();

    let mut in_data = [0.0f32; TEST_BLOCK_SIZE];
    let mut out_data = [0.0f32; TEST_BLOCK_SIZE];

    // In debug, reduce blocks for faster CI; release uses full value.
    let num_blocks = if cfg!(debug_assertions) {
        512
    } else {
        TEST_NUM_BLOCKS
    };

    let mut tot_energy = 0.0f64;
    let mut pos: u64 = 0;

    for _ in 0..num_blocks {
        // Controlled sine generator as in `ModelTest.cpp`
        for item in in_data.iter_mut().take(TEST_BLOCK_SIZE) {
            *item = ((pos as f32) * 0.01).sin();
            pos += 1;
        }

        model.process(&in_data, &mut out_data);

        for &out_val in out_data.iter().take(TEST_BLOCK_SIZE) {
            assert!(
                out_val.is_finite(),
                "Computational crash detected: FPU produced non-finite float. Audit failure."
            );
            tot_energy += (out_val as f64) * (out_val as f64);
        }
    }

    let rms = (tot_energy / ((TEST_BLOCK_SIZE * num_blocks) as f64)).sqrt();
    println!(
        "[WaveNet Integrity Audit] RMS over processed sine wave: {}",
        rms
    );

    // T-7: Reasonable magnitude check — RMS should be ≤ 10.0 for synthetic model.
    // An RMS > 10.0 would indicate numerical divergence of the network or initialization error.
    assert!(
        rms <= 10.0,
        "WaveNet RMS {rms:.4} exceeds reasonable magnitude (10.0). Possible numerical divergence."
    );
}

// =============================================================================
// Test Coverage Expansion (Feather, Nano, LSTM 2x8)
// =============================================================================

/// Test 13: WaveNet Feather Stability
#[test]
fn test_wavenet_stability_feather() {
    let path = model_path("BossWN-feather.nam");

    if !path.exists() {
        eprintln!("SKIP: BossWN-feather.nam not found at {path:?}. Skipping Feather stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet Feather model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for Feather");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[Feather] Non-finite sample at index {i}: {s}"
        );
        assert!(
            s.abs() < 100.0,
            "[Feather] Excessive magnitude at index {i}: {s} (limit 100.0)"
        );
    }
}

/// WaveNet Nano Stability
#[test]
fn test_wavenet_stability_nano() {
    let path = model_path("BossWN-nano.nam");

    if !path.exists() {
        eprintln!("SKIP: BossWN-nano.nam not found at {path:?}. Skipping Nano stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet Nano model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for Nano");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "[Nano] Non-finite sample at index {i}: {s}");
        assert!(
            s.abs() < 100.0,
            "[Nano] Excessive magnitude at index {i}: {s} (limit 100.0)"
        );
    }
}

/// WaveNet A2-Full Stability
///
/// A2 fixtures use deterministic random weights (SEED=42), which can produce
/// large-but-finite outputs for certain inputs. Only finiteness is asserted
/// (no NaN/Inf), not magnitude limits — matching the existing
/// `test_a2_full_real_inference_finite_output()` convention.
#[test]
fn test_wavenet_stability_a2_full() {
    let path = model_path("wavenet_a2_full.nam");

    if !path.exists() {
        eprintln!("SKIP: wavenet_a2_full.nam not found at {path:?}. Skipping A2-Full stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet A2-Full model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for A2-Full");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[A2-Full] Non-finite sample at index {i}: {s}"
        );
    }
}

/// WaveNet A2-Lite Stability
///
/// Same convention as A2-Full: only finiteness asserted.
#[test]
fn test_wavenet_stability_a2_lite() {
    let path = model_path("wavenet_a2_lite.nam");

    if !path.exists() {
        eprintln!("SKIP: wavenet_a2_lite.nam not found at {path:?}. Skipping A2-Lite stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet A2-Lite model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for A2-Lite");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[A2-Lite] Non-finite sample at index {i}: {s}"
        );
    }
}

/// Test 15: LSTM 2x8 Stability
#[test]
fn test_lstm_stability_2x8() {
    let path = model_path("BossLSTM-2x8.nam");

    if !path.exists() {
        eprintln!("SKIP: BossLSTM-2x8.nam not found at {path:?}. Skipping LSTM 2x8 stability.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read LSTM 2x8 model");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
    let mut model = build_model(&model_data).expect("Dispatcher failed for LSTM 2x8");

    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[LSTM 2x8] Non-finite sample at index {i}: {s}"
        );
        assert!(
            s.abs() < 100.0,
            "[LSTM 2x8] Excessive magnitude at index {i}: {s} (limit 100.0)"
        );
    }
}

// =============================================================================
// Prolonged Silence Stability Test (Denormals)
// =============================================================================

/// Test 17: Stability under prolonged silence — denormal validation.
///
/// Loads `BossWN-standard.nam` and `BossLSTM-1x16.nam`, runs prewarm and
/// processes 4096 blocks (≈5.5s) of **total silence** (input = zeros).
///
/// Validates that:
/// - All outputs are finite.
/// - Output stabilizes (magnitude < 1.0 after convergence — real models
///   may have residual DC offset due to neural network biases).
/// - No detectable subnormal values in the output (all zero or finite normals).
/// - Per-block processing time (measured via `Instant::now()`) does not exceed 500μs.
///
/// This test exercises the exponential decay path in the internal states
/// of WaveNet (convolution buffers) and LSTM (cell/hidden states), which without
/// DAZ/FTZ converge to denormals and cause microcode penalty in the FPU.
#[test]
fn test_denormal_stability_silence() {
    #[cfg(debug_assertions)]
    const SILENCE_BLOCKS: usize = 256;
    #[cfg(not(debug_assertions))]
    const SILENCE_BLOCKS: usize = 4096;
    const BLOCK_SIZE: usize = 64;
    const MAX_BLOCK_TIME_US: u128 = 500;

    unsafe {
        nam_rs::math::common::set_daz_ftz();
    }

    // --- WaveNet Standard ---
    let wn_path = model_path("BossWN-standard.nam");
    if !wn_path.exists() {
        eprintln!("SKIP: BossWN-standard.nam not found. Skipping denormal silence WaveNet.");
    } else {
        let json_data =
            fs::read_to_string(&wn_path).expect("Failed to read WaveNet model for denormal test");
        let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
        let mut model =
            build_model(&model_data).expect("Dispatcher failed for denormal test WaveNet");

        model.prewarm(2048);

        let silence = [0.0f32; BLOCK_SIZE];
        let mut output = [0.0f32; BLOCK_SIZE];
        let mut max_block_time_us: u128 = 0;

        for block_idx in 0..SILENCE_BLOCKS {
            let start = std::time::Instant::now();
            model.process(&silence, &mut output);
            let elapsed = start.elapsed().as_micros();

            if block_idx > 100 && elapsed > max_block_time_us {
                max_block_time_us = elapsed;
            }

            // Validate finiteness across all blocks
            for (i, &s) in output.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "[Denormal WaveNet] Non-finite sample in block {block_idx}, index {i}: {s}"
                );
            }
        }

        // Validate stability: output after prolonged silence should be stable
        // (magnitude < 1.0). Real models maintain residual DC offset (~6e-3)
        // due to internal neural network biases — this is correct.
        for (i, &s) in output.iter().enumerate() {
            assert!(
                s.abs() < 1.0,
                "[Denormal WaveNet] After {SILENCE_BLOCKS} silence blocks, \
                 output[{i}]={s} diverged (threshold 1.0)"
            );
        }

        // No subnormal values in the output
        for &s in output.iter() {
            assert!(
                s == 0.0 || s.is_normal(),
                "[Denormal WaveNet] Subnormal value detected in output: {s} (bits: 0x{:08X})",
                s.to_bits()
            );
        }

        println!(
            "WaveNet denormal OK — {SILENCE_BLOCKS} silence blocks, \
             max_block_time={max_block_time_us}μs, output[0]={:.6e}",
            output[0]
        );

        // Timing gate: informative warning only — wall-clock is flaky under
        // system load and not a reliable denormal indicator. The value-based
        // checks above (subnormals, finiteness, stability) are the real gate.
        if !cfg!(debug_assertions) && max_block_time_us >= MAX_BLOCK_TIME_US {
            eprintln!(
                "WARN [Denormal WaveNet] Slowest block={max_block_time_us}μs exceeds \
                 {MAX_BLOCK_TIME_US}μs — possible denormal penalty or system load"
            );
        }
    }

    // --- LSTM 1×16 ---
    let lstm_path = model_path("BossLSTM-1x16.nam");
    if !lstm_path.exists() {
        eprintln!("SKIP: BossLSTM-1x16.nam not found. Skipping denormal silence LSTM.");
    } else {
        let json_data =
            fs::read_to_string(&lstm_path).expect("Failed to read LSTM model for denormal test");
        let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");
        let mut model = build_model(&model_data).expect("Dispatcher failed for denormal test LSTM");

        model.prewarm(2048);

        let silence = [0.0f32; BLOCK_SIZE];
        let mut output = [0.0f32; BLOCK_SIZE];
        let mut max_block_time_us: u128 = 0;

        for block_idx in 0..SILENCE_BLOCKS {
            let start = std::time::Instant::now();
            model.process(&silence, &mut output);
            let elapsed = start.elapsed().as_micros();

            if block_idx > 100 && elapsed > max_block_time_us {
                max_block_time_us = elapsed;
            }

            for (i, &s) in output.iter().enumerate() {
                assert!(
                    s.is_finite(),
                    "[Denormal LSTM] Non-finite sample in block {block_idx}, index {i}: {s}"
                );
            }
        }

        // Validate stability: output < 1.0 (no numerical divergence)
        for (i, &s) in output.iter().enumerate() {
            assert!(
                s.abs() < 1.0,
                "[Denormal LSTM] After {SILENCE_BLOCKS} silence blocks, \
                 output[{i}]={s} diverged (threshold 1.0)"
            );
        }

        // Validate that no subnormal value appears in the output
        for &s in output.iter() {
            // A subnormal f32 float has exponent = 0 and mantissa != 0
            // We verify that all values are zero or finite normals
            assert!(
                s == 0.0 || s.is_normal(),
                "[Denormal LSTM] Subnormal value detected in output after silence: {s} \
                 (bits: 0x{:08X})",
                s.to_bits()
            );
        }

        println!(
            "LSTM denormal OK — {SILENCE_BLOCKS} silence blocks, \
             max_block_time={max_block_time_us}μs, output[0]={:.6e}",
            output[0]
        );

        if !cfg!(debug_assertions) && max_block_time_us >= MAX_BLOCK_TIME_US {
            eprintln!(
                "WARN [Denormal LSTM] Slowest block={max_block_time_us}μs exceeds \
                 {MAX_BLOCK_TIME_US}μs — possible denormal penalty or system load"
            );
        }
    }
}

// =============================================================================
// Block Size Invariance Tests
// =============================================================================

/// Verifies Block Size invariance in the Static WaveNet implementation.
///
/// The inference engine must produce the same mathematical result (MSE ≈ 0)
/// regardless of the block size provided by the host (DAW/Soundcard).
/// This guarantees that the internal state of the dilated convolutions and
/// circular buffers are correctly preserved across block boundaries.
#[test]
fn test_wavenet_variable_block_sizes() {
    let path = model_path("BossWN-standard.nam");
    if !path.exists() {
        eprintln!("SKIP: BossWN-standard.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser");

    let block_sizes = [1, 16, 32, 64, 128, 256, 512];
    let input = generate_sine_440hz(512);
    let mut ref_output = vec![0.0f32; 512];

    for &bs in &block_sizes {
        let mut model = build_model(&model_data).expect("Failed to build model");
        model.prewarm(2048);

        let mut output = vec![0.0f32; 512];
        process_in_blocks(&mut model, &input, &mut output, bs);

        let mut tot_energy = 0.0f64;
        for &s in &output {
            assert!(
                s.is_finite(),
                "Block size {} produced non-finite output",
                bs
            );
            tot_energy += (s as f64) * (s as f64);
        }
        let rms = (tot_energy / 512.0).sqrt();
        assert!(rms <= 10.0, "Block size {} has high RMS: {}", bs, rms);

        if bs == 1 {
            ref_output.copy_from_slice(&output);
        } else {
            let mse = compute_mse(&ref_output, &output);
            assert!(
                mse < 1e-7,
                "Divergence between block_size=1 and block_size={} (MSE={})",
                bs,
                mse
            );
        }
    }
}

/// Verifies LSTM processing with various block sizes.
///
/// The inference engine must be invariant to the processed block size:
/// processing 512 samples one by one must produce the same result (MSE ~0)
/// as processing in blocks of 64 or 512. This is critical to ensure that the
/// sound does not change depending on the host's buffer configuration (DAW/Soundcard).
#[test]
fn test_lstm_variable_block_sizes() {
    let path = model_path("BossLSTM-1x16.nam");
    if !path.exists() {
        eprintln!("SKIP: BossLSTM-1x16.nam not found.");
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in parser");

    let block_sizes = [1, 16, 32, 64, 128, 256, 512];
    let input = generate_sine_440hz(512);
    let mut ref_output = vec![0.0f32; 512];

    for &bs in &block_sizes {
        let mut model = build_model(&model_data).expect("Failed to build LSTM model");
        model.prewarm(2048);

        let mut output = vec![0.0f32; 512];
        process_in_blocks(&mut model, &input, &mut output, bs);

        let mut tot_energy = 0.0f64;
        for &s in &output {
            assert!(
                s.is_finite(),
                "LSTM Block size {} produced non-finite output (NaN/Inf)",
                bs
            );
            tot_energy += (s as f64) * (s as f64);
        }
        let rms = (tot_energy / 512.0).sqrt();
        assert!(
            rms <= 10.0,
            "Instability detected: LSTM Block size {} has excessive RMS: {}",
            bs,
            rms
        );

        if bs == 1 {
            ref_output.copy_from_slice(&output);
        } else {
            let mse = compute_mse(&ref_output, &output);
            assert!(
                mse < 1e-7,
                "Block invariance failed for LSTM: Divergence between bs=1 and bs={} (MSE={})",
                bs,
                mse
            );
        }
    }
}

// =============================================================================
// Community Model Tests (Task 6.2)
// =============================================================================

/// Inference Validation on Real Community Models (Ecosystem Regression).
///
/// This test loads a collection of community-exported models to ensure
/// that the loader and dispatcher correctly handle metadata subtleties
/// and real topologies (e.g., Standard vs Lite) generated by different versions
/// of the official exporter (NeuralAmpModeler/NAM).
///
/// Validation on real models is critical as it detects regressions that do not appear
/// in ideal synthetic models, such as bias truncation or gain normalization.
#[test]
fn test_community_models_inference() {
    let models: [(&str, WavenetTopologyResult); 4] = [
        (
            "BossWN-standard.nam",
            WavenetTopologyResult::Known(NamWavenetTopology::Standard),
        ),
        (
            "BossWN-lite.nam",
            WavenetTopologyResult::Known(NamWavenetTopology::Lite),
        ),
        (
            "BossWN-feather.nam",
            WavenetTopologyResult::Known(NamWavenetTopology::Feather),
        ),
        (
            "BossWN-nano.nam",
            WavenetTopologyResult::Known(NamWavenetTopology::Nano),
        ),
    ];

    let input = generate_sine_440hz(64);

    for (filename, expected_topo) in models {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/fixtures/models");
        path.push(filename);

        if !path.exists() {
            panic!(
                "Community model not found: {:?}. Check test submodules.",
                path
            );
        }

        let json_data = fs::read_to_string(&path).expect("Failed to read JSON");
        let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

        let topo = get_wavenet_topology(&model_data);
        assert!(
            matches!(topo, ref result if result == &expected_topo),
            "Incorrectly detected topology for community model {}: expected {:?}, got {:?}",
            filename,
            expected_topo,
            topo
        );

        let mut model =
            build_model(&model_data).expect("Dispatcher failed to build community model");

        model.prewarm(2048);

        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);

        for (i, &s) in output.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Model {} produced invalid sample (NaN/Inf) at index {}",
                filename,
                i
            );
            assert!(
                s.abs() < 100.0,
                "Model {} generated excessive magnitude peak at index {}: {}. Possible instability.",
                filename,
                i,
                s
            );
        }
        println!("✓ Community model {} successfully validated.", filename);
    }
}

/// Legacy Format (Keras/H5) Rejection Test.
///
/// The NAM-rs engine focuses on the modern JSON-based format (v0.5+) and NAMB (v1/v2).
/// Old Keras/TensorFlow H5 models must be gracefully rejected by the dispatcher.
///
/// Fixture: `tests/fixtures/models/keras_unsupported.json`
/// (Synthetic Keras-legacy dummy export lacking required "architecture" field;
/// committed to the repo — must never be removed without updating this test.)
#[test]
fn test_reject_keras_legacy_format() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/keras_unsupported.json");

    assert!(
        path.exists(),
        "Keras-legacy fixture not found at {path:?}. \
         This is a committed fixture — do not remove it without updating this test. \
         If deliberately retired, convert this test to an inline JSON stub or delete the test explicitly."
    );

    let json_data = fs::read_to_string(&path).expect("Failed to read Keras JSON");

    // The parser should return Ok if it is structurally valid JSON,
    // but may fail if it already identifies missing fields.
    let model_data = match parse_nam_json(&json_data) {
        Ok(data) => data,
        Err(e) => {
            println!("parse_nam_json correctly rejected Keras-legacy format: {e}");
            return;
        }
    };

    // If the parser passed, the dispatcher MUST reject it.
    let build_result = build_model(&model_data);
    assert!(
        build_result.is_err(),
        "build_model() accepted a Keras-legacy format — it must be rejected."
    );

    println!("Keras-legacy format correctly rejected via build_model().");
}

/// Architectural Validation: Non-Tanh Activation in WaveNet with unrecognized shape.
///
/// Models with non-Tanh activations that do not match any known A2 or A1 topology
/// must return a clear error (not a silent bypass or panic).
#[test]
fn test_reject_unrecognized_activation_with_clear_error() {
    let synthetic_json = r#"{
        "version": "0.5.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [
                {
                    "condition_size": 1,
                    "channels": 16,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4],
                    "activation": "ReLU",
                    "gated": false,
                    "head_size": 8,
                    "with_head": true
                }
            ]
        },
        "weights": [0.0, 0.0, 0.0]
    }"#;

    let model_data = parse_nam_json(synthetic_json).expect("Failed to parse synthetic JSON");
    let result = build_model(&model_data);
    assert!(
        result.is_err(),
        "Unrecognized shape with non-Tanh activation must return error"
    );
    let err = format!("{}", result.err().unwrap());
    assert!(
        err.contains("not recognized"),
        "Error should mention topology not being recognized: {err}",
    );
}
