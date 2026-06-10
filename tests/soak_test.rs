// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Numerical Stability Suite (Soak Test) for NAM-rs.
//!
//! These tests are designed to run for millions of iterations, checking
//! numerical drift, filter stability, absence of NaNs/Infs, and circular
//! buffer integrity under long-duration execution.
//!
//! Execution: `cargo test --release -- --ignored --nocapture`

use nam_rs::dsp::gate::*;
use nam_rs::dsp::mirror_buf::*;
use nam_rs::dsp::resampler::*;
use nam_rs::math::common::AlignedVec;
use nam_rs::models::a2::{A2_KERNEL_SIZES, WaveNetA2};
use nam_rs::models::lstm::*;
use nam_rs::models::wavenet::*;
use nam_rs::models::wavenet::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use std::time::Instant;

/// Simple deterministic PRNG (Linear Congruential Generator - LCG).
///
/// Implemented manually to keep the project free of rand crate dependencies
/// used only for tests. The goal is to provide a reproducible noise stream
/// for diagnosing failures in soak tests.
struct SimplePcg {
    state: u64,
}

impl SimplePcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates the next f32 in the range [-1.0, 1.0].
    fn next_f32(&mut self) -> f32 {
        // Simplified variant of PCG-XSH-RR for 32-bit output.
        let old_state = self.state;
        self.state = old_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        let res = (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31));
        // Normalization to the floating-point audio range
        (res as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Helper to build a synthetic WaveNetModel<16, 3, 8> for soak testing.
///
/// This model simulates the "Standard" topology (Array1 + Array2) with 10 million implicit
/// parameters via constant weights. Initialization with small values (0.01) ensures
/// that the audio does not explode immediately, allowing the FPU to process real values
/// across all layers for millions of iterations.
fn build_soak_wavenet() -> WaveNetModel<16, 3, 8> {
    // Inner layer of Array1: CH=16, K=3
    let make_layer = |dilation: usize| -> WaveNetLayer<1, 16, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: AlignedVec::from_vec(vec![
                    half::f16::from_f32(0.01).to_bits();
                    16 * 3 * 16
                ]),
                bias: AlignedVec::from_vec(vec![0.001; 16]),
                do_bias: true,
                dilation,
                // Prefetch strategy switches for larger dilations to test
                // the engine's cache-miss handling.
                prefetch_fn: if dilation >= 128 {
                    nam_rs::math::common::prefetch_strategy_2stage
                } else {
                    nam_rs::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 16]),
                bias: AlignedVec::from_vec(vec![0.0; 16]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 16 * 16]),
                bias: AlignedVec::from_vec(vec![0.0; 16]),
                do_bias: false,
            },
        }
    };

    // Default dilations for the Standard model (total RF = 2046 samples)
    let dilations = [1, 2, 4, 8, 16, 32, 64, 128];
    let rf = 128 * (3 - 1);

    let layers_1: Vec<WaveNetLayer<1, 16, 3>> = dilations.iter().map(|&d| make_layer(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(16, rf, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    // Array 1: Responsible for extracting deep temporal features
    let array1 = WaveNetLayerArray::<1, 1, 16, 3, 8> {
        layers: layers_1,
        states: states_1,
        rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 16]),
            bias: AlignedVec::from_vec(vec![0.0; 16]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8 * 16]),
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf,
        block_size: 16,
        block_buffer: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: dilations.len(),
    };

    // Array 2: Final spectral refinement (CH=8, K=3)
    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 8, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8 * 3 * 8]),
                bias: AlignedVec::from_vec(vec![0.001; 8]),
                do_bias: true,
                dilation,
                prefetch_fn: nam_rs::math::common::prefetch_strategy_simple,
            },
            input_mixin: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8]),
                bias: AlignedVec::from_vec(vec![0.0; 8]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                f32_weights: None,
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8 * 8]),
                bias: AlignedVec::from_vec(vec![0.0; 8]),
                do_bias: false,
            },
        }
    };

    let layers_2: Vec<WaveNetLayer<1, 8, 3>> = [1, 2].iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| {
            WaveNetLayerState::new(8, 2 * (3 - 1), i).expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let layers_2_len = layers_2.len();
    let array2 = WaveNetLayerArray::<16, 1, 8, 3, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 16 * 8]),
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            f32_weights: None,
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: 2,
        block_size: 8,
        block_buffer: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
        effective_layers: layers_2_len,
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.1,
        receptive_field_size: rf,
    }
}

/// Soak Test: Infinite Silence Processing via WaveNet.
///
/// The goal is to verify whether error accumulation in feedback models (if any)
/// or f16 precision instability in SIMD kernels causes NaNs or audio
/// "blow-ups" after millions of iterations.
#[test]
#[ignore] // Run manually via --ignored
fn test_wavenet_silence_soak() {
    let mut model = build_soak_wavenet();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        // black_box prevents the compiler from noticing that the input is always zero
        // and optimizing the entire inference loop into a no-op.
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            // Basic numerical integrity check (RT Safety)
            assert!(
                v.is_finite(),
                "Numerical Instability (NaN/Inf) after {} frames",
                processed
            );
            // Ensures the model is not gaining energy out of nowhere (divergence)
            assert!(
                (-2.0..=2.0).contains(&v),
                "Output diverged excessively: {} after {} frames",
                v,
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- WaveNet Silence Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: White Noise Processing via WaveNet.
///
/// Unlike silence, white noise excites all filter coefficients
/// and all activation branches (Tanh) simultaneously. This is vital
/// for detecting numerical overflows in deep layers that only occur
/// under high signal energy.
#[test]
#[ignore]
fn test_wavenet_noise_soak() {
    let mut model = build_soak_wavenet();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(42);
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "NaN/Inf detected under noise after {} frames",
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- WaveNet Noise Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: LSTM state stability under silence.
///
/// LSTMs have recurrent states (cell state) that can accumulate residual errors.
/// This test monitors internal state sanity over millions of iterations.
#[test]
#[ignore]
fn test_lstm_silence_soak() {
    let mut model = LstmModel2::<16, 17, 32, 64>::new();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;

    let start = Instant::now();
    while processed < num_frames {
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "NaN/Inf detected in LSTM output after {} frames",
                processed
            );
        }
        // Checks divergence of internal states in the 2-layer architecture
        for &v in model.layer1.cell_state.iter() {
            assert!(v.is_finite(), "Layer 1 LSTM internal state corrupted");
        }
        for &v in model.layer2.cell_state.iter() {
            assert!(v.is_finite(), "Layer 2 LSTM internal state corrupted");
        }

        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- LSTM Silence Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
}

/// Soak Test: LSTM resilience under stochastic input.
///
/// Random signals force constant forgetting and updating of the LSTM
/// gates. This test ensures that the "Forget/Input/Output" logic
/// does not cause infinite drift under signal stress conditions.
#[test]
#[ignore]
fn test_lstm_noise_soak() {
    let mut model = LstmModel2::<16, 17, 32, 64>::new();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(1337);
    let num_frames = 10_000_000;
    let mut processed = 0;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "LSTM diverged under noise after {} frames",
                processed
            );
        }
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- LSTM Noise Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
}

/// Soak Test: Resampler stability under long-duration asynchronous conversions.
///
/// The resampler uses polynomial interpolation and phase accumulation. Small rounding
/// errors in the phase increment can accumulate over millions of samples,
/// causing "clicks" or NaNs. This test validates Upsampling and Downsampling scenarios.
#[test]
#[ignore]
fn test_resampler_drift_soak() {
    // Scenario 1: Upsampling (22050 -> 48000)
    let mut resampler = NamResampler::new(22050, 48000, 64).unwrap();
    let mut pcg = SimplePcg::new(777);
    let num_samples = 50_000_000; // 50 million samples (~17 minutes at 48kHz)
    let mut processed_in = 0;
    let mut processed_out = 0;

    let mut in_l = vec![0.0f32; 1024];
    let mut in_r = vec![0.0f32; 1024];
    let mut out_l = vec![0.0f32; 2048];
    let mut out_r = vec![0.0f32; 2048];

    let start = Instant::now();
    while processed_in < num_samples {
        for i in 0..1024 {
            in_l[i] = pcg.next_f32();
            in_r[i] = pcg.next_f32();
        }
        let n = resampler.process_input(
            std::hint::black_box(&in_l),
            std::hint::black_box(&in_r),
            std::hint::black_box(&mut out_l),
            std::hint::black_box(&mut out_r),
        );
        processed_in += 1024;
        processed_out += n;

        for i in 0..n {
            assert!(out_l[i].is_finite(), "NaN in Resampler Out L (Upsampling)");
            assert!(out_r[i].is_finite(), "NaN in Resampler Out R (Upsampling)");
        }
    }
    let duration = start.elapsed();

    println!("--- Resampler Drift Soak (22050->48000) ---");
    println!("Duration: {:?}", duration);
    println!("Samples In: {}, Out: {}", processed_in, processed_out);

    // Scenario 2: Downsampling (96000 -> 48000)
    let mut resampler = NamResampler::new(96000, 48000, 64).unwrap();
    processed_in = 0;
    processed_out = 0;
    let start = Instant::now();
    while processed_in < num_samples {
        for i in 0..1024 {
            in_l[i] = pcg.next_f32();
            in_r[i] = pcg.next_f32();
        }
        let n = resampler.process_input(
            std::hint::black_box(&in_l),
            std::hint::black_box(&in_r),
            std::hint::black_box(&mut out_l[..512]),
            std::hint::black_box(&mut out_r[..512]),
        );
        processed_in += 1024;
        processed_out += n;
        for i in 0..n {
            assert!(
                out_l[i].is_finite(),
                "NaN in Resampler Out L (Downsampling)"
            );
            assert!(
                out_r[i].is_finite(),
                "NaN in Resampler Out R (Downsampling)"
            );
        }
    }
    println!("--- Resampler Drift Soak (96000->48000) ---");
    println!("Duration: {:?}", start.elapsed());
    println!("Samples In: {}, Out: {}", processed_in, processed_out);
}

/// Soak Test: MirroredBuffer memory integrity.
///
/// MirroredBuffer uses mmap to mirror memory, enabling continuous linear
/// accesses that cross the buffer boundary. This test verifies that mirroring
/// remains consistent after billions of writes.
#[test]
#[ignore]
#[cfg(target_os = "linux")]
fn test_mirror_buf_long_run() {
    let mut buf = MirroredBuffer::<f32>::new(1024 * 1024).expect("Failed to create MirroredBuffer"); // 1M elements
    let size = buf.size();
    let num_cycles = 100_000_000;
    let mut pos = 0;
    let chunk = 64;

    let start = Instant::now();
    for i in 0..(num_cycles / chunk) {
        // Simulates write and advance in the mirrored buffer
        for j in 0..chunk {
            std::hint::black_box(&mut buf)[pos + j] = std::hint::black_box((i * chunk + j) as f32);
        }

        // Checks integrity at the boundary: the value written at the end
        // must appear identically at the beginning (mmap mirroring).
        if pos + chunk >= size {
            let offset = (pos + chunk) - size;
            for j in 0..offset {
                assert_eq!(
                    buf[j],
                    buf[size + j],
                    "Critical memory mirroring failure in MirroredBuffer at index {}",
                    j
                );
            }
        }

        pos += chunk;
        if pos >= size {
            pos -= size;
        }
    }
    let duration = start.elapsed();

    println!("--- MirroredBuffer Long Run ---");
    println!("Duration: {:?}", duration);
    println!("Cycles (samples): {}", num_cycles);
}

/// Soak Test: Noise Gate State Machine (FSM) endurance.
///
/// This test forces rapid alternations between the `Open`, `Hold`, and `Release` states
/// using a stochastic binary signal. The goal is to ensure that the gain
/// multiplier never leaves the [0, 1] range and that the FSM does not lock up
/// in invalid states after millions of transitions.
#[test]
#[ignore]
fn test_gate_fsm_endurance() {
    let mut gate = DynamicHysteresis::new();
    let params = GateParams::default();
    let mut pcg = SimplePcg::new(999);
    let num_alternations = 10_000_000; // 10 million transition cycles

    // Typical real-world thresholds (-60dB and -80dB)
    let threshold_open = -60.0f32.powf(10.0 / 20.0);
    let threshold_close = -80.0f32.powf(10.0 / 20.0);

    let start = Instant::now();
    for _ in 0..num_alternations {
        // Aggressively alternates between full signal (1.0) and silence (0.0)
        let val = if pcg.next_f32() > 0.0 { 1.0 } else { 0.0 };
        gate.update(
            std::hint::black_box(val),
            threshold_open,
            threshold_close,
            &params,
            64,
        );

        // The gate's gain multiplier must be strictly well-behaved [0.0, 1.0]
        let m = std::hint::black_box(gate.multiplier());
        assert!(
            (0.0..=1.0).contains(&m),
            "Gate multiplier diverged: {} outside [0, 1]",
            m
        );
    }
    let duration = start.elapsed();

    println!("--- Gate FSM Endurance ---");
    println!("Duration: {:?}", duration);
    println!("Alternations: {}", num_alternations);
}

// =============================================================================
// A2 Architecture Soak Tests — WaveNetA2 numerical stability
// =============================================================================

/// Computes total A2 weight count.
const fn a2_total_weights<const CH: usize>() -> usize {
    let mut total = CH;
    let mut i = 0;
    while i < 23 {
        let k = A2_KERNEL_SIZES[i];
        total += CH * CH * k + CH + CH + CH * CH + CH;
        i += 1;
    }
    total += 16 * CH + 2;
    total
}

/// Assembles synthetic A2 weight stream.
fn a2_synth_weights<const CH: usize>(weight_val: f32) -> Vec<f32> {
    let num = a2_total_weights::<CH>();
    let mut w = Vec::with_capacity(num);
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
    assert_eq!(w.len(), num);
    w
}

/// Builds a synthetically-weighted A2 model.
fn build_soak_a2<const CH: usize>(weight_val: f32) -> WaveNetA2<CH> {
    let weights = a2_synth_weights::<CH>(weight_val);
    let mut model = WaveNetA2::<CH>::new();
    model.set_weights(&weights).expect("A2 set_weights failed");
    model
}

/// Soak Test: A2-Full silence processing — 10M frames.
#[test]
#[ignore]
fn test_a2_full_silence_soak() {
    let mut model = build_soak_a2::<8>(0.01);
    model.prewarm();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(v.is_finite(), "A2-Full NaN/Inf after {} frames", processed);
            assert!(
                (-2.0..=2.0).contains(&v),
                "A2-Full diverged: {} after {} frames",
                v,
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();
    println!("--- A2-Full Silence Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: A2-Lite silence processing — 10M frames.
#[test]
#[ignore]
fn test_a2_lite_silence_soak() {
    let mut model = build_soak_a2::<3>(0.01);
    model.prewarm();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(v.is_finite(), "A2-Lite NaN/Inf after {} frames", processed);
            assert!(
                (-2.0..=2.0).contains(&v),
                "A2-Lite diverged: {} after {} frames",
                v,
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();
    println!("--- A2-Lite Silence Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: A2-Full white noise processing — 10M frames.
#[test]
#[ignore]
fn test_a2_full_noise_soak() {
    let mut model = build_soak_a2::<8>(0.01);
    model.prewarm();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(42);
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "A2-Full NaN/Inf under noise after {} frames",
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();
    println!("--- A2-Full Noise Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: A2-Lite white noise processing — 10M frames.
#[test]
#[ignore]
fn test_a2_lite_noise_soak() {
    let mut model = build_soak_a2::<3>(0.01);
    model.prewarm();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(1337);
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "A2-Lite NaN/Inf under noise after {} frames",
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();
    println!("--- A2-Lite Noise Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}
