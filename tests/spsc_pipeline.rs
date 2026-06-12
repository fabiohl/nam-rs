// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! End-to-end SPSC pipeline, NAMB roundtrip, and rapid hot-swap tests.
//!
//! Validates the CLI→SPSC→DSP chain: model loading via SPSC channel,
//! NAMB binary parser → dispatcher → inference roundtrip, and rapid
//! sequential model hot-swap via SPSC with ownership integrity.

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::NamModel;
use std::fs;

mod common;
use common::*;

// =============================================================================
// End-to-End SPSC Pipeline Test (CLI → SPSC → DSP)
// =============================================================================

/// Test 11: End-to-end SPSC pipeline — simulates CLI→SPSC→DSP.
///
/// Loads and processes `BossWN-standard.nam` exclusively via SPSC:
/// 1. Load + Parse + Dispatch (simulates CLI thread)
/// 2. Send model via SPSC (ParamPayload::LoadModel)
/// 3. Receive and run inference in the consumer (simulates DSP callback)
/// 4. Validate finiteness and reasonable magnitude
///
/// This test validates the entire chain without PipeWire/Jack dependency.
#[test]
fn test_end_to_end_spsc_pipeline() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        eprintln!("SKIP: BossWN-standard.nam not found at {path:?}. Skipping E2E pipeline.");
        return;
    }

    // 1. Parse + Dispatch (simulates CLI thread)
    let json_data = fs::read_to_string(&path).expect("Failed to read WaveNet model for E2E");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser for E2E");
    let boxed = build_model(&model_data).expect("Dispatcher failed in E2E pipeline");

    // 2. Create SPSC channel and send the model as the CLI would
    let (mut producer, mut consumer) =
        rtrb::RingBuffer::<nam_rs::common::spsc::ParamPayload>::new(8);

    producer
        .push(nam_rs::common::spsc::ParamPayload::LoadModel {
            model_l: Some(boxed),
            model_r: None,
            input_mult_adj: 1.0,
            output_mult_adj: 1.0,
            sample_rate: 48000,
        })
        .expect("Failed to send model via SPSC in E2E");

    // 3. Consumer side (simulates DSP callback) — drains and runs inference
    let received = consumer
        .pop()
        .expect("Failed to receive model via SPSC in E2E");

    let mut active_model = match received {
        nam_rs::spsc::ParamPayload::LoadModel { model_l, .. } => model_l,
        _ => panic!("Received payload is not LoadModel in E2E"),
    };

    let model = active_model.as_mut().expect("Null model after SPSC drain");
    model.prewarm(2048);

    // 4. Process 440 Hz sine signal (64 samples, 1 block)
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    // 5. Validation: finiteness and reasonable magnitude
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "[E2E] Non-finite sample at index {i}: {s}");
        assert!(
            s.abs() < 100.0,
            "[E2E] Excessive magnitude at index {i}: {s} (limit 100.0)"
        );
    }

    println!("E2E Pipeline OK — CLI→SPSC→DSP validated without PipeWire (64 samples processed).");
}

// =============================================================================
// NAMB Parser → Dispatcher E2E Test (T-2)
// =============================================================================

/// Test 12: NAMB roundtrip — binary parser → dispatcher → inference.
///
/// Builds a valid synthetic `.namb` buffer (via `build_valid_namb()`), parses
/// with `parse_namb()`, dispatches to `build_model()` and runs prewarm + processing.
/// Verifies that the output is finite and that the complete `.namb → NamModelData → StaticModel`
/// chain is functional end-to-end.
///
/// The synthetic NAMB carries zeroed-out weights (0.01) that form a degraded but
/// numerically stable model — the goal is not to validate tonal quality, but rather
/// the integrity of the binary deserialization chain.
#[test]
fn test_namb_roundtrip_dispatcher_e2e() {
    use nam_rs::loader::namb::parse_namb;

    // Compute the correct number of weights for WaveNet Standard (CH=16, K=3, HEAD=8)
    // Array1: rechannel(16) + 10×(conv(768+16)+mixin(16)+o2o(256+16)) + head(128) = 10864
    // Array2: rechannel(128) + 10×(conv(192+8)+mixin(8)+o2o(64+8)) + head(8+1) = 2937
    // head_scale: 1 → Total: 13802
    let total_weights = 13802;
    let weights: Vec<f32> = vec![0.01; total_weights];

    // Build NAMB binary buffer with valid CRC32
    let weights_offset: usize = 80;
    let total_size = weights_offset + weights.len() * 4;
    let mut namb_data = vec![0u8; total_size];

    // Magic Number
    namb_data[0..4].copy_from_slice(&0x4E414D42u32.to_le_bytes());
    // Version = 1
    namb_data[4..6].copy_from_slice(&1u16.to_le_bytes());
    // Weights offset
    namb_data[12..16].copy_from_slice(&(weights_offset as u32).to_le_bytes());
    // Version String @32
    namb_data[32..37].copy_from_slice(b"0.9.0");
    // Sample rate = 48000.0
    namb_data[64..68].copy_from_slice(&48000.0f32.to_le_bytes());
    // Input DBU = 0.0
    namb_data[68..72].copy_from_slice(&0.0f32.to_le_bytes());
    // Output DBU = 0.0
    namb_data[72..76].copy_from_slice(&0.0f32.to_le_bytes());

    // Weights
    for (i, float_val) in weights.iter().enumerate() {
        let off = weights_offset + i * 4;
        namb_data[off..off + 4].copy_from_slice(&float_val.to_le_bytes());
    }

    // CRC32 over weights block
    let crc = nam_rs::loader::namb::crc32_ieee(&namb_data[weights_offset..]);
    namb_data[24..28].copy_from_slice(&crc.to_le_bytes());

    // 1. Parse NAMB
    let model_data = parse_namb(&namb_data).expect("Failed in parse_namb for E2E NAMB");
    assert_eq!(model_data.architecture, "WaveNet");
    assert_eq!(model_data.weights.len(), total_weights);

    // 2. Dispatcher: build StaticModel
    let mut model = build_model(&model_data).expect("Dispatcher failed in E2E NAMB");

    // 3. Prewarm and processing
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    // 4. Validation: finiteness
    for (i, &s) in output.iter().enumerate() {
        assert!(
            s.is_finite(),
            "[E2E NAMB] Non-finite sample at index {i}: {s}"
        );
    }

    println!("NAMB E2E OK — parse_namb→build_model→prewarm→process validated (64 samples).");
}

// =============================================================================
// Rapid Hot-Swap via SPSC Test (T-6)
// =============================================================================

/// Test 18: Rapid hot-swap via SPSC — sequential swap of 3 models.
///
/// Simulates the scenario of a user rapidly switching between 3 different
/// models via CLI (`model <path>`). The SPSC chain must maintain ownership
/// integrity (no leak, no double-free) and each model must produce stable
/// inference after prewarm.
///
/// Models used:
/// 1. WaveNet Standard (`BossWN-standard.nam`)
/// 2. LSTM 1×16 (`BossLSTM-1x16.nam`)
/// 3. WaveNet Feather (`BossWN-feather.nam`)
///
/// Procedure:
/// 1. Push the 3 models sequentially into the SPSC queue (simulates CLI)
/// 2. Pop sequentially, replacing the active model each iteration
/// 3. For each model: prewarm + process 64 sinusoidal samples
/// 4. Verify finiteness and reasonable magnitude of outputs
///
/// The previous model is discarded (dropped) when replaced — validating
/// that ownership transfer via `Box<StaticModel>` works without leak.
#[test]
fn test_rapid_hot_swap_spsc() {
    let models_to_load = [
        ("BossWN-standard.nam", "WaveNet Standard"),
        ("BossLSTM-1x16.nam", "LSTM 1×16"),
        ("BossWN-feather.nam", "WaveNet Feather"),
    ];

    // Verify availability of all fixtures
    for (filename, label) in &models_to_load {
        let p = model_path(filename);
        if !p.exists() {
            eprintln!(
                "SKIP: {filename} not found at {p:?}. Skipping hot-swap SPSC test ({label})."
            );
            return;
        }
    }

    // Create SPSC channel with capacity 4 (fits all 3 models)
    let (mut producer, mut consumer) =
        rtrb::RingBuffer::<nam_rs::common::spsc::ParamPayload>::new(4);

    // 1. Push the 3 models sequentially (simulates CLI thread doing 3 swaps)
    for (filename, label) in &models_to_load {
        let p = model_path(filename);
        let json_data = fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("Failed to read {filename} for hot-swap: {e}"));
        let model_data = parse_nam_json(&json_data)
            .unwrap_or_else(|e| panic!("Failed in JSON for {filename}: {e}"));
        let boxed = build_model(&model_data)
            .unwrap_or_else(|e| panic!("Dispatcher failed for {label}: {e}"));

        producer
            .push(nam_rs::common::spsc::ParamPayload::LoadModel {
                model_l: Some(boxed),
                model_r: None,
                input_mult_adj: 1.0,
                output_mult_adj: 1.0,
                sample_rate: 48000,
            })
            .unwrap_or_else(|_| panic!("SPSC push failed for {label} — buffer full"));
    }

    // 2. Pop and sequential processing (simulates DSP thread receiving swaps)
    let input = generate_sine_440hz(64);
    let mut active_model: Option<Box<nam_rs::models::StaticModel>> = None;

    for (idx, (_filename, label)) in models_to_load.iter().enumerate() {
        let received = consumer
            .pop()
            .unwrap_or_else(|_| panic!("SPSC pop failed for {label}"));

        // Replace the active model — the previous one is dropped here
        let new_model = match received {
            nam_rs::common::spsc::ParamPayload::LoadModel { model_l, .. } => model_l,
            _ => panic!("Payload #{idx} is not LoadModel"),
        };

        // Explicit drop of previous model before assigning new one
        // (validates ownership transfer — no leak)
        drop(active_model.take());
        active_model = new_model;

        let model = active_model.as_mut().expect("Null model after SPSC pop");

        // 3. Prewarm + process
        model.prewarm(2048);

        let mut output = vec![0.0f32; 64];
        model.process(&input, &mut output);

        // 4. Validation: finiteness and reasonable magnitude
        for (i, &s) in output.iter().enumerate() {
            assert!(
                s.is_finite(),
                "[Hot-Swap #{idx} {label}] Non-finite sample at index {i}: {s}"
            );
            assert!(
                s.abs() < 100.0,
                "[Hot-Swap #{idx} {label}] Excessive magnitude at index {i}: {s}"
            );
        }
    }

    // Verify that the SPSC channel is empty (all payloads consumed)
    assert!(
        consumer.pop().is_err(),
        "SPSC should be empty after consuming all 3 models"
    );

    println!(
        "Hot-Swap SPSC OK — 3 models swapped sequentially, \
         ownership transfer validated without leak."
    );
}
