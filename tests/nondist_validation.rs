// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

mod common;
use common::*;

use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::{WavenetTopologyResult, get_wavenet_topology, parse_nam_json};
use nam_rs::models::NamModel;
use std::fs;
use std::path::{Path, PathBuf};

/// Finds all `.nam` and `.json` files in the given directory recursively.
fn find_models_in_dir(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_models_in_dir(&path));
            } else if path
                .extension()
                .is_some_and(|ext| ext == "nam" || ext == "json")
            {
                files.push(path);
            }
        }
    }
    files
}

#[test]
fn test_nondist_models_validation() {
    let mut nondist_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    nondist_path.push("tests/fixtures/models-nondist");

    if !nondist_path.exists() {
        println!(
            "SKIP: Non-distributable models directory {:?} not found.",
            nondist_path
        );
        return;
    }

    let models = find_models_in_dir(&nondist_path);
    if models.is_empty() {
        println!("SKIP: No .nam or .json models found in {:?}", nondist_path);
        return;
    }

    println!(
        "Found {} non-distributable models to validate.",
        models.len()
    );

    for model_file in models {
        let filename = model_file
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        println!("Validating non-distributable model: {}", filename);

        // 1. Parsing & Loading
        let json_data = fs::read_to_string(&model_file)
            .unwrap_or_else(|e| panic!("Failed to read model file {}: {}", filename, e));
        let model_data = parse_nam_json(&json_data)
            .unwrap_or_else(|e| panic!("Failed to parse model JSON for {}: {}", filename, e));

        // Skip free-geometry WaveNet A1 models with >2 layer arrays — the dynamic
        // engine (T3.1) currently supports exactly 2 arrays. Models with N≠2 arrays
        // (edge geometry outside standard A1) are skipped gracefully.
        if model_data.architecture == "WaveNet"
            && !model_data.is_wavenet_a2()
            && matches!(
                get_wavenet_topology(&model_data),
                WavenetTopologyResult::Free(ref g) if g.num_arrays != 2
            )
        {
            println!("SKIP (free geometry, N≠2 arrays not yet supported): {filename}");
            continue;
        }

        // Let's verify that we can dispatch and build the model
        let mut model = build_model(&model_data)
            .unwrap_or_else(|e| panic!("Failed to dispatch/build model for {}: {}", filename, e));

        // Prewarm the model
        model.prewarm(2048);

        // 2. Determinism (Self-Consistency)
        let mut model_dup = build_model(&model_data)
            .expect("Failed to build duplicate model for determinism check");
        model_dup.prewarm(2048);

        let num_samples = 512;
        let input = generate_sine_440hz(num_samples);
        let mut out_a = vec![0.0f32; num_samples];
        let mut out_b = vec![0.0f32; num_samples];

        process_in_blocks(&mut model, &input, &mut out_a, 64);
        process_in_blocks(&mut model_dup, &input, &mut out_b, 64);

        let mse = compute_mse(&out_a, &out_b);
        assert_eq!(
            mse, 0.0,
            "Model {} is non-deterministic (MSE={:.6e})",
            filename, mse
        );

        // 3. Block Size Invariance.
        // Capped at 64 frames: that is the universal per-call processing contract
        // (`WAVENET_MAX_NUM_FRAMES`). A2 conv kernels (`conv1d_ch8`/`ch3`) use a fixed
        // 64-frame stack scratch, so feeding a single block > 64 is unsupported by A2
        // regardless of `set_max_buffer_size` (see audit finding in TODO-problemas.md).
        // WaveNet/LSTM re-chunk internally and are invariant for any size, so 1..=64
        // fully exercises streaming invariance across all architectures.
        let block_sizes = [1, 16, 32, 64];
        let mut ref_model = build_model(&model_data).unwrap();
        ref_model.prewarm(2048);
        let mut ref_output = vec![0.0f32; num_samples];
        process_in_blocks(&mut ref_model, &input, &mut ref_output, 1);

        for &bs in &block_sizes {
            let mut test_model = build_model(&model_data).unwrap();
            test_model.prewarm(2048);
            let mut test_output = vec![0.0f32; num_samples];
            process_in_blocks(&mut test_model, &input, &mut test_output, bs);

            let test_mse = compute_mse(&ref_output, &test_output);
            assert!(
                test_mse < 1e-7,
                "Block size invariance violated for {filename} at \
                 block_size={bs} (MSE={test_mse:.6e})"
            );
        }

        // 4. Stability / Finiteness Check
        for (i, &s) in out_a.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Model {} produced non-finite output at sample index {} ({})",
                filename,
                i,
                s
            );
        }

        // 5. Denormal/Subnormal Silence Test
        let silence = vec![0.0f32; num_samples];
        let mut silence_output = vec![0.0f32; num_samples];

        // Build a fresh model to avoid bias from the previous sine wave
        let mut silence_model = build_model(&model_data).unwrap();
        silence_model.prewarm(2048);
        process_in_blocks(&mut silence_model, &silence, &mut silence_output, 64);

        for (i, &s) in silence_output.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Model {} produced non-finite output under silence at index {} ({})",
                filename,
                i,
                s
            );
            assert!(
                s.abs() < 1.0,
                "Model {} output diverged under silence at index {} ({})",
                filename,
                i,
                s
            );
            assert!(
                s == 0.0 || s.is_normal(),
                "Model {} produced subnormal float value under silence at index {} (bits: 0x{:08X})",
                filename,
                i,
                s.to_bits()
            );
        }

        println!(
            "✓ Model {} validated successfully across all checks.",
            filename
        );
    }
}
