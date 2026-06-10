// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use nam_rs::models::NamModel;

/// Reads a `.golden.bin` file in the specified binary format.
///
/// Returns `Some((input, expected_output))` or `None` if the file does not exist
/// or is malformed.
///
/// ## Format
/// ```text
/// [u32 num_samples LE]
/// [f32×N input samples LE]
/// [f32×N expected output LE]
/// ```
pub fn read_golden_bin(path: &Path) -> Option<(Vec<f32>, Vec<f32>)> {
    let data = fs::read(path).ok()?;

    if data.len() < 12 {
        eprintln!(
            "WARN: golden file {path:?} too small ({} bytes)",
            data.len()
        );
        return None;
    }

    let num_samples = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    let expected_size = 4 + num_samples * 4 * 2;
    if data.len() < expected_size {
        eprintln!(
            "WARN: golden {path:?} declares {num_samples} samples but has {} bytes (expected {expected_size})",
            data.len()
        );
        return None;
    }

    let input_start = 4;
    let output_start = 4 + num_samples * 4;

    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let offset = input_start + i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    let output: Vec<f32> = (0..num_samples)
        .map(|i| {
            let offset = output_start + i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    Some((input, output))
}

/// Writes a `.golden.bin` file with the standard binary format:
///
/// ```text
/// [u32 num_samples LE]
/// [f32×N input samples LE]
/// [f32×N expected output LE]
/// ```
pub fn write_golden_bin(path: &Path, input: &[f32], output: &[f32]) -> std::io::Result<()> {
    assert_eq!(
        input.len(),
        output.len(),
        "write_golden_bin: input and output must have same length"
    );
    let mut file = std::fs::File::create(path)?;
    let num_samples = input.len() as u32;
    file.write_all(&num_samples.to_le_bytes())?;
    for sample in input {
        file.write_all(&sample.to_le_bytes())?;
    }
    for sample in output {
        file.write_all(&sample.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

/// Resolves the path to a test model in `tests/fixtures/models/`.
pub fn model_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models");
    path.push(filename);
    path
}

/// Processes an input block through the model in chunks of `block_size`.
pub fn process_in_blocks(
    model: &mut nam_rs::models::StaticModel,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}
