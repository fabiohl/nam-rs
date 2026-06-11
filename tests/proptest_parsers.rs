// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use nam_rs::loader::nam_json::{
    NamConfig, NamDate, NamLayerConfig, NamMetadata, NamModelData, WeightsLayout,
    get_wavenet_topology, parse_nam_json,
};
use nam_rs::loader::namb::{FLAG_HAS_CRC32, crc32_ieee, parse_namb};
use proptest::prelude::*;
use std::fs;

// Fuzz 1: Sends fully arbitrary bytes to the JSON parser.
// Ensures the parser handles data that is not valid UTF-8 without panicking.
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_nam_json_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        let json_str = String::from_utf8_lossy(&bytes);
        // The goal here is only to ensure absence of panics (memory safety).
        let _ = parse_nam_json(&json_str);
    }
}

/// Strategy: Generates a "near-valid" JSON.
/// Contains required fields but with mixed data types or corrupted structures.
/// Tests whether the parser correctly rejects semantically invalid data.
fn near_valid_json_strategy() -> impl Strategy<Value = String> {
    (
        any::<Option<String>>(),                     // version (optional)
        any::<String>(),                             // architecture (arbitrary string)
        prop::collection::vec(any::<f32>(), 0..100), // weights (arbitrary amount)
        any::<bool>(),                               // flag to decide whether to corrupt the JSON
    )
        .prop_map(|(ver, arch, weights, scramble)| {
            let mut json = serde_json::json!({
                "architecture": arch,
                "weights": weights,
                "config": {
                    "layers": [],
                    "head": null,
                    "head_scale": 0.02
                }
            });

            if let Some(v) = ver {
                json["version"] = serde_json::Value::String(v);
            }

            if scramble {
                // Forces type error: transforms string into number and object into array
                json["architecture"] = serde_json::Value::Number(123.into());
                json["config"] = serde_json::Value::Array(vec![]);
            }

            serde_json::to_string(&json).unwrap()
        })
}

// Fuzz 2: Near valid but randomized values
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_nam_json_near_valid(json_str in near_valid_json_strategy()) {
        let _ = parse_nam_json(&json_str);
    }
}

// Fuzz 3: Truncation of valid JSONs.
// Simulates an incomplete download or corrupted file on disk.
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_nam_json_truncated(cut_idx in 0usize..400_000) {
        // Uses a real fixture as the base for truncation
        let fixture_content = fs::read_to_string("tests/nam_files/EVH-5150-Lite.nam")
            .unwrap_or_else(|_| "{}".to_string());

        let mut idx = cut_idx;
        if idx > fixture_content.len() {
            idx = fixture_content.len();
        }

        // We truncate raw bytes and convert lossy to UTF-8
        // so as not to break multi-byte characters.
        let bytes = fixture_content.as_bytes();
        let truncated = &bytes[0..idx];
        let json_str = String::from_utf8_lossy(truncated);

        let _ = parse_nam_json(&json_str);
    }
}

/// Strategy: Generates JSONs with extreme or invalid numeric values (NaN/Inf).
/// Tests the system's resilience against mathematical instabilities originating from the weights file.
fn weight_overflow_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            Just(f32::MAX),
            Just(f32::MIN),
            Just(f32::INFINITY),
            Just(f32::NEG_INFINITY),
            Just(f32::NAN),
            Just(f32::MIN_POSITIVE),
            any::<f32>(),
        ],
        0..200
    ).prop_map(|weights| {
        // Builds a valid WaveNet topology, but populated with numeric "garbage".
        let json = serde_json::json!({
            "version": "0.5.4",
            "architecture": "WaveNet",
            "config": {
                "layers": [
                    {
                        "input_size": 1, "condition_size": 1, "head_size": 4,
                        "channels": 12, "kernel_size": 3, "dilations": [1,2,4,8,16,32,64],
                        "activation": "Tanh", "gated": false, "head_bias": false
                    },
                    {
                        "input_size": 1, "condition_size": 1, "head_size": 4,
                        "channels": 12, "kernel_size": 3, "dilations": [128,256,512,1,2,4,8,16,32,64,128,256,512],
                        "activation": "Tanh", "gated": false, "head_bias": true
                    }
                ],
                "head": null,
                "head_scale": 0.02
            },
            "weights": weights,
            "sample_rate": 48000
        });

        serde_json::to_string(&json).unwrap()
    })
}

// Fuzz 4: Weight overflow / weird f32s
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_nam_json_weight_overflow(json_str in weight_overflow_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            // Just ensure accessing topology doesn't crash
            let _topo = get_wavenet_topology(&parsed);
        }
    }
}

// Fuzz 5: Arbitrary bytes in the binary parser (NAMB).
// The NAMB parser must be extremely resilient to malformed binary files.
proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]
    #[test]
    #[ignore]
    fn prop_fuzz_namb_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..8192)) {
        let _ = parse_namb(&bytes);
    }
}

// Strategy: Generates a valid synthetic NAMB file.
// Allows corrupting specific fields (Magic, CRC, Offsets) in the tests below.
prop_compose! {
    fn valid_namb_strategy()(weights in prop::collection::vec(any::<f32>(), 0..200)) -> Vec<u8> {
        let weights_offset: usize = 80;
        let total_size = weights_offset + weights.len() * 4;
        let mut sim_data = vec![0u8; total_size];

        // Header: Magic 'NAMB'
        sim_data[0..4].copy_from_slice(&0x4E414D42u32.to_le_bytes());
        // Version: 1
        sim_data[4..6].copy_from_slice(&1u16.to_le_bytes());
        // Weights offset
        sim_data[12..16].copy_from_slice(&(weights_offset as u32).to_le_bytes());
        // Short metadata
        sim_data[32..37].copy_from_slice(b"1.0.0");
        // DSP Params
        sim_data[64..68].copy_from_slice(&48000.0f32.to_le_bytes());
        sim_data[68..72].copy_from_slice(&12.0f32.to_le_bytes());
        sim_data[72..76].copy_from_slice(&(-6.0f32).to_le_bytes());

        // Write weights in Little Endian
        for (i, float_val) in weights.iter().enumerate() {
            let off = weights_offset + i * 4;
            sim_data[off..off + 4].copy_from_slice(&float_val.to_le_bytes());
        }

        // Compute actual CRC32 IEEE to ensure the file starts valid
        let crc = nam_rs::loader::namb::crc32_ieee(&sim_data[weights_offset..]);
        sim_data[24..28].copy_from_slice(&crc.to_le_bytes());

        sim_data
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(5_000)
    })]

    // Tests rejection of files with invalid 'Magic Bytes'.
    #[test]
    #[ignore]
    fn prop_fuzz_namb_bad_magic(mut namb in valid_namb_strategy(), bad_magic in any::<u32>()) {
        prop_assume!(bad_magic != 0x4E414D42);
        let bad_bytes = bad_magic.to_le_bytes();
        prop_assume!(&bad_bytes != b"NAMB" && &bad_bytes != b"BMAN");
        namb[0..4].copy_from_slice(&bad_bytes);
        assert!(parse_namb(&namb).is_err(), "Parser accepted invalid Magic Byte!");
    }

    // Tests integrity via CRC. Flipping 1 bit should cause a parsing error.
    #[test]
    #[ignore]
    fn prop_fuzz_namb_bad_crc(mut namb in valid_namb_strategy(), bit_flip in 0..32usize) {
        let byte_idx = 24 + (bit_flip / 8);
        let bit_idx = bit_flip % 8;
        namb[byte_idx] ^= 1 << bit_idx;
        assert!(parse_namb(&namb).is_err(), "Parser accepted file with corrupted CRC!");
    }

    // Tests resilience to prematurely truncated files.
    #[test]
    #[ignore]
    fn prop_fuzz_namb_truncated(namb in valid_namb_strategy(), truncate_idx in any::<usize>()) {
        let idx = std::cmp::min(truncate_idx, namb.len().saturating_sub(1));
        let truncated = &namb[0..idx];
        assert!(parse_namb(truncated).is_err(), "Parser accepted truncated binary file!");
    }

    // Tests out-of-bounds offset attacks (Oversized Offset).
    #[test]
    #[ignore]
    fn prop_fuzz_namb_oversized_offset(mut namb in valid_namb_strategy(), offset_add in 1..10000u32) {
        let new_offset = namb.len() as u32 + offset_add;
        namb[12..16].copy_from_slice(&new_offset.to_le_bytes());
        assert!(parse_namb(&namb).is_err(), "Parser accepted out-of-bounds weights offset!");
    }
}

// ---------------------------------------------------------------------------
// S13.T02 — Shrinking strategy for NamModelData (via strategy functions)
// ---------------------------------------------------------------------------

/// Strategy for `NamLayerConfig` with shrinking.
fn arbitrary_layer_config() -> impl Strategy<Value = NamLayerConfig> {
    let channels_s = prop_oneof![
        Just(8usize),
        Just(12),
        Just(16),
        Just(20),
        Just(24),
        any::<usize>(),
    ];
    let kernel_s = prop_oneof![Just(1usize), Just(2), Just(3), any::<usize>()];
    let dilation_s = prop_oneof![
        Just(1usize),
        Just(2),
        Just(4),
        Just(8),
        Just(16),
        Just(32),
        Just(64),
        Just(128),
        Just(256),
        Just(512),
        any::<usize>(),
    ];
    let activation_s = prop_oneof![
        Just("Tanh".to_string()),
        Just("ReLU".to_string()),
        any::<String>()
    ];

    (
        any::<Option<usize>>(),
        any::<Option<usize>>(),
        any::<Option<usize>>(),
        channels_s,
        kernel_s,
        prop::collection::vec(dilation_s, 1..20),
        activation_s,
        any::<bool>(),
        any::<bool>(),
    )
        .prop_map(
            |(
                input_size,
                condition_size,
                head_size,
                channels,
                kernel_size,
                dilations,
                activation,
                gated,
                head_bias,
            )| NamLayerConfig {
                input_size,
                condition_size,
                head_size,
                channels: Some(channels),
                kernel_size: Some(kernel_size),
                dilations: Some(dilations),
                activation: Some(activation),
                gated: Some(gated),
                head_bias: Some(head_bias),
            },
        )
}

/// Strategy for `NamConfig` with shrinking.
fn arbitrary_nam_config() -> impl Strategy<Value = NamConfig> {
    (
        prop::collection::vec(arbitrary_layer_config(), 1..6),
        any::<Option<Option<String>>>(),
        prop_oneof![Just(Some(0.02f32)), any::<Option<f32>>()],
        any::<Option<usize>>(),
        any::<Option<usize>>(),
    )
        .prop_map(
            |(layers, head, head_scale, num_layers, hidden_size)| NamConfig {
                layers,
                head,
                head_scale,
                num_layers,
                hidden_size,
                receptive_field: None,
                bias: None,
                submodels: None,
            },
        )
}

/// Strategy for `NamDate` with shrinking.
fn arbitrary_nam_date() -> impl Strategy<Value = NamDate> {
    (
        any::<Option<i32>>(),
        any::<Option<i32>>(),
        any::<Option<i32>>(),
        any::<Option<i32>>(),
        any::<Option<i32>>(),
        any::<Option<i32>>(),
    )
        .prop_map(|(year, month, day, hour, minute, second)| NamDate {
            year,
            month,
            day,
            hour,
            minute,
            second,
        })
}

/// Strategy for `NamMetadata` with shrinking.
fn arbitrary_nam_metadata() -> impl Strategy<Value = NamMetadata> {
    (
        arbitrary_nam_date().prop_map(Some),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<String>>(),
        any::<Option<bool>>(), // training presence flag
        any::<Option<f32>>(),
        any::<Option<f32>>(),
        any::<Option<f32>>(),
    )
        .prop_map(
            |(
                date,
                name,
                modeled_by,
                gear_make,
                gear_model,
                gear_type,
                tone_type,
                training_has,
                input_level_dbu,
                output_level_dbu,
                loudness,
            )| {
                let training = training_has.and_then(|has| {
                    if has {
                        Some(serde_json::json!({"epochs": 100, "lr": 0.001}))
                    } else {
                        None
                    }
                });
                NamMetadata {
                    date,
                    name,
                    modeled_by,
                    gear_make,
                    gear_model,
                    gear_type,
                    tone_type,
                    training,
                    input_level_dbu,
                    output_level_dbu,
                    loudness,
                }
            },
        )
}

/// Shrinking strategy for `NamModelData`.
///
/// Generates synthetic models (WaveNet or LSTM) with random weights, metadata,
/// and configuration. Proptest shrinking automatically reduces the model to the
/// smallest counter-example when an assertion fails.
pub fn arbitrary_nam_model_data() -> impl Strategy<Value = NamModelData> {
    let arch = prop_oneof![Just("WaveNet".to_string()), Just("LSTM".to_string()),];

    let layout = prop_oneof![
        Just(WeightsLayout::Original),
        Just(WeightsLayout::GateMajorLstm),
        Just(WeightsLayout::Interleaved4WaveNet),
    ];

    (
        any::<Option<String>>(),
        arch,
        arbitrary_nam_config(),
        prop::collection::vec(any::<f32>(), 0..500),
        any::<Option<f32>>(),
        arbitrary_nam_metadata().prop_map(Some),
        layout,
    )
        .prop_map(
            |(version, architecture, config, weights, sample_rate, metadata, weights_layout)| {
                NamModelData {
                    version,
                    architecture,
                    config,
                    weights,
                    sample_rate,
                    metadata,
                    weights_layout,
                }
            },
        )
}

// ---------------------------------------------------------------------------
// S13.T02 — 100k iterations: valid NAMB header + random body
// ---------------------------------------------------------------------------

/// Generates a byte-array with a syntactically valid NAMB v2 header
/// (magic, version, flags, offset, crc) followed by a completely random
/// weights body.
fn arbitrary_namb_bytes_strategy() -> impl Strategy<Value = Vec<u8>> {
    const HEADER_SIZE: usize = 80;

    (
        any::<u32>(),      // sample_rate (reinterpreted as bytes)
        any::<u32>(),      // input_level_dbu
        any::<u32>(),      // output_level_dbu
        any::<[u8; 32]>(), // version_str
        any::<u8>(),       // layout_type
        prop::collection::vec(any::<u8>(), 0..16384),
    )
        .prop_map(
            |(
                sample_rate_raw,
                input_level_raw,
                output_level_raw,
                version_str,
                layout_type,
                body,
            )| {
                let offset = HEADER_SIZE;
                let total_len = offset + body.len();
                let mut data = vec![0u8; total_len];

                // Magic 'NAMB'
                data[0..4].copy_from_slice(&0x4E414D42u32.to_le_bytes());
                // Version 2
                data[4..6].copy_from_slice(&2u16.to_le_bytes());
                // Layout type (0=Original, 1=GateMajorLstm, 2=Interleaved4WaveNet)
                data[6] = layout_type % 3;
                // Flags: FLAG_HAS_CRC32 set
                data[7] = FLAG_HAS_CRC32;
                // Weights offset = 80 (header only, no JSON)
                data[12..16].copy_from_slice(&(offset as u32).to_le_bytes());
                // CRC32 computed from the body
                let crc = crc32_ieee(&body);
                data[24..28].copy_from_slice(&crc.to_le_bytes());
                // Version string
                data[32..64].copy_from_slice(&version_str);
                // Sample rate
                data[64..68].copy_from_slice(&sample_rate_raw.to_le_bytes());
                // Input level dBu
                data[68..72].copy_from_slice(&input_level_raw.to_le_bytes());
                // Output level dBu
                data[72..76].copy_from_slice(&output_level_raw.to_le_bytes());

                // Copy body into weights section
                data[offset..].copy_from_slice(&body);

                data
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(100_000)
    })]

    /// Fuzz 6: Valid NAMB v2 header + completely random weights body.
    ///
    /// Ensures the parser never panics with 100k combinations of arbitrary
    /// body, even when the header is syntactically correct.
    /// — Triggered exclusively via `utils/tests-long.sh`.
    #[test]
    #[ignore]
    fn prop_fuzz_namb_arbitrary_valid_header(bytes in arbitrary_namb_bytes_strategy()) {
        let _ = parse_namb(&bytes);
    }
}
