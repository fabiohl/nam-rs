// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use nam_rs::loader::nam_json::{
    NamConfig, NamDate, NamLayerConfig, NamMetadata, NamModelData, WeightsLayout,
    get_wavenet_topology, parse_nam_json,
};
use nam_rs::loader::namb::{FLAG_HAS_CRC32, crc32_ieee_update, parse_namb};
use proptest::prelude::*;
use std::fs;

use nam_rs::loader::nam_json::{
    MAX_CONVNET_CHANNELS, MAX_CONVNET_KERNEL_SIZE, MAX_DILATION, MAX_DILATIONS_PER_ARRAY,
    MAX_HEAD_SIZE, MAX_KERNEL_SIZE, MAX_RECEPTIVE_FIELD, MAX_TOTAL_STATE_FRAMES,
    MAX_WAVENET_ARRAYS, MAX_WAVENET_FREE_CHANNELS,
};

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
        let fixture_content = fs::read_to_string("tests/fixtures/models-nondist/EVH-5150-Lite.nam")
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
                ..Default::default()
            },
        )
}

/// Strategy for `NamConfig` head field.
/// Generates `None` (absent), `Some(Value::Null)` (null),
/// or `Some(Value::Object)` (head config with fields).
fn arbitrary_head_value() -> impl Strategy<Value = Option<serde_json::Value>> {
    prop::option::of(prop_oneof![
        Just(serde_json::Value::Null),
        (any::<u16>(), any::<bool>(), any::<u16>(), any::<u16>(),).prop_map(
            |(ch, bias, out_ch, ks)| {
                let mut map = serde_json::Map::new();
                map.insert(
                    "channels".into(),
                    serde_json::Value::Number((ch as u64).into()),
                );
                map.insert("bias".into(), serde_json::Value::Bool(bias));
                map.insert(
                    "out_channels".into(),
                    serde_json::Value::Number((out_ch as u64).into()),
                );
                map.insert(
                    "activation".into(),
                    serde_json::Value::String("Tanh".into()),
                );
                map.insert(
                    "kernel_size".into(),
                    serde_json::Value::Number((ks as u64).into()),
                );
                serde_json::Value::Object(map)
            }
        ),
    ])
}

/// Strategy for `NamConfig` with shrinking.
fn arbitrary_nam_config() -> impl Strategy<Value = NamConfig> {
    (
        prop::collection::vec(arbitrary_layer_config(), 1..6),
        arbitrary_head_value(),
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
                ..Default::default()
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

                // Compute CRC32 of the entire file excluding CRC32 field itself
                let crc = {
                    let mut crc_val = crc32_ieee_update(0xFFFFFFFFu32, &data[..24]);
                    crc_val = crc32_ieee_update(crc_val, &data[28..]);
                    crc_val ^ 0xFFFFFFFFu32
                };
                data[24..28].copy_from_slice(&crc.to_le_bytes());

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

// ---------------------------------------------------------------------------
// F2 — Adversarial dimension proptest: kernel_size, dilations, channels,
//      head_size, receptive_field — ensure Err, never abort/panic.
// ---------------------------------------------------------------------------

/// Strategy: generates a WaveNet model JSON with one or more adversarial
/// dimension fields (kernel_size, dilations count/value, head_size, array count).
fn adversarial_wavenet_json_strategy() -> impl Strategy<Value = String> {
    (
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
    )
        .prop_map(
            |(_ch_raw, k_raw, d_cnt_raw, d_val_raw, head_raw, arrays_raw, _pattern)| {
                // Select an attack pattern based on _pattern
                let pattern = _pattern % 4;

                let (ch, k, dil_cnt, dil_val, head, arrays) = match pattern {
                    0 => (8, MAX_KERNEL_SIZE + 1 + (k_raw % 1024), 10, 2, 4, 2), // extreme kernel
                    1 => (
                        8,
                        3,
                        MAX_DILATIONS_PER_ARRAY + 1 + (d_cnt_raw % 64),
                        2,
                        4,
                        2,
                    ), // many dilations
                    2 => (8, 3, 10, MAX_DILATION + 1 + (d_val_raw % 1024), 4, 2), // extreme dilation value
                    3 => (8, 3, 10, 2, MAX_HEAD_SIZE + 1 + (head_raw % 256), 2), // extreme head_size
                    _ => (8, 3, 10, 2, 4, MAX_WAVENET_ARRAYS + 1 + (arrays_raw % 4)), // many arrays
                };

                let dil_vec: Vec<usize> = if dil_cnt == 0 {
                    vec![1]
                } else {
                    let cnt = dil_cnt.min(256);
                    vec![dil_val.min(usize::MAX - 1); cnt]
                };
                let k_val = k.min(usize::MAX / 4096);
                let head_val = head.min(usize::MAX / 4096);

                let mut layers = Vec::new();
                let n = arrays.min(16);
                for i in 0..n {
                    let layer = serde_json::json!({
                        "channels": ch.min(MAX_WAVENET_FREE_CHANNELS),
                        "kernel_size": k_val,
                        "dilations": dil_vec.clone(),
                        "head_size": head_val,
                        "activation": "Tanh",
                        "gated": false,
                        "head_bias": i == n - 1,
                    });
                    layers.push(layer);
                }

                let json = serde_json::json!({
                    "version": "0.5.4",
                    "architecture": "WaveNet",
                    "config": {
                        "layers": layers,
                        "head": null,
                        "head_scale": 0.02
                    },
                    "weights": vec![0.0f32; 16],
                    "sample_rate": 48000
                });
                serde_json::to_string(&json).unwrap()
            },
        )
}

/// Strategy: generates a ConvNet model JSON with adversarial channels/kernel/dilations.
fn adversarial_convnet_json_strategy() -> impl Strategy<Value = String> {
    (
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
        any::<usize>(),
    )
        .prop_map(|(ch_raw, k_raw, d_raw, pattern)| {
            let (ch, k, dil_val, dil_cnt) = match pattern % 3 {
                0 => (MAX_CONVNET_CHANNELS + 1 + (ch_raw % 64), 3, 2, 1), // extreme channels
                1 => (8, MAX_CONVNET_KERNEL_SIZE + 1 + (k_raw % 64), 2, 1), // extreme kernel
                _ => (
                    8,
                    3,
                    MAX_DILATION + 1 + (d_raw % 1024),
                    MAX_DILATIONS_PER_ARRAY + 1,
                ), // extreme dilation
            };

            let dil_vec: Vec<usize> = {
                let cnt = dil_cnt.min(64);
                vec![dil_val.min(usize::MAX / 4096); cnt]
            };

            let json = serde_json::json!({
                "version": "0.7.0",
                "architecture": "ConvNet",
                "config": {
                    "layers": [{
                        "channels": ch.min(usize::MAX / 4096),
                        "kernel_size": k.min(usize::MAX / 4096),
                        "dilations": dil_vec,
                        "activation": "Tanh",
                    }],
                    "head": null,
                    "head_scale": 0.02
                },
                "weights": vec![0.0f32; 16],
                "sample_rate": 48000
            });
            serde_json::to_string(&json).unwrap()
        })
}

/// Strategy: generates a Linear model JSON with adversarial receptive_field.
fn adversarial_linear_json_strategy() -> impl Strategy<Value = String> {
    any::<usize>().prop_map(|rf_raw| {
        let rf = MAX_RECEPTIVE_FIELD + 1 + (rf_raw % 1024);
        let weights: Vec<f32> = vec![0.0f32; 16.min(rf)];
        let json = serde_json::json!({
            "version": "0.5.4",
            "architecture": "Linear",
            "config": {
                "layers": [],
                "head": null,
                "receptive_field": rf.min(usize::MAX / 4096),
                "bias": false
            },
            "weights": weights,
            "sample_rate": 48000
        });
        serde_json::to_string(&json).unwrap()
    })
}

// ---------------------------------------------------------------------------
// F12 — Adversarial state budget. Generates models that combine near-max
//       kernel_size × dilation × channels × layer count to stress the new
//       MAX_TOTAL_STATE_FRAMES bound.
// ---------------------------------------------------------------------------

/// Strategy: generates a WaveNet model JSON with high aggregate state budget.
/// Targets the combination of kernel_size × dilation × channels across many
/// layers — the F12 DoS vector that F2 individual bounds alone don't close.
fn adversarial_state_budget_strategy() -> impl Strategy<Value = String> {
    (
        any::<usize>(), // pattern selector
        any::<u8>(),    // jitter for near-max values
    )
        .prop_map(|(pattern, jitter)| {
            let j = (jitter as usize) % 16;

            match pattern % 3 {
                // Case 0: just under the budget — should be Free (or Rejected if channels push over)
                0 => {
                    // Target ~60 Mi frames (under 64 Mi cap)
                    // 8 arrays × 4 dilations × (63 × 256 × 512) ≈ 63 × 256 × 512 × 32 = ~264M
                    // Too high. Let's use: 4 arrays, 8 dilations, k=16, d=1024, ch=128
                    // => 4 × 8 × 15 × 1024 × 128 = 62,914,560 ≈ 60 Mi (under 64 Mi)
                    let dil_vec: Vec<usize> = vec![1024usize.saturating_sub(j); 8];
                    let layers: Vec<serde_json::Value> = (0..4)
                        .map(|i| {
                            serde_json::json!({
                                "channels": 128,
                                "kernel_size": 16,
                                "dilations": dil_vec.clone(),
                                "head_size": 4,
                                "activation": "Tanh",
                                "gated": false,
                                "head_bias": i == 3,
                            })
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.5.4",
                        "architecture": "WaveNet",
                        "config": { "layers": layers, "head": null, "head_scale": 0.02 },
                        "weights": vec![0.0f32; 16],
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // Case 1: over the budget — should be Rejected
                1 => {
                    // Target ~100 Mi frames (over 64 Mi cap)
                    // 8 arrays × 4 dilations × (63 × 512 × 64) = 8 × 4 × 63 × 512 × 64 = ~262M
                    // Too high. Use: 8 arrays, 4 dilations, high k, d, ch combo
                    let cnt = (4 + j / 4).min(16);
                    let dil_val = 2048usize.saturating_add(j * 128).min(MAX_DILATION);
                    let dil_vec: Vec<usize> = vec![dil_val; cnt];
                    let arrays = (3 + j / 8).min(8);
                    let layers: Vec<serde_json::Value> = (0..arrays)
                        .map(|i| {
                            serde_json::json!({
                                "channels": 128usize.saturating_add(j * 8).min(MAX_WAVENET_FREE_CHANNELS),
                                "kernel_size": (32usize + j).min(MAX_KERNEL_SIZE),
                                "dilations": dil_vec.clone(),
                                "head_size": 4,
                                "activation": "Tanh",
                                "gated": false,
                                "head_bias": i == arrays - 1,
                            })
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.5.4",
                        "architecture": "WaveNet",
                        "config": { "layers": layers, "head": null, "head_scale": 0.02 },
                        "weights": vec![0.0f32; 16],
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
                // Case 2: channels=1 amplification — tiny weight file exploiting receptive field
                _ => {
                    // Worst case from F12 description: k=64, d=4096, ch=1, 8 arrays × 64 dilations
                    // => 512 layers × ((63 * 4096) + 1600) × 1 = 512 × ~259,648 = ~133 Mi frames
                    let dil_cnt = (MAX_DILATIONS_PER_ARRAY / 2 + j / 2).min(MAX_DILATIONS_PER_ARRAY);
                    let dil_val = (MAX_DILATION / 2 + j * 128).min(MAX_DILATION);
                    let dil_vec: Vec<usize> = vec![dil_val; dil_cnt];
                    let arrays = 2usize.saturating_add(j / 4).min(MAX_WAVENET_ARRAYS);
                    let layers: Vec<serde_json::Value> = (0..arrays)
                        .map(|i| {
                            serde_json::json!({
                                "channels": 1,
                                "kernel_size": (32usize + j).min(MAX_KERNEL_SIZE),
                                "dilations": dil_vec.clone(),
                                "head_size": 4,
                                "activation": "Tanh",
                                "gated": false,
                                "head_bias": i == arrays - 1,
                            })
                        })
                        .collect();
                    serde_json::to_string(&serde_json::json!({
                        "version": "0.5.4",
                        "architecture": "WaveNet",
                        "config": { "layers": layers, "head": null, "head_scale": 0.02 },
                        "weights": vec![0.0f32; 16],
                        "sample_rate": 48000
                    }))
                    .unwrap()
                }
            }
        })
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(10_000)
    })]

    /// F2 — Adversarial dimensions: ensures topology detection rejects
    /// models with dimensions exceeding safe bounds (never abort/panic).
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_wavenet_dims(json_str in adversarial_wavenet_json_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let result = get_wavenet_topology(&parsed);
            // Must never match Known — adversarial dims should be Free (if valid)
            // or Rejected (if exceeding bounds).
            match result {
                nam_rs::loader::nam_json::WavenetTopologyResult::Known(_) => {
                    panic!("adversarial dimensions should not match a catalog SKU");
                }
                nam_rs::loader::nam_json::WavenetTopologyResult::Free(geom) => {
                    // If accepted as Free, dimensions must be within bounds
                    assert!(
                        geom.channels.iter().all(|&c| c <= MAX_WAVENET_FREE_CHANNELS),
                        "free geometry channels within bounds"
                    );
                    // F12: verify state budget — no Free model should exceed MAX_TOTAL_STATE_FRAMES
                    let total_state = geom
                        .dilations
                        .iter()
                        .zip(geom.kernel_sizes.iter())
                        .zip(geom.channels.iter())
                        .fold(0usize, |acc, ((dils, &k), &ch)| {
                            let rf = k.saturating_sub(1);
                            acc.wrapping_add(
                                dils
                                    .iter()
                                    .map(|&d| rf.saturating_mul(d).saturating_mul(ch))
                                    .sum::<usize>(),
                            )
                        });
                    assert!(
                        total_state <= MAX_TOTAL_STATE_FRAMES,
                        "Free geometry exceeded state budget: {total_state} > {MAX_TOTAL_STATE_FRAMES}"
                    );
                }
                nam_rs::loader::nam_json::WavenetTopologyResult::Rejected(_) => {
                    // Expected for adversarial dims — nothing to check
                }
            }
        }
    }

    /// F2 — Adversarial ConvNet dimensions.
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_convnet_dims(json_str in adversarial_convnet_json_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let topo = nam_rs::loader::nam_json::get_convnet_topology(&parsed);
            // Adversarial dimensions should be rejected (None)
            assert!(topo.is_none(),
                "ConvNet with adversarial dimensions should be rejected, got: {topo:?}");
        }
    }

    /// F2 — Adversarial Linear receptive_field.
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_linear_dims(json_str in adversarial_linear_json_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let topo = nam_rs::loader::nam_json::get_linear_topology(&parsed);
            // Adversarial receptive_field should be rejected (None)
            assert!(topo.is_none(),
                "Linear with adversarial receptive_field should be rejected, got: {topo:?}");
        }
    }

    /// F12 — Adversarial state budget: ensures models exceeding
    /// MAX_TOTAL_STATE_FRAMES are Rejected, while others may be Free or Rejected
    /// but never cause panic or incorrect acceptance.
    #[test]
    #[ignore]
    fn prop_fuzz_adversarial_state_budget(json_str in adversarial_state_budget_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            let result = get_wavenet_topology(&parsed);
            match result {
                nam_rs::loader::nam_json::WavenetTopologyResult::Known(_) => {
                    panic!("adversarial state budget should not match a catalog SKU");
                }
                nam_rs::loader::nam_json::WavenetTopologyResult::Free(geom) => {
                    // If accepted, must not exceed budget
                    let total_state = geom
                        .dilations
                        .iter()
                        .zip(geom.kernel_sizes.iter())
                        .zip(geom.channels.iter())
                        .fold(0usize, |acc, ((dils, &k), &ch)| {
                            let rf = k.saturating_sub(1);
                            acc.wrapping_add(
                                dils
                                    .iter()
                                    .map(|&d| rf.saturating_mul(d).saturating_mul(ch))
                                    .sum::<usize>(),
                            )
                        });
                    assert!(
                        total_state <= MAX_TOTAL_STATE_FRAMES,
                        "Free geometry exceeded state budget: {total_state} > {MAX_TOTAL_STATE_FRAMES}"
                    );
                }
                nam_rs::loader::nam_json::WavenetTopologyResult::Rejected(_) => {
                    // Rejection is acceptable — some combos exceed budget
                }
            }
        }
    }
}
