// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use nam_rs::loader::nam_json::{get_wavenet_topology, parse_nam_json};
use proptest::prelude::*;
use std::fs;

// Fuzz 1: Arbitrary bytes lossy to string
proptest! {
    #![proptest_config(ProptestConfig::with_cases(5_000))]
    #[test]
    fn prop_fuzz_nam_json_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
        let json_str = String::from_utf8_lossy(&bytes);
        // Ensure it doesn't panic.
        let _ = parse_nam_json(&json_str);
    }
}

// Helper para Near Valid:
fn near_valid_json_strategy() -> impl Strategy<Value = String> {
    (
        any::<Option<String>>(),                     // version
        any::<String>(),                             // architecture
        prop::collection::vec(any::<f32>(), 0..100), // weights
        any::<bool>(),                               // random choice for valid/invalid structure
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
                // Misturar campos
                json["architecture"] = serde_json::Value::Number(123.into());
                json["config"] = serde_json::Value::Array(vec![]);
            }

            serde_json::to_string(&json).unwrap()
        })
}

// Fuzz 2: Near valid but randomized values
proptest! {
    #![proptest_config(ProptestConfig::with_cases(5_000))]
    #[test]
    fn prop_fuzz_nam_json_near_valid(json_str in near_valid_json_strategy()) {
        let _ = parse_nam_json(&json_str);
    }
}

// Fuzz 3: Truncated valid JSON
proptest! {
    #![proptest_config(ProptestConfig::with_cases(5_000))]
    #[test]
    fn prop_fuzz_nam_json_truncated(cut_idx in 0usize..400_000) {
        // Load fixture directly
        let fixture_content = fs::read_to_string("tests/nam_files/EVH-5150-Lite.nam")
            .unwrap_or_else(|_| "{}".to_string());

        let mut idx = cut_idx;
        if idx > fixture_content.len() {
            idx = fixture_content.len();
        }

        // We truncate bytes and convert lossy to avoid slicing inside a char boundary
        let bytes = fixture_content.as_bytes();
        let truncated = &bytes[0..idx];
        let json_str = String::from_utf8_lossy(truncated);

        let _ = parse_nam_json(&json_str);
    }
}

// Helper para Weight Overflow:
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
        // Build a perfectly valid WaveNet Lite JSON structure but with weird weights
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
    #![proptest_config(ProptestConfig::with_cases(5_000))]
    #[test]
    fn prop_fuzz_nam_json_weight_overflow(json_str in weight_overflow_strategy()) {
        if let Ok(parsed) = parse_nam_json(&json_str) {
            // Just ensure accessing topology doesn't crash
            let _topo = get_wavenet_topology(&parsed);
        }
    }
}
