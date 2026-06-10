// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fabio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::fs;
use std::path::PathBuf;

fn fixture_path(filename: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/models");
    p.push(filename);
    p
}

#[test]
fn test_a2_full_fixture_loads() {
    let json = fs::read_to_string(fixture_path("wavenet_a2_full.nam")).unwrap();
    let data = nam_rs::loader::nam_json::parse_nam_json(&json)
        .expect("Failed to parse wavenet_a2_full.nam");

    assert_eq!(data.architecture, "WaveNet");
    assert_eq!(data.weights.len(), 12146);

    let ch =
        nam_rs::loader::nam_json::is_a2_shape(&data).expect("Should be recognized as A2 shape");
    assert_eq!(ch, 8);

    let _model =
        nam_rs::loader::dispatcher::build_model(&data).expect("Should dispatch to A2-Full");
}

#[test]
fn test_a2_lite_fixture_loads() {
    let json = fs::read_to_string(fixture_path("wavenet_a2_lite.nam")).unwrap();
    let data = nam_rs::loader::nam_json::parse_nam_json(&json)
        .expect("Failed to parse wavenet_a2_lite.nam");

    assert_eq!(data.architecture, "WaveNet");
    assert_eq!(data.weights.len(), 1871);

    let ch =
        nam_rs::loader::nam_json::is_a2_shape(&data).expect("Should be recognized as A2 shape");
    assert_eq!(ch, 3);

    let _model =
        nam_rs::loader::dispatcher::build_model(&data).expect("Should dispatch to A2-Lite");
}
