// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use baseview::DropData;
use std::path::PathBuf;

#[test]
fn test_get_valid_model_file_none() {
    let data = DropData::None;
    assert_eq!(get_valid_model_file(&data), None);
}

#[test]
fn test_get_valid_model_file_invalid_extensions() {
    let files = vec![
        PathBuf::from("model.wav"),
        PathBuf::from("config.json"),
        PathBuf::from("readme.txt"),
    ];
    let data = DropData::Files(files);
    assert_eq!(get_valid_model_file(&data), None);
}

#[test]
fn test_get_valid_model_file_valid_nam() {
    let files = vec![PathBuf::from("my_amp_model.nam")];
    let data = DropData::Files(files);
    assert_eq!(
        get_valid_model_file(&data),
        Some(PathBuf::from("my_amp_model.nam"))
    );
}

#[test]
fn test_get_valid_model_file_valid_namb() {
    let files = vec![PathBuf::from("my_amp_model.namb")];
    let data = DropData::Files(files);
    assert_eq!(
        get_valid_model_file(&data),
        Some(PathBuf::from("my_amp_model.namb"))
    );
}

#[test]
fn test_get_valid_model_file_case_insensitive() {
    let files = vec![PathBuf::from("MY_AMP_MODEL.NAM")];
    let data = DropData::Files(files);
    assert_eq!(
        get_valid_model_file(&data),
        Some(PathBuf::from("MY_AMP_MODEL.NAM"))
    );

    let files_namb = vec![PathBuf::from("another_model.Namb")];
    let data_namb = DropData::Files(files_namb);
    assert_eq!(
        get_valid_model_file(&data_namb),
        Some(PathBuf::from("another_model.Namb"))
    );
}

#[test]
fn test_get_valid_model_file_multiple_mixed() {
    let files = vec![
        PathBuf::from("invalid.wav"),
        PathBuf::from("sweet_tone.nam"),
        PathBuf::from("other.namb"),
    ];
    let data = DropData::Files(files);
    // Should skip the first invalid file and return the first valid model file
    assert_eq!(
        get_valid_model_file(&data),
        Some(PathBuf::from("sweet_tone.nam"))
    );
}
