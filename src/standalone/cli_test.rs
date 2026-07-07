// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_parse_args_diagnose() {
    let args = vec!["nam-rs", "--diagnose"];
    let parser = lexopt::Parser::from_iter(args);
    let cli_args = parse_args_from(parser);
    assert!(cli_args.diagnose);
    assert!(!cli_args.diagnose_full);
}

#[test]
fn test_parse_args_diagnose_full() {
    let args = vec!["nam-rs", "--diagnose-full"];
    let parser = lexopt::Parser::from_iter(args);
    let cli_args = parse_args_from(parser);
    assert!(!cli_args.diagnose);
    assert!(cli_args.diagnose_full);
}

#[test]
fn test_parse_args_model_and_gains() {
    let args = vec![
        "nam-rs",
        "-m",
        "my_model.nam",
        "-i",
        "6.0",
        "-o",
        "-3.5",
        "-b",
        "512",
    ];
    let parser = lexopt::Parser::from_iter(args);
    let cli_args = parse_args_from(parser);
    assert_eq!(cli_args.model_path, Some(PathBuf::from("my_model.nam")));
    assert_eq!(cli_args.input_gain, 6.0);
    assert_eq!(cli_args.output_gain, -3.5);
    assert_eq!(cli_args.buffer_size, 512);
    assert!(!cli_args.diagnose);
    assert!(!cli_args.diagnose_full);
}

#[test]
fn test_parse_args_activation_standard() {
    let args = vec!["nam-rs", "--activation", "standard"];
    let parser = lexopt::Parser::from_iter(args);
    let cli_args = parse_args_from(parser);
    assert_eq!(
        cli_args.activation as usize,
        ActivationPrecision::Standard as usize
    );
}

#[test]
fn test_parse_args_activation_hf() {
    let args = vec!["nam-rs", "--activation", "hf"];
    let parser = lexopt::Parser::from_iter(args);
    let cli_args = parse_args_from(parser);
    assert_eq!(
        cli_args.activation as usize,
        ActivationPrecision::HighFidelity as usize
    );
}

#[test]
fn test_parse_args_activation_highfidelity() {
    let args = vec!["nam-rs", "--activation", "highfidelity"];
    let parser = lexopt::Parser::from_iter(args);
    let cli_args = parse_args_from(parser);
    assert_eq!(
        cli_args.activation as usize,
        ActivationPrecision::HighFidelity as usize
    );
}

#[test]
fn test_parse_args_activation_default() {
    let args: Vec<&str> = vec!["nam-rs"];
    let parser = lexopt::Parser::from_iter(args);
    let cli_args = parse_args_from(parser);
    assert_eq!(
        cli_args.activation as usize,
        ActivationPrecision::Standard as usize
    );
}
