// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Helper functions for Command Line Interface (CLI).
//!
//! Handles the display of help and parsing of arguments
//! provided by the user via terminal.

use crate::math::constants::{GAIN_MAX_DB, GAIN_MIN_DB};

use crate::standalone::colors::Colorize;
use lexopt::prelude::*;

use std::path::PathBuf;

/// Prints usage instructions and help in the terminal.
pub fn print_help() {
    println!(
        "{}",
        format!(
            "🎸 NAM-rs Standalone v{} — Neural Amp Modeler",
            env!("CARGO_PKG_VERSION")
        )
        .bright_green()
        .bold()
    );
    println!("\n{}", "Usage:".yellow().bold());
    println!("  nam-rs [OPTIONS]");
    println!("\n{}", "Options:".yellow().bold());
    println!("  -m, --model <FILE>      Path to the model (.nam or .namb). Supports ~, ../, etc.");
    println!("  -i, --input-gain <DB>   Input gain in dB (e.g. -3.5, 12, 0) [default: 0]");
    println!("  -o, --output-gain <DB>  Output gain in dB (e.g. 5.0, -10) [default: 0]");
    println!(
        "  -b, --buffer-size <SAMPLES> Fixed block size (e.g. 64, 256, 512). Use 0 for auto [default: 256]"
    );
    println!("  -h, --help              Show this help message and exit");
}

/// Displays a styled error message and exits the process with code 1.
pub fn exit_with_error(msg: impl std::fmt::Display) -> ! {
    eprintln!("{} {}", "❌ Argument error:".red().bold(), msg);
    eprintln!("{}", "👉 Use '-h' to show the help screen".yellow());
    std::process::exit(1);
}

/// Parses command-line arguments.
///
/// Returns a tuple containing:
/// - Optional path to the model (`PathBuf`)
/// - Input gain in dB (`f32`)
/// - Output gain in dB (`f32`)
/// - Desired buffer size (`u32`)
pub fn parse_args() -> (Option<PathBuf>, f32, f32, u32) {
    let mut model_path = None;
    let mut input_gain = 0.0;
    let mut output_gain = 0.0;
    let mut buffer_size = 256;
    let mut has_args = false;

    let mut parser = lexopt::Parser::from_env();

    while let Some(arg) = parser.next().unwrap_or_else(|e| exit_with_error(e)) {
        has_args = true;
        match arg {
            Short('h') | Long("help") => {
                print_help();
                std::process::exit(0);
            }
            Short('m') | Long("model") => {
                let val = parser.value().unwrap_or_else(|e| exit_with_error(e));
                let p_str = val
                    .into_string()
                    .unwrap_or_else(|_| exit_with_error("Invalid model path (UTF-8)."));

                // Simplified tilde expansion implementation
                let expanded = if p_str.starts_with("~/") {
                    if let Ok(home) = std::env::var("HOME") {
                        p_str.replacen("~", &home, 1)
                    } else {
                        p_str
                    }
                } else {
                    p_str
                };
                model_path = Some(PathBuf::from(expanded));
            }
            Short('i') | Long("input-gain") => {
                let val = parser.value().unwrap_or_else(|e| exit_with_error(e));
                let val_str = val
                    .into_string()
                    .unwrap_or_else(|_| exit_with_error("Invalid input gain value."));
                input_gain = val_str.parse::<f32>().unwrap_or_else(|_| {
                    exit_with_error(format!(
                        "Invalid input gain: '{}'. Must be a number.",
                        val_str
                    ))
                });

                if !(GAIN_MIN_DB..=GAIN_MAX_DB).contains(&input_gain) {
                    exit_with_error(format!(
                        "Input gain ({:.1} dB) out of range [{:.1}, {:.1}].",
                        input_gain, GAIN_MIN_DB, GAIN_MAX_DB
                    ));
                }
            }
            Short('o') | Long("output-gain") => {
                let val = parser.value().unwrap_or_else(|e| exit_with_error(e));
                let val_str = val
                    .into_string()
                    .unwrap_or_else(|_| exit_with_error("Invalid output gain value."));
                output_gain = val_str.parse::<f32>().unwrap_or_else(|_| {
                    exit_with_error(format!(
                        "Invalid output gain: '{}'. Must be a number.",
                        val_str
                    ))
                });

                if !(GAIN_MIN_DB..=GAIN_MAX_DB).contains(&output_gain) {
                    exit_with_error(format!(
                        "Output gain ({:.1} dB) out of range [{:.1}, {:.1}].",
                        output_gain, GAIN_MIN_DB, GAIN_MAX_DB
                    ));
                }
            }
            Short('b') | Long("buffer-size") => {
                let val = parser.value().unwrap_or_else(|e| exit_with_error(e));
                let val_str = val
                    .into_string()
                    .unwrap_or_else(|_| exit_with_error("Invalid buffer size value."));
                buffer_size = val_str.parse::<u32>().unwrap_or_else(|_| {
                    exit_with_error(format!(
                        "Invalid buffer size: '{}'. Must be an integer.",
                        val_str
                    ))
                });
            }
            _ => exit_with_error(arg.unexpected()),
        }
    }

    if !has_args {
        print_help();
        std::process::exit(0);
    }

    (model_path, input_gain, output_gain, buffer_size)
}
