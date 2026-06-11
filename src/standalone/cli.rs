// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Helper functions for Command Line Interface (CLI).
//!
//! Handles the display of help and parsing of arguments
//! provided by the user via terminal.

use crate::math::constants::{GAIN_MAX_DB, GAIN_MIN_DB};

use crate::dsp::adaptive::SlimOverride;
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
    println!("  -c, --cab <FILE>        Path to the cab-sim IR (.wav). Supports ~, ../, etc.");
    println!("  -i, --input-gain <DB>   Input gain in dB (e.g. -3.5, 12, 0) [default: 0]");
    println!("  -i, --input-gain <DB>   Input gain in dB (e.g. -3.5, 12, 0) [default: 0]");
    println!("  -o, --output-gain <DB>  Output gain in dB (e.g. 5.0, -10) [default: 0]");
    println!(
        "  -b, --buffer-size <SAMPLES> Fixed block size (e.g. 64, 256, 512). Use 0 for auto [default: 256]"
    );
    println!("      --diagnose          Print technical support block and exit");
    println!("      --diagnose-full     Print technical support block with raw paths and exit");
    println!(
        "      --slim auto|full|lite  Force quality level (overrides adaptive FSM) [default: auto]"
    );
    println!("  -h, --help              Show this help message and exit");
}

/// Displays a styled error message and exits the process with code 1.
pub fn exit_with_error(msg: impl std::fmt::Display) -> ! {
    eprintln!("{} {}", "❌ Argument error:".red().bold(), msg);
    eprintln!("{}", "👉 Use '-h' to show the help screen".yellow());
    std::process::exit(1);
}

/// Parsed command line arguments.
#[derive(Debug, Clone)]
pub struct CliArgs {
    /// Path to the neural model file.
    pub model_path: Option<PathBuf>,
    /// Path to the cab-sim impulse response (.wav) file.
    pub cab_path: Option<PathBuf>,
    /// Input gain in dB.
    pub input_gain: f32,
    /// Output gain in dB.
    pub output_gain: f32,
    /// Desired buffer size.
    pub buffer_size: u32,
    /// Immediate diagnostic flag.
    pub diagnose: bool,
    /// Immediate diagnostic flag with full (unredacted) paths.
    pub diagnose_full: bool,
    /// Manual slim override quality level.
    pub slim_override: SlimOverride,
}

/// Parses command-line arguments.
pub fn parse_args() -> CliArgs {
    parse_args_from(lexopt::Parser::from_env())
}

/// Helper parsing function that accepts any lexopt::Parser to facilitate testing.
pub fn parse_args_from(mut parser: lexopt::Parser) -> CliArgs {
    let mut model_path = None;
    let mut cab_path = None;
    let mut input_gain = 0.0;
    let mut output_gain = 0.0;
    let mut buffer_size = 256;
    let mut diagnose = false;
    let mut diagnose_full = false;
    let mut slim_override = SlimOverride::Auto;
    let mut has_args = false;

    while let Some(arg) = parser.next().unwrap_or_else(|e| exit_with_error(e)) {
        has_args = true;
        match arg {
            Short('h') | Long("help") => {
                print_help();
                std::process::exit(0);
            }
            Long("diagnose") => {
                diagnose = true;
            }
            Long("diagnose-full") => {
                diagnose_full = true;
            }
            Long("slim") => {
                let val = parser.value().unwrap_or_else(|e| exit_with_error(e));
                let val_str = val
                    .into_string()
                    .unwrap_or_else(|_| exit_with_error("Invalid slim override value."));
                slim_override = match val_str.to_lowercase().as_str() {
                    "auto" => SlimOverride::Auto,
                    "full" => SlimOverride::ForceFull,
                    "lite" => SlimOverride::ForceLite,
                    other => exit_with_error(format!(
                        "Invalid slim override: '{}'. Expected 'auto', 'full', or 'lite'.",
                        other
                    )),
                };
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
            Short('c') | Long("cab") => {
                let val = parser.value().unwrap_or_else(|e| exit_with_error(e));
                let p_str = val
                    .into_string()
                    .unwrap_or_else(|_| exit_with_error("Invalid cab IR path (UTF-8)."));

                let expanded = if p_str.starts_with("~/") {
                    if let Ok(home) = std::env::var("HOME") {
                        p_str.replacen("~", &home, 1)
                    } else {
                        p_str
                    }
                } else {
                    p_str
                };
                cab_path = Some(PathBuf::from(expanded));
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

    CliArgs {
        model_path,
        cab_path,
        input_gain,
        output_gain,
        buffer_size,
        diagnose,
        diagnose_full,
        slim_override,
    }
}

#[cfg(test)]
mod tests {
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
}
