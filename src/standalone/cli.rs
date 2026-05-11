// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Funções auxiliares para interface de Linha de Comando (CLI).
//!
//! Lida com a exibição de ajuda e a interpretação dos argumentos
//! fornecidos pelo usuário via terminal.

use crate::math::fastmath::{GAIN_MAX_DB, GAIN_MIN_DB};

use crate::standalone::colors::Colorize;
use lexopt::prelude::*;

use std::path::PathBuf;

/// Imprime as instruções de uso e ajuda no terminal.
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
    println!("\n{}", "Uso:".yellow().bold());
    println!("  nam-rs [OPÇÕES]");
    println!("\n{}", "Opções:".yellow().bold());
    println!(
        "  -m, --model <ARQUIVO>   Caminho para o modelo (.nam ou .namb). Suporta ~, ../, etc."
    );
    println!("  -i, --input-gain <DB>   Ganho de entrada em dB (ex: -3.5, 12, 0) [padrão: 0]");
    println!("  -o, --output-gain <DB>  Ganho de saída em dB (ex: 5.0, -10) [padrão: 0]");
    println!(
        "  -b, --buffer-size <SAMPLES> Tamanho fixo do bloco (ex: 64, 256, 512). Use 0 para automático [padrão: 256]"
    );
    println!("  -h, --help              Mostra esta ajuda e sai");
}

/// Exibe uma mensagem de erro estilizada e encerra o processo com código 1.
fn exit_with_error(msg: impl std::fmt::Display) -> ! {
    eprintln!("{} {}", "❌ Erro no argumento:".red().bold(), msg);
    eprintln!("{}", "👉 Use '-h' para ver a tela de ajuda".yellow());
    std::process::exit(1);
}

/// Faz o parsing dos argumentos de linha de comando.
///
/// Retorna uma tupla contendo:
/// - Opcional caminho para o modelo (`PathBuf`)
/// - Ganho de entrada em dB (`f32`)
/// - Ganho de saída em dB (`f32`)
/// - Tamanho de buffer desejado (`u32`)
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
                    .unwrap_or_else(|_| exit_with_error("Caminho do modelo inválido (UTF-8)."));

                // Implementação simplificada de tilde expansion
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
                    .unwrap_or_else(|_| exit_with_error("Valor de ganho de entrada inválido."));
                input_gain = val_str.parse::<f32>().unwrap_or_else(|_| {
                    exit_with_error(format!(
                        "Ganho de entrada inválido: '{}'. Deve ser um número.",
                        val_str
                    ))
                });

                if !(GAIN_MIN_DB..=GAIN_MAX_DB).contains(&input_gain) {
                    exit_with_error(format!(
                        "Ganho de entrada ({:.1} dB) fora do intervalo [{:.1}, {:.1}].",
                        input_gain, GAIN_MIN_DB, GAIN_MAX_DB
                    ));
                }
            }
            Short('o') | Long("output-gain") => {
                let val = parser.value().unwrap_or_else(|e| exit_with_error(e));
                let val_str = val
                    .into_string()
                    .unwrap_or_else(|_| exit_with_error("Valor de ganho de saída inválido."));
                output_gain = val_str.parse::<f32>().unwrap_or_else(|_| {
                    exit_with_error(format!(
                        "Ganho de saída inválido: '{}'. Deve ser um número.",
                        val_str
                    ))
                });

                if !(GAIN_MIN_DB..=GAIN_MAX_DB).contains(&output_gain) {
                    exit_with_error(format!(
                        "Ganho de saída ({:.1} dB) fora do intervalo [{:.1}, {:.1}].",
                        output_gain, GAIN_MIN_DB, GAIN_MAX_DB
                    ));
                }
            }
            Short('b') | Long("buffer-size") => {
                let val = parser.value().unwrap_or_else(|e| exit_with_error(e));
                let val_str = val
                    .into_string()
                    .unwrap_or_else(|_| exit_with_error("Valor de tamanho do buffer inválido."));
                buffer_size = val_str.parse::<u32>().unwrap_or_else(|_| {
                    exit_with_error(format!(
                        "Tamanho do buffer inválido: '{}'. Deve ser um inteiro.",
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
