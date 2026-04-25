// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Interface de Linha de Comando (CLI) para o NAM-rs.
//!
//! Lida com a exibição de ajuda e a interpretação dos argumentos
//! fornecidos pelo usuário via terminal.

use lexopt::prelude::*;
use nam_rs::colors::Colorize;
use std::path::PathBuf;

/// Imprime as instruções de uso e ajuda no terminal.
pub fn print_help() {
    println!("{}", "🎸 NAM-rs Standalone".bright_green().bold());
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

/// Faz o parsing dos argumentos de linha de comando.
///
/// Retorna uma tupla contendo:
/// - Opcional caminho para o modelo (`PathBuf`)
/// - Ganho de entrada em dB (`f32`)
/// - Ganho de saída em dB (`f32`)
/// - Tamanho de buffer desejado (`u32`)
pub fn parse_args() -> Result<(Option<PathBuf>, f32, f32, u32), String> {
    // Valores padrão e inicialização do estado capturado da linha de comando.
    let mut model_path = None;
    let mut input_gain = 0.0;
    let mut output_gain = 0.0;
    let mut buffer_size = 256;
    let mut has_args = false;

    // Inicializa o parser lexopt a partir dos argumentos fornecidos pelo sistema operacional.
    let mut parser = lexopt::Parser::from_env();

    // Loop de iteração sobre os argumentos fornecidos.
    loop {
        match parser.next() {
            Ok(Some(arg)) => {
                has_args = true;
                match arg {
                    // Exibe a tela de ajuda estilizada e encerra o processo com sucesso.
                    Short('h') | Long("help") => {
                        print_help();
                        std::process::exit(0);
                    }
                    // Captura o caminho do modelo. Suporta expansão manual de '~'.
                    Short('m') | Long("model") => {
                        let val = parser.value().map_err(|e| e.to_string())?;
                        let p_str = val.into_string().map_err(|_| {
                            "Caminho de modelo inválido (caracteres não-UTF8).".to_string()
                        })?;

                        // Implementação simplificada de tilde expansion para Linux/macOS.
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
                    // Configuração de ganho de entrada (dB).
                    Short('i') | Long("input-gain") => {
                        let val = parser.value().map_err(|e| e.to_string())?;
                        let val_str = val
                            .into_string()
                            .map_err(|_| "Valor de ganho de entrada inválido.".to_string())?;
                        input_gain = val_str.parse::<f32>().map_err(|_| format!("Ganho de entrada inválido: '{}'. Deve ser um número em dB (ex: 3.5, -12).", val_str))?;
                    }
                    // Configuração de ganho de saída (dB).
                    Short('o') | Long("output-gain") => {
                        let val = parser.value().map_err(|e| e.to_string())?;
                        let val_str = val
                            .into_string()
                            .map_err(|_| "Valor de ganho de saída inválido.".to_string())?;
                        output_gain = val_str.parse::<f32>().map_err(|_| format!("Ganho de saída inválido: '{}'. Deve ser um número em dB (ex: 3.5, -12).", val_str))?;
                    }
                    // Ajuste do tamanho do buffer do DSP (em samples).
                    Short('b') | Long("buffer-size") => {
                        let val = parser.value().map_err(|e| e.to_string())?;
                        let val_str = val
                            .into_string()
                            .map_err(|_| "Valor de tamanho do buffer inválido.".to_string())?;
                        buffer_size = val_str.parse::<u32>().map_err(|_| format!("Tamanho do buffer inválido: '{}'. Deve ser um número inteiro (ex: 256).", val_str))?;
                    }
                    // Retorna erro caso encontre uma opção desconhecida.
                    _ => return Err(arg.unexpected().to_string()),
                }
            }
            // Fim dos argumentos.
            Ok(None) => break,
            // Erro genérico no parsing (ex: opção sem valor esperado).
            Err(e) => return Err(e.to_string()),
        }
    }

    // Comportamento amigável: se rodar sem nada, mostra a ajuda em vez de erro.
    if !has_args {
        print_help();
        std::process::exit(0);
    }

    Ok((model_path, input_gain, output_gain, buffer_size))
}
