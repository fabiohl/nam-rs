// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Interface de Linha de Comando (CLI) para o NAM-rs.
//!
//! Lida com a exibição de ajuda e a interpretação dos argumentos
//! fornecidos pelo usuário via terminal.

use crate::diagnostics::SystemSnapshot;
use crate::math::fastmath::{GAIN_MAX_DB, GAIN_MIN_DB, get_gain_lut};

use crate::spsc::{ParamPayload, SHUTDOWN};
use crate::standalone::colors::Colorize;
use lexopt::prelude::*;

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

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
    println!("  -I, --interactive       Ativa o modo interativo (ajuste de ganhos em tempo real)");
    println!("  -h, --help              Mostra esta ajuda e sai");
}

/// Faz o parsing dos argumentos de linha de comando.
///
/// Retorna uma tupla contendo:
/// - Opcional caminho para o modelo (`PathBuf`)
/// - Ganho de entrada em dB (`f32`)
/// - Ganho de saída em dB (`f32`)
/// - Tamanho de buffer desejado (`u32`)
/// - Se o modo interativo deve ser ativado (`bool`)
pub fn parse_args() -> Result<(Option<PathBuf>, f32, f32, u32, bool), String> {
    // Valores padrão e inicialização do estado capturado da linha de comando.
    let mut model_path = None;
    let mut input_gain = 0.0;
    let mut output_gain = 0.0;
    let mut buffer_size = 256;
    let mut interactive = false;
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

                        if !(GAIN_MIN_DB..=GAIN_MAX_DB).contains(&input_gain) {
                            return Err(format!(
                                "Ganho de entrada ({:.1} dB) fora do intervalo permitido [{:.1}, {:.1}] dB.",
                                input_gain, GAIN_MIN_DB, GAIN_MAX_DB
                            ));
                        }
                    }
                    // Configuração de ganho de saída (dB).
                    Short('o') | Long("output-gain") => {
                        let val = parser.value().map_err(|e| e.to_string())?;
                        let val_str = val
                            .into_string()
                            .map_err(|_| "Valor de ganho de saída inválido.".to_string())?;
                        output_gain = val_str.parse::<f32>().map_err(|_| format!("Ganho de saída inválido: '{}'. Deve ser um número em dB (ex: 3.5, -12).", val_str))?;

                        if !(GAIN_MIN_DB..=GAIN_MAX_DB).contains(&output_gain) {
                            return Err(format!(
                                "Ganho de saída ({:.1} dB) fora do intervalo permitido [{:.1}, {:.1}] dB.",
                                output_gain, GAIN_MIN_DB, GAIN_MAX_DB
                            ));
                        }
                    }
                    // Ativa o modo interativo (CLI).
                    Short('I') | Long("interactive") => {
                        interactive = true;
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

    Ok((
        model_path,
        input_gain,
        output_gain,
        buffer_size,
        interactive,
    ))
}

/// Loop interativo de comandos (TUI simplificada).
///
/// Permite ao usuário ajustar parâmetros em tempo real sem reiniciar o processo.
/// Comandos suportados: `gain <db>`, `out <db>`, `load <path>`, `help`, `exit`.
pub fn cli_loop(
    mut producer: rtrb::Producer<ParamPayload>,
    sys: SystemSnapshot,
) -> anyhow::Result<()> {
    println!(
        "\n{} {}",
        "🚀".bright_green(),
        "Modo Interativo Ativo. Digite 'help' para comandos.".bright_cyan()
    );

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        if SHUTDOWN.load(Ordering::Relaxed) {
            break;
        }

        print!("{} ", "nam-rs>".yellow().bold());
        let _ = io::stdout().flush();

        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            break; // EOF
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0].to_lowercase().as_str() {
            "help" => {
                println!("\n{}", "Comandos Interativos:".yellow().bold());
                println!("  gain <db>    Ajusta o ganho de entrada (ex: gain -3.5)");
                println!("  out <db>     Ajusta o ganho de saída (ex: out 6)");
                println!("  load <path>  Carrega um novo modelo (ex: load clean.namb)");
                println!("  status       Mostra o estado atual (placeholder)");
                println!("  exit, quit   Encerra o NAM-rs\n");
            }
            "exit" | "quit" => {
                SHUTDOWN.store(true, Ordering::SeqCst);
                break;
            }
            "gain" => {
                if let Some(db_str) = parts.get(1) {
                    if let Ok(db) = db_str.parse::<f32>() {
                        if (GAIN_MIN_DB..=GAIN_MAX_DB).contains(&db) {
                            let mult = get_gain_lut().db_to_linear(db);
                            let _ = producer.push(ParamPayload::InputGain(mult));
                            println!("{} Ganho de entrada: {:+.1} dB", "✅".green(), db);
                        } else {
                            println!("{} Ganho fora do intervalo permitido.", "❌".red());
                        }
                    } else {
                        println!("{} Valor inválido para ganho.", "❌".red());
                    }
                } else {
                    println!("{} Uso: gain <db> (ex: gain -3.5)", "💡".yellow());
                }
            }
            "out" => {
                if let Some(db_str) = parts.get(1) {
                    if let Ok(db) = db_str.parse::<f32>() {
                        if (GAIN_MIN_DB..=GAIN_MAX_DB).contains(&db) {
                            let mult = get_gain_lut().db_to_linear(db);
                            let _ = producer.push(ParamPayload::OutputGain(mult));
                            println!("{} Ganho de saída: {:+.1} dB", "✅".green(), db);
                        } else {
                            println!("{} Ganho fora do intervalo permitido.", "❌".red());
                        }
                    } else {
                        println!("{} Valor inválido para ganho.", "❌".red());
                    }
                } else {
                    println!("{} Uso: out <db> (ex: out 6)", "💡".yellow());
                }
            }
            "load" => {
                if let Some(path_str) = parts.get(1) {
                    let path = Path::new(path_str);
                    println!("{} Carregando: {} ...", "📂".cyan(), path_str);
                    match crate::loader::load_and_build_model(path, &sys) {
                        Ok(crate::loader::LoadedModelPair {
                            model_l,
                            model_r,
                            input_mult_adj,
                            output_mult_adj,
                            sample_rate,
                        }) => {
                            let _ = producer.push(ParamPayload::LoadModel {
                                model_l,
                                model_r,
                                input_mult_adj,
                                output_mult_adj,
                                sample_rate,
                            });
                            println!("{} Modelo carregado com sucesso.", "✅".green());
                        }
                        Err(e) => {
                            println!("{} Erro ao carregar modelo: {}", "❌".red(), e);
                        }
                    }
                } else {
                    println!("{} Uso: load <path> (ex: load clean.namb)", "💡".yellow());
                }
            }
            _ => {
                println!(
                    "{} Comando desconhecido: '{}'. Digite 'help'.",
                    "❓".red(),
                    parts[0]
                );
            }
        }
    }

    Ok(())
}
