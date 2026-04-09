// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![warn(missing_docs)]

//! Ponto de entrada principal do NAM-rs.
//! Inicializa o host base de PipeWire para processamento de áudio
//! e coordena o shutdown gracioso quando CTRL+C é pressionado.
//!
//! # Regras de Arquitetura para Desenvolvedores
//! - **ZERO LOCKS** na thread DSP (módulo pw_host).
//! - **ZERO ALOCAÇÕES** na thread DSP (o loop `process()` deve ser zero-allocation).

use lexopt::prelude::*;
use nam_rs::{loader, pw_host, spsc, spsc::ParamPayload};
use std::path::PathBuf;
use std::sync::atomic::Ordering;

fn parse_args() -> Result<(Option<PathBuf>, f32, f32), lexopt::Error> {
    let mut model_path = None;
    let mut input_gain = 0.0;
    let mut output_gain = 0.0;

    let mut parser = lexopt::Parser::from_env();
    while let Some(arg) = parser.next()? {
        match arg {
            Short('m') | Long("model") => {
                model_path = Some(PathBuf::from(parser.value()?));
            }
            Short('i') | Long("input-gain") => {
                input_gain = parser.value()?.parse()?;
            }
            Short('o') | Long("output-gain") => {
                output_gain = parser.value()?.parse()?;
            }
            Long("help") => {
                println!("NAM-rs Standalone\n\nOptions:");
                println!("  -m, --model <PATH>      Model file (.nam ou .namb)");
                println!("  -i, --input-gain <DB>   Input gain in dB (float)");
                println!("  -o, --output-gain <DB>  Output gain in dB (float)");
                std::process::exit(0);
            }
            _ => return Err(arg.unexpected()),
        }
    }
    Ok((model_path, input_gain, output_gain))
}

fn load_and_send_model(path: &std::path::Path, producer: &mut rtrb::Producer<ParamPayload>) {
    let path_str = path.to_string_lossy();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let ext_lower = ext.to_lowercase();
    let result = if ext_lower == "namb" {
        std::fs::read(path)
            .map_err(anyhow::Error::from)
            .and_then(|bytes| loader::namb::parse_namb(&bytes))
    } else if ext_lower == "nam" {
        std::fs::read_to_string(path)
            .map_err(anyhow::Error::from)
            .and_then(|json| loader::nam_json::parse_nam_json(&json))
    } else {
        println!("[CLI] Erro: Extensão de arquivo desconhecida (deve ser .nam ou .namb)");
        return;
    };

    match result {
        Ok(model_data) => {
            let meta = model_data
                .metadata
                .clone()
                .unwrap_or(loader::nam_json::NamMetadata {
                    input_level_dbu: None,
                    output_level_dbu: None,
                    loudness: None,
                });
            let in_level = meta.input_level_dbu.unwrap_or(0.0);
            let loudness = meta.loudness.unwrap_or(-18.0);

            let input_db_adj = 12.0 - in_level;
            let output_db_adj = -18.0 - loudness;

            // Dispatcher: converte NamModelData → Box<DynamicModel> (thread CLI)
            let boxed_model = match loader::dispatcher::build_model(&model_data) {
                Ok(mut model) => {
                    // Prewarm na thread CLI — estabiliza estados internos antes do DSP
                    model.0.prewarm(2048);
                    println!("[CLI] Modelo prewarmado com 2048 amostras.");
                    Some(model)
                }
                Err(e) => {
                    println!("[CLI] Falha ao construir modelo: {}", e);
                    None
                }
            };

            if producer
                .push(ParamPayload::LoadModel {
                    model: boxed_model,
                    input_db_adj,
                    output_db_adj,
                })
                .is_ok()
            {
                println!(
                    "[CLI] Payload enviado. Modelo: {}, InputAdj: {:.2}dB, OutputAdj: {:.2}dB",
                    path_str, input_db_adj, output_db_adj
                );
            } else {
                println!("[CLI] Falha ao enviar LoadModel (buffer cheio).");
            }
        }
        Err(e) => {
            println!("[CLI] Erro ao carregar modelo {}: {}", path_str, e);
        }
    }
}

fn cli_loop(mut producer: rtrb::Producer<ParamPayload>) {
    println!("\n[CLI] Interface interativa iniciada. Digite 'help' para comandos.");

    let stdin = std::io::stdin();
    for line in stdin.lines() {
        let Ok(cmd) = line else { break };
        let cmd = cmd.trim();
        if cmd.is_empty() {
            continue;
        }

        let mut parts = cmd.split_whitespace();
        match parts.next().unwrap() {
            "quit" | "q" | "exit" => {
                spsc::SHUTDOWN.store(true, Ordering::SeqCst);
                println!("[CLI] Solicitando desligamento gracioso...");
                break;
            }
            "model" | "m" => {
                if let Some(p) = parts.next() {
                    let path = std::path::Path::new(p);
                    load_and_send_model(path, &mut producer);
                } else {
                    println!("[CLI] Uso: model <caminho>");
                }
            }
            "gain_in" | "gi" => {
                if let Some(val_str) = parts.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        if producer.push(ParamPayload::InputGain(val)).is_ok() {
                            println!("[CLI] Input Gain configurado para {:.2} dB.", val);
                        } else {
                            println!("[CLI] Falha ao enviar InputGain (buffer cheio).");
                        }
                    } else {
                        println!("[CLI] Valor de ganho inválido: {}", val_str);
                    }
                } else {
                    println!("[CLI] Uso: gain_in <valor_db>");
                }
            }
            "gain_out" | "go" => {
                if let Some(val_str) = parts.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        if producer.push(ParamPayload::OutputGain(val)).is_ok() {
                            println!("[CLI] Output Gain configurado para {:.2} dB.", val);
                        } else {
                            println!("[CLI] Falha ao enviar OutputGain (buffer cheio).");
                        }
                    } else {
                        println!("[CLI] Valor de ganho inválido: {}", val_str);
                    }
                } else {
                    println!("[CLI] Uso: gain_out <valor_db>");
                }
            }
            "help" | "h" => {
                println!("[CLI] Comandos disponíveis:");
                println!("  model <caminho>   : Carrega um arquivo de modelo (.nam ou .namb)");
                println!("  gain_in <db>      : Ajusta o ganho de entrada (ex: -3.0)");
                println!("  gain_out <db>     : Ajusta o ganho de saída (ex: 2.5)");
                println!("  quit              : Encerra o engine");
            }
            _ => {
                println!("[CLI] Comando desconhecido. Digite 'help'.");
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let (model_path, initial_in_gain, initial_out_gain) = parse_args()?;

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        println!("Engine NAM-rs: AVX2 + FMA ativado computando vetores de 256 bits.");
    } else {
        eprintln!(
            "AVISO CRÍTICO: AVX2 ou FMA não presentes nativamente (X86-64-V3 ausente).\nO rendimento DSP será castigado pelo fallback escalar."
        );
    }

    pipewire::init();

    ctrlc::set_handler(|| {
        if spsc::SHUTDOWN.load(Ordering::SeqCst) {
            std::process::exit(1);
        }
        spsc::SHUTDOWN.store(true, Ordering::SeqCst);
    })
    .expect("Erro ao configurar handler de Ctrl-C");

    let channels = spsc::setup_spsc(64);
    let mut producer = channels.param_producer;
    let consumer = channels.param_consumer;
    let gc_producer = channels.gc_producer;
    let mut gc_consumer = channels.gc_consumer;
    let resampler_producer = channels.resampler_producer;
    let resampler_consumer = channels.resampler_consumer;
    let rt_status = channels.rt_status;

    // 2. Thread GC para "Drop-Delegation" lock-free
    std::thread::spawn(move || {
        while !spsc::SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
            while let Ok(model) = gc_consumer.pop() {
                drop(model);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    if initial_in_gain != 0.0 {
        let _ = producer.push(ParamPayload::InputGain(initial_in_gain));
    }
    if initial_out_gain != 0.0 {
        let _ = producer.push(ParamPayload::OutputGain(initial_out_gain));
    }
    if let Some(path) = model_path {
        load_and_send_model(&path, &mut producer);
    }

    std::thread::spawn(move || {
        cli_loop(producer);
    });

    pw_host::run_pipewire_host(
        consumer,
        gc_producer,
        resampler_consumer,
        resampler_producer,
        rt_status,
    )?;

    unsafe {
        pipewire::deinit();
    }

    Ok(())
}
