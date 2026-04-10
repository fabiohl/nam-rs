// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![warn(missing_docs)]

//! Ponto de entrada principal do NAM-rs.
//!
//! Pense neste arquivo como a "recepção" do nosso estúdio virtual. Ele é responsável por:
//! 1. Ler o que o usuário digita no terminal (qual amplificador carregar e os volumes de entrada/saída).
//! 2. Abrir a conexão de áudio com o sistema (PipeWire), conectando a guitarra ao motor sonoro.
//! 3. Garantir que, quando o usuário apertar CTRL+C, tudo seja desligado com segurança, sem deixar ruídos.
//!
//! # Regras de Arquitetura para Desenvolvedores
//! - **ZERO LOCKS** na thread de Áudio (módulo `pw_host`): O áudio não "espera" pela interface visual. Se não houver instrução nova, ele continua usando a anterior. Evita "engasgos" no som.
//! - **ZERO ALOCAÇÕES** na thread de Áudio: A memória do canal de áudio (`process()`) é sempre preparada 100% de antemão. O áudio nunca "pede por mais memória RAM" de supetão.

use lexopt::prelude::*;
use nam_rs::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
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

/// Carrega um arquivo de modelo (.nam ou .namb) e o envia para o callback DSP via SPSC.
///
/// Este é o pipeline completo de carregamento, executado na thread CLI:
/// 1. Verifica existência e extensão do arquivo.
/// 2. Lê e parseia o formato (JSON via [`loader::nam_json`] ou binário via [`loader::namb`]).
/// 3. Constrói o modelo neural via [`loader::dispatcher::build_model`].
/// 4. Prewarm do modelo (2048 amostras) para estabilizar estados internos.
/// 5. Envia o payload completo (`LoadModel` + ajustes de ganho) pela fila SPSC.
///
/// Todos os erros emitem diagnósticos estruturados via [`NamDiagnostic`].
/// Mensagens informativas de sucesso usam `println!` mundano.
fn load_and_send_model(
    path: &std::path::Path,
    producer: &mut rtrb::Producer<ParamPayload>,
    sys: &SystemSnapshot,
) {
    let path_str = path.to_string_lossy();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Verificação de existência do arquivo
    if !path.exists() {
        NamDiagnostic::new(NamErrorCode::FileNotFound, sys)
            .message(format!(
                "Arquivo de modelo não encontrado: \"{}\"",
                path_str
            ))
            .hint("Verifique se o caminho está correto e o arquivo existe.")
            .param("file", &path_str)
            .emit();
        return;
    }

    let ext_lower = ext.to_lowercase();
    let result = if ext_lower == "namb" {
        std::fs::read(path).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Erro ao ler o arquivo \"{}\"", path_str))
                .hint("Verifique as permissões do arquivo e se o disco está acessível.")
                .param("file", &path_str)
                .param("io_error", &e)
                .emit();
            anyhow::Error::from(e)
        }).and_then(|bytes| {
            loader::namb::parse_namb(&bytes).inspect_err(|e| {
                // Mapeamento de erros específicos do parser NAMB
                let msg = e.to_string();
                let code = if msg.contains("muito pequeno") {
                    NamErrorCode::NambTruncated
                } else if msg.contains("mágica inválida") || msg.contains("Assinatura") {
                    NamErrorCode::NambInvalidMagic
                } else if msg.contains("Versão") {
                    NamErrorCode::NambUnsupportedVersion
                } else if msg.contains("CRC32") {
                    NamErrorCode::NambCrc32Mismatch
                } else {
                    NamErrorCode::ModelBuildFailed
                };

                let file_size = std::fs::metadata(path)
                    .map(|m| m.len())
                    .unwrap_or(0);

                NamDiagnostic::new(code, sys)
                    .message(format!(
                        "O arquivo \"{}\" não é um modelo .namb válido.",
                        path_str
                    ))
                    .hint(
                        "O arquivo pode estar corrompido. Tente baixá-lo novamente do site original.",
                    )
                    .param("file", &path_str)
                    .param("size", file_size)
                    .param("detail", &msg)
                    .emit();
            })
        })
    } else if ext_lower == "nam" {
        std::fs::read_to_string(path).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Erro ao ler o arquivo \"{}\"", path_str))
                .hint("Verifique as permissões do arquivo e se o disco está acessível.")
                .param("file", &path_str)
                .param("io_error", &e)
                .emit();
            anyhow::Error::from(e)
        }).and_then(|json| {
            loader::nam_json::parse_nam_json(&json).inspect_err(|e| {
                let file_size = std::fs::metadata(path)
                    .map(|m| m.len())
                    .unwrap_or(0);

                NamDiagnostic::new(NamErrorCode::NamJsonParseError, sys)
                    .message(format!(
                        "Falha ao interpretar o modelo \"{}\".",
                        path_str
                    ))
                    .hint(
                        "O arquivo JSON pode estar incompleto ou malformado. Verifique se foi baixado corretamente.",
                    )
                    .param("file", &path_str)
                    .param("size", file_size)
                    .param("detail", e)
                    .emit();
            })
        })
    } else {
        NamDiagnostic::new(NamErrorCode::UnknownExtension, sys)
            .message(format!(
                "Extensão de arquivo desconhecida: \".{}\"",
                ext_lower
            ))
            .hint("O NAM-rs aceita arquivos .nam (JSON) ou .namb (binário).")
            .param("file", &path_str)
            .param("extension", &ext_lower)
            .emit();
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
                    let msg = e.to_string();
                    let code = if msg.contains("Arquitetura não suportada") {
                        NamErrorCode::UnsupportedArchitecture
                    } else if msg.contains("pesos inconsistentes") || msg.contains("insuficientes")
                    {
                        NamErrorCode::WeightCountMismatch
                    } else if msg.contains("Geometria LSTM") || msg.contains("Dilatações") {
                        NamErrorCode::TopologyDetectionFailed
                    } else {
                        NamErrorCode::ModelBuildFailed
                    };

                    NamDiagnostic::new(code, sys)
                        .message(format!(
                            "Não foi possível construir o modelo a partir de \"{}\".",
                            path_str
                        ))
                        .hint(
                            "O modelo pode ser incompatível com esta versão do NAM-rs. \
                             Verifique se a arquitetura (WaveNet/LSTM) e os pesos são compatíveis.",
                        )
                        .param("file", &path_str)
                        .param("architecture", &model_data.architecture)
                        .param("weight_count", model_data.weights.len())
                        .param("detail", &msg)
                        .emit();
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
                // Mensagem mundana informativa — mantida como antes
                println!(
                    "[CLI] Payload enviado. Modelo: {}, InputAdj: {:.2}dB, OutputAdj: {:.2}dB",
                    path_str, input_db_adj, output_db_adj
                );
            } else {
                NamDiagnostic::new(NamErrorCode::ParamChannelFull, sys)
                    .message("Canal de parâmetros cheio. O modelo não foi enviado ao DSP.")
                    .hint(
                        "Tente novamente em alguns instantes. \
                         Se o problema persistir, o motor de áudio pode estar sobrecarregado.",
                    )
                    .param("file", &path_str)
                    .emit();
            }
        }
        Err(_) => {
            // Diagnóstico já emitido dentro do bloco de parsing acima — não duplicar.
        }
    }
}

/// Loop interativo de comandos CLI (thread separada).
///
/// Lê comandos do stdin em loop bloqueante e os converte em operações SPSC:
/// - `model <path>` — carrega um novo modelo neural.
/// - `gain_in <dB>` / `gain_out <dB>` — ajusta ganhos de entrada/saída.
/// - `quit` — sinaliza shutdown gracioso via [`spsc::SHUTDOWN`].
///
/// Erros de parsing e canal cheio emitem diagnósticos estruturados.
/// O loop encerra quando stdin fecha (EOF) ou o usuário digita `quit`.
fn cli_loop(mut producer: rtrb::Producer<ParamPayload>, sys: SystemSnapshot) {
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
                    load_and_send_model(path, &mut producer, &sys);
                } else {
                    println!("[CLI] Uso: model <caminho>");
                }
            }
            "gain_in" | "gi" => {
                if let Some(val_str) = parts.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        if producer.push(ParamPayload::InputGain(val)).is_ok() {
                            // Mensagem mundana informativa
                            println!("[CLI] Input Gain configurado para {:.2} dB.", val);
                        } else {
                            NamDiagnostic::new(NamErrorCode::ParamChannelFull, &sys)
                                .message("Canal de parâmetros cheio. O ganho não foi alterado.")
                                .hint("Tente novamente em alguns instantes.")
                                .param("param", "InputGain")
                                .param("value_db", val)
                                .emit();
                        }
                    } else {
                        NamDiagnostic::new(NamErrorCode::InvalidGainValue, &sys)
                            .message(format!("Valor de ganho inválido: \"{}\"", val_str))
                            .hint(
                                "Use um número decimal, por exemplo: gain_in -3.0 ou gain_in 12.5",
                            )
                            .param("raw_input", val_str)
                            .emit();
                    }
                } else {
                    println!("[CLI] Uso: gain_in <valor_db>");
                }
            }
            "gain_out" | "go" => {
                if let Some(val_str) = parts.next() {
                    if let Ok(val) = val_str.parse::<f32>() {
                        if producer.push(ParamPayload::OutputGain(val)).is_ok() {
                            // Mensagem mundana informativa
                            println!("[CLI] Output Gain configurado para {:.2} dB.", val);
                        } else {
                            NamDiagnostic::new(NamErrorCode::ParamChannelFull, &sys)
                                .message("Canal de parâmetros cheio. O ganho não foi alterado.")
                                .hint("Tente novamente em alguns instantes.")
                                .param("param", "OutputGain")
                                .param("value_db", val)
                                .emit();
                        }
                    } else {
                        NamDiagnostic::new(NamErrorCode::InvalidGainValue, &sys)
                            .message(format!("Valor de ganho inválido: \"{}\"", val_str))
                            .hint(
                                "Use um número decimal, por exemplo: gain_out 2.5 ou gain_out -6.0",
                            )
                            .param("raw_input", val_str)
                            .emit();
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
            other => {
                NamDiagnostic::new(NamErrorCode::UnknownCommand, &sys)
                    .message(format!("Comando desconhecido: \"{}\"", other))
                    .hint("Digite 'help' para ver os comandos disponíveis.")
                    .param("input", other)
                    .emit();
            }
        }
    }
}

/// Ponto de entrada do NAM-rs.
///
/// Orquestra o startup completo do engine:
/// 1. Parse de argumentos CLI via [`lexopt`].
/// 2. Captura do [`SystemSnapshot`] (propagado para todos os diagnósticos).
/// 3. Detecção de features SIMD (AVX2+FMA) com aviso se ausentes.
/// 4. Inicialização do PipeWire e handler de Ctrl-C.
/// 5. Setup dos canais SPSC (parâmetros, GC, resampler) via [`spsc::setup_spsc`].
/// 6. Spawn da thread GC (drop-delegation fora do RT).
/// 7. Carga do modelo inicial (se especificado via `-m`).
/// 8. Spawn da thread CLI ([`cli_loop`]) e execução do host PipeWire ([`pw_host::run_pipewire_host`]).
fn main() -> anyhow::Result<()> {
    let (model_path, initial_in_gain, initial_out_gain) = parse_args()?;

    // Captura snapshot do sistema uma vez — propagado para todas as funções de diagnóstico
    let sys = SystemSnapshot::capture();

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // Mensagem mundana informativa
        println!("Engine NAM-rs: AVX2 + FMA ativado computando vetores de 256 bits.");
    } else {
        NamDiagnostic::new(NamErrorCode::Avx2FmaMissing, &sys)
            .message("AVX2 ou FMA não detectados neste processador (x86-64-v3 ausente).")
            .hint(
                "O NAM-rs continuará funcionando com fallback escalar, mas o desempenho DSP \
                 será significativamente inferior. Para melhor performance, execute em um \
                 processador com suporte a AVX2+FMA (Intel Haswell+ / AMD Zen+).",
            )
            .emit_warning();
    }

    pipewire::init();

    ctrlc::set_handler(|| {
        if spsc::SHUTDOWN.load(Ordering::SeqCst) {
            std::process::exit(1);
        }
        spsc::SHUTDOWN.store(true, Ordering::SeqCst);
    })
    .map_err(|e| {
        NamDiagnostic::new(NamErrorCode::CtrlCHandlerFailed, &sys)
            .message("Falha ao configurar o handler de Ctrl-C.")
            .hint("O programa pode não responder corretamente ao sinal de interrupção.")
            .param("detail", &e)
            .emit();
        anyhow::anyhow!("Ctrl-C handler setup failed: {e}")
    })?;

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
        load_and_send_model(&path, &mut producer, &sys);
    }

    let sys_for_cli = sys.clone();
    std::thread::spawn(move || {
        cli_loop(producer, sys_for_cli);
    });

    pw_host::run_pipewire_host(
        consumer,
        gc_producer,
        resampler_consumer,
        resampler_producer,
        rt_status,
        sys,
    )?;

    unsafe {
        pipewire::deinit();
    }

    Ok(())
}
