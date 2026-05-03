// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![warn(missing_docs)]

//! Ponto de entrada principal do NAM-rs.
//!
//! Pense neste arquivo como a "recepção" do nosso estúdio virtual. Ele é responsável por:
//! 1. Ler o que o usuário digita no terminal (qual amplificador carregar e os volumes de entrada/saída).
//! 2. Abrir a conexão de áudio com o sistema (PipeWire), conectando o sinal de áudio ao motor sonoro.
//! 3. Garantir que, quando o usuário apertar CTRL+C, tudo seja desligado com segurança, sem deixar ruídos.
//!
//! # Regras de Arquitetura para Desenvolvedores
//! - **ZERO LOCKS** na thread de Áudio (módulo `pw_host`): O áudio não "espera" pela interface visual. Se não houver instrução nova, ele continua usando a anterior. Evita "engasgos" no som.
//! - **ZERO ALOCAÇÕES** na thread de Áudio: A memória do canal de áudio (`process()`) é sempre preparada 100% de antemão. O áudio nunca "pede por mais memória RAM" de supetão.

pub mod cli;

use nam_rs::colors::Colorize;
use nam_rs::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use nam_rs::{loader, models::NamModel, pw_host, spsc, spsc::ParamPayload};
use std::sync::atomic::Ordering;

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
/// Mensagens informativas de sucesso são emitidas via `log::info!` (visíveis com `RUST_LOG=info`).
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
                "Não foi possível encontrar o arquivo do modelo em \"{}\".",
                path_str
            ))
            .hint("Verifique se o caminho digitado está correto e se o arquivo realmente existe.")
            .param("file", &path_str)
            .emit();
        return;
    }

    // Determina o parser adequado (binário ou JSON) com base na extensão do arquivo.
    let ext_lower = ext.to_lowercase();
    let result = if ext_lower == "namb" {
        // Caso .namb: Leitura e processamento de modelos em formato binário.
        std::fs::read(path).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Não conseguimos ler o arquivo \"{}\".", path_str))
                .hint("Verifique se o aplicativo tem permissão de leitura nesta pasta ou se o arquivo não está bloqueado.")
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
                        "O arquivo \"{}\" não parece ser um modelo .namb válido.",
                        path_str
                    ))
                    .hint(
                        "Ele pode estar corrompido ou incompleto. Sugerimos baixar o modelo novamente do site original.",
                    )
                    .param("file", &path_str)
                    .param("size", file_size)
                    .param("detail", &msg)
                    .emit();
            })
        })
    } else if ext_lower == "nam" {
        // Caso .nam: Leitura e processamento de modelos em formato JSON.
        std::fs::read_to_string(path).map_err(|e| {
            NamDiagnostic::new(NamErrorCode::FileReadError, sys)
                .message(format!("Não conseguimos ler o arquivo \"{}\".", path_str))
                .hint("Verifique as permissões de acesso ao arquivo ou problemas no disco rígido.")
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
                        "Não foi possível entender o conteúdo do modelo \"{}\".",
                        path_str
                    ))
                    .hint(
                        "O arquivo JSON pode estar danificado ou incompleto. Recomendamos fazer um novo download da fonte original.",
                    )
                    .param("file", &path_str)
                    .param("size", file_size)
                    .param("detail", e)
                    .emit();
            })
        })
    } else {
        // Extensão desconhecida: emite diagnóstico de erro e encerra o pipeline.
        NamDiagnostic::new(NamErrorCode::UnknownExtension, sys)
            .message(format!(
                "Não suportamos arquivos do tipo \".{}\".",
                ext_lower
            ))
            .hint("Por favor, selecione um arquivo de modelo válido do NAM (.nam ou .namb).")
            .param("file", &path_str)
            .param("extension", &ext_lower)
            .emit();
        return;
    };

    // Extrai os metadados do modelo carregado, definindo valores padrão caso o arquivo
    // não os contenha ou campos específicos (ganho de entrada, loudness) estejam ausentes.
    match result {
        Ok(model_data) => {
            let meta = model_data
                .metadata
                .clone()
                .unwrap_or(loader::nam_json::NamMetadata {
                    date: None,
                    name: None,
                    modeled_by: None,
                    gear_make: None,
                    gear_model: None,
                    gear_type: None,
                    tone_type: None,
                    training: None,
                    input_level_dbu: None,
                    output_level_dbu: None,
                    loudness: None,
                });
            let in_level = meta.input_level_dbu.unwrap_or(12.0);
            let loudness = meta.loudness.unwrap_or(-18.0);

            // Calibração de ganho baseada em metadados do modelo:
            // - input_db_adj: O cálculo `12.0 - input_level_dbu` reflete com exatidão a física.
            //   Se o modelo foi treinado a 15 dBu (sinal quente), a nossa referência (12 dBu)
            //   está mais baixa e o modelo precisa atenuar o sinal para compensar? Não,
            //   a regra de calibração dita estritamente: 12.0 - input_level_dbu.
            // - output_db_adj: normalização de loudness. Alvo = −18 LUFS.
            //   Se o modelo declara −15 LUFS, output_db_adj = −18 − (−15) = −3 dB (atenua).
            let input_db_adj = 12.0 - in_level;
            let output_db_adj = -18.0 - loudness;
            let nam_rate = model_data.sample_rate.unwrap_or(48000.0) as u32;

            // Converte os ajustes de dB para multiplicadores lineares na Main Thread.
            let lut = nam_rs::math::fastmath::get_gain_lut();
            let input_mult_adj = lut.db_to_linear(input_db_adj);
            let output_mult_adj = lut.db_to_linear(output_db_adj);

            // Dispatcher: converte NamModelData → Box<DynamicModel> (thread CLI)
            // Para "True Stereo", instanciamos dois caminhos (L e R) de estados estritamente independentes.
            let model_l = match loader::dispatcher::build_model(&model_data) {
                Ok(mut model) => {
                    model.prewarm(2048);
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
                            "Não foi possível construir o amplificador neural a partir de \"{}\".",
                            path_str
                        ))
                        .hint(
                            "Este modelo pode exigir recursos não suportados ou não ser compatível com a nossa versão do aplicativo.",
                        )
                        .param("file", &path_str)
                        .param("architecture", &model_data.architecture)
                        .param("weight_count", model_data.weights.len())
                        .param("detail", &msg)
                        .emit();
                    None
                }
            };

            // Para suporte a "True Stereo", instanciamos uma segunda via idêntica para o canal direito.
            let model_r = if model_l.is_some() {
                match loader::dispatcher::build_model(&model_data) {
                    Ok(mut model) => {
                        model.prewarm(2048);
                        Some(model)
                    }
                    Err(_) => None,
                }
            } else {
                None
            };

            if model_l.is_some() {
                log::info!(
                    "{} Modelo preparado em True Stereo (L+R) com 2048 amostras pré-aquecidas.",
                    "🔥 [CLI]".yellow()
                );
            }

            // Tenta enviar o modelo carregado e os ajustes de calibração para o motor de áudio via SPSC.
            if producer
                .push(ParamPayload::LoadModel {
                    model_l,
                    model_r,
                    input_mult_adj,
                    output_mult_adj,
                    sample_rate: nam_rate,
                })
                .is_ok()
            {
                log::info!(
                    "{} Payload enviado. Modelo: {}",
                    "🚀 [CLI]".green(),
                    path_str.bright_cyan(),
                );
                log::info!(
                    "{} Calibração do modelo: input_level_dbu={:+.1}dB, loudness={:+.1}dB",
                    "📐 [CLI]".blue(),
                    input_db_adj,
                    output_db_adj
                );
            } else {
                // Caso a fila SPSC esteja cheia (sistema sobrecarregado), emite um diagnóstico de erro.
                NamDiagnostic::new(NamErrorCode::ParamChannelFull, sys)
                    .message("O sistema de áudio está temporariamente ocupado.")
                    .hint(
                        "Aguarde um instante e tente carregar o modelo novamente. Caso persista, pode haver sobrecarga no processamento.",
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

/// Ponto de entrada do NAM-rs.
///
/// Orquestra o startup completo do engine:
/// 1. Parse de argumentos CLI via [`lexopt`].
/// 2. Captura do [`SystemSnapshot`] (propagado para todos os diagnósticos).
/// 3. Detecção de features SIMD avançadas (AVX-512, VNNI) para multiversioning.
/// 4. Inicialização do PipeWire e handler de Ctrl-C.
/// 5. Setup dos canais SPSC (parâmetros, GC, resampler) via [`spsc::setup_spsc`].
/// 6. Spawn da thread GC (drop-delegation fora do RT).
/// 7. Carga do modelo inicial (se especificado via `-m`).
/// 8. Spawn da thread CLI ([`cli_loop`]) e execução do host PipeWire ([`pw_host::run_pipewire_host`]).
fn main() -> anyhow::Result<()> {
    // Inicializa o backend de logging (respeita RUST_LOG; padrão: info)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let (model_path, initial_in_gain, initial_out_gain, buffer_size) = match cli::parse_args() {
        Ok(args) => args,
        Err(e) => {
            eprintln!(
                "\n{} {}",
                "❌ Erro ao ler argumentos da CLI:".red().bold(),
                e
            );
            eprintln!(
                "{}",
                "💡 Dica: Rode 'nam-rs --help' para ver os parâmetros e formatos corretos.\n"
                    .yellow()
            );
            std::process::exit(1);
        }
    };

    // Captura snapshot do sistema uma vez — propagado para todas as funções de diagnóstico
    let sys = SystemSnapshot::capture();

    // Banner de startup
    log::info!(
        "🎸 {}",
        format!(
            "NAM-rs v{} — Neural Amp Modeler (Rust PipeWire native)",
            sys.version
        )
        .bright_green()
        .bold()
    );

    pipewire::init();
    log::info!("{} PipeWire inicializado.", "🔌".bright_blue());

    // Exibe features de CPU avançadas detectadas (acima do baseline v3)
    if !sys.features.is_empty() {
        log::info!(
            "{} SIMD avançado detectado: {}",
            "⚡".yellow(),
            sys.features.join(", ").bright_cyan()
        );
    }

    // Handler de SIGINT direto via libc::sigaction — substitui crate `ctrlc`.
    // Usa apenas operações async-signal-safe: AtomicBool + libc::_exit.
    extern "C" fn sigint_handler(_sig: libc::c_int) {
        if spsc::SHUTDOWN.load(Ordering::SeqCst) {
            // Segundo CTRL+C → force-quit imediato (signal-safe)
            unsafe { libc::_exit(1) };
        }
        spsc::SHUTDOWN.store(true, Ordering::SeqCst);
    }

    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sigint_handler as *const () as libc::sighandler_t;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigemptyset(&mut sa.sa_mask);
        let ret = libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        if ret != 0 {
            NamDiagnostic::new(NamErrorCode::CtrlCHandlerFailed, &sys)
                .message("Falha ao preparar o sistema para interceptar o CTRL+C.")
                .hint(
                    "O aplicativo pode não encerrar suavemente se você tentar fechá-lo pelo terminal.",
                )
                .param("detail", "sigaction(SIGINT) retornou erro")
                .emit();
            return Err(anyhow::anyhow!("sigaction(SIGINT) failed"));
        }
    }

    // Inicializa a infraestrutura de comunicação lock-free (SPSC) entre a CLI e a thread de áudio.
    let channels = spsc::setup_spsc(64);
    let mut producer = channels.param_producer;
    let consumer = channels.param_consumer;
    let gc_producer = channels.gc_producer;
    let mut gc_consumer = channels.gc_consumer;
    let gc_resampler_producer = channels.gc_resampler_producer;
    let mut gc_resampler_consumer = channels.gc_resampler_consumer;
    let resampler_producer = channels.resampler_producer;
    let resampler_consumer = channels.resampler_consumer;
    let rt_status = channels.rt_status;

    // Thread GC para "Drop-Delegation" lock-free
    std::thread::spawn(move || {
        while !spsc::SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
            // Drena modelos obsoletos
            while let Ok(model) = gc_consumer.pop() {
                drop(model);
            }
            // Drena resamplers obsoletos
            while let Ok(rs) = gc_resampler_consumer.pop() {
                drop(rs);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    if let Some(ref path) = model_path {
        log::info!(
            "{} Carregando modelo: {} ...",
            "📂 [CLI]".cyan(),
            path.to_string_lossy().bright_cyan()
        );
        load_and_send_model(path, &mut producer, &sys);
    } else {
        log::info!(
            "{} Nenhum modelo especificado. Use '--model <caminho>' para carregar.",
            "ℹ️".blue()
        );
    }
    if initial_in_gain != 0.0 {
        let lut = nam_rs::math::fastmath::get_gain_lut();
        let _ = producer.push(ParamPayload::InputGain(lut.db_to_linear(initial_in_gain)));
        log::info!(
            "{} Ganho de entrada: {:+.1} dB",
            "🎚️ [CLI]".cyan(),
            initial_in_gain
        );
    }
    if initial_out_gain != 0.0 {
        let lut = nam_rs::math::fastmath::get_gain_lut();
        let _ = producer.push(ParamPayload::OutputGain(lut.db_to_linear(initial_out_gain)));
        log::info!(
            "{} Ganho de saída: {:+.1} dB",
            "🎚️ [CLI]".cyan(),
            initial_out_gain
        );
    }

    // Mantém o producer vivo sem thread TUI — o canal SPSC precisa existir
    // enquanto o consumer (RT thread) estiver ativo. A infraestrutura de
    // parâmetros em tempo de execução permanece intacta para uso futuro.
    std::mem::forget(producer);

    // Inicializa a âncora de tempo para conversão RDTSC -> Duration fora da thread RT.
    let tsc_anchor = minstant::Anchor::new();

    // Executa o host PipeWire. Esta é uma chamada bloqueante que orquestra o processamento de áudio.
    pw_host::run_pipewire_host(
        consumer,
        gc_producer,
        gc_resampler_producer,
        resampler_consumer,
        resampler_producer,
        rt_status,
        pw_host::PipewireHostConfig {
            buffer_size,
            tsc_anchor,
            sys,
        },
    )?;

    // Finaliza a sessão do PipeWire e libera os recursos antes de sair.
    unsafe {
        pipewire::deinit();
    }

    Ok(())
}
