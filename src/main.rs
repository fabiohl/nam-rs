// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![warn(missing_docs)]

//! Ponto de entrada principal do binário CLI/Standalone do NAM-rs.
//!
//! **NOTA:** Este arquivo é utilizado apenas para o modo executável (terminal/PipeWire).
//! O plugin CLAP ignora este arquivo e utiliza diretamente a biblioteca definida em `lib.rs`.
//!
//! Pense neste arquivo como a "recepção" do nosso estúdio virtual. Ele é responsável por:
//! 1. Ler o que o usuário digita no terminal (qual amplificador carregar e os volumes de entrada/saída).
//! 2. Abrir a conexão de áudio com o sistema (PipeWire), conectando o sinal de áudio ao motor sonoro.
//! 3. Garantir que, quando o usuário apertar CTRL+C, tudo seja desligado com segurança, sem deixar ruídos.
//!
//! # Regras de Arquitetura para Desenvolvedores
//! - **ZERO LOCKS** na thread de Áudio (módulo `pw_host`): O áudio não "espera" pela interface visual. Se não houver instrução nova, ele continua usando a anterior. Evita "engasgos" no som.
//! - **ZERO ALOCAÇÕES** na thread de Áudio: A memória do canal de áudio (`process()`) é sempre preparada 100% de antemão. O áudio nunca "pede por mais memória RAM" de supetão.

use nam_rs::diagnostics::SystemSnapshot;
use nam_rs::standalone::{cli, colors::Colorize, pw_host, rt_setup};
use nam_rs::{loader, spsc, spsc::ParamPayload};

use std::sync::atomic::Ordering;

/// Ponto de entrada do NAM-rs.
fn main() -> anyhow::Result<()> {
    // Inicializa o backend de logging (respeita RUST_LOG; padrão: info)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // 1. LER CONFIGURAÇÕES: O sistema começa lendo o que você digitou no terminal.
    // Ele descobre qual arquivo de "amplificador" (.nam) você quer usar e os volumes iniciais.
    let (model_path, initial_in_gain, initial_out_gain, buffer_size) = cli::parse_args();

    // 2. CONHECER O COMPUTADOR: Captura uma "foto" das capacidades do seu processador.
    // Isso ajuda o NAM-rs a escolher a forma mais rápida de processar a matemática do áudio.
    let sys = SystemSnapshot::capture();
    log::info!(
        "🎸 {}",
        format!("NAM-rs Standalone v{} — Neural Amp Modeler", sys.version)
            .bright_green()
            .bold()
    );

    // 3. PREPARAR O ÁUDIO: Inicializa o PipeWire (o sistema de som do Linux)
    // e calibra os "relógios" internos para garantir que o som saia sem atrasos (latência).
    pipewire::init();
    rt_setup::calibrate_tsc();

    // 4. BOTÃO DE EMERGÊNCIA: Configura o "Ctrl+C".
    // Se você quiser fechar o programa, ele garante que o áudio pare de forma suave, sem estalos.
    extern "C" fn sigint_handler(_sig: libc::c_int) {
        if spsc::SHUTDOWN.load(Ordering::SeqCst) {
            unsafe { libc::_exit(1) };
        }
        spsc::SHUTDOWN.store(true, Ordering::SeqCst);
    }
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sigint_handler as *const () as libc::sighandler_t;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
    }

    // 5. CANAIS DE COMUNICAÇÃO: Cria "tubos" de comunicação ultra-rápidos.
    // Imagine que a Interface (você) e o Motor de Som (o áudio) moram em salas separadas.
    // Esses canais permitem enviar comandos (ex: "aumenta o volume") sem nunca fazer o som "engasgar".
    let channels = spsc::setup_spsc(64);
    let mut producer = channels.param_producer;
    let consumer = channels.param_consumer;
    let gc_producer = channels.gc_producer;
    let gc_consumer = channels.gc_consumer;
    let gc_overflow = channels.gc_overflow;
    let resampler_producer = channels.resampler_producer;
    let resampler_consumer = channels.resampler_consumer;
    let rt_status = channels.rt_status;

    // 6. CARREGAR O SOM: Se você disse "use o amplificador X",
    // aqui é onde o computador abre aquele arquivo e prepara as contas matemáticas (Redes Neurais).
    if let Some(path) = model_path {
        log::info!("{} Loading model...", "📂".cyan());
        match loader::load_and_build_model(&path, &sys) {
            Ok(nam_rs::loader::LoadedModelPair {
                model_l,
                model_r,
                input_mult_adj,
                output_mult_adj,
                sample_rate,
                ..
            }) => {
                let _ = producer.push(ParamPayload::LoadModel {
                    model_l,
                    model_r,
                    input_mult_adj,
                    output_mult_adj,
                    sample_rate,
                });
            }
            Err(e) => cli::exit_with_error(format!("Model load failed: {}", e)),
        }
    }

    // Ganhos iniciais
    if initial_in_gain != 0.0 {
        let _ = producer.push(ParamPayload::InputGain(
            nam_rs::math::dsp::gain_lut::get_gain_lut().db_to_linear(initial_in_gain),
        ));
    }
    if initial_out_gain != 0.0 {
        let _ = producer.push(ParamPayload::OutputGain(
            nam_rs::math::dsp::gain_lut::get_gain_lut().db_to_linear(initial_out_gain),
        ));
    }

    let tsc_anchor = minstant::Anchor::new();

    // Configurações process-wide (THP disable + mlockall) antes de iniciar o PipeWire.
    // Executadas aqui (fora do cold-path do primeiro frame DSP) para evitar
    // syscalls que causariam jitter no momento crítico da primeira entrega de áudio.
    rt_setup::configure_process_wide();

    // Executa o host PipeWire (bloqueante)
    pw_host::run_pipewire_host(
        consumer,
        gc_producer,
        gc_overflow,
        resampler_consumer,
        resampler_producer,
        rt_status,
        pw_host::PipewireHostConfig {
            buffer_size,
            tsc_anchor,
            sys,
        },
        gc_consumer,
    )?;

    unsafe {
        pipewire::deinit();
    }
    Ok(())
}
