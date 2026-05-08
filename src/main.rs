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
use nam_rs::diagnostics::SystemSnapshot;
use nam_rs::{loader, pw_host, spsc, spsc::ParamPayload};
use std::sync::atomic::Ordering;

/// Ponto de entrada do NAM-rs.
fn main() -> anyhow::Result<()> {
    // Inicializa o backend de logging (respeita RUST_LOG; padrão: info)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let (model_path, initial_in_gain, initial_out_gain, buffer_size, interactive) =
        match cli::parse_args() {
            Ok(args) => args,
            Err(e) => {
                eprintln!("\n{} {}", "❌ Erro nos argumentos:".red().bold(), e);
                std::process::exit(1);
            }
        };

    let sys = SystemSnapshot::capture();
    log::info!(
        "🎸 {}",
        format!("NAM-rs v{} — Neural Amp Modeler", sys.version)
            .bright_green()
            .bold()
    );

    pipewire::init();
    nam_rs::rt_setup::calibrate_tsc();

    // Handler de SIGINT (Ctrl-C)
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

    // Setup de canais SPSC
    let channels = spsc::setup_spsc(64);
    let mut producer = channels.param_producer;
    let consumer = channels.param_consumer;
    let gc_producer = channels.gc_producer;
    let gc_consumer = channels.gc_consumer;
    let gc_overflow = channels.gc_overflow;
    let resampler_producer = channels.resampler_producer;
    let resampler_consumer = channels.resampler_consumer;
    let rt_status = channels.rt_status;

    // Carga inicial do modelo
    if let Some(path) = model_path {
        log::info!("{} Carregando modelo inicial...", "📂".cyan());
        match loader::load_and_build_model(&path, &sys) {
            Ok(nam_rs::loader::LoadedModelPair {
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
            }
            Err(e) => log::error!("Falha na carga inicial: {}", e),
        }
    }

    // Ganhos iniciais
    if initial_in_gain != 0.0 {
        let _ = producer.push(ParamPayload::InputGain(
            nam_rs::math::fastmath::get_gain_lut().db_to_linear(initial_in_gain),
        ));
    }
    if initial_out_gain != 0.0 {
        let _ = producer.push(ParamPayload::OutputGain(
            nam_rs::math::fastmath::get_gain_lut().db_to_linear(initial_out_gain),
        ));
    }

    // Spawn da thread CLI interativa (apenas se solicitado)
    if interactive {
        let sys_clone = sys.clone();
        std::thread::spawn(move || {
            if let Err(e) = cli::cli_loop(producer, sys_clone) {
                log::error!("Erro no loop CLI: {}", e);
            }
        });
    }

    let tsc_anchor = minstant::Anchor::new();

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
