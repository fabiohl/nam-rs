// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![warn(missing_docs)]

//! Ponto de entrada principal do NAM-rs.
//! Inicializa o host base de PipeWire para processamento de áudio
//! e coordena o shutdown gracioso quando CTRL+C é pressionado.
//!
//! # Regras de Arquitetura para Desenvolvedores
//! - **ZERO LOCKS** na thread DSP (módulo audio).
//! - **ZERO ALOCAÇÕES** na thread DSP (o loop `process()` deve ser zero-allocation).

mod audio;
mod buffer;

use std::sync::atomic::Ordering;

fn main() -> anyhow::Result<()> {
    // Inicializa as APIs do PipeWire nativo
    pipewire::init();

    // 1️⃣ Configura o tratamento de shutdown gracioso
    // Um handler de sinal intercepta a intenção de encerramento de forma global.
    ctrlc::set_handler(|| {
        if buffer::SHUTDOWN.load(Ordering::SeqCst) {
            // Segundo sinal: aborta imediatamente.
            std::process::exit(1);
        }
        buffer::SHUTDOWN.store(true, Ordering::SeqCst);
    })
    .expect("Erro ao configurar handler de Ctrl-C");

    // 2️⃣ Inicializa o SPSC Ring Buffer para a comunicação atômica CLI <-> DSP
    let (_producer, consumer) = buffer::setup_spsc(64);

    // 3️⃣ Inicia a topologia PipeWire e bloqueia a thread
    audio::run_pipewire_host(consumer)?;

    // Desaloca componentes nativos após graceful shutdown
    unsafe {
        pipewire::deinit();
    }

    Ok(())
}
