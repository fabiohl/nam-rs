// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![warn(missing_docs)]

//! Ponto de entrada principal do NAM-rs.
//! Inicializa o backend standalone do `nih_plug` para processamento de áudio via PipeWire
//! e coordena o shutdown gracioso quando CTRL+C é pressionado.
//!
//! # Regras de Arquitetura para Desenvolvedores
//! - **ZERO LOCKS** na thread DSP (módulo audio).
//! - **ZERO ALOCAÇÕES** na thread DSP (o loop `process()` deve ser zero-allocation).

mod audio;
mod buffer;

use audio::NamRsPlugin;
use nih_plug::wrapper::standalone::nih_export_standalone;
use std::sync::atomic::Ordering;
use std::time::Duration;

fn main() {
    // ------------------------------------------------------------
    // 1️⃣ Configura o tratamento de shutdown gracioso
    // ------------------------------------------------------------
    // Um handler de sinal intercepta a intenção de encerramento. O nih_plug gerencia
    // o lifecycle do áudio; SHUTDOWN é lido pela thread coordenadora para encerrar o processo.
    ctrlc::set_handler(|| {
        if buffer::SHUTDOWN.load(Ordering::SeqCst) {
            // Segundo sinal: aborta imediatamente.
            std::process::exit(1);
        }
        buffer::SHUTDOWN.store(true, Ordering::SeqCst);
    })
    .expect("Erro ao configurar handler de Ctrl-C");

    // 2️⃣ Thread coordenadora de shutdown.
    // O nih_plug bloqueia a main indefinidamente; esta thread monitora o SHUTDOWN
    // e encerra o processo de forma limpa quando CTRL+C é pressionado.
    std::thread::spawn(move || {
        while !buffer::SHUTDOWN.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
        }
        // Breve pausa para permitir que o buffer de stdout esvazie mensagens de forma limpa.
        std::thread::sleep(Duration::from_millis(50));
        std::process::exit(0);
    });

    // 3️⃣ Inicializa o motor DSP de Tempo Real via emulação de plugin PipeWire.
    // Isso bloqueia a thread main indefinidamente.
    nih_export_standalone::<NamRsPlugin>();
}
