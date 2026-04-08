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

pub mod loader;
pub mod math;
pub mod models;
mod pw_host;
mod spsc;

use std::sync::atomic::Ordering;

fn main() -> anyhow::Result<()> {
    // Verificando suporte de instruções matemáticas fundamentais de AVX2/FMA
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        println!("Engine NAM-rs: AVX2 + FMA ativado computando vetores de 256 bits.");
    } else {
        eprintln!(
            "AVISO CRÍTICO: AVX2 ou FMA não presentes nativamente (X86-64-V3 ausente).\nO rendimento DSP será castigado pelo fallback escalar."
        );
    }

    // Inicializa as APIs do PipeWire nativo
    pipewire::init();

    // 1️⃣ Configura o tratamento de shutdown gracioso
    // Um handler de sinal intercepta a intenção de encerramento de forma global.
    ctrlc::set_handler(|| {
        if spsc::SHUTDOWN.load(Ordering::SeqCst) {
            // Segundo sinal: aborta imediatamente.
            std::process::exit(1);
        }
        spsc::SHUTDOWN.store(true, Ordering::SeqCst);
    })
    .expect("Erro ao configurar handler de Ctrl-C");

    // 2️⃣ Inicializa o SPSC Ring Buffer para a comunicação atômica CLI <-> DSP
    let (_producer, consumer) = spsc::setup_spsc(64);

    // 3️⃣ Inicia a topologia PipeWire e bloqueia a thread
    pw_host::run_pipewire_host(consumer)?;

    // Desaloca componentes nativos após graceful shutdown
    unsafe {
        pipewire::deinit();
    }

    Ok(())
}
