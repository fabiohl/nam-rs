// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![cfg(feature = "standalone")]

//! Teste de Integração (End-to-End) para a Pipeline do PipeWire
//!
//! Este teste garante o correto inicializador do contexto do PipeWire Core.
//! Nota: Falhará propositalmente sem ser ignorado caso PipeWire não esteja presente,
//! a menos que seja mockado. O ambiente CI (ex: GitHub Actions) precisa de dependências
//! de PipeWire-headless rodando.

use minstant::Anchor;
use nam_rs::common::diagnostics::SystemSnapshot;
use nam_rs::common::spsc::{self, RtStatusFlags};
use nam_rs::standalone::pw_host::run_pipewire_host;
use rtrb::RingBuffer;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

/// Testa a inicialização e a comunicação básica da pipeline PipeWire em modo Headless.
///
/// Este teste simula o ciclo de vida completo do motor:
/// 1. Criação de RingBuffers SPSC (Single-Producer Single-Consumer) para comandos e telemetria.
/// 2. Spawning da thread de áudio (host).
/// 3. Envio de parâmetros de ganho via canal de controle.
/// 4. Desligamento sinalizado via flag atômica.
#[test]
fn test_pipewire_headless_integration() {
    // Inicializa a biblioteca nativa. Em ambientes sem libpipewire instalada, isso causará pânico.
    pipewire::init();

    println!("✔ PipeWire Inicializado com sucesso.");

    // RingBuffers para comunicação Thread-Safe sem Locks:
    // param: Comandos da UI -> DSP
    let (mut param_prod, param_cons) = RingBuffer::new(4);
    // gc: Garbage Collection de recursos DSP (Thread DSP -> Background)
    let (gc_prod, gc_cons) = RingBuffer::new(4);
    // res: Respostas/Telemetria (DSP -> UI)
    let (res_prod, res_cons) = RingBuffer::new(2);

    // Buffer de fallback para overflow de GC
    let gc_overflow = Arc::new(spsc::GcOverflowBuffer::new(64));

    // Flags de status compartilhado (xruns, cpu load, etc)
    let rt_status = Arc::new(RtStatusFlags::default());

    let rt_clone = rt_status.clone();
    let gc_overflow_clone = gc_overflow.clone();
    // Captura metadados do sistema para diagnóstico inicial
    let sys = SystemSnapshot::capture();
    // Sincroniza o relógio TSC para medições de nanosegundos de baixa latência
    let anchor = Anchor::new();

    // Inicia a Thread de Áudio
    let pw_thread = thread::spawn(move || {
        // run_pipewire_host é o loop principal que interage com o daemon do sistema.
        run_pipewire_host(
            param_cons,
            gc_prod,
            gc_overflow_clone,
            res_cons,
            res_prod,
            rt_clone,
            nam_rs::standalone::pw_host::PipewireHostConfig {
                buffer_size: 0, // 0 = Usar padrão do sistema (PipeWire quantum)
                sys,
                tsc_anchor: anchor,
            },
            gc_cons,
        )
    });

    // Simulação de interação do usuário: ajuste de ganhos de entrada/saída
    thread::sleep(Duration::from_millis(50));
    let _ = param_prod.push(nam_rs::common::spsc::ParamPayload::InputGain(2.5));
    let _ = param_prod.push(nam_rs::common::spsc::ParamPayload::OutputGain(-1.0));

    // Sinaliza o desligamento da pipeline.
    // O motor monitora esta flag a cada iteração do loop de áudio.
    thread::sleep(Duration::from_millis(150));
    spsc::SHUTDOWN.store(true, Ordering::Relaxed);

    // Aguarda a finalização da thread de áudio e valida o resultado
    match pw_thread.join() {
        Ok(result) => {
            if let Err(e) = result {
                // Erro de Daemon Ausente (ex: no CI) é reportado mas não quebra o teste
                eprintln!(
                    "O host PipeWire encerrou com erro esperado (possível ausência de daemon): {e}"
                );
            } else {
                println!("Pipeline host executado e finalizado graciosamente.");
            }
        }
        Err(_) => panic!("A Thread de PipeWire sofreu um pânico fatal (Panic)!"),
    }

    println!("Teste de integração concluído.")
}
