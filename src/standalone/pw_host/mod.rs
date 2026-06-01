// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Núcleo de processamento de áudio DSP usando `pipewire-rs`.
//!
//! Este é o "coração" do modo Standalone do NAM-rs: o módulo que processa o áudio em
//! tempo real. Ele recebe amostras brutas de áudio do PipeWire (o servidor de som
//! do Linux), passa pelo "motor neural", e entrega o resultado final processado
//! ao hardware via arquitetura dual-stream.
//!
//! ## Arquitetura Dual-Stream com DspBridge
//!
//! O PipeWire copia os buffers para o monitor port **antes** de chamar o `process()`.
//! Portanto, modificações in-place numa única stream `Audio/Sink` seriam invisíveis
//! ao hardware. A solução usa duas streams:
//!
//! 1. **Capture stream** (`Audio/Sink`, `Direction::Input`) — atua como um Virtual Sink
//!    que recebe áudio dos apps, aplica a cadeia DSP (ganho + inferência neural)
//!    e escreve o resultado no [`DspBridge`].
//! 2. **Playback stream** (`Stream/Output/Audio`, `Direction::Output`) — atua como um
//!    cliente de reprodução que lê do [`DspBridge`] e entrega ao hardware.
//!
//! O [`DspBridge`] é um buffer `#[repr(align(128))]` compartilhado entre as duas
//! closures via ponteiro raw, com sincronização lock-free via `Ordering::Release/Acquire`
//! e contador de geração atômico.
//!
//! ## Regras absolutas deste módulo (por que são tão rigorosas?)
//!
//! No `process()` callback (a função chamada centenas de vezes por segundo pelo PipeWire):
//! - **Zero alocação na heap** — nunca pedimos memória nova ao sistema durante o processamento.
//! - **Zero I/O** — nunca escrevemos no terminal ou em arquivos; status é reportado via flags atômicas.
//! - **Zero mutexes** — nunca travamos/esperamos por outras threads.
//!
//! Essas regras existem porque qualquer pausa, por menor que seja, causaria estalos e cortes no
//! áudio — inaceitável para um músico tocando ao vivo.
//!
//! ## Fluxo de processamento (Capture callback)
//!
//! O callback `process()` segue esta sequência para cada bloco de áudio:
//! 1. **Noise Gate e Ganho de Entrada** — Avalia a energia do sinal e aplica o ganho inicial (pré-DSP).
//! 2. `NamResampler::process_input()` — Converte o sample rate para a taxa compatível (geralmente 48 kHz).
//! 3. **Inferência neural WaveNet/LSTM** — O motor neural que processa o sinal sonoro.
//! 4. `NamResampler::process_output()` — Converte de volta para o sample rate original do host.
//! 5. **Ganho de Saída e Clipping** — Aplica o volume final e detecta saturação digital.
//! 6. **Escrita no [`DspBridge`]** — Publica o resultado com `Ordering::Release` para o playback callback.
//!
//! Quando nenhum modelo está carregado, o motor opera em **True-Bypass** (o sinal de entrada passa limpo).
//! Quando o sample rate do PipeWire é o mesmo do modelo nam, o resampler opera em bypass sem overhead.

mod bridge;
mod capture;
mod playback;
mod rt_callback;

pub use crate::dsp::pipeline::PipewireHostConfig;

use crate::common::spsc::{GcItem, GcOverflowBuffer, ParamPayload, RtStatusFlags, SHUTDOWN};
use crate::dsp::pipeline::AppState;
use crate::dsp::resampler::NamResampler;
use crate::standalone::colors::Colorize;
use crate::standalone::rt_setup;

use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::Ordering;

// Re-exports para compatibilidade do módulo de testes (pw_host_test.rs).
#[cfg(test)]
pub(crate) use crate::dsp::pipeline::{BridgeBuffer, DspBridge, MAX_BRIDGE_BUF};

/// Inicializa a topologia PipeWire dual-stream (Capture + Playback).
///
/// Arquitetura: Apps → [Capture: Audio/Sink] → process(DSP) → DspBridge → [Playback: Stream/Output] → Hardware.
/// O monitor port do `Audio/Sink` copia o buffer *antes* do `process()` — portanto, a única
/// forma de entregar o áudio processado ao hardware é via uma segunda stream de playback
/// que lê do `DspBridge` pós-DSP.
///
/// ## Parâmetros de canais SPSC
///
/// - `consumer`: Consumidor do canal de parâmetros CLI→DSP (gain, modelo, etc.).
/// - `gc_producer`: Produtor do canal GC para drop-delegation de modelos obsoletos.
/// - `resampler_consumer`: Canal dedicado para receber resamplers pré-construídos
///   pela thread principal — **zero alocações no callback RT**.
/// - `resampler_producer`: Produtor do canal de resamplers — a thread principal
///   constrói `NamResampler::new()` aqui (alocação fora do RT) e envia para o callback.
/// - `rt_status`: Flags atômicas para comunicação silenciosa RT→Main.
#[allow(clippy::too_many_arguments)]
pub fn run_pipewire_host(
    consumer: Consumer<ParamPayload>,
    gc_producer: rtrb::Producer<GcItem>,
    gc_overflow: Arc<GcOverflowBuffer>,
    resampler_consumer: Consumer<NamResampler>,
    mut resampler_producer: rtrb::Producer<NamResampler>,
    rt_status: Arc<RtStatusFlags>,
    config: PipewireHostConfig,
    mut gc_consumer: Consumer<GcItem>,
) -> anyhow::Result<()> {
    let PipewireHostConfig {
        buffer_size,
        tsc_anchor,
        sys,
    } = config;

    // =========================================================
    // 1. INICIALIZAÇÃO DO LOOP PIPEWIRE
    // =========================================================
    let thread_loop =
        unsafe { pipewire::thread_loop::ThreadLoopBox::new(Some("nam-rs-loop"), None) }?;
    let context = pipewire::context::ContextBox::new(thread_loop.loop_(), None)?;
    let core = context.connect(None)?;

    // =========================================================
    // 2. ALOCAÇÃO DO DSP BRIDGE (Comunicação Lock-Free)
    // =========================================================
    let bridge_ptr = bridge::allocate_dsp_bridge();

    // =========================================================
    // 3. OTIMIZAÇÃO DE NÚCLEO (CPU Affinity)
    // =========================================================
    let target_cpu = rt_setup::select_optimal_cpu().unwrap_or(0);

    // =========================================================
    // 4. ESCOPO DE CONFIGURAÇÃO PROTEGIDA (RAII)
    // =========================================================
    let (capture_stream, capture_listener, playback_stream, playback_listener);
    {
        let _lock = thread_loop.lock();

        let latency_str = format!("{}/48000", buffer_size);

        let (cs, cl) = capture::setup_capture_stream(
            &core,
            bridge_ptr,
            buffer_size,
            &sys,
            target_cpu,
            consumer,
            gc_producer,
            gc_overflow.clone(),
            resampler_consumer,
            rt_status.clone(),
        )?;
        capture_stream = cs;
        capture_listener = cl;

        let (ps, pl) =
            playback::setup_playback_stream(&core, bridge_ptr, buffer_size, &latency_str)?;
        playback_stream = ps;
        playback_listener = pl;
    }

    let _app_state = AppState {
        capture_stream,
        capture_listener,
        playback_stream,
        playback_listener,
    };

    let _cpu_dma_lock = rt_setup::lock_cpu_c_states();

    sys.emit_irq_advisory(target_cpu);

    // =========================================================
    // 5. INÍCIO DA THREAD RT (Background)
    // =========================================================
    thread_loop.start();

    // =========================================================
    // 6. LOOP DE CONTROLE PRINCIPAL (Thread Main, Non-RT)
    // =========================================================
    let mut was_silent = false;
    let mut was_fading = false;
    while !SHUTDOWN.load(Ordering::Relaxed) {
        if rt_status.check_flag(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD) {
            let target_pw_rate = rt_status.requested_pw_rate.load(Ordering::Relaxed);
            let target_nam_rate = rt_status.requested_nam_rate.load(Ordering::Relaxed);

            if target_pw_rate != 0 && target_nam_rate != 0 {
                match NamResampler::new(target_pw_rate, target_nam_rate, 2048) {
                    Ok(new_rs) => {
                        rt_status
                            .clear_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED);

                        log::info!(
                            "{} Sample rate atualizado: PW={} Hz, NAM={} Hz (bypass={})",
                            "🔄".cyan(),
                            target_pw_rate,
                            target_nam_rate,
                            new_rs.is_bypass()
                        );

                        if resampler_producer.push(new_rs).is_err() {
                            crate::common::diagnostics::NamDiagnostic::new(
                                crate::common::diagnostics::NamErrorCode::ResamplerChannelFull,
                                &sys,
                            )
                            .message("Resampler channel full. Rebuild discarded.")
                            .hint(
                                "The audio engine is overloaded. \
                                 If the problem persists, restart NAM-rs.",
                            )
                            .param("target_pw_rate", target_pw_rate)
                            .param("target_nam_rate", target_nam_rate)
                            .emit_warning();
                        }
                    }
                    Err(e) => {
                        crate::common::diagnostics::NamDiagnostic::new(
                            crate::common::diagnostics::NamErrorCode::ResamplerBuildFailed,
                            &sys,
                        )
                        .message(format!(
                            "Failed to rebuild resampler for PW={} Hz and NAM={} Hz.",
                            target_pw_rate, target_nam_rate
                        ))
                        .hint(
                            "Audio will continue with the previous resampler. \
                             If the sample rate is incorrect, restart NAM-rs.",
                        )
                        .param("target_pw_rate", target_pw_rate)
                        .param("target_nam_rate", target_nam_rate)
                        .param("detail", &e)
                        .emit();

                        rt_status.set_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED);
                    }
                }
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD);
            }
        }

        (was_silent, was_fading) = rt_setup::poll_rt_status(
            &rt_status,
            &sys,
            was_silent,
            was_fading,
            &tsc_anchor,
            unsafe { &*(bridge_ptr.as_ptr()) },
        );

        crate::common::spsc::drain_gc_channels(&mut gc_consumer, &gc_overflow);

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // =========================================================
    // 7. GRACEFUL SHUTDOWN
    // =========================================================
    thread_loop.stop();

    Ok(())
}

#[cfg(test)]
#[path = "pw_host_test.rs"]
mod pw_host_test;
