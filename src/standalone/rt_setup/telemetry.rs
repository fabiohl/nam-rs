// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Telemetria e monitoramento de status RT do motor de áudio.
//!
//! Traduz sinais atômicos da thread DSP em logs de diagnóstico para
//! o loop principal, atuando como "painel de instrumentos" do NAM-rs.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::common::spsc::RtStatusFlags;
use crate::standalone::colors::Colorize;
use minstant::Anchor;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

/// Lê flags atômicas de status RT e emite logs de monitoramento para o usuário.
///
/// Esta função atua como o "painel de instrumentos" do NAM-rs. Ela é chamada periodicamente
/// para traduzir os sinais técnicos vindos da thread de áudio (que é silenciosa e ultra-veloz)
/// em mensagens compreensíveis, avisos de performance e telemetria de latência.
///
/// Retorna uma tupla (current_silent, current_fading) para controle de estado no loop principal.
pub fn poll_rt_status(
    rt_status: &RtStatusFlags,
    sys: &SystemSnapshot,
    was_silent: bool,
    was_fading: bool,
    _tsc_anchor: &Anchor,
    bridge: &crate::dsp::pipeline::DspBridge,
) -> (bool, bool) {
    // 1. GERENCIAMENTO DE MEMÓRIA (Garbage Collection):
    // Se o canal de limpeza estiver cheio, significa que estamos trocando modelos neurais
    // mais rápido do que o sistema consegue descartar os antigos. Priorizamos o áudio
    // "vazando" memória temporariamente para evitar estalos (drops) no som.
    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW) {
        NamDiagnostic::new(NamErrorCode::GcOverflow, sys)
            .message("Garbage Collection (GC) channel overflow detected.")
            .hint(
                "The audio thread had to leak memory to avoid dropouts in the hot-path. \
                   This can occur during rapid model swaps. \
                   NAM-rs will drain the buffer aggressively now.",
            )
            .emit_warning();
    }

    // 1.5. MODELO A2 PLACEHOLDER
    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_A2_PLACEHOLDER) {
        log::info!(
            "{} Modelo A2 não suportado — bypass ativo. A implementação real está em desenvolvimento.",
            "⚠️".yellow()
        );
    }

    // 2. MUDANÇA DE RITMO (Sample Rate):
    // Avisa quando o servidor de áudio (PipeWire) altera a frequência de amostragem
    // (ex: mudou de 44.100 para 48.000 batidas por segundo).
    let rate_notif = rt_status.active_rate_changed.swap(0, Ordering::Relaxed);
    if rate_notif != 0 {
        log::info!(
            "{} RT callback activated resampler with rate = {} Hz",
            "✅".green(),
            rate_notif
        );
    }

    // 3. DISTORÇÃO DIGITAL (Clipping):
    // O equivalente ao "LED vermelho" em mesas de som. Indica que o volume do sinal
    // ultrapassou o limite máximo do processamento digital.
    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED) {
        log::warn!(
            "{} Clipping detected! Consider reducing the input and/or output gain.",
            "🔥".bright_red().bold()
        );
    }

    // 4. PRIORIDADE DE TEMPO REAL (Real-Time Priority):
    // Verifica se o Linux permitiu que o NAM-rs rode com "prioridade máxima".
    // Isso impede que outros programas (como o navegador) causem interrupções no áudio.
    let prio = rt_status.rt_priority.load(Ordering::Relaxed);
    if prio != -1 {
        let is_fifo = rt_status.check_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);

        rt_status.rt_priority.store(-1, Ordering::Relaxed);

        if is_fifo {
            log::info!(
                "{} DSP thread confirmed: SCHED_FIFO active, RT priority = {}",
                "✅".green(),
                prio
            );
        } else {
            NamDiagnostic::new(NamErrorCode::SchedFifoDenied, sys)
                .message(format!(
                    "DSP thread is NOT in SCHED_FIFO (priority = {}). \
                     Audio may experience jitter and xruns.",
                    prio
                ))
                .hint(
                    "Check if your user has RT permission (ulimit -r) \
                     or if the system has rtkit/PipeWire configured correctly.",
                )
                .emit_warning();
        }
    }

    // 5. SOBRECARGA DE PROCESSAMENTO (Overloads):
    // Avisa se o processador (CPU) não está sendo rápido o suficiente para calcular
    // a rede neural antes do próximo bloco de áudio ser necessário.
    let overloads = rt_status.dsp_overloads.swap(0, Ordering::Relaxed);
    if overloads > 0 {
        log::warn!(
            "{} CPU overload ({} buffers). Consider using a lighter model or a faster processor.",
            "🚨".red(),
            overloads
        );
    }

    // 6. SINCRONIA DE PLACAS (Clock Drifting):
    // Ocorre quando você usa dispositivos diferentes para entrada e saída (ex: Microfone USB
    // e Fone P2). Se um for levemente mais rápido que o outro, o sistema precisa descartar
    // alguns pequenos pedaços de áudio para manter a sincronia.
    let drops = bridge.drain_dropped_frames();
    if drops > 0 {
        log::warn!(
            "{} Drifting detected: {} audio blocks discarded (capture > playback).",
            "⚠️".yellow(),
            drops
        );
    }

    // 7. TELEMETRIA DE PERFORMANCE (Latência):
    // Mostra estatísticas de quanto tempo a CPU leva para processar cada bloco.
    // - Mediana (P50): O tempo "comum" de processamento.
    // - P99: O pior caso em 99% das vezes (indica estabilidade).
    // - Máx: O maior pico de atraso já registrado.
    let nanos = rt_status.dsp_cycle_time.load(Ordering::Relaxed);
    if nanos > 0 {
        let duration = Duration::from_nanos(nanos);
        static TELEMETRY_THROTTLE: AtomicU32 = AtomicU32::new(0);
        if TELEMETRY_THROTTLE
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_rem(100)
            == 0
        {
            let p50 = rt_status.latency_hist.get_percentile(0.50) / 1000;
            let p99 = rt_status.latency_hist.get_percentile(0.99) / 1000;
            let max = rt_status.latency_hist.get_max() / 1000;
            let total_calls = rt_status.latency_hist.total_count();

            log::info!(
                "{} DSP Telemetry (10s): {}µs (Median) | {}µs (P99) | {}µs (Max) [{} blocks]",
                "📊".bright_blue(),
                p50,
                p99,
                max,
                total_calls
            );
            rt_status.latency_hist.reset();
        }

        // 8. VERIFICAÇÃO DE PRAZO (Deadline):
        // Se o tempo de execução ultrapassar o "orçamento" dado pelo sistema de áudio,
        // geramos um erro de diagnóstico explicando o que falhou.
        let rate_val = rt_status.active_rate.load(Ordering::Relaxed);
        let samples_val = rt_status.last_n_samples.load(Ordering::Relaxed);

        if rate_val > 0 && samples_val > 0 {
            let budget_us = (samples_val as f64 / rate_val as f64) * 1_000_000.0;
            let elapsed_us = duration.as_micros() as f64;

            if elapsed_us > budget_us {
                NamDiagnostic::new(NamErrorCode::DeadlineExceeded, sys)
                    .message("PipeWire deadline exceeded (Possible Xrun detected)")
                    .hint("Verify model topology or reduce system load.")
                    .param("exec_time_us", elapsed_us as u64)
                    .param("budget_us", budget_us as u64)
                    .param("n_samples", samples_val)
                    .param("rate", rate_val)
                    .emit();
            }
        }
    }

    // Detecção de transição de silêncio
    let current_silent = rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_SILENT);

    if current_silent != was_silent {
        if current_silent {
            log::info!(
                "{} Silent Mode: Input below threshold (Gate Closed).",
                "🔇".blue()
            );
        } else {
            log::info!(
                "{} Audio Signal Detected: DSP processing resumed.",
                "🔊".green()
            );
        }
    }

    // Detecção de transição de fading
    let current_fading = rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING);

    if current_fading != was_fading && current_fading {
        log::info!("{} Signal Transition: Gate in Fade-In/Out.", "🌓".yellow());
    }

    (current_silent, current_fading)
}
