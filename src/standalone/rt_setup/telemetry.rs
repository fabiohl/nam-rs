// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Telemetry and RT status monitoring for the audio engine.
//!
//! Translates atomic signals from the DSP thread into diagnostic logs for
//! the main loop, acting as the "dashboard" of NAM-rs.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::common::spsc::RtStatusFlags;
use crate::standalone::colors::Colorize;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

/// Reads atomic RT status flags and emits monitoring logs to the user.
///
/// This function acts as the "dashboard" of NAM-rs. It is called periodically
/// to translate the technical signals coming from the audio thread (which is silent and ultra-fast)
/// into understandable messages, performance warnings, and latency telemetry.
///
/// Returns a tuple (current_silent, current_fading) for state control in the main loop.
pub fn poll_rt_status(
    rt_status: &RtStatusFlags,
    sys: &SystemSnapshot,
    was_silent: bool,
    was_fading: bool,
    bridge: &crate::dsp::pipeline::DspBridge,
) -> (bool, bool) {
    let current_bits = rt_status.status_bits.load(Ordering::Relaxed);
    rt_status
        .flags_seen
        .fetch_or(current_bits, Ordering::Relaxed);

    // 1. MEMORY MANAGEMENT (Garbage Collection):
    // If the cleanup channel is full, it means we are swapping neural models
    // faster than the system can discard old ones. We prioritize audio
    // "leaking" memory temporarily to avoid clicks (drops) in the sound.
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

    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_GC_CORRUPTED) {
        NamDiagnostic::new(NamErrorCode::GcCorrupted, sys)
            .message("Garbage Collection overflow buffer corruption detected.")
            .hint(
                "A GC slot had inconsistent type/pointer data. The pointer was leaked \
                   to avoid undefined behavior (Box::from_raw with wrong type). \
                   This should never happen — report it.",
            )
            .emit();
    }

    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_SLIMMABLE_SLICE_FAILED) {
        log::error!(
            "WaveNet slimmable slice_channels rebuild failed — model may run in reduced state."
        );
    }

    // 2. RATE CHANGE (Sample Rate):
    // Warns when the audio server (PipeWire) changes the sampling frequency
    // (e.g. changed from 44.100 to 48.000 beats per second).
    let rate_notif = rt_status.active_rate_changed.swap(0, Ordering::Relaxed);
    if rate_notif != 0 {
        log::info!(
            "{} RT callback activated resampler with rate = {} Hz",
            "✅".green(),
            rate_notif
        );
    }

    // 3. DIGITAL DISTORTION (Clipping):
    // The equivalent of the "red LED" on mixing consoles. Indicates that the signal volume
    // exceeded the maximum limit of digital processing.
    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED) {
        log::warn!(
            "{} Clipping detected! Consider reducing the input and/or output gain.",
            "🔥".bright_red().bold()
        );
    }

    // 3.5 HUGE PAGE STATUS:
    // Sync from mirror buffer global and log once.
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static HUGEPAGE_SYNCED: AtomicBool = AtomicBool::new(false);
        if !HUGEPAGE_SYNCED.load(Ordering::Relaxed) {
            crate::dsp::mirror_buf::sync_huge_page_flag(rt_status);
            HUGEPAGE_SYNCED.store(true, Ordering::Relaxed);
        }
    }
    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_HUGEPAGE_OK) {
        log::info!(
            "{} Huge pages (2MB) active — reduced TLB pressure on DSP thread.",
            "✅".green()
        );
    }

    // 4. REAL-TIME PRIORITY:
    // Checks whether Linux allowed NAM-rs to run with "maximum priority".
    // This prevents other programs (like the browser) from causing audio interruptions.
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

    // 5. PROCESSING OVERLOAD (Overloads):
    // Warns if the processor (CPU) is not fast enough to compute
    // the neural network before the next audio block is needed.
    let overloads = rt_status.dsp_overloads.swap(0, Ordering::Relaxed);
    if overloads > 0 {
        rt_status.xruns.fetch_add(overloads, Ordering::Relaxed);
        log::warn!(
            "{} CPU overload ({} buffers). Consider using a lighter model or a faster processor.",
            "🚨".red(),
            overloads
        );
    }

    // 6. CLOCK DRIFTING:
    // Occurs when you use different devices for input and output (e.g. USB Microphone
    // and P2 Headphones). If one is slightly faster than the other, the system needs to discard
    // some small audio chunks to maintain synchronization.
    let drops = bridge.drain_dropped_frames();
    if drops > 0 {
        log::warn!(
            "{} Drifting detected: {} audio blocks discarded (capture > playback).",
            "⚠️".yellow(),
            drops
        );
    }

    // 7. PERFORMANCE TELEMETRY (Latency):
    // Shows statistics of how long the CPU takes to process each block.
    // - Median (P50): The "common" processing time.
    // - P99: The worst case 99% of the time (indicates stability).
    // - Max: The highest delay peak ever recorded.
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
            let exact = rt_status.latency_hist.take_exact_max() / 1000;
            let total_calls = rt_status.latency_hist.total_count();

            log::info!(
                "{} DSP Telemetry (10s): {}µs (Median) | {}µs (P99) | {}µs (Exact Max) [{} blocks]",
                "📊".bright_blue(),
                p50,
                p99,
                exact,
                total_calls
            );
            rt_status.latency_hist.reset();
        }

        // 8. DEADLINE CHECK:
        // If execution time exceeds the "budget" given by the audio system,
        // we generate a diagnostic error explaining what failed.
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

    // Silence transition detection
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

    // Fading transition detection
    let current_fading = rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING);

    if current_fading != was_fading && current_fading {
        log::info!("{} Signal Transition: Gate in Fade-In/Out.", "🌓".yellow());
    }

    (current_silent, current_fading)
}
