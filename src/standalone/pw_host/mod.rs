// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP audio processing core using `pipewire-rs`.
//!
//! This is the "heart" of NAM-rs Standalone mode: the module that processes audio in
//! real time. It receives raw audio samples from PipeWire (the Linux sound
//! server), passes them through the "neural engine", and delivers the processed
//! final result to the hardware via a dual-stream architecture.
//!
//! ## Dual-Stream Architecture with DspBridge
//!
//! PipeWire copies buffers to the monitor port **before** calling `process()`.
//! Therefore, in-place modifications on a single `Audio/Sink` stream would be invisible
//! to the hardware. The solution uses two streams:
//!
//! 1. **Capture stream** (`Audio/Sink`, `Direction::Input`) — acts as a Virtual Sink
//!    that receives audio from apps, applies the DSP chain (gain + neural inference)
//!    and writes the result to [`DspBridge`].
//! 2. **Playback stream** (`Stream/Output/Audio`, `Direction::Output`) — acts as a
//!    playback client that reads from [`DspBridge`] and delivers to hardware.
//!
//! The [`DspBridge`] is a `#[repr(align(128))]` buffer shared between the two
//! closures via raw pointer, with lock-free synchronization via `Ordering::Release/Acquire`
//! and an atomic generation counter.
//!
//! ## Absolute rules of this module (why are they so strict?)
//!
//! In the `process()` callback (the function called hundreds of times per second by PipeWire):
//! - **Zero heap allocation** — we never request new memory from the system during processing.
//! - **Zero I/O** — we never write to the terminal or files; status is reported via atomic flags.
//! - **Zero mutexes** — we never lock/wait for other threads.
//!
//! These rules exist because any pause, no matter how small, would cause clicks and glitches in
//! the audio — unacceptable for a musician playing live.
//!
//! ## Processing flow (Capture callback)
//!
//! The `process()` callback follows this sequence for each audio block:
//! 1. **Noise Gate and Input Gain** — Evaluates signal energy and applies the initial gain (pre-DSP).
//! 2. `NamResampler::process_input()` — Converts sample rate to the compatible rate (usually 48 kHz).
//! 3. **WaveNet/LSTM neural inference** — The neural engine that processes the audio signal.
//! 4. `NamResampler::process_output()` — Converts back to the original host sample rate.
//! 5. **Output Gain and Clipping** — Applies the final volume and detects digital saturation.
//! 6. **Write to [`DspBridge`]** — Publishes the result with `Ordering::Release` to the playback callback.
//!
//! When no model is loaded, the engine operates in **True-Bypass** (the input signal passes clean).
//! When the PipeWire sample rate is the same as the nam model, the resampler operates in bypass without overhead.

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

// Re-exports for test module compatibility (pw_host_test.rs).
#[cfg(test)]
pub(crate) use crate::dsp::pipeline::{BridgeBuffer, DspBridge, MAX_BRIDGE_BUF};

/// Initializes the PipeWire dual-stream topology (Capture + Playback).
///
/// Architecture: Apps → [Capture: Audio/Sink] → process(DSP) → DspBridge → [Playback: Stream/Output] → Hardware.
/// The monitor port of `Audio/Sink` copies the buffer *before* `process()` — therefore, the only
/// way to deliver the processed audio to hardware is via a second playback stream
/// that reads from `DspBridge` post-DSP.
///
/// ## SPSC channel parameters
///
/// - `consumer`: Consumer of the CLI→DSP parameter channel (gain, model, etc.).
/// - `gc_producer`: Producer of the GC channel for drop-delegation of obsolete models.
/// - `resampler_consumer`: Dedicated channel for receiving pre-built resamplers
///   from the main thread — **zero allocations in the RT callback**.
/// - `resampler_producer`: Producer of the resampler channel — the main thread
///   builds `NamResampler::new()` here (allocation outside RT) and sends to the callback.
/// - `rt_status`: Atomic flags for silent RT→Main communication.
#[allow(clippy::too_many_arguments)]
pub fn run_pipewire_host(
    consumer: Consumer<ParamPayload>,
    gc_producer: rtrb::Producer<GcItem>,
    gc_overflow: Arc<GcOverflowBuffer>,
    resampler_consumer: Consumer<Box<NamResampler>>,
    mut resampler_producer: rtrb::Producer<Box<NamResampler>>,
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
    // 1. PIPEWIRE LOOP INITIALIZATION
    // =========================================================
    let thread_loop =
        unsafe { pipewire::thread_loop::ThreadLoopBox::new(Some("nam-rs-loop"), None) }?;
    let context = pipewire::context::ContextBox::new(thread_loop.loop_(), None)?;
    let core = context.connect(None)?;

    // =========================================================
    // 2. DSP BRIDGE ALLOCATION (Lock-Free Communication)
    // =========================================================
    let bridge_ptr = bridge::allocate_dsp_bridge();

    // =========================================================
    // 3. CORE OPTIMIZATION (CPU Affinity)
    // =========================================================
    let target_cpu = rt_setup::select_optimal_cpu().unwrap_or(0);

    // =========================================================
    // 4. PROTECTED CONFIGURATION SCOPE (RAII)
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
    // 5. RT THREAD START (Background)
    // =========================================================
    thread_loop.start();

    // =========================================================
    // 6. MAIN CONTROL LOOP (Main Thread, Non-RT)
    // =========================================================
    let mut was_silent = false;
    let mut was_fading = false;
    while !SHUTDOWN.load(Ordering::Relaxed) {
        let active = rt_status.active_rate.load(Ordering::Relaxed);
        if active != 0 {
            crate::common::diagnostics::ACTIVE_SAMPLE_RATE.store(active, Ordering::Relaxed);
        }

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

                        if resampler_producer.push(Box::new(new_rs)).is_err() {
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

        let drained = crate::common::spsc::drain_gc_channels(&mut gc_consumer, &gc_overflow);
        rt_status
            .drains
            .fetch_add(drained as u32, Ordering::Relaxed);

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
