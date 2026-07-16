// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! PipeWire host execution — dual-stream topology setup, DSP bridge allocation,
//! CPU affinity locking, main control loop, and graceful shutdown.

use crate::common::spsc::{self, GcItem, GcOverflowBuffer, ParamPayload, RtStatusFlags, SHUTDOWN};
use crate::dsp::pipeline::AppState;
use crate::dsp::resampler::NamResampler;
use crate::models::StaticModel;
use crate::models::slimmable::slice_wavenet_model;
use crate::standalone::colors::Colorize;
use crate::standalone::rt_setup;

use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::PipewireHostConfig;
use super::bridge;
use super::capture;
use super::playback;

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
///   builds `NamResampler::new().expect("construction should succeed for test-sized buffers")` here (allocation outside RT) and sends to the callback.
/// - `rt_status`: Atomic flags for silent RT→Main communication.
#[allow(clippy::too_many_arguments)]
pub fn run_pipewire_host(
    consumer: Consumer<ParamPayload>,
    gc_producer: rtrb::Producer<GcItem>,
    gc_overflow: Arc<GcOverflowBuffer>,
    resampler_consumer: Consumer<Box<NamResampler>>,
    mut resampler_producer: rtrb::Producer<Box<NamResampler>>,
    cabsim_consumer: Consumer<Option<Box<crate::dsp::cabsim::conv::ConvEngine>>>,
    mut cabsim_producer: rtrb::Producer<Option<Box<crate::dsp::cabsim::conv::ConvEngine>>>,
    rt_status: Arc<RtStatusFlags>,
    config: PipewireHostConfig,
    mut gc_consumer: Consumer<GcItem>,
    slimmable_consumer: Consumer<Option<Box<StaticModel>>>,
    os_consumer: Consumer<Box<crate::dsp::oversample::OsEnginePair>>,
) -> anyhow::Result<()> {
    let PipewireHostConfig {
        buffer_size,
        sys,
        ir_raw_samples,
        full_wavenet_model,
        mut slimmable_producer,
        mut os_producer,
        oversample,
    } = config;

    let full_wavenet_model = full_wavenet_model;

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
            ir_raw_samples.clone(),
            &sys,
            target_cpu,
            consumer,
            gc_producer,
            gc_overflow.clone(),
            resampler_consumer,
            cabsim_consumer,
            rt_status.clone(),
            slimmable_consumer,
            os_consumer,
            oversample,
        )?;
        capture_stream = cs;
        capture_listener = cl;

        let (ps, pl) = playback::setup_playback_stream(
            &core,
            bridge_ptr,
            buffer_size,
            &latency_str,
            rt_status.clone(),
        )?;
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
    while !SHUTDOWN.load(Ordering::Acquire) {
        // pairs with Release store em main.rs:90
        let active = rt_status.active_rate.load(Ordering::Relaxed);
        if active != 0 {
            crate::common::diagnostics::ACTIVE_SAMPLE_RATE.store(active, Ordering::Relaxed);
        }

        if rt_status.check_flag_acquire(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD) {
            let target_pw_rate = rt_status.requested_pw_rate.load(Ordering::Relaxed);
            let target_nam_rate = rt_status.requested_nam_rate.load(Ordering::Relaxed);

            if target_pw_rate != 0 && target_nam_rate != 0 {
                match NamResampler::new(target_pw_rate, target_nam_rate, 2048) {
                    Ok(new_rs) => {
                        rt_status.clear_flag_relaxed(
                            crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED,
                        );

                        log::info!(
                            "{} Sample rate updated: PW={} Hz, NAM={} Hz (bypass={})",
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
                        .param("detail", e)
                        .emit();

                        rt_status.set_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED);
                    }
                }
                rt_status
                    .clear_flag_relaxed(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD);
            }
        }

        if rt_status.check_flag_acquire(crate::common::spsc::RT_STATUS_NEEDS_CABSIM_REBUILD) {
            let partition_size = rt_status
                .requested_cabsim_partition_size
                .load(Ordering::Relaxed) as usize;
            if partition_size > 0 {
                if let Some(ref samples) = ir_raw_samples {
                    use crate::dsp::cabsim::conv::ConvEngine;
                    let engine = ConvEngine::new(samples, partition_size)
                        .map_err(|e| anyhow::anyhow!("Cab-sim engine: {e}"))?;
                    log::info!(
                        "{} Cab-sim IR rebuilt: partition_size={} ({} partitions, FFT={})",
                        "🔄".cyan(),
                        partition_size,
                        engine.num_partitions(),
                        engine.fft_size(),
                    );
                    if cabsim_producer.push(Some(Box::new(engine))).is_err() {
                        crate::common::diagnostics::NamDiagnostic::new(
                            crate::common::diagnostics::NamErrorCode::ParamChannelFull,
                            &sys,
                        )
                        .message("Cab-sim rebuild channel full. Rebuild discarded.")
                        .hint("The audio engine is overloaded. If the problem persists, restart NAM-rs.")
                        .param("partition_size", partition_size)
                        .emit_warning();
                    }
                }
                rt_status.clear_flag_relaxed(crate::common::spsc::RT_STATUS_NEEDS_CABSIM_REBUILD);
            }
        }

        // WaveNet slimmable rebuild: main thread performs all allocation,
        // prewarm, and mmap outside the audio-thread callback.
        if rt_status.check_flag_acquire(spsc::RT_STATUS_NEEDS_SLIMMABLE_REBUILD) {
            let target_ch = rt_status.requested_slimmable_ch.load(Ordering::Relaxed) as usize;
            if target_ch >= 4
                && let Some(ref m) = full_wavenet_model
                && let StaticModel::WavenetDyn(w) = m.as_ref()
            {
                // Build L channel model
                match slice_wavenet_model(w, target_ch) {
                    Ok(mut slimmed) => {
                        slimmed.prewarm();
                        let model_l = Box::new(StaticModel::WavenetDyn(Box::new(slimmed)));
                        let _ = slimmable_producer.push(Some(model_l));
                    }
                    Err(_) => {
                        rt_status.set_flag(spsc::RT_STATUS_SLIMMABLE_SLICE_FAILED);
                    }
                }
                // Build R channel model (same weights, same target_ch)
                match slice_wavenet_model(w, target_ch) {
                    Ok(mut slimmed) => {
                        slimmed.prewarm();
                        let model_r = Box::new(StaticModel::WavenetDyn(Box::new(slimmed)));
                        let _ = slimmable_producer.push(Some(model_r));
                    }
                    Err(_) => {
                        rt_status.set_flag(spsc::RT_STATUS_SLIMMABLE_SLICE_FAILED);
                    }
                }
            }
            rt_status.clear_flag_relaxed(spsc::RT_STATUS_NEEDS_SLIMMABLE_REBUILD);
        }

        if rt_status.check_flag_acquire(spsc::RT_STATUS_NEEDS_OS_REBUILD) {
            let factor_val = rt_status.requested_os_factor.load(Ordering::Relaxed);
            let factor = crate::dsp::oversample::OversampleFactor::from_f32(factor_val as f32);
            match (
                crate::dsp::oversample::OversampleEngine::new(
                    factor,
                    crate::dsp::pipeline::MAX_RESAMP_BUF,
                ),
                crate::dsp::oversample::OversampleEngine::new(
                    factor,
                    crate::dsp::pipeline::MAX_RESAMP_BUF,
                ),
            ) {
                (Ok(os_l), Ok(os_r)) => {
                    let pair = Box::new(crate::dsp::oversample::OsEnginePair {
                        l: Box::new(os_l),
                        r: Box::new(os_r),
                    });
                    log::info!(
                        "{} Oversampling factor changed to {:?}",
                        "🔄".cyan(),
                        factor,
                    );
                    if os_producer.push(pair).is_err() {
                        crate::common::diagnostics::NamDiagnostic::new(
                            crate::common::diagnostics::NamErrorCode::ParamChannelFull,
                            &sys,
                        )
                        .message("OS engine channel full. Rebuild discarded.")
                        .hint("The audio engine is overloaded. If the problem persists, restart NAM-rs.")
                        .emit_warning();
                    } else {
                        rt_status.clear_flag_relaxed(spsc::RT_STATUS_NEEDS_OS_REBUILD);
                    }
                }
                (Err(e), _) | (_, Err(e)) => {
                    crate::common::diagnostics::NamDiagnostic::new(
                        crate::common::diagnostics::NamErrorCode::OutOfMemory,
                        &sys,
                    )
                    .message("Failed to rebuild oversample engine (OOM).")
                    .hint("Audio will continue with the previous oversampling state.")
                    .param("detail", e)
                    .emit();
                }
            }
        }

        (was_silent, was_fading) =
            rt_setup::poll_rt_status(&rt_status, &sys, was_silent, was_fading, unsafe {
                &*(bridge_ptr.as_ptr())
            });

        let drained =
            crate::common::spsc::drain_gc_channels(&mut gc_consumer, &gc_overflow, &rt_status);
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
