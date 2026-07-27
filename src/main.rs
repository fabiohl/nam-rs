// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![warn(missing_docs)]

//! Main entry point for the NAM-rs CLI/Standalone binary.
//!
//! **NOTE:** This file is only used for executable mode (terminal/PipeWire).
//! The CLAP plugin ignores this file and directly uses the library defined in `lib.rs`.
//!
//! Think of this file as the "reception" of our virtual studio. It is responsible for:
//! 1. Reading what the user types in the terminal (which amp to load and the input/output volumes).
//! 2. Opening the audio connection to the system (PipeWire), connecting the audio signal to the sound engine.
//! 3. Ensuring that when the user presses CTRL+C, everything shuts down safely, without leaving noise.
//!
//! # Architecture Rules for Developers
//! - **ZERO LOCKS** in the Audio thread (`pw_host` module): Audio does not "wait" for the visual interface. If there is no new instruction, it continues using the previous one. This avoids sound "glitching".
//! - **ZERO ALLOCATIONS** in the Audio thread: The audio channel memory (`process()`) is always prepared 100% in advance. Audio never "requests more RAM" out of nowhere.

use nam_rs::diagnostics::{SystemSnapshot, logger::NamLogger};
use nam_rs::dsp::cabsim::adapter::CabSimAdapter;
use nam_rs::dsp::cabsim::conv::ConvEngine;
use nam_rs::dsp::cabsim::loader::CabSimIr;
use nam_rs::math::activations::set_activation_tls;
use nam_rs::models::StaticModel;
use nam_rs::models::slimmable::clone_wavenet_for_slimmable_storage;
use nam_rs::standalone::{cli, colors::Colorize, pw_host, rt_setup};
use nam_rs::{loader, spsc, spsc::ParamPayload};

use std::sync::atomic::Ordering;

/// Entry point for NAM-rs.
fn main() -> anyhow::Result<()> {
    // Install panic hook to capture crash diagnostics
    nam_rs::common::panic_hook::install_panic_hook("standalone");

    // Initialize the NamLogger backend (respects RUST_LOG/NAM_LOG_LEVEL; default: info)
    let level_filter = std::env::var("RUST_LOG")
        .or_else(|_| std::env::var("NAM_LOG_LEVEL"))
        .unwrap_or_else(|_| "info".to_string())
        .parse::<log::LevelFilter>()
        .unwrap_or(log::LevelFilter::Info);
    NamLogger::init_standalone(level_filter).expect("Failed to initialize NamLogger backend");

    #[cfg(feature = "testing")]
    if std::env::var("NAM_DISABLE_GATE").is_ok() {
        nam_rs::dsp::pipeline::DISABLE_GATE.store(true, Ordering::Relaxed);
        log::info!("⚡ Noise gate disabled via NAM_DISABLE_GATE environment variable.");
    }

    // 1. READ CONFIGURATIONS: The system starts by reading what you typed in the terminal.
    // It figures out which "amplifier" file (.nam) you want to use and the initial volumes.
    let args = cli::parse_args();

    // 1.1. IMMEDIATE DIAGNOSTIC EXITS: If the user requested an immediate diagnostic dump,
    // we print it to stdout and exit immediately with code 0 (without initializing audio).
    if args.diagnose || args.diagnose_full {
        let bundle = nam_rs::diagnostics::DiagnosticBundle::capture().with_full(args.diagnose_full);
        println!("{}", bundle.render());
        std::process::exit(0);
    }

    let model_path = args.model_path;
    let initial_in_gain = args.input_gain;
    let initial_out_gain = args.output_gain;
    let buffer_size = args.buffer_size;

    // 2. KNOW THE COMPUTER: Captures a "snapshot" of your processor's capabilities.
    // This helps NAM-rs choose the fastest way to process the audio math.
    let sys = SystemSnapshot::capture();
    log::info!(
        "🎸 {}",
        format!(
            "NAM-rs Standalone v{} [x86-64-v3] — Neural Amp Modeler",
            sys.version
        )
        .bright_green()
        .bold()
    );

    // 3. PREPARE THE AUDIO: Initialize PipeWire (the Linux sound system)
    // and calibrate the internal "clocks" to ensure sound output without delays (latency).
    pipewire::init();
    rt_setup::calibrate_tsc();

    // 4. EMERGENCY BUTTON: Configures "Ctrl+C".
    // If you want to close the program, it ensures the audio stops smoothly, without clicks.
    extern "C" fn sigint_handler(_sig: libc::c_int) {
        if spsc::SHUTDOWN.load(Ordering::Acquire) {
            // pairs with Release store em linha 90
            // Segundo Ctrl-C: o graceful shutdown não respondeu a tempo.
            // `_exit(1)` encerra o processo sem rodar destrutores.
            // O kernel recolhe TODOS os recursos abertos (fds, mapeamentos, PM QoS,
            // THP advice) — nada persiste após o exit do processo (R3, verificado
            // via /proc/<pid>/fd e pm_qos_constraint após o exit).
            // Preferir `_exit` a `abort` para evitar core dump desnecessário;
            // usar `abort()` apenas se core dump for desejado para diagnóstico.
            unsafe { libc::_exit(1) };
        }
        spsc::SHUTDOWN.store(true, Ordering::Release); // pairs with Acquire loads em panic_hook.rs:30, run.rs:141
    }
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        // SAFETY: sigint_handler has the 1-arg signature expected by the kernel
        // when SA_SIGINFO is not set; SA_RESTART alone triggers the 1-arg handler
        // path. The cast to sighandler_t (usize) is a no-op ABI-compatible
        // conversion — both `extern "C" fn(i32)` and `sighandler_t` (usize) are
        // pointer-sized integers on this target.
        sa.sa_sigaction = sigint_handler as *const () as libc::sighandler_t;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
    }

    // 5. COMMUNICATION CHANNELS: Creates ultra-fast "pipes" for communication.
    // Imagine the Interface (you) and the Sound Engine (audio) live in separate rooms.
    // These channels allow sending commands (e.g. "increase volume") without ever making the sound "glitch".
    let channels = spsc::setup_spsc(spsc::SPSC_CAPACITY);
    let mut producer = channels.param_producer;
    let consumer = channels.param_consumer;
    let gc_producer = channels.gc_producer;
    let gc_consumer = channels.gc_consumer;
    let gc_overflow = channels.gc_overflow;
    let resampler_producer = channels.resampler_producer;
    let resampler_consumer = channels.resampler_consumer;
    let mut cabsim_producer = channels.cabsim_producer;
    let cabsim_consumer = channels.cabsim_consumer;
    let slimmable_producer = channels.slimmable_producer;
    let slimmable_consumer = channels.slimmable_consumer;
    let os_producer = channels.os_producer;
    let os_consumer = channels.os_consumer;
    let rt_status = channels.rt_status;

    // 6. LOAD THE CAB-SIM IR: If you said "use cabinet X",
    // this is where the computer opens that WAV file and builds the convolution engine.
    let mut ir_raw_samples = None;
    if let Some(ref cab_path) = args.cab_path {
        let target_rate = nam_rs::diagnostics::ACTIVE_SAMPLE_RATE
            .load(Ordering::Relaxed)
            .max(48000);
        let partition_size = if buffer_size > 0 {
            buffer_size as usize
        } else {
            256
        };
        match CabSimIr::load(cab_path, target_rate, true) {
            Ok(cabsim) => {
                let engine = ConvEngine::new(&cabsim.samples, partition_size)
                    .map_err(|e| anyhow::anyhow!("Cab-sim engine init: {e}"))?;
                let adapter = CabSimAdapter::new(Box::new(engine))
                    .map_err(|e| anyhow::anyhow!("Cab-sim adapter init: {e:?}"))?;
                log::info!(
                    "{} Cab-sim IR loaded: {} ({} partitions, FFT={})",
                    "🎛️".cyan(),
                    cab_path.display(),
                    adapter.num_partitions(),
                    adapter.engine().fft_size(),
                );
                ir_raw_samples = Some(cabsim.samples);
                let _ = cabsim_producer.push(Some(adapter));
            }
            Err(e) => {
                log::warn!(
                    "{} Cab-sim IR load failed: {} — continuing without cab-sim",
                    "⚠️".yellow(),
                    e
                );
            }
        }
    }
    let mut full_wavenet_model: Option<Box<StaticModel>> = None;
    let mut model_architecture = String::new();
    if let Some(ref path) = model_path {
        log::info!("{} Loading model...", "📂".cyan());
        match loader::load_and_build_model(path, &sys, true, loader::LoadOptions::default()) {
            Ok(loaded) => {
                // Populate active model path and sample rate
                if let Ok(mut name) = nam_rs::diagnostics::ACTIVE_MODEL_NAME.write() {
                    *name = path.to_string_lossy().into_owned();
                }
                nam_rs::diagnostics::ACTIVE_SAMPLE_RATE
                    .store(loaded.sample_rate, Ordering::Relaxed);

                model_architecture = loaded.architecture.clone();

                let model_info = loaded.model_info(path);
                if let Ok(mut info_guard) = nam_rs::diagnostics::ACTIVE_MODEL_INFO.write() {
                    *info_guard = Some(model_info);
                }

                full_wavenet_model = loaded.model_l.as_ref().and_then(|m| {
                    if let StaticModel::WavenetDyn(w) = m.as_ref() {
                        clone_wavenet_for_slimmable_storage(w).ok()
                    } else {
                        None
                    }
                });

                let _ = producer.push(ParamPayload::LoadModel {
                    model_l: loaded.model_l,
                    model_r: loaded.model_r,
                    input_mult_adj: loaded.input_mult_adj,
                    output_mult_adj: loaded.output_mult_adj,
                    sample_rate: loaded.sample_rate,
                });
            }
            Err(e) => cli::exit_with_error(format!("Model load failed: {}", e)),
        }
    }

    // Initial gains
    if initial_in_gain != 0.0 {
        let _ = producer.push(ParamPayload::InputGain(
            nam_rs::math::dsp::gain_lut::get_gain_lut().db_to_linear(initial_in_gain),
        ));
    }
    if initial_out_gain != 0.0 {
        let _ = producer.push(ParamPayload::OutputGain(
            nam_rs::math::dsp::gain_lut::get_gain_lut().db_to_linear(initial_out_gain),
        ));
    }

    let _ = producer.push(ParamPayload::SlimOverride(args.slim_override));
    let _ = producer.push(ParamPayload::SetOversample(args.oversample));

    // Apply activation precision mode before any audio processing if explicitly overridden by the user
    if let Some(activation) = args.activation {
        set_activation_tls(activation);
        log::info!(
            "{} Activation precision explicitly set to {:?}",
            "⚡".yellow(),
            activation
        );

        if activation == nam_rs::math::activations::ActivationPrecision::Fast
            && model_architecture.eq_ignore_ascii_case("LSTM")
        {
            log::warn!(
                "{} Fast activation (Padé) with LSTM architecture is NOT recommended — \
                 measured degradation is ~−13 dB ESR (clearly audible). \
                 Standard (exact-grade) activation is the universal default and costs only \
                 +10–15% CPU for LSTM models. See docs/audio_fidelity_map.md §2.",
                "⚠️".yellow(),
            );
        }
    }

    // Process-wide settings (THP disable + mlockall) before starting PipeWire.
    // Executed here (outside the cold-path of the first DSP frame) to avoid
    // syscalls that would cause jitter at the critical moment of the first audio delivery.
    rt_setup::configure_process_wide();

    // Run the PipeWire host (blocking)
    let res = pw_host::run_pipewire_host(
        consumer,
        gc_producer,
        gc_overflow,
        resampler_consumer,
        resampler_producer,
        cabsim_consumer,
        cabsim_producer,
        rt_status,
        pw_host::PipewireHostConfig {
            buffer_size,
            sys,
            ir_raw_samples,
            full_wavenet_model,
            slimmable_producer,
            os_producer,
            oversample: args.oversample,
        },
        gc_consumer,
        slimmable_consumer,
        os_consumer,
    );

    // Signal shutdown to bypass panic hook during cleanup
    nam_rs::common::panic_hook::set_shutdown_in_progress();

    unsafe {
        pipewire::deinit();
    }
    res?;
    Ok(())
}
