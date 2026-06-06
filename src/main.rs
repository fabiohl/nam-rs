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

use nam_rs::diagnostics::SystemSnapshot;
use nam_rs::standalone::{cli, colors::Colorize, pw_host, rt_setup};
use nam_rs::{loader, spsc, spsc::ParamPayload};

use std::sync::atomic::Ordering;

/// Entry point for NAM-rs.
fn main() -> anyhow::Result<()> {
    // Initialize the logging backend (respects RUST_LOG; default: info)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    if std::env::var("NAM_DISABLE_GATE").is_ok() {
        nam_rs::dsp::pipeline::DISABLE_GATE.store(true, Ordering::Relaxed);
        log::info!("⚡ Noise gate disabled via NAM_DISABLE_GATE environment variable.");
    }

    // 1. READ CONFIGURATIONS: The system starts by reading what you typed in the terminal.
    // It figures out which "amplifier" file (.nam) you want to use and the initial volumes.
    let (model_path, initial_in_gain, initial_out_gain, buffer_size) = cli::parse_args();

    // 2. KNOW THE COMPUTER: Captures a "snapshot" of your processor's capabilities.
    // This helps NAM-rs choose the fastest way to process the audio math.
    let sys = SystemSnapshot::capture();
    log::info!(
        "🎸 {}",
        format!("NAM-rs Standalone v{} — Neural Amp Modeler", sys.version)
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
        if spsc::SHUTDOWN.load(Ordering::SeqCst) {
            unsafe { libc::_exit(1) };
        }
        spsc::SHUTDOWN.store(true, Ordering::SeqCst);
    }
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sigint_handler as *const () as libc::sighandler_t;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
    }

    // 5. COMMUNICATION CHANNELS: Creates ultra-fast "pipes" for communication.
    // Imagine the Interface (you) and the Sound Engine (audio) live in separate rooms.
    // These channels allow sending commands (e.g. "increase volume") without ever making the sound "glitch".
    let channels = spsc::setup_spsc(64);
    let mut producer = channels.param_producer;
    let consumer = channels.param_consumer;
    let gc_producer = channels.gc_producer;
    let gc_consumer = channels.gc_consumer;
    let gc_overflow = channels.gc_overflow;
    let resampler_producer = channels.resampler_producer;
    let resampler_consumer = channels.resampler_consumer;
    let rt_status = channels.rt_status;

    // 6. LOAD THE SOUND: If you said "use amplifier X",
    // this is where the computer opens that file and prepares the math (Neural Networks).
    if let Some(path) = model_path {
        log::info!("{} Loading model...", "📂".cyan());
        match loader::load_and_build_model(&path, &sys) {
            Ok(nam_rs::loader::LoadedModelPair {
                model_l,
                model_r,
                input_mult_adj,
                output_mult_adj,
                sample_rate,
                ..
            }) => {
                let _ = producer.push(ParamPayload::LoadModel {
                    model_l,
                    model_r,
                    input_mult_adj,
                    output_mult_adj,
                    sample_rate,
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

    let tsc_anchor = minstant::Anchor::new();

    // Process-wide settings (THP disable + mlockall) before starting PipeWire.
    // Executed here (outside the cold-path of the first DSP frame) to avoid
    // syscalls that would cause jitter at the critical moment of the first audio delivery.
    rt_setup::configure_process_wide();

    // Run the PipeWire host (blocking)
    pw_host::run_pipewire_host(
        consumer,
        gc_producer,
        gc_overflow,
        resampler_consumer,
        resampler_producer,
        rt_status,
        pw_host::PipewireHostConfig {
            buffer_size,
            tsc_anchor,
            sys,
        },
        gc_consumer,
    )?;

    unsafe {
        pipewire::deinit();
    }
    Ok(())
}
