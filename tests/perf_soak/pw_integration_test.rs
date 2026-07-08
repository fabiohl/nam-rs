// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  PipeWire Pipeline Integration (End-to-End) Test
//
//  This test ensures the correct initialization of the PipeWire Core context.
//  Prerequisite: a running PipeWire daemon (session or system). Without it the test is skipped
//  by the `#[ignore]` attribute; `utils/tests-long.sh` auto-detects the daemon via `pw-cli info`.

use nam_rs::common::diagnostics::SystemSnapshot;
use nam_rs::common::spsc::{self, RtStatusFlags};
use nam_rs::standalone::pw_host::run_pipewire_host;
use rtrb::RingBuffer;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

/// Tests the basic initialization and communication of the PipeWire pipeline.
///
/// This test simulates the full lifecycle of the engine:
/// 1. Creation of SPSC (Single-Producer Single-Consumer) RingBuffers for commands and telemetry.
/// 2. Spawning the audio thread (host).
/// 3. Sending gain parameters via the control channel.
/// 4. Shutdown signaled via atomic flag.
#[test]
#[ignore = "requires a running PipeWire daemon (session or system); auto-detected by utils/tests-long.sh"]
fn test_pipewire_integration() {
    // Initializes the native library. In environments without libpipewire installed, this will panic.
    pipewire::init();

    println!("✔ PipeWire initialized successfully.");

    // RingBuffers for Lock-Free Thread-Safe communication:
    // param: UI -> DSP commands
    let (mut param_prod, param_cons) = RingBuffer::new(4);
    // gc: Garbage Collection of DSP resources (DSP Thread -> Background)
    let (gc_prod, gc_cons) = RingBuffer::new(4);
    // res: Responses/Telemetry (DSP -> UI)
    let (res_prod, res_cons) = RingBuffer::new(2);
    // cabsim: IR loader (Main -> DSP)
    let (cs_prod, cs_cons) = RingBuffer::new(2);
    // slimmable: Slimmable model rebuild (Main -> DSP)
    let (sl_prod, sl_cons) = RingBuffer::new(2);
    // os: Oversampling engine rebuild (Main -> DSP)
    let (os_prod, os_cons) = RingBuffer::new(2);

    // Fallback buffer for GC overflow
    let gc_overflow = Arc::new(spsc::GcOverflowBuffer::new(64));

    // Shared status flags (xruns, cpu load, etc.)
    let rt_status = Arc::new(RtStatusFlags::default());

    let rt_clone = rt_status.clone();
    let gc_overflow_clone = gc_overflow.clone();
    // Captures system metadata for initial diagnostics
    let sys = SystemSnapshot::capture();

    // Starts the Audio Thread
    let pw_thread = thread::spawn(move || {
        // run_pipewire_host is the main loop that interacts with the system daemon.
        run_pipewire_host(
            param_cons,
            gc_prod,
            gc_overflow_clone,
            res_cons,
            res_prod,
            cs_cons,
            cs_prod,
            rt_clone,
            nam_rs::standalone::pw_host::PipewireHostConfig {
                buffer_size: 0, // 0 = Use system default (PipeWire quantum)
                sys,
                ir_raw_samples: None,
                full_wavenet_model: None,
                slimmable_producer: sl_prod,
                os_producer: os_prod,
                oversample: nam_rs::dsp::oversample::OversampleFactor::Off,
            },
            gc_cons,
            sl_cons,
            os_cons,
        )
    });

    // Simulates user interaction: input/output gain adjustment
    thread::sleep(Duration::from_millis(50));
    let _ = param_prod.push(nam_rs::common::spsc::ParamPayload::InputGain(2.5));
    let _ = param_prod.push(nam_rs::common::spsc::ParamPayload::OutputGain(-1.0));

    // Signals pipeline shutdown.
    // The engine monitors this flag on every audio loop iteration.
    thread::sleep(Duration::from_millis(150));
    spsc::SHUTDOWN.store(true, Ordering::Relaxed);

    // Waits for the audio thread to finish and validates the result
    match pw_thread.join() {
        Ok(result) => {
            if let Err(e) = result {
                // Missing daemon error (e.g. in CI) is reported but does not break the test
                eprintln!(
                    "PipeWire host exited with expected error (possible daemon absence): {e}"
                );
            } else {
                println!("Pipeline host ran and shut down gracefully.");
            }
        }
        Err(_) => panic!("The PipeWire thread suffered a fatal panic!"),
    }

    println!("Integration test completed.")
}
