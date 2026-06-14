// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![cfg(feature = "clap-plugin")]

use clack_host::prelude::*;
use std::env;
use std::path::PathBuf;

struct MyHostShared;
impl SharedHandler<'_> for MyHostShared {
    fn request_restart(&self) {}
    fn request_process(&self) {}
    fn request_callback(&self) {}
}

struct MyHost;
impl HostHandlers for MyHost {
    type Shared<'a> = MyHostShared;
    type MainThread<'a> = ();
    type AudioProcessor<'a> = ();
}

#[test]
fn test_clap_lifecycle() {
    let path = if let Ok(custom_path) = env::var("CLAP_PLUGIN_PATH") {
        PathBuf::from(custom_path)
    } else {
        // Try to load from the default install location used by build-release.sh/tests-cargo.sh
        let mut p = PathBuf::from(env::var("HOME").expect("HOME env var not set"));
        p.push(".clap/nam-rs.clap");

        if !p.exists() {
            // Fallback to the build directory if not installed under HOME
            let current_dir = env::current_dir().expect("Failed to get current dir");
            p = current_dir.join("target/release/libnam_rs.so");
            if !p.exists() {
                p = current_dir.join("target/debug/libnam_rs.so");
                if !p.exists() {
                    p = current_dir.join("target/clap/release/libnam_rs.so");
                    if !p.exists() {
                        p = current_dir.join("target/clap/debug/libnam_rs.so");
                        if !p.exists() {
                            p = current_dir.join("target/clap-test/release/libnam_rs.so");
                            if !p.exists() {
                                p = current_dir.join("target/clap-test/debug/libnam_rs.so");
                            }
                        }
                    }
                }
            }
        }
        p
    };

    if !path.exists() {
        panic!(
            "Plugin binary not found. Set CLAP_PLUGIN_PATH or run build first. Checked path: {:?}",
            path
        );
    }

    // SAFETY: Loading a CLAP plugin is inherently unsafe because it executes code from a dynamic library.
    let entry = unsafe { PluginEntry::load(&path).expect("Failed to load plugin entry") };
    let host_info = HostInfo::new(
        "NAM-rs Test Host",
        "Fabio Lima",
        "https://github.com/fabiohl/nam-rs",
        "0.1.0",
    )
    .unwrap();

    let mut plugin_instance = PluginInstance::<MyHost>::new(
        |_| MyHostShared,
        |_| (),
        &entry,
        c"br.eti.fabiolima.nam-rs",
        &host_info,
    )
    .expect("Failed to create plugin instance");

    let audio_config = PluginAudioConfiguration {
        sample_rate: 48000.0,
        min_frames_count: 512,
        max_frames_count: 512,
    };

    let stopped_processor = plugin_instance
        .activate(|_, _| (), audio_config)
        .expect("Failed to activate plugin");

    let mut started_processor = stopped_processor
        .start_processing()
        .expect("Failed to start processing");

    // Prepare audio buffers (silence)
    let mut input_audio_buffers = [[0.0f32; 512]; 2];
    let mut output_audio_buffers = [[0.0f32; 512]; 2];

    let mut input_ports = AudioPorts::with_capacity(2, 1);
    let mut output_ports = AudioPorts::with_capacity(2, 1);

    let mut output_events_buffer = EventBuffer::with_capacity(10);

    // Simulate 4 processing blocks
    for _ in 0..4 {
        let input_events = InputEvents::empty();
        let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

        let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_audio_buffers.iter_mut().map(InputChannel::constant),
            ),
        }]);

        let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(
                output_audio_buffers.iter_mut().map(|b| b.as_mut_slice()),
            ),
        }]);

        let status = started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .expect("Process failed");

        // Plugin must return Continue or Sleep (currently implemented as Continue for bypass)
        assert!(
            status == ProcessStatus::Continue || status == ProcessStatus::Sleep,
            "Unexpected process status: {:?}",
            status
        );
    }

    let stopped_processor = started_processor.stop_processing();
    plugin_instance.deactivate(stopped_processor);
}
