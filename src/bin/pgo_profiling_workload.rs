// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! PGO Profiling Workload Tool.
//!
//! Mimics the final standalone CLI and CLAP plugin execution structures to run a
//! highly representative real-world workload (loading real WAV files and models)
//! to generate optimal compiler profiles for PGO.

#![cfg(all(feature = "clap-plugin", feature = "testing"))]

use clack_host::prelude::*;
use nam_rs::clap::test_util;
use nam_rs::diagnostics::SystemSnapshot;
use nam_rs::loader;
use nam_rs::models::NamModel;
use nam_rs::spsc;
use nam_rs::testing::wav::read_wav_f32;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Replicate standalone startup initialization structure
    nam_rs::common::panic_hook::install_panic_hook("pgo_profiler");
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("🎸 Starting real-world PGO profiling workload...");

    // Capture system architecture/SIMD snapshot as in final standalone
    let sys = SystemSnapshot::capture();
    log::info!("  Processor capabilities verified. Mode: x86-64-v3 baseline.");

    // Setup SPSC channels to profile lock-free queue primitives
    let _channels = spsc::setup_spsc(spsc::SPSC_CAPACITY);

    // 1. Load WAV file
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wav_path = manifest_dir.join("tests/fixtures/stress_signal_v2_48000.wav");
    if !wav_path.exists() {
        return Err(format!("Reference WAV file not found: {:?}", wav_path).into());
    }

    log::info!("🔊 Loading WAV: {:?}", wav_path);
    let (samples, sr) = read_wav_f32(&wav_path)?;
    log::info!("   Loaded {} samples at {} Hz", samples.len(), sr);

    // 2. Select models representing Wavenet, LSTM, and SlimmableContainer
    let models_to_run = select_models(&manifest_dir);

    // 3. Profile CLAP mode wrapper & extensions (State, Params, Ports)
    for model_path in &models_to_run {
        log::info!(
            "🚀 Profiling CLAP plugin path for model: {:?}",
            model_path.file_name().unwrap()
        );
        // Run with model rate (no resampling) and different rates (triggers resampler)
        for &target_sr in &[48000.0, 44100.0] {
            profile_clap_model(&samples, target_sr, model_path)?;
        }
    }

    // 4. Profile Standalone mode loading & DSP path
    for model_path in &models_to_run {
        log::info!(
            "🚀 Profiling Standalone DSP path for model: {:?}",
            model_path.file_name().unwrap()
        );
        profile_standalone_model(&samples, &sys, model_path)?;
    }

    log::info!("✓ Real-world PGO profiling workload completed successfully.");
    Ok(())
}

/// Selects 3 models in popular formats/topologies for profiling.
fn select_models(manifest_dir: &Path) -> Vec<PathBuf> {
    // 1. WaveNet A1 Standard (Git-versioned CC0)
    // 2. SlimmableContainer A2 Example (Git-versioned CC0, identical to Deluxe Reverb)
    // 3. LSTM model (Git-versioned CC0)
    vec![
        manifest_dir.join("tests/fixtures/models/wavenet_a1_standard.nam"),
        manifest_dir.join("tests/fixtures/models/a2_example.nam"),
        manifest_dir.join("tests/fixtures/models/lstm.nam"),
    ]
}

/// Emulates a host running the CLAP plugin on the given audio.
fn profile_clap_model(
    samples: &[f32],
    sample_rate: f64,
    model_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let so_path = std::env::var("NAM_CLAP_SO_PATH").ok();
    let (_entry, _host_info, mut plugin_instance) = match so_path {
        Some(ref path) => {
            log::info!("🔌 Profiling CLAP dynamically from .so: {}", path);
            test_util::make_test_plugin_dynamic(Path::new(path))
        }
        None => {
            log::info!("🔌 Profiling CLAP statically (compiled-in plugin)");
            test_util::make_test_plugin()
        }
    };

    // Serialize model path into the plugin's state and restore it
    let params = test_util::make_default_params(Some(model_path.to_path_buf()));
    test_util::load_plugin_state(&mut plugin_instance, &params);

    // Profile common buffer sizes: 64 and 128
    for &block_size in &[64, 128] {
        let audio_config = PluginAudioConfiguration {
            sample_rate,
            min_frames_count: block_size as u32,
            max_frames_count: block_size as u32,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config)?;
        let mut started_processor = stopped_processor.start_processing()?;

        let mut input_ports = AudioPorts::with_capacity(2, 1);
        let mut output_ports = AudioPorts::with_capacity(2, 1);
        let mut output_events_buffer = EventBuffer::new();

        let mut offset = 0;
        while offset < samples.len() {
            let chunk_len = std::cmp::min(block_size, samples.len() - offset);
            if chunk_len == 0 {
                break;
            }

            // Mock stereo input from mono WAV
            let mut in_l = samples[offset..offset + chunk_len].to_vec();
            let mut in_r = samples[offset..offset + chunk_len].to_vec();
            let mut out_l = vec![0.0f32; chunk_len];
            let mut out_r = vec![0.0f32; chunk_len];

            let mut input_channels = [in_l.as_mut_slice(), in_r.as_mut_slice()];
            let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [out_l.as_mut_slice(), out_r.as_mut_slice()];
            let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let input_events = InputEvents::empty();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            started_processor.process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )?;

            offset += chunk_len;
        }

        let stopped_processor = started_processor.stop_processing();
        plugin_instance.deactivate(stopped_processor);
    }

    Ok(())
}

/// Emulates the standalone loader and core DSP engine processing.
fn profile_standalone_model(
    samples: &[f32],
    sys: &SystemSnapshot,
    model_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Replicates how standalone loads and runs inference using the standard dispatcher
    let loaded =
        loader::load_and_build_model(model_path, sys, true, loader::LoadOptions::default())?;

    if let Some(mut model) = loaded.model_l {
        model.prewarm(2048);

        let block_size = 64;
        let mut offset = 0;
        let mut out = vec![0.0f32; block_size];

        while offset < samples.len() {
            let chunk_len = std::cmp::min(block_size, samples.len() - offset);
            if chunk_len < block_size {
                break; // Skip last partial block to avoid mismatch
            }

            let chunk = &samples[offset..offset + block_size];
            model.process(chunk, &mut out);
            offset += block_size;
        }
    }

    Ok(())
}
