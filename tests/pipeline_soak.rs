// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pipeline Integration Soak Test (Numerical, Telemetry, and RSS Stability).
//!
//! Evaluates the full DSP pipeline context (Capture -> DSP -> Bridge -> Playback)
//! under prolonged load of 10M frames with alternating white noise and silence.
//!
//! Run manually via: `cargo test --release --features standalone --test pipeline_soak -- --ignored --nocapture`

mod common;

#[cfg(feature = "standalone")]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
    use std::time::Instant;

    use nam_rs::common::params::AdaptiveComputeMode;
    use nam_rs::common::spsc::RtStatusFlags;
    use nam_rs::dsp::adaptive::AdaptiveCompute;
    use nam_rs::dsp::gate::{DynamicHysteresis, GateParams};
    use nam_rs::dsp::pipeline::{
        BridgeBuffer, DspBridge, DspBridgeWriter, DspBuffers, DspPipelineContext, MAX_BRIDGE_BUF,
        MAX_RESAMP_BUF, capture_dsp_pipeline,
    };
    use nam_rs::dsp::resampler::NamResampler;
    use nam_rs::loader::dispatcher::build_model;
    use nam_rs::loader::nam_json::parse_nam_json;
    use nam_rs::models::NamModel;

    use super::common::model_path;

    /// Simple deterministic LCG PRNG for reproducible white noise.
    struct SimplePcg {
        state: u64,
    }

    impl SimplePcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f32(&mut self) -> f32 {
            let old_state = self.state;
            self.state = old_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
            let rot = (old_state >> 59) as u32;
            let res = (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31));
            (res as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// Read Resident Set Size (RSS) in bytes on Linux by parsing `/proc/self/status`.
    fn get_rss_bytes() -> usize {
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(kb) = parts.get(1).and_then(|part| part.parse::<usize>().ok()) {
                        return kb * 1024;
                    }
                }
            }
        }
        0
    }

    fn run_pipeline_soak_core(
        model_file: &str,
        total_frames: usize,
        budget_mult: f64,
        label: &str,
    ) {
        let path = model_path(model_file);
        if !path.exists() {
            eprintln!("SKIP: {} not found at {:?}", model_file, path);
            return;
        }

        println!("--- {} Pipeline Soak ({} frames) ---", label, total_frames);

        let json_data = fs::read_to_string(&path).expect("Failed to read model file");
        let model_data = parse_nam_json(&json_data).expect("Failed to parse JSON");
        let mut model_l = build_model(&model_data).expect("Failed to build model (L)");
        let mut model_r = build_model(&model_data).expect("Failed to build model (R)");

        model_l.prewarm(2048);
        model_r.prewarm(2048);

        let input_sr = 96000;
        let output_sr = 48000;
        let block_size = 64;

        let mut resampler = NamResampler::new(input_sr, output_sr, block_size)
            .expect("Failed to create NamResampler");

        let rt_status = RtStatusFlags::default();

        let mut bridge = Box::new(DspBridge {
            buffers: [
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
            ],
            active_read_idx: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            consumed_gen: AtomicU64::new(0),
            dropped_frames: AtomicU32::new(0),
        });

        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = vec![0.0; MAX_RESAMP_BUF];
        let mut model_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut model_out_r = vec![0.0; MAX_RESAMP_BUF];

        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut opt_model_l = Some(model_l);
        let mut opt_model_r = Some(model_r);

        let mut samples_l = vec![0.0f32; block_size];
        let mut samples_r = vec![0.0f32; block_size];

        let mut pcg = SimplePcg::new(42);

        let mut processed_frames = 0;

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);
        for _ in 0..1000 {
            for j in 0..block_size {
                samples_l[j] = pcg.next_f32();
                samples_r[j] = pcg.next_f32();
            }
            let ctx = DspPipelineContext {
                resampler: &mut resampler,
                active_model_l: &mut opt_model_l,
                active_model_r: &mut opt_model_r,
                input_gain_mult: 1.0,
                output_gain_mult: 1.0,
                gate_params: &gate_params,
                silence_hysteresis: &mut silence_hysteresis,
                mono_hysteresis: &mut mono_hysteresis,
                threshold_open_sq: 0.0,
                threshold_close_sq: 0.0,
                process_mono: &mut process_mono,
                rt_status: &rt_status,
                adaptive: &mut adaptive,
                bridge_writer: unsafe {
                    Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge))
                },
                conv: None,
            };
            let bufs = DspBuffers {
                resamp_mid_l: &mut resamp_mid_l,
                resamp_mid_r: &mut resamp_mid_r,
                resamp_out_l: &mut resamp_out_l,
                resamp_out_r: &mut resamp_out_r,
                model_out_l: &mut model_out_l,
                model_out_r: &mut model_out_r,
            };
            capture_dsp_pipeline(
                &mut samples_l,
                &mut samples_r,
                block_size,
                ctx,
                bufs,
                output_sr,
            );
        }

        let initial_rss = get_rss_bytes();
        println!("Initial RSS: {} bytes", initial_rss);

        let audio_duration_s = total_frames as f64 / input_sr as f64;
        let start_time = Instant::now();

        let mut generation_counter: u64 = bridge.generation.load(Ordering::Relaxed);

        while processed_frames < total_frames {
            let is_noise = (processed_frames / 50_000) % 2 == 0;

            if is_noise {
                for j in 0..block_size {
                    samples_l[j] = pcg.next_f32();
                    samples_r[j] = pcg.next_f32();
                }
            } else {
                for j in 0..block_size {
                    samples_l[j] = 0.0;
                    samples_r[j] = 0.0;
                }
            }

            let start_cycle = Instant::now();

            let ctx = DspPipelineContext {
                resampler: &mut resampler,
                active_model_l: &mut opt_model_l,
                active_model_r: &mut opt_model_r,
                input_gain_mult: 1.0,
                output_gain_mult: 1.0,
                gate_params: &gate_params,
                silence_hysteresis: &mut silence_hysteresis,
                mono_hysteresis: &mut mono_hysteresis,
                threshold_open_sq: -60.0f32.powf(10.0 / 20.0),
                threshold_close_sq: -80.0f32.powf(10.0 / 20.0),
                process_mono: &mut process_mono,
                rt_status: &rt_status,
                adaptive: &mut adaptive,
                bridge_writer: unsafe {
                    Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge))
                },
                conv: None,
            };

            let bufs = DspBuffers {
                resamp_mid_l: &mut resamp_mid_l,
                resamp_mid_r: &mut resamp_mid_r,
                resamp_out_l: &mut resamp_out_l,
                resamp_out_r: &mut resamp_out_r,
                model_out_l: &mut model_out_l,
                model_out_r: &mut model_out_r,
            };

            capture_dsp_pipeline(
                &mut samples_l,
                &mut samples_r,
                block_size,
                ctx,
                bufs,
                output_sr,
            );

            let elapsed_cycle_ns = start_cycle.elapsed().as_nanos() as u64;
            rt_status.latency_hist.record(elapsed_cycle_ns);

            for &s in &resamp_out_l[..block_size / 2] {
                assert!(s.is_finite(), "NaN/Inf detected in resamp_out_l!");
            }
            for &s in &resamp_out_r[..block_size / 2] {
                assert!(s.is_finite(), "NaN/Inf detected in resamp_out_r!");
            }

            let new_generation = bridge.generation.load(Ordering::Relaxed);
            assert!(
                new_generation >= generation_counter,
                "DspBridge generation count decreased! ({} < {})",
                new_generation,
                generation_counter
            );
            generation_counter = new_generation;

            processed_frames += block_size;
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        println!(
            "Elapsed time: {:.3}s for {:.3}s of audio",
            elapsed, audio_duration_s
        );

        assert!(
            elapsed < budget_mult * audio_duration_s,
            "Real-time processing took too long: {:.3}s (allowed: {:.3}s)",
            elapsed,
            budget_mult * audio_duration_s
        );

        let final_rss = get_rss_bytes();
        println!("Final RSS: {} bytes", final_rss);
        if initial_rss > 0 && final_rss > 0 {
            let rss_diff = (final_rss as isize - initial_rss as isize).unsigned_abs();
            let limit = 10 * 1024 * 1024;
            println!("Memory variation: {} bytes", rss_diff);
            assert!(
                rss_diff < limit,
                "Memory leak detected! RSS variation ({} bytes) exceeds limit (10 MB)",
                rss_diff
            );
        }

        let count = rt_status.latency_hist.total_count();
        assert!(count > 0, "Telemetry latency_hist was never populated!");
        println!("Telemetry total count: {}", count);
        println!(
            "Telemetry max latency: {} ns",
            rt_status.latency_hist.get_max()
        );
        println!(
            "Telemetry P50 latency: {} ns",
            rt_status.latency_hist.get_percentile(0.50)
        );
        println!(
            "Telemetry P99 latency: {} ns",
            rt_status.latency_hist.get_percentile(0.99)
        );
    }

    /// Integration Soak Test for the integrated DSP pipeline (A1 Nano).
    #[test]
    #[ignore]
    fn test_pipeline_soak_integrated() {
        run_pipeline_soak_core("BossWN-nano.nam", 10_000_000, 2.5, "A1-Nano");
    }

    /// Integration Soak Test for A2-Full (CH=8) through the full DSP pipeline.
    #[test]
    #[ignore]
    fn test_pipeline_soak_a2_full() {
        run_pipeline_soak_core("wavenet_a2_full.nam", 2_000_000, 10.0, "A2-Full");
    }

    /// Integration Soak Test for A2-Lite (CH=3) through the full DSP pipeline.
    #[test]
    #[ignore]
    fn test_pipeline_soak_a2_lite() {
        run_pipeline_soak_core("wavenet_a2_lite.nam", 5_000_000, 5.0, "A2-Lite");
    }
}
