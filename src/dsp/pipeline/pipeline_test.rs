// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use super::super::test_util::infra::{ALLOC_COUNT, TrackingGuard};
    use super::super::*;
    use crate::common::params::AdaptiveComputeMode;
    use crate::common::spsc::RtStatusFlags;
    use crate::dsp::adaptive::AdaptiveCompute;
    use crate::dsp::gate::{DynamicHysteresis, GateParams};
    use crate::dsp::resampler::NamResampler;

    use std::sync::atomic::Ordering;

    /// Helper function that simulates a lab for testing the audio engine (pipeline).
    /// It sets up everything needed to check if sound enters and exits correctly.
    fn run_pipeline_test(
        pw_rate: u32,
        nam_rate: u32,
        input_l: &[f32],
        input_r: &[f32],
        force_hold_zero: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = input_l.len();
        // Prepares the sample rate converter (e.g.: transform 44100 to 48000 Hz).
        let mut resampler = NamResampler::new(pw_rate, nam_rate, n).unwrap();
        let rt_status = RtStatusFlags::default();

        // Creates the "bridge" that stores processed sounds so we can read them later.
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        // Prepares temporary "drawers" to hold the sound while it is being processed.
        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        // Configures the noise reducer (Noise Gate) options for the test.
        let mut gate_params = GateParams::default();
        if force_hold_zero {
            gate_params.hold_frames = 0;
            gate_params.mono_epsilon = 1.0; // Makes mono detection easier (equal on both sides).
        }
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = input_l.to_vec();
        let mut samples_r = input_r.to_vec();

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        // Combines all settings into a single "Instruction Manual" (Context).
        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0, // Keeps sound always passing.
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            adaptive: &mut adaptive,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
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

        // TURNS ON THE MEMORY WATCHDOG.
        let _guard = TrackingGuard::new();
        // RUNS THE REAL SOUND PROCESSING.
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
        // Checks if the watchdog caught any forbidden memory request.
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        // If the program "stopped to think" (requested memory) while playing sound, the test fails.
        assert_eq!(
            allocs, 0,
            "Allocation detected on hot-path! The system cannot allocate memory while processing audio."
        );

        // Retrieves the final sound that ended up in our "bridge" to check the result.
        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[read_idx];
        let n_out = out_buf.n_samples as usize;

        (
            out_buf.buf_l[..n_out].to_vec(),
            out_buf.buf_r[..n_out].to_vec(),
        )
    }

    /// TEST: Direct Sound (Bypass) in Stereo.
    /// Ensures that, if no effects are active, the input sound is exactly equal to the output.
    #[test]
    #[cfg(feature = "stereo")]
    fn test_bypass_no_resampler_stereo() {
        let n = 64;
        // Creates a test sound for the left (L) and right (R) sides.
        let input_l: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let input_r: Vec<f32> = (0..n).map(|i| (i as f32 + 50.0) * 0.01).collect();

        // Runs the lab with the same input and output rate (no Hz conversion).
        let (out_l, out_r) = run_pipeline_test(48000, 48000, &input_l, &input_r, false);

        assert_eq!(out_l.len(), n);
        assert_eq!(out_r.len(), n);
        // Checks if the sounds are identical, sample by sample.
        assert_eq!(
            out_l, input_l,
            "L channel must be identical to the original"
        );
        assert_eq!(
            out_r, input_r,
            "R channel must be identical to the original"
        );
    }

    /// TEST: Direct Sound (Bypass) in Mono.
    /// Verifies that the system identifies identical sounds and keeps the output synchronized.
    #[test]
    fn test_bypass_no_resampler_mono() {
        let n = 64;
        let input_l: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        // For the mono test, we make the right side an exact copy of the left.
        let input_r = input_l.clone();

        let (out_l, out_r) = run_pipeline_test(48000, 48000, &input_l, &input_r, true);

        assert_eq!(out_l.len(), n);
        assert_eq!(out_r.len(), n);
        assert_eq!(out_l, input_l);
        // In mono mode, the right side (R) must come out exactly equal to the left (L).
        assert_eq!(
            out_r, input_l,
            "In mono mode, the R side must be a copy of L"
        );
    }

    /// TEST: Direct Sound with Quality Change (Resampling).
    /// Tests whether the sound continues to pass correctly even when the "speed" (Hz rate) changes.
    #[test]
    #[cfg(feature = "stereo")]
    fn test_bypass_with_resampler_stereo() {
        // Example: Sound enters at 44100Hz (CD) and leaves at 48000Hz (Video).
        let n = 256;
        let input_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let input_r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1 + 0.5).sin()).collect();

        let (out_l, _out_r) = run_pipeline_test(44100, 48000, &input_l, &input_r, false);

        assert!(!out_l.is_empty());

        // Calculates the "strength" (energy) of the original sound.
        let mut energy_in = 0.0;
        for &x in &input_l {
            energy_in += x * x;
        }

        // Calculates the "strength" of the sound that came out after conversion.
        let mut energy_out = 0.0;
        for &x in &out_l {
            energy_out += x * x;
        }

        // The sound doesn't need to be identical (due to the mathematical conversion),
        // but it should keep a similar volume and not be silent.
        assert!(
            energy_out > energy_in * 0.5,
            "The output sound is too weak or silent after conversion"
        );
    }

    /// TEST: Mono with Quality Change (Resampling).
    /// Ensures that, even after converting the Hz rate, both channels remain identical.
    #[test]
    fn test_bypass_with_resampler_mono() {
        let n = 256;
        let input_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let input_r = input_l.clone();

        let (out_l, out_r) = run_pipeline_test(44100, 48000, &input_l, &input_r, true);

        assert!(!out_l.is_empty());
        assert_eq!(
            out_l, out_r,
            "Even with Hz conversion, mono sound must be equal on both sides (L == R)"
        );
    }

    /// TEST: Processing Economy (Gate Closed and Silence).
    /// If there is no sound and the gate is closed, the system should not waste time processing anything.
    #[test]
    fn test_hotpath_gate_closed_and_silence() {
        let n = 64; // Block size of samples (64 "little pieces" of sound).
        let input_l = vec![0.0; n]; // Silent input on left channel.
        let input_r = vec![0.0; n]; // Silent input on right channel.

        // We prepare the audio tools (resampler and data bridge).
        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        // Temporary working buffers for DSP calculations.
        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        // Noise gate configuration.
        let gate_params = GateParams::new(-70.0, -80.0, 0, 0, 1e-4);
        let mut silence_hysteresis = DynamicHysteresis::new();
        // Forces the gate to close manually for the test.
        // We simulate that the sound is very low (0.0) so that the gate closes.
        silence_hysteresis.update(0.0, 0.1, 0.01, &gate_params, 1000);
        silence_hysteresis.update(0.0, 0.1, 0.01, &gate_params, 1000);
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = input_l.clone();
        let mut samples_r = input_r.clone();

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        // We group everything into the "Context" for processing.
        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.1,
            threshold_close_sq: 0.01,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            adaptive: &mut adaptive,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
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

        // Memory allocation watchdog.
        let _guard = TrackingGuard::new();
        // We run the audio orchestra (Pipeline).
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        // Verification: Allocating memory in the middle of audio is forbidden (causes pops).
        assert_eq!(allocs, 0, "Allocation on the critical path!");

        // Verification: The system must mark that the current state is silence.
        assert!(
            rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_SILENT),
            "The gate should be closed (silence)"
        );
        assert!(!rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING));

        // Final verification: Since the gate is closed, no sample should have been sent to the bridge.
        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[1 - read_idx];
        assert_eq!(
            out_buf.n_samples, 0,
            "There should be no processed samples when the gate is in absolute silence"
        );
    }

    /// TEST: Smooth Transition (FadeOut).
    /// Verifies that the system correctly detects when the volume is gradually decreasing (fading).
    #[test]
    fn test_hotpath_gate_fading() {
        let n = 64;
        let mut input_l = vec![0.0; n];
        let mut input_r = vec![0.0; n];
        // We simulate a very weak signal, which should trigger smooth sound closing.
        for i in 0..n {
            input_l[i] = 0.05; // This value is between the "open" and "close" thresholds.
            input_r[i] = 0.05;
        }

        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        // We configure FadeOut to last 100 sound frames.
        let gate_params = GateParams::new(-70.0, -80.0, 0, 100, 1e-4);
        let mut silence_hysteresis = DynamicHysteresis::new();
        // First we open the sound (1.0) and then inject silence to start FadeOut.
        silence_hysteresis.update(1.0, 0.1, 0.0001, &gate_params, 100);

        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = vec![0.0; n];
        let mut samples_r = vec![0.0; n];

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.1,
            threshold_close_sq: 0.01,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            adaptive: &mut adaptive,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
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

        let _guard = TrackingGuard::new();
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        // Checks: No allocation and must indicate the "FADING" state.
        assert_eq!(allocs, 0);
        assert!(
            rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING),
            "The system should indicate it is in the middle of a smooth close (FadeOut)"
        );
        assert!(!rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_SILENT));
    }

    /// TEST: Distortion Detection (Clipping).
    /// Verifies that the system warns when volume exceeds the digital limit (1.0), causing unwanted noise.
    #[test]
    fn test_hotpath_clipping_detection() {
        let n = 64;
        let mut input_l = vec![0.0; n];
        let input_r = vec![0.0; n];
        // We force an impossible volume (1.5) on a specific sample to cause distortion.
        input_l[10] = 1.5;

        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        // Temporary working buffers for DSP calculations.
        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = input_l.clone();
        let mut samples_r = input_r.clone();

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
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
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
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

        let _guard = TrackingGuard::new();
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        // Verification: Must detect the Clipping flag (RT_STATUS_HAS_CLIPPED).
        assert_eq!(allocs, 0);
        assert!(
            rt_status.check_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED),
            "The system should have detected that the sound exceeded the limit (clipping)"
        );
    }

    /// TEST: Dropped Frames Detection.
    /// Verifies that the system detects when the computer is too slow and starts dropping audio packets
    /// because whoever should process the sound can't keep up with reading in time.
    #[test]
    fn test_hotpath_dropped_frames() {
        let n = 64;
        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            // Simulates that whoever should "listen" to the sound (consumer) hasn't read anything yet.
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];
        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();

        // First pass: The pipeline processes the sound and stores it in the "bridge".
        let mut process_mono = false;
        let mut samples_l = vec![1.0; n];
        let mut samples_r = vec![1.0; n];

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
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
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
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
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs, 48000);

        // Second pass: The system tries to process more sound, but sees that the bridge is still occupied
        // with the sound from the previous pass that nobody read.
        let mut process_mono2 = false;
        let mut samples_l2 = vec![1.0; n];
        let mut samples_r2 = vec![1.0; n];
        let ctx2 = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0,
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono2,
            rt_status: &rt_status,
            adaptive: &mut adaptive,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
            conv: None,
        };

        let bufs2 = DspBuffers {
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        let _guard = TrackingGuard::new();
        // Here the system should be forced to discard this new sound packet.
        capture_dsp_pipeline(&mut samples_l2, &mut samples_r2, n, ctx2, bufs2, 48000);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        assert_eq!(allocs, 0);

        // The system must have incremented the dropped packets counter.
        let dropped = bridge.dropped_frames.load(Ordering::Relaxed);
        assert_eq!(dropped, 1, "Should have detected 1 dropped audio packet");
    }

    #[test]
    fn test_denormal_dither_mono_symmetry() {
        use super::super::stages::{apply_input_stage, apply_output_stage};

        let n = 64;
        let mut samples_l = vec![0.0_f32; n];
        let mut samples_r = vec![0.0_f32; n];
        let rt_status = RtStatusFlags::default();
        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
        let gate_params = GateParams::new(-70.0, -80.0, 0, 0, 1e-4);
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = true;

        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        let mut ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
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
            bridge_writer: None,
            conv: None,
        };

        // 1. Run input stage (under mono mode, R shouldn't get dither)
        apply_input_stage(&mut samples_l, &mut samples_r, n, &mut ctx);

        // Sanity check: verify process_mono is indeed true
        assert!(*ctx.process_mono);

        // Under mono mode, samples_l should have DENORMAL_DITHER_OFFSET, samples_r should not.
        for &val in &samples_l {
            assert!((val - 1.0e-11_f32).abs() < 1e-15_f32);
        }
        for &val in &samples_r {
            assert!(val.abs() < 1e-15_f32);
        }

        // 2. Run output stage (with process_mono = true, R shouldn't have dither subtracted)
        apply_output_stage(
            &mut samples_l,
            &mut samples_r,
            n,
            1.0,
            &mut silence_hysteresis,
            &rt_status,
            true,
            &mut adaptive,
            48000,
        );

        // After output stage, both should be back to exactly 0.0 (or within float epsilon).
        for &val in &samples_l {
            assert!(
                val.abs() < 1e-15_f32,
                "L channel DC offset is too high: {}",
                val
            );
        }
        for &val in &samples_r {
            assert!(
                val.abs() < 1e-15_f32,
                "R channel DC offset is too high: {}",
                val
            );
        }
    }

    /// TEST: Dither addition SIMD vs Scalar parity.
    /// Ensures that the SIMD-optimized dither implementation is bit-exact with the
    /// scalar reference implementation, across various buffer sizes and offsets,
    /// and performs zero heap allocations (RT-safe).
    #[test]
    fn test_dither_simd_vs_scalar_bit_exact() {
        use crate::math::common::scalar_ref::apply_dither_add_fallback;
        use crate::math::dsp::gain::apply_dither_add_simd;

        let lengths = [
            0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 256, 512, 1024,
        ];
        let offsets = [1.0e-11_f32, -1.0e-11_f32, 0.5_f32, -0.5_f32];

        for &len in &lengths {
            for &offset in &offsets {
                let mut buf_simd: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();
                let mut buf_scalar = buf_simd.clone();

                // Track allocations to ensure RT-safety of dither operations.
                let _guard = TrackingGuard::new();
                let start_allocs = ALLOC_COUNT.load(Ordering::Relaxed);

                // Apply SIMD-dispatched dither.
                apply_dither_add_simd(&mut buf_simd, offset);

                // Apply scalar reference dither.
                // SAFETY: buffer is valid for its lifetime, size is matching.
                unsafe {
                    apply_dither_add_fallback(&mut buf_scalar, offset);
                }

                let end_allocs = ALLOC_COUNT.load(Ordering::Relaxed);
                drop(_guard);

                assert_eq!(
                    end_allocs - start_allocs,
                    0,
                    "Allocation detected in dither hot-path for len {}",
                    len
                );

                assert_eq!(
                    buf_simd, buf_scalar,
                    "Dither SIMD output is not bit-exact with scalar reference for len {} and offset {}",
                    len, offset
                );
            }
        }
    }
}
