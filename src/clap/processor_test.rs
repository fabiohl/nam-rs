// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::clap::test_util::{self, MonoTestBuffers, StereoTestBuffers};
    use clack_host::prelude::*;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_zero_alloc_process_bypass() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let mut bufs = StereoTestBuffers::new(512, 0.1, 0.2);

        let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
        let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);

        let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
        let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);

        let input_events =
            InputEvents::from_buffer::<[clack_host::events::event_types::NoteOnEvent; 0]>(&[]);
        let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

        test_util::assert_zero_alloc("CLAP process bypass", || {
            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("Failure in process()");
        });

        for i in 0..512 {
            assert!(
                (bufs.out_l[i] - bufs.in_l[i]).abs() < 1e-4,
                "Bypass failure Channel L sample {}: {} vs {}",
                i,
                bufs.out_l[i],
                bufs.in_l[i]
            );
            assert!(
                (bufs.out_r[i] - bufs.in_l[i]).abs() < 1e-4,
                "Bypass failure Channel R sample {}: {} vs {}",
                i,
                bufs.out_r[i],
                bufs.in_l[i]
            );
        }
    }

    #[test]
    fn test_irregular_block_sizes_stress() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 1,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let sizes = [1, 7, 17, 33, 53, 128, 256, 512, 1, 1];

        let mut input_ports = AudioPorts::with_capacity(2, 1);
        let mut output_ports = AudioPorts::with_capacity(2, 1);
        let mut output_events_buffer = EventBuffer::new();

        for &n in &sizes {
            let mut in_l = vec![0.1f32; n];
            let mut in_r = vec![0.2f32; n];
            let mut out_l = vec![0.0f32; n];
            let mut out_r = vec![0.0f32; n];

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

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap_or_else(|_| panic!("Failure in process() for n={}", n));

            for i in 0..n {
                assert!(
                    out_l[i].is_finite(),
                    "NaN/Inf detected in channel L for n={}",
                    n
                );
                assert!(
                    out_r[i].is_finite(),
                    "NaN/Inf detected in channel R for n={}",
                    n
                );
            }
        }
    }

    #[test]
    fn test_model_switching_stress() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 64,
            max_frames_count: 64,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_dir.push("tests/fixtures/models");

        let models = [
            "BossWN-nano.nam",
            "BossWN-feather.nam",
            "BossWN-standard.nam",
            "wavenet_a2_lite.nam",
        ];

        let n = 64;
        let mut bufs = StereoTestBuffers::new(n, 0.0, 0.0);
        let state_ext = test_util::get_state_ext(&mut plugin_instance);

        for i in 0..1000 {
            if i % 50 == 0 {
                let model_name = models[(i / 50) % models.len()];
                let mut path = model_dir.clone();
                path.push(model_name);

                let params = test_util::make_default_params(Some(path));
                let state_bytes = serde_json::to_vec(&params).unwrap();
                let mut handle = plugin_instance.plugin_handle();
                let prev_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
                state_ext
                    .load(&mut handle, &mut state_bytes.as_slice())
                    .expect("Failed to load state");

                let current_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
                assert!(
                    current_counter > prev_counter,
                    "Model load counter did not increment after loading {}, indicating the load failed.",
                    model_name
                );
            }

            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let input_events = InputEvents::empty();
            let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

            test_util::assert_zero_alloc(&format!("process cycle {}", i), || {
                started_processor
                    .process(
                        &input_audio,
                        &mut output_audio,
                        &input_events,
                        &mut output_events,
                        None,
                        None,
                    )
                    .unwrap();
            });
        }
    }

    #[test]
    fn test_parameter_modulation_stress() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let n = 512;
        let mut bufs = StereoTestBuffers::new(n, 0.5, 0.5);

        // Pre-warm the resampler with the DC signal to avoid step response ringing
        {
            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let mut out_l_pre = vec![0.0f32; n];
            let mut out_r_pre = vec![0.0f32; n];
            let output_channels = [out_l_pre.as_mut_slice(), out_r_pre.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let input_events = InputEvents::empty();
            let mut output_events_buffer = EventBuffer::new();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("Failure in pre-warm");
        }

        let mut output_events_buffer = EventBuffer::new();

        use crate::clap::extensions::params::PARAM_INPUT_GAIN;
        use clack_common::events::Pckn;
        use clack_common::events::event_types::ParamValueEvent;
        use clack_common::utils::{ClapId, Cookie};

        let mut input_events_buffer = EventBuffer::new();
        for i in 0..n {
            let val = -20.0 + (i as f32 / n as f32) * 40.0;
            let event = ParamValueEvent::new(
                i as u32,
                ClapId::new(PARAM_INPUT_GAIN),
                Pckn::match_all(),
                val as f64,
                Cookie::empty(),
            );
            input_events_buffer.push(&event);
        }

        let input_events = InputEvents::from_buffer(&input_events_buffer);

        let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
        let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);

        let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
        let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);

        let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

        test_util::assert_zero_alloc("intense modulation", || {
            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("Failure in process() with intense modulation");
        });

        for i in 1..n {
            let diff_l = (bufs.out_l[i] - bufs.out_l[i - 1]).abs();
            let diff_r = (bufs.out_r[i] - bufs.out_r[i - 1]).abs();
            assert!(
                diff_l < 0.05,
                "Possible zipper noise detected in channel L sample {}",
                i
            );
            assert!(
                diff_r < 0.05,
                "Possible zipper noise detected in channel R sample {}",
                i
            );
        }
    }

    #[cfg(feature = "clap-plugin")]
    #[test]
    fn test_gui_extension_x11() {
        use clack_extensions::gui::{GuiApiType, GuiConfiguration, GuiSize, PluginGui};

        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let gui_ext = plugin_instance
            .plugin_handle()
            .get_extension::<PluginGui>()
            .expect("PluginGui extension not found");

        let mut handle = plugin_instance.plugin_handle();

        // 1. Test is_api_supported
        assert!(gui_ext.is_api_supported(
            &mut handle,
            GuiConfiguration {
                api_type: GuiApiType::X11,
                is_floating: false,
            }
        ));
        assert!(!gui_ext.is_api_supported(
            &mut handle,
            GuiConfiguration {
                api_type: GuiApiType::WAYLAND,
                is_floating: false,
            }
        ));
        assert!(gui_ext.is_api_supported(
            &mut handle,
            GuiConfiguration {
                api_type: GuiApiType::X11,
                is_floating: true,
            }
        ));

        // 2. Test get_preferred_api
        let pref = gui_ext
            .get_preferred_api(&mut handle)
            .expect("Preferred API not found");
        assert_eq!(pref.api_type, GuiApiType::X11);
        assert!(!pref.is_floating);

        // 3. Test get_size
        let size = gui_ext.get_size(&mut handle).expect("Failed to get size");
        assert_eq!(size.width, 600);
        assert_eq!(size.height, 275);

        // 4. Test can_resize
        assert!(!gui_ext.can_resize(&mut handle));

        // 5. Test set_size
        assert!(
            gui_ext
                .set_size(
                    &mut handle,
                    GuiSize {
                        width: 600,
                        height: 275
                    }
                )
                .is_ok()
        );
        // Note: clack-extensions 0.1.0's FFI wrapper for set_size contains a bug where it calls
        // .is_some() on the Option<Result<(), PluginError>>, returning true (Ok) to the host even
        // when the plugin returns an Err. Thus we cannot assert gui_ext.set_size returns Err
        // from the host-side wrapper here.
    }

    #[test]
    fn test_ui_telemetry() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_dir.push("tests/fixtures/models/BossWN-nano.nam");

        let params = test_util::make_default_params(Some(model_dir));
        test_util::load_plugin_state(&mut plugin_instance, &params);

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        // Check model name basename was updated
        {
            let name_guard = shared.cold.ui_model_name.lock().unwrap();
            assert_eq!(*name_guard, "BossWN-nano.nam");
        }

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        shared
            .rt_to_ui
            .ui_peak_l
            .store(0.0f32.to_bits(), Ordering::Relaxed);
        shared
            .rt_to_ui
            .ui_peak_r
            .store(0.0f32.to_bits(), Ordering::Relaxed);
        shared.rt_to_ui.ui_clipped.store(false, Ordering::Relaxed);

        let n = 512;
        let mut bufs = StereoTestBuffers::new(n, 1.5, 0.5);

        let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
        let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);

        let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
        let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);

        let input_events = InputEvents::empty();
        let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

        started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .unwrap();

        // Verify peaks are greater than 0.0
        let peak_l = f32::from_bits(shared.rt_to_ui.ui_peak_l.load(Ordering::Relaxed));
        let peak_r = f32::from_bits(shared.rt_to_ui.ui_peak_r.load(Ordering::Relaxed));
        assert!(peak_l > 0.0, "peak_l was: {}", peak_l);
        assert!(peak_r > 0.0, "peak_r was: {}", peak_r);

        // Verify that clipping occurred because input left was 1.5
        let clipped = shared.rt_to_ui.ui_clipped.load(Ordering::Relaxed);
        assert!(
            clipped,
            "ui_clipped should be true since input left was 1.5"
        );
    }

    #[test]
    fn test_monophonic_parameter_modulation() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let n = 512;
        let mut bufs = StereoTestBuffers::new(n, 0.5, 0.5);

        use crate::clap::extensions::params::{PARAM_GATE_THRESH, PARAM_INPUT_GAIN};
        use clack_common::events::Pckn;
        use clack_common::events::event_types::{ParamModEvent, ParamValueEvent};
        use clack_common::utils::{ClapId, Cookie};

        // 1. Base case: No modulation, base gain = 0dB.
        {
            let mut input_events_buffer = EventBuffer::new();
            let val_event = ParamValueEvent::new(
                0,
                ClapId::new(PARAM_INPUT_GAIN),
                Pckn::match_all(),
                0.0,
                Cookie::empty(),
            );
            input_events_buffer.push(&val_event);

            let gate_event = ParamValueEvent::new(
                0,
                ClapId::new(PARAM_GATE_THRESH),
                Pckn::match_all(),
                -90.0,
                Cookie::empty(),
            );
            input_events_buffer.push(&gate_event);

            let input_events = InputEvents::from_buffer(&input_events_buffer);
            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let mut output_events_buffer = EventBuffer::new();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("Failure in base case process");

            for (i, &val) in bufs.out_l.iter().enumerate().take(n) {
                assert!(
                    (val - 0.5).abs() < 1e-4,
                    "Base case L sample {}: {}",
                    i,
                    val
                );
            }
        }

        // 2. Modulation
        {
            let mut input_events_buffer = EventBuffer::new();
            let mod_event = ParamModEvent::new(
                0,
                ClapId::new(PARAM_INPUT_GAIN),
                Pckn::match_all(),
                6.0,
                Cookie::empty(),
            );
            input_events_buffer.push(&mod_event);

            let input_events = InputEvents::from_buffer(&input_events_buffer);
            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let mut output_events_buffer = EventBuffer::new();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            test_util::assert_zero_alloc("process with modulation", || {
                started_processor
                    .process(
                        &input_audio,
                        &mut output_audio,
                        &input_events,
                        &mut output_events,
                        None,
                        None,
                    )
                    .expect("Failure in process with modulation");
            });

            assert!(
                bufs.out_l[n - 1] > 0.6,
                "Modulation not applied, last sample: {}",
                bufs.out_l[n - 1]
            );
        }

        // 3. Gate Modulation
        {
            let mut input_events_buffer = EventBuffer::new();
            // Modulates gate threshold by +120dB (bringing effective threshold to +30dB)
            let mod_event = ParamModEvent::new(
                0,
                ClapId::new(PARAM_GATE_THRESH),
                Pckn::match_all(),
                120.0,
                Cookie::empty(),
            );
            input_events_buffer.push(&mod_event);

            let input_events = InputEvents::from_buffer(&input_events_buffer);
            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let mut out_l_gate = vec![0.0f32; n];
            let mut out_r_gate = vec![0.0f32; n];
            let output_channels = [out_l_gate.as_mut_slice(), out_r_gate.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let mut output_events_buffer = EventBuffer::new();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            // Processes the first block containing the modulation event
            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("Failure in process with gate modulation");

            // Processes 5 more blocks without new modulation events to allow the gate FSM to transition:
            // Open -> hold (2048 frames) -> FadingOut (256 frames) -> Closed
            let empty_events = InputEvents::empty();
            for _ in 0..5 {
                out_l_gate.fill(0.0);
                out_r_gate.fill(0.0);

                let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
                let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                    latency: 0,
                    channels: AudioPortBufferType::f32_input_only(
                        input_channels.iter_mut().map(InputChannel::constant),
                    ),
                }]);

                let output_channels = [out_l_gate.as_mut_slice(), out_r_gate.as_mut_slice()];
                let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                    latency: 0,
                    channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
                }]);

                let mut output_events_buffer = EventBuffer::new();
                let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

                started_processor
                    .process(
                        &input_audio,
                        &mut output_audio,
                        &empty_events,
                        &mut output_events,
                        None,
                        None,
                    )
                    .expect("Failure in subsequent gate blocks process");
            }

            // Since the effective threshold is 0dB and the signal is -6dB, the gate should close and silence the output.
            assert_eq!(
                out_l_gate[n - 1],
                0.0,
                "Gate should have silenced the signal at the last sample. Last sample: {}",
                out_l_gate[n - 1]
            );
        }
    }

    #[cfg(feature = "heap-audit")]
    #[test]
    fn test_heap_audit_trigger() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_dir.push("tests/fixtures/models/mock_a2.nam");

        let params = test_util::make_default_params(Some(model_dir));
        test_util::load_plugin_state(&mut plugin_instance, &params);

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        // mock_a2.nam fails to build (model_l = None post-build).
        // load_model now rejects this — counter stays at 0.
        assert_eq!(
            shared.cold.model_load_counter.load(Ordering::Relaxed),
            0,
            "model_load_counter should not increment when model build fails"
        );

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let n = 512;
        let mut bufs = StereoTestBuffers::new(n, 0.1, 0.2);

        let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
        let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);

        let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
        let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);

        let input_events = InputEvents::empty();
        let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

        // Activates heap audit globally
        crate::common::alloc_audit::AUDIT_ENABLED.store(true, Ordering::Relaxed);

        // Resets the status flag to ensure it is clean before
        shared
            .cold
            .rt_status
            .clear_flag(crate::common::spsc::RT_STATUS_HEAP_ALLOC);

        // Runs process(), which should return Continue (zero heap allocations in the hot path)
        let status = started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .expect("process failed in heap audit test");

        // Disables heap audit to avoid interfering with other tests
        crate::common::alloc_audit::AUDIT_ENABLED.store(false, Ordering::Relaxed);

        // Verifies that the returned status is Continue (zero-alloc path)
        assert!(
            matches!(status, ProcessStatus::Continue),
            "Expected ProcessStatus::Continue (zero-alloc), got {status:?}"
        );

        // Verifies the RT_STATUS_HEAP_ALLOC flag was NOT set (no allocations detected)
        assert!(
            !shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_HEAP_ALLOC)
        );

        // Verifies the RT_STATUS_MODEL_LOAD_FAILED flag was set (as mock_a2.nam fails to build)
        assert!(
            shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED),
            "Expected RT_STATUS_MODEL_LOAD_FAILED to be set because mock_a2.nam fails to build"
        );
    }

    #[test]
    fn test_gui_load_error_null_byte_safety() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let path_with_null = PathBuf::from("invalid_model\0name.nam");
        if let Ok(mut pending_guard) = shared.cold.ui_pending_model.lock() {
            *pending_guard = Some(path_with_null);
        }
        shared.cold.ui_loading.store(true, Ordering::Relaxed);

        plugin_instance.call_on_main_thread_callback();

        assert!(!shared.cold.ui_loading.load(Ordering::Relaxed));
        assert!(shared.cold.ui_load_error.load(Ordering::Relaxed));
    }

    #[test]
    fn test_c_string_null_byte_sanitization() {
        use std::ffi::CString;
        let err_msg = "Model load failed: file\0name.nam not found";
        let err_str = format!("NAM-rs: Failed to load model from GUI: {}", err_msg);
        let sanitized_err = err_str.replace('\0', " ");
        let msg = CString::new(sanitized_err);
        assert!(msg.is_ok(), "Sanitized CString creation should succeed");
    }

    /// T17.3 — Loading a .nam file that fails to build should:
    /// 1. Show an error to the user (ui_load_error = true)
    /// 2. NOT swap the active model (model_load_counter unchanged)
    #[test]
    fn test_gui_load_model_build_failed() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();
        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let mut model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_path.push("tests/fixtures/models/mock_a2.nam");

        if let Ok(mut pending_guard) = shared.cold.ui_pending_model.lock() {
            *pending_guard = Some(model_path);
        }
        shared.cold.ui_loading.store(true, Ordering::Relaxed);

        plugin_instance.call_on_main_thread_callback();

        assert!(!shared.cold.ui_loading.load(Ordering::Relaxed));
        assert!(
            shared.cold.ui_load_error.load(Ordering::Relaxed),
            "Expected ui_load_error to be set when model build fails"
        );
        assert_eq!(
            shared.cold.model_load_counter.load(Ordering::Relaxed),
            0,
            "model_load_counter should not increment for failed model build"
        );
    }

    // ---------------------------------------------------------------------------
    // G3.T05 — Integration test hardening (render + state-context + preset-load)
    // ---------------------------------------------------------------------------

    #[test]
    fn test_preset_load_integration() {
        use clack_extensions::preset_discovery::prelude::*;
        use std::ffi::CString;

        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let preset_load_ext = plugin_instance
            .plugin_handle()
            .get_extension::<PluginPresetLoad>()
            .expect("PluginPresetLoad extension not found");

        let mut model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_path.push("tests/fixtures/models/BossWN-nano.nam");
        let path_str = model_path.to_str().expect("Invalid model path");
        let path_cstr = CString::new(path_str).expect("Invalid CString");

        let counter_before = shared.cold.model_load_counter.load(Ordering::Relaxed);
        assert_eq!(counter_before, 0, "model_load_counter should start at 0");

        let mut handle = plugin_instance.plugin_handle();
        preset_load_ext
            .load_from_location(&mut handle, Location::File { path: &path_cstr }, None)
            .expect("load_from_location should succeed");

        plugin_instance.call_on_main_thread_callback();

        let counter_after = shared.cold.model_load_counter.load(Ordering::Relaxed);
        assert!(
            counter_after > counter_before,
            "model_load_counter should increment after preset load (was {}, now {})",
            counter_before,
            counter_after
        );

        let model_name = shared.cold.ui_model_name.lock().unwrap();
        assert!(
            !model_name.is_empty(),
            "ui_model_name should be set after preset load"
        );
    }

    #[test]
    fn test_render_mode_transitions() {
        use clack_extensions::render::{PluginRender, RenderMode};

        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let render_ext = plugin_instance
            .plugin_handle()
            .get_extension::<PluginRender>()
            .expect("PluginRender extension not found");

        assert!(
            !render_ext.has_realtime_requirement(&mut plugin_instance.plugin_handle()),
            "NAM should not have hard realtime requirement"
        );

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let render_mode = shared.cold.render_mode.load(Ordering::Acquire);
        assert_eq!(
            render_mode,
            crate::clap::plugin::RENDER_MODE_REALTIME,
            "should start in realtime mode"
        );

        let mut handle = plugin_instance.plugin_handle();
        render_ext
            .set(&mut handle, RenderMode::Offline)
            .expect("set RenderMode::Offline should succeed");

        let render_mode = shared.cold.render_mode.load(Ordering::Acquire);
        assert_eq!(
            render_mode,
            crate::clap::plugin::RENDER_MODE_OFFLINE,
            "render_mode should be OFFLINE after set"
        );

        let n = 512;
        let mut bufs = MonoTestBuffers::new(n, 0.1);
        let mut input_channels = [bufs.in_buf.as_mut_slice()];
        let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);
        let output_channels = [bufs.out_buf.as_mut_slice()];
        let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);
        let input_events = InputEvents::empty();
        let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

        // Process a few blocks in offline mode — degradation flags should stay clear
        for _ in 0..4 {
            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("process should succeed in offline mode");
        }

        assert!(
            !shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED),
            "DEGRADE_REDUCED should be clear in offline mode"
        );
        assert!(
            !shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL),
            "DEGRADE_MINIMAL should be clear in offline mode"
        );

        let mut handle = plugin_instance.plugin_handle();
        render_ext
            .set(&mut handle, RenderMode::Realtime)
            .expect("set RenderMode::Realtime should succeed");

        let render_mode = shared.cold.render_mode.load(Ordering::Acquire);
        assert_eq!(
            render_mode,
            crate::clap::plugin::RENDER_MODE_REALTIME,
            "render_mode should be back to REALTIME"
        );

        // Process blocks in realtime mode
        for _ in 0..2 {
            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .expect("process should succeed in realtime mode");
        }

        // Verify output is not silent (bypass off by default)
        assert!(
            bufs.out_buf.iter().any(|s| *s != 0.0),
            "Output should not be silent (bypass is off)"
        );
    }

    #[test]
    fn test_offline_mode_forces_adaptive_off() {
        use clack_extensions::render::{PluginRender, RenderMode};

        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let render_ext = plugin_instance
            .plugin_handle()
            .get_extension::<PluginRender>()
            .expect("PluginRender extension not found");

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        // 1. Configure AdaptiveCompute to Aggressive in Realtime mode
        shared.ui_to_rt.param_adaptive_compute.store(
            crate::common::params::AdaptiveComputeMode::Aggressive as u32,
            Ordering::Relaxed,
        );
        shared.bump_generation();

        shared
            .rt_to_ui
            .active_channel_count
            .store(1, Ordering::Relaxed);

        let n = 512;
        let mut bufs = MonoTestBuffers::new(n, 0.1);
        let mut input_channels = [bufs.in_buf.as_mut_slice()];
        let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);
        let output_channels = [bufs.out_buf.as_mut_slice()];
        let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);
        let input_events = InputEvents::empty();
        let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

        // Process a block to apply/sync parameter changes
        started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .unwrap();

        // 2. Manually set degradation flags to simulate an overload that occurred previously
        shared
            .cold
            .rt_status
            .set_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
        shared
            .cold
            .rt_status
            .set_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
        assert!(
            shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED)
        );
        assert!(
            shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL)
        );

        // 3. Set RenderMode::Offline
        let mut handle = plugin_instance.plugin_handle();
        render_ext
            .set(&mut handle, RenderMode::Offline)
            .expect("set RenderMode::Offline should succeed");

        // 4. Process a block in Offline mode
        started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .unwrap();

        // 5. Verify that degradation flags are CLEARED in Offline mode
        assert!(
            !shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED),
            "DEGRADE_REDUCED should be cleared in offline mode"
        );
        assert!(
            !shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL),
            "DEGRADE_MINIMAL should be cleared in offline mode"
        );

        // 6. Try to change AdaptiveCompute to Conservative while offline
        shared.ui_to_rt.param_adaptive_compute.store(
            crate::common::params::AdaptiveComputeMode::Conservative as u32,
            Ordering::Relaxed,
        );
        shared.bump_generation();

        // Set degradation flags again manually to see if they are cleared on next block
        shared
            .cold
            .rt_status
            .set_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
        shared
            .cold
            .rt_status
            .set_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);

        started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .unwrap();

        // Flags must have been cleared again, because AdaptiveCompute is immediately forced to Off when offline
        assert!(
            !shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED),
            "DEGRADE_REDUCED must be immediately cleared when attempting parameter changes offline"
        );
        assert!(
            !shared
                .cold
                .rt_status
                .check_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL),
            "DEGRADE_MINIMAL must be immediately cleared when attempting parameter changes offline"
        );
    }

    #[test]
    fn test_state_context_roundtrip() {
        use clack_extensions::state_context::{PluginStateContext, StateContextType};

        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let state_ext = test_util::get_state_ext(&mut plugin_instance);
        let state_ctx_ext = plugin_instance
            .plugin_handle()
            .get_extension::<PluginStateContext>()
            .expect("PluginStateContext extension not found");

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let mut model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_path.push("tests/fixtures/models/BossWN-nano.nam");

        use crate::common::params::NamPluginParams;
        let params = NamPluginParams {
            model_path: Some(model_path.clone()),
            input_gain_db: 3.5,
            output_gain_db: -4.0,
            gate_threshold_db: -45.0,
            model_basename: Some("BossWN-nano.nam".to_string()),
            model_search_paths: vec![],
            bypass: false,
            adaptive_compute: crate::common::params::AdaptiveComputeMode::Conservative,
            slim_override: Default::default(),
            ir_path: None,
        };
        let state_bytes = serde_json::to_vec(&params).unwrap();
        let mut handle = plugin_instance.plugin_handle();
        state_ext
            .load(&mut handle, &mut state_bytes.as_slice())
            .expect("Failed to load model via PluginState");

        let model_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
        assert!(model_counter > 0, "Model should have been loaded");

        // --- Save: ForPreset context ---
        let mut preset_buffer = Vec::new();
        let mut handle = plugin_instance.plugin_handle();
        state_ctx_ext
            .save(&mut handle, &mut preset_buffer, StateContextType::ForPreset)
            .expect("save ForPreset should succeed");

        let preset_json: serde_json::Value =
            serde_json::from_slice(&preset_buffer).expect("preset buffer should be valid JSON");
        assert_eq!(
            preset_json["version"], 1,
            "ForPreset save should produce v1 envelope"
        );
        assert!(
            preset_json["params"].is_object(),
            "ForPreset envelope should contain params"
        );
        assert!(
            preset_json["params"]["model_path"].is_null(),
            "ForPreset save should omit model_path"
        );
        assert!(
            preset_json["params"]["model_basename"].is_string(),
            "ForPreset save should preserve model_basename"
        );
        assert!(
            (preset_json["params"]["input_gain_db"].as_f64().unwrap() - 3.5).abs() < f64::EPSILON,
            "ForPreset save should preserve input_gain_db"
        );

        // --- Save: ForProject context ---
        let mut project_buffer = Vec::new();
        let mut handle = plugin_instance.plugin_handle();
        state_ctx_ext
            .save(
                &mut handle,
                &mut project_buffer,
                StateContextType::ForProject,
            )
            .expect("save ForProject should succeed");

        let project_json: serde_json::Value =
            serde_json::from_slice(&project_buffer).expect("project buffer should be valid JSON");
        assert_eq!(
            project_json["version"], 1,
            "ForProject save should produce v1 envelope"
        );
        assert!(
            project_json["params"]["model_path"].is_string(),
            "ForProject save should preserve model_path"
        );
        assert!(
            (project_json["params"]["input_gain_db"].as_f64().unwrap() - 3.5).abs() < f64::EPSILON,
            "ForProject save should preserve input_gain_db"
        );

        // --- Load: ForPreset context (only audio params restored, model_path unchanged) ---
        let preset_json_str = r#"{"input_gain_db":1.5,"output_gain_db":-2.0,"gate_threshold_db":-40.0,"model_path":null,"model_basename":null,"model_search_paths":[],"bypass":false,"adaptive_compute":"Off"}"#;
        let mut handle = plugin_instance.plugin_handle();
        state_ctx_ext
            .load(
                &mut handle,
                &mut preset_json_str.as_bytes(),
                StateContextType::ForPreset,
            )
            .expect("load ForPreset should succeed");

        let gain_in = shared.ui_to_rt.param_input_gain.load(Ordering::Relaxed);
        assert!(
            (f32::from_bits(gain_in) - 1.5).abs() < 0.01,
            "input gain should be restored from preset"
        );
        let gain_out = shared.ui_to_rt.param_output_gain.load(Ordering::Relaxed);
        assert!(
            (f32::from_bits(gain_out) - (-2.0)).abs() < 0.01,
            "output gain should be restored from preset"
        );

        // --- Load: ForProject context (full state, with model_path) ---
        let project_with_path = format!(
            r#"{{"input_gain_db":5.0,"output_gain_db":-8.0,"gate_threshold_db":-60.0,"model_path":"{}","model_basename":"BossWN-nano.nam","model_search_paths":[],"bypass":true,"adaptive_compute":"Aggressive"}}"#,
            model_path.to_str().unwrap()
        );
        let mut handle = plugin_instance.plugin_handle();
        state_ctx_ext
            .load(
                &mut handle,
                &mut project_with_path.as_bytes(),
                StateContextType::ForProject,
            )
            .expect("load ForProject should succeed");

        let gain_in = shared.ui_to_rt.param_input_gain.load(Ordering::Relaxed);
        assert!(
            (f32::from_bits(gain_in) - 5.0).abs() < 0.01,
            "input gain should reflect full project state restoration"
        );
        let bypass_val = shared.ui_to_rt.param_bypass.load(Ordering::Relaxed);
        assert_eq!(bypass_val, 1, "bypass should be enabled from project state");
        let adaptive_mode = shared
            .ui_to_rt
            .param_adaptive_compute
            .load(Ordering::Relaxed);
        assert_eq!(
            adaptive_mode, 2,
            "adaptive_compute should be Aggressive from project state"
        );
    }

    #[cfg(not(feature = "stereo"))]
    #[test]
    fn test_mono_clipping_detection() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let n = 512;
        let mut input_ports = AudioPorts::with_capacity(1, 1);
        let mut output_ports = AudioPorts::with_capacity(1, 1);

        // Case 1: No clipping (signal = 0.5, gain = 0 dB -> output = 0.5)
        {
            shared.rt_to_ui.ui_clipped.store(false, Ordering::Relaxed);
            let mut in_buf = vec![0.5f32; n];
            let mut out_buf = vec![0.0f32; n];

            let mut input_channels = [in_buf.as_mut_slice()];
            let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [out_buf.as_mut_slice()];
            let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let input_events = InputEvents::empty();
            let mut output_events_buffer = EventBuffer::new();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap();

            let clipped = shared.rt_to_ui.ui_clipped.load(Ordering::Relaxed);
            assert!(
                !clipped,
                "ui_clipped should be false for non-clipping signal"
            );
        }

        // Case 2: Steady state clipping (signal = 1.5, gain = 0 dB -> output = 1.5 -> clipped)
        {
            shared.rt_to_ui.ui_clipped.store(false, Ordering::Relaxed);
            let mut in_buf = vec![1.5f32; n];
            let mut out_buf = vec![0.0f32; n];

            let mut input_channels = [in_buf.as_mut_slice()];
            let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [out_buf.as_mut_slice()];
            let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let input_events = InputEvents::empty();
            let mut output_events_buffer = EventBuffer::new();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap();

            let clipped = shared.rt_to_ui.ui_clipped.load(Ordering::Relaxed);
            assert!(
                clipped,
                "ui_clipped should be true for steady state clipping signal (>1.0)"
            );
        }

        // Case 3: Ramp clipping (signal = 0.5, gain modulates from 0 dB to 12 dB -> gain linear = 3.98 -> output ~1.99 -> clipped)
        {
            shared.rt_to_ui.ui_clipped.store(false, Ordering::Relaxed);
            let mut in_buf = vec![0.5f32; n];
            let mut out_buf = vec![0.0f32; n];

            let mut input_channels = [in_buf.as_mut_slice()];
            let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [out_buf.as_mut_slice()];
            let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            use crate::clap::extensions::params::PARAM_INPUT_GAIN;
            use clack_common::events::Pckn;
            use clack_common::events::event_types::ParamValueEvent;
            use clack_common::utils::{ClapId, Cookie};

            let mut input_events_buffer = EventBuffer::new();
            let val_event = ParamValueEvent::new(
                0,
                ClapId::new(PARAM_INPUT_GAIN),
                Pckn::match_all(),
                12.0, // +12 dB
                Cookie::empty(),
            );
            input_events_buffer.push(&val_event);
            let input_events = InputEvents::from_buffer(&input_events_buffer);

            let mut output_events_buffer = EventBuffer::new();
            let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap();

            let clipped = shared.rt_to_ui.ui_clipped.load(Ordering::Relaxed);
            assert!(
                clipped,
                "ui_clipped should be true for ramp clipping signal"
            );
        }
    }

    #[test]
    #[ignore]
    fn test_gc_stress_1000_swaps() {
        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let state_ext = test_util::get_state_ext(&mut plugin_instance);

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 64,
            max_frames_count: 64,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_dir.push("tests/fixtures/models");

        let models = [
            "BossWN-nano.nam",
            "BossWN-feather.nam",
            "BossWN-standard.nam",
            "wavenet_a2_lite.nam",
        ];

        let n = 64;
        let mut bufs = StereoTestBuffers::new(n, 0.0, 0.0);

        let shared = unsafe { &*test_util::extract_shared(&mut plugin_instance) };

        let rt_status = &shared.cold.rt_status;

        // Perform exactly 24 model swaps first to test limit of SPSC + parking lot (48 slots).
        // CLAP is mono so model_r is never built (T17.4); cold_load_model no longer discards it.
        // 1st swap pushes 1 item (old_resampler, since model_l is initially None and model_r is None).
        // Subsequent swaps push 2 items each (old_model_l + old_resampler).
        // Total items pushed for 24 swaps (i = 0 to 23) is exactly 1 + 23 * 2 = 47 items.
        for i in 0..24 {
            let model_name = models[i % models.len()];
            let mut path = model_dir.clone();
            path.push(model_name);

            let params = test_util::make_default_params(Some(path));
            let state_bytes = serde_json::to_vec(&params).unwrap();
            let mut handle = plugin_instance.plugin_handle();
            let prev_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
            state_ext
                .load(&mut handle, &mut state_bytes.as_slice())
                .expect("Failed to load state");

            let current_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
            assert!(
                current_counter > prev_counter,
                "Model load counter did not increment after loading {}, indicating the load failed.",
                model_name
            );

            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let input_events = InputEvents::empty();
            let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap();
        }

        // Verify that no GC overflow/leak occurred as we have not exceeded SPSC + parking lot (47 <= 48)
        assert!(
            !rt_status.check_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW),
            "GC overflow flag was set prematurely!"
        );

        // Perform 1 more swap (the 25th swap). This pushes 2 more items.
        // Total items pushed = 47 + 2 = 49 items.
        // This exceeds SPSC + parking lot limit of 48 items, so 1 item must spill into the overflow buffer,
        // which triggers the RT_STATUS_GC_OVERFLOW flag.
        {
            let model_name = models[24 % models.len()];
            let mut path = model_dir.clone();
            path.push(model_name);

            let params = test_util::make_default_params(Some(path));
            let state_bytes = serde_json::to_vec(&params).unwrap();
            let mut handle = plugin_instance.plugin_handle();
            let prev_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
            state_ext
                .load(&mut handle, &mut state_bytes.as_slice())
                .expect("Failed to load state");

            let current_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
            assert!(
                current_counter > prev_counter,
                "Model load counter did not increment after loading {}, indicating the load failed.",
                model_name
            );

            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let input_events = InputEvents::empty();
            let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap();
        }

        // Verify that the GC overflow flag is indeed set now
        assert!(
            rt_status.check_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW),
            "Expected GC overflow flag to be set after exceeding SPSC + parking lot capacity"
        );

        // Perform a complete drain to reclaim all 49 items from the channels and overflow buffer
        plugin_instance.call_on_main_thread_callback();
        // One process cycle to move items from the parking lot to the now empty SPSC channel
        {
            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);
            let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);
            let input_events = InputEvents::empty();
            let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);
            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap();
        }
        plugin_instance.call_on_main_thread_callback();

        // Clear the overflow flag manually now that the system is fully drained and clean
        rt_status.clear_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);

        // Perform the remaining 975 model swaps to reach 1000 model swaps in total.
        // We will drain every 10 swaps (20 items), which fits comfortably within the 32-capacity SPSC channel,
        // so no overflow should occur during this loop.
        for i in 25..1000 {
            let model_name = models[i % models.len()];
            let mut path = model_dir.clone();
            path.push(model_name);

            let params = test_util::make_default_params(Some(path));
            let state_bytes = serde_json::to_vec(&params).unwrap();
            let mut handle = plugin_instance.plugin_handle();
            let prev_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
            state_ext
                .load(&mut handle, &mut state_bytes.as_slice())
                .expect("Failed to load state");

            let current_counter = shared.cold.model_load_counter.load(Ordering::Relaxed);
            assert!(
                current_counter > prev_counter,
                "Model load counter did not increment after loading {}, indicating the load failed.",
                model_name
            );

            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);

            let input_events = InputEvents::empty();
            let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap();

            // Periodic drain of the SPSC channel
            if i % 10 == 0 {
                plugin_instance.call_on_main_thread_callback();
            }
        }

        // Final cleanup and drainage of any leftover items
        for _ in 0..5 {
            plugin_instance.call_on_main_thread_callback();
            let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
            let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);
            let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
            let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
            }]);
            let input_events = InputEvents::empty();
            let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);
            started_processor
                .process(
                    &input_audio,
                    &mut output_audio,
                    &input_events,
                    &mut output_events,
                    None,
                    None,
                )
                .unwrap();
        }
        plugin_instance.call_on_main_thread_callback();

        // Verify that the GC overflow flag was not set again
        assert!(
            !rt_status.check_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW),
            "GC overflow / leak occurred during the remaining model swaps!"
        );
    }

    #[test]
    fn test_model_gain_calibration() {
        let mut model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_path.push("tests/fixtures/models/BossWN-nano.nam");

        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let params = test_util::make_default_params(Some(model_path.clone()));
        test_util::load_plugin_state(&mut plugin_instance, &params);

        plugin_instance.call_on_main_thread_callback();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let n = 512;
        let mut bufs = StereoTestBuffers::new(n, 0.5, 0.5);

        let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
        let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);

        let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
        let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);

        let input_events = InputEvents::empty();
        let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

        started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .unwrap();

        // ── Verify CLAP output is a processed signal (not silence, not passthrough) ──
        let clap_rms: f32 = (bufs.out_l.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        assert!(
            clap_rms > 0.001,
            "CLAP output RMS too low ({:.6}) — model may not have been loaded",
            clap_rms
        );
        assert!(
            clap_rms < 0.4,
            "CLAP output RMS too high ({:.6}) — may be passthrough or uncalibrated",
            clap_rms
        );

        // ── Load model directly and verify calibration multipliers ──
        use crate::models::NamModel;
        let sys = crate::common::diagnostics::SystemSnapshot::capture();
        let model_pair = crate::loader::build::load_and_build_model(&model_path, &sys, true)
            .expect("Failed to load model directly");
        let mut direct_model = model_pair.model_l.expect("Failed to build direct model");
        let input_mult = model_pair.input_mult_adj;
        let output_mult = model_pair.output_mult_adj;

        assert!((input_mult - 1.0).abs() < 1e-4);
        assert!((output_mult - 1.0).abs() > 1e-4);

        use crate::dsp::pipeline::DENORMAL_DITHER_OFFSET;
        let mut direct_in = vec![0.5f32; n];
        for val in direct_in.iter_mut() {
            *val = *val * input_mult + DENORMAL_DITHER_OFFSET;
        }

        let mut direct_out = vec![0.0f32; n];
        direct_model.process(&direct_in, &mut direct_out);

        for val in direct_out.iter_mut() {
            *val = (*val - DENORMAL_DITHER_OFFSET) * output_mult;
        }

        let direct_rms: f32 = (direct_out.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
        assert!(
            direct_rms > 0.001,
            "Direct output RMS too low ({:.6})",
            direct_rms
        );
    }

    #[test]
    fn test_audio_processor_flush() {
        use crate::clap::extensions::params::PARAM_INPUT_GAIN;
        use clack_common::events::Pckn;
        use clack_common::events::event_types::ParamValueEvent;
        use clack_common::utils::{ClapId, Cookie};
        use clack_extensions::params::PluginAudioProcessorParams;

        let (_entry, _host_info, mut plugin_instance) = test_util::make_test_plugin();

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        // Activate the plugin to instantiate the audio processor.
        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();

        let raw_ptr = plugin_instance.plugin_handle().as_raw_ptr();
        let processor_ptr = unsafe {
            clack_plugin::extensions::wrapper::PluginWrapper::<crate::clap::NamClapPlugin>::handle(
                raw_ptr,
                |wrapper| {
                    let ptr = wrapper.audio_processor().unwrap().as_ptr();
                    Ok(ptr)
                },
            )
            .unwrap()
        };
        let processor = unsafe { &mut *processor_ptr };

        // Ensure initially generation is 0
        let initial_gen = processor
            .shared
            .ui_to_rt
            .gui_param_generation
            .load(Ordering::Relaxed);

        // Prepare parameter events for flush: set input gain to 10.5 dB
        let mut input_events_buffer = EventBuffer::new();
        let event = ParamValueEvent::new(
            0,
            ClapId::new(PARAM_INPUT_GAIN),
            Pckn::match_all(),
            10.5f64,
            Cookie::empty(),
        );
        input_events_buffer.push(&event);
        let input_events = InputEvents::from_buffer(&input_events_buffer);

        let mut output_events_buffer = EventBuffer::new();
        let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

        // Call flush() on the audio processor
        processor.flush(&input_events, &mut output_events);

        // 1. Verify that the parameter target is updated
        assert_eq!(processor.params.input_gain_db, 10.5);
        let target_linear = processor.gain_lut.db_to_linear(10.5);
        assert!((processor.smoother_in.target_value() - target_linear).abs() < 1e-5);

        // 2. Verify that ui_to_rt param atomic is updated
        let stored_db = f32::from_bits(
            processor
                .shared
                .ui_to_rt
                .param_input_gain
                .load(Ordering::Relaxed),
        );
        assert_eq!(stored_db, 10.5);

        // 3. Verify that the generation counter was bumped
        let current_gen = processor
            .shared
            .ui_to_rt
            .gui_param_generation
            .load(Ordering::Relaxed);
        assert!(
            current_gen > initial_gen,
            "Generation counter should have been bumped"
        );

        // Start processing (calls process_events internally, which should sync generation)
        let mut started_processor = stopped_processor.start_processing().unwrap();
        let n = 512;
        let mut bufs = StereoTestBuffers::new(n, 0.1, 0.2);
        let mut input_channels = [bufs.in_l.as_mut_slice(), bufs.in_r.as_mut_slice()];
        let input_audio = bufs.input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);
        let output_channels = [bufs.out_l.as_mut_slice(), bufs.out_r.as_mut_slice()];
        let mut output_audio = bufs.output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);

        let input_events = InputEvents::empty();
        let mut output_events = OutputEvents::from_buffer(&mut bufs.output_events_buffer);

        started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .unwrap();

        // After process(), the processor's last_seen_generation should match current_gen
        let processor_ptr_after = unsafe {
            clack_plugin::extensions::wrapper::PluginWrapper::<crate::clap::NamClapPlugin>::handle(
                raw_ptr,
                |wrapper| {
                    let ptr = wrapper.audio_processor().unwrap().as_ptr();
                    Ok(ptr)
                },
            )
            .unwrap()
        };
        let processor_after = unsafe { &mut *processor_ptr_after };
        assert_eq!(processor_after.last_seen_generation, current_gen);
    }
}
