// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::clap::NamClapPlugin;

    use crate::common::params::NamPluginParams;
    use crate::dsp::pipeline::test_util::infra::{ALLOC_COUNT, TrackingGuard};
    use clack_extensions::state::PluginState;
    use clack_host::prelude::*;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    #[allow(dead_code)]
    struct TestHostShared;
    impl<'a> SharedHandler<'a> for TestHostShared {
        fn request_restart(&self) {}
        fn request_process(&self) {}
        fn request_callback(&self) {}
    }

    #[allow(dead_code)]
    struct TestHost;
    impl HostHandlers for TestHost {
        type Shared<'a> = TestHostShared;
        type MainThread<'a> = ();
        type AudioProcessor<'a> = ();
    }

    #[test]
    fn test_zero_alloc_process_bypass() {
        let entry = PluginEntry::load_from_clack::<
            clack_plugin::entry::SinglePluginEntry<NamClapPlugin>,
        >(c"/test")
        .expect("Falha ao carregar PluginEntry");

        let host_info = HostInfo::new(
            "NAM-rs-Test",
            "NAM",
            "https://github.com/fabiohl/nam-rs",
            "0.1.0",
        )
        .expect("Falha ao criar HostInfo");

        let mut plugin_instance = PluginInstance::<TestHost>::new(
            |_| TestHostShared,
            |_| (),
            &entry,
            c"br.eti.fabiolima.nam-rs",
            &host_info,
        )
        .expect("Falha ao instanciar plugin");

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let mut input_l = [0.1f32; 512];
        let mut input_r = [0.2f32; 512];
        let mut output_l = [0.0f32; 512];
        let mut output_r = [0.0f32; 512];

        let mut input_ports = AudioPorts::with_capacity(2, 1);
        let mut output_ports = AudioPorts::with_capacity(2, 1);

        let mut input_channels = [input_l.as_mut_slice(), input_r.as_mut_slice()];
        let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_input_only(
                input_channels.iter_mut().map(InputChannel::constant),
            ),
        }]);

        let output_channels = [output_l.as_mut_slice(), output_r.as_mut_slice()];
        let mut output_audio = output_ports.with_output_buffers([AudioPortBuffer {
            latency: 0,
            channels: AudioPortBufferType::f32_output_only(output_channels.into_iter()),
        }]);

        let input_events =
            InputEvents::from_buffer::<[clack_host::events::event_types::NoteOnEvent; 0]>(&[]);
        let mut output_events_buffer = EventBuffer::new();
        let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

        let _guard = TrackingGuard::new();
        let before = ALLOC_COUNT.load(Ordering::Relaxed);

        started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .expect("Falha no process()");

        let after = ALLOC_COUNT.load(Ordering::Relaxed);
        let diff = after - before;

        println!(
            "[CLAP Zero-Alloc Test] Alocações detectadas no process(): {}",
            diff
        );
        assert_eq!(
            diff, 0,
            "Alocações detectadas no hot-path do CLAP via clack-host!"
        );

        for i in 0..512 {
            assert!(
                (output_l[i] - input_l[i]).abs() < 1e-4,
                "Falha no bypass Canal L amostra {}: {} vs {}",
                i,
                output_l[i],
                input_l[i]
            );
            assert!(
                (output_r[i] - input_r[i]).abs() < 1e-4,
                "Falha no bypass Canal R amostra {}: {} vs {}",
                i,
                output_r[i],
                input_r[i]
            );
        }
    }

    #[test]
    fn test_irregular_block_sizes_stress() {
        let entry = PluginEntry::load_from_clack::<
            clack_plugin::entry::SinglePluginEntry<NamClapPlugin>,
        >(c"/test")
        .expect("Falha ao carregar PluginEntry");

        let host_info = HostInfo::new("Test", "Test", "Test", "0.1.0").unwrap();

        let mut plugin_instance = PluginInstance::<TestHost>::new(
            |_| TestHostShared,
            |_| (),
            &entry,
            c"br.eti.fabiolima.nam-rs",
            &host_info,
        )
        .expect("Falha ao instanciar plugin");

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 1,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        // Sequência expandida conforme Tarefa 3.2.2
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
                .unwrap_or_else(|_| panic!("Falha no process() para n={}", n));

            for i in 0..n {
                assert!(
                    out_l[i].is_finite(),
                    "NaN/Inf detectado no canal L para n={}",
                    n
                );
                assert!(
                    out_r[i].is_finite(),
                    "NaN/Inf detectado no canal R para n={}",
                    n
                );
            }
        }
    }

    #[test]
    fn test_model_switching_stress() {
        let entry = PluginEntry::load_from_clack::<
            clack_plugin::entry::SinglePluginEntry<NamClapPlugin>,
        >(c"/test")
        .expect("Falha ao carregar PluginEntry");

        let host_info = HostInfo::new("Test", "Test", "Test", "0.1.0").unwrap();

        let mut plugin_instance = PluginInstance::<TestHost>::new(
            |_| TestHostShared,
            |_| (),
            &entry,
            c"br.eti.fabiolima.nam-rs",
            &host_info,
        )
        .expect("Falha ao instanciar plugin");

        let state_ext = plugin_instance
            .plugin_handle()
            .get_extension::<PluginState>()
            .expect("Extensão de Estado não encontrada");

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_dir.push("tests/fixtures/models");

        let models = [
            "BossWN-nano.nam",
            "BossWN-feather.nam",
            "BossWN-standard.nam",
            "mock_a2.nam",
        ];

        let n = 512;
        let mut in_l = vec![0.0f32; n];
        let mut in_r = vec![0.0f32; n];
        let mut out_l = vec![0.0f32; n];
        let mut out_r = vec![0.0f32; n];

        let mut input_ports = AudioPorts::with_capacity(2, 1);
        let mut output_ports = AudioPorts::with_capacity(2, 1);
        let mut output_events_buffer = EventBuffer::new();

        // 1000 ciclos de process()
        for i in 0..1000 {
            // Troca de modelo a cada 50 ciclos
            if i % 50 == 0 {
                let model_name = models[(i / 50) % models.len()];
                let mut path = model_dir.clone();
                path.push(model_name);

                let params = NamPluginParams {
                    model_path: Some(path),
                    input_gain_db: 0.0,
                    output_gain_db: 0.0,
                    gate_threshold_db: -70.0,
                    bypass: false,
                };
                let state_bytes = serde_json::to_vec(&params).unwrap();
                let mut handle = plugin_instance.plugin_handle();
                state_ext
                    .load(&mut handle, &mut state_bytes.as_slice())
                    .expect("Falha ao carregar estado");
            }

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

            // Verificação de Zero-Alloc durante a troca de modelo (no hot-path)
            let _guard = TrackingGuard::new();
            let before = ALLOC_COUNT.load(Ordering::Relaxed);

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

            let after = ALLOC_COUNT.load(Ordering::Relaxed);
            assert_eq!(
                after - before,
                0,
                "Alocação detectada no process() durante ciclo {}",
                i
            );
        }
    }

    #[test]
    fn test_parameter_modulation_stress() {
        let entry = PluginEntry::load_from_clack::<
            clack_plugin::entry::SinglePluginEntry<NamClapPlugin>,
        >(c"/test")
        .expect("Falha ao carregar PluginEntry");

        let host_info = HostInfo::new("Test", "Test", "Test", "0.1.0").unwrap();

        let mut plugin_instance = PluginInstance::<TestHost>::new(
            |_| TestHostShared,
            |_| (),
            &entry,
            c"br.eti.fabiolima.nam-rs",
            &host_info,
        )
        .expect("Falha ao instanciar plugin");

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        let n = 512;
        let mut in_l = vec![0.5f32; n];
        let mut in_r = vec![0.5f32; n];
        let mut out_l = vec![0.0f32; n];
        let mut out_r = vec![0.0f32; n];

        let mut input_ports = AudioPorts::with_capacity(2, 1);
        let mut output_ports = AudioPorts::with_capacity(2, 1);

        // Pre-warm the resampler with the DC signal to avoid step response ringing
        {
            let mut input_channels = [in_l.as_mut_slice(), in_r.as_mut_slice()];
            let input_audio = input_ports.with_input_buffers([AudioPortBuffer {
                latency: 0,
                channels: AudioPortBufferType::f32_input_only(
                    input_channels.iter_mut().map(InputChannel::constant),
                ),
            }]);

            let mut out_l_pre = vec![0.0f32; n];
            let mut out_r_pre = vec![0.0f32; n];
            let output_channels = [out_l_pre.as_mut_slice(), out_r_pre.as_mut_slice()];
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
                .expect("Falha no pre-warm");
        }

        let mut output_events_buffer = EventBuffer::new();

        use crate::clap::extensions::params::PARAM_INPUT_GAIN;
        use clack_common::events::Pckn;
        use clack_common::events::event_types::ParamValueEvent;
        use clack_common::utils::{ClapId, Cookie};

        // Cria eventos de modulação intensa (um por sample)
        let mut input_events_buffer = EventBuffer::new();
        for i in 0..n {
            // Modula o ganho de -20dB a +20dB linearmente
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

        let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);

        let _guard = TrackingGuard::new();
        let before = ALLOC_COUNT.load(Ordering::Relaxed);

        started_processor
            .process(
                &input_audio,
                &mut output_audio,
                &input_events,
                &mut output_events,
                None,
                None,
            )
            .expect("Falha no process() com modulação intensa");

        let after = ALLOC_COUNT.load(Ordering::Relaxed);
        assert_eq!(
            after - before,
            0,
            "Alocação detectada no process() durante modulação"
        );

        // Verificação de "Zipper Noise" (continuidade)
        for i in 1..n {
            let diff_l = (out_l[i] - out_l[i - 1]).abs();
            let diff_r = (out_r[i] - out_r[i - 1]).abs();
            assert!(
                diff_l < 0.05,
                "Possível zipper noise detectado no canal L amostra {}",
                i
            );
            assert!(
                diff_r < 0.05,
                "Possível zipper noise detectado no canal R amostra {}",
                i
            );
        }
    }

    #[cfg(feature = "clap-plugin-gui")]
    #[test]
    fn test_gui_extension_x11() {
        use clack_extensions::gui::{GuiApiType, GuiConfiguration, GuiSize, PluginGui};

        let entry = PluginEntry::load_from_clack::<
            clack_plugin::entry::SinglePluginEntry<NamClapPlugin>,
        >(c"/test")
        .expect("Falha ao carregar PluginEntry");

        let host_info = HostInfo::new("Test", "Test", "Test", "0.1.0").unwrap();

        let mut plugin_instance = PluginInstance::<TestHost>::new(
            |_| TestHostShared,
            |_| (),
            &entry,
            c"br.eti.fabiolima.nam-rs",
            &host_info,
        )
        .expect("Falha ao instanciar plugin");

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
        assert!(!gui_ext.is_api_supported(
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
        assert_eq!(size.height, 280);

        // 4. Test can_resize
        assert!(!gui_ext.can_resize(&mut handle));

        // 5. Test set_size
        assert!(
            gui_ext
                .set_size(
                    &mut handle,
                    GuiSize {
                        width: 600,
                        height: 280
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
        let entry = PluginEntry::load_from_clack::<
            clack_plugin::entry::SinglePluginEntry<NamClapPlugin>,
        >(c"/test")
        .expect("Falha ao carregar PluginEntry");

        let host_info = HostInfo::new("Test", "Test", "Test", "0.1.0").unwrap();

        let mut plugin_instance = PluginInstance::<TestHost>::new(
            |_| TestHostShared,
            |_| (),
            &entry,
            c"br.eti.fabiolima.nam-rs",
            &host_info,
        )
        .expect("Falha ao instanciar plugin");

        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        // 1. Check basename update on load_model
        let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        model_dir.push("tests/fixtures/models/BossWN-nano.nam");

        let state_ext = plugin_instance
            .plugin_handle()
            .get_extension::<PluginState>()
            .expect("PluginState extension not found");

        let params = NamPluginParams {
            model_path: Some(model_dir),
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            gate_threshold_db: -70.0,
            bypass: false,
        };
        let state_bytes = serde_json::to_vec(&params).unwrap();
        let mut handle = plugin_instance.plugin_handle();
        state_ext
            .load(&mut handle, &mut state_bytes.as_slice())
            .expect("Falha ao carregar estado");

        // Obtain shared instance
        let raw_plugin_ptr = plugin_instance.plugin_handle().as_raw_ptr();
        let shared_ptr = unsafe {
            clack_plugin::extensions::wrapper::PluginWrapper::<NamClapPlugin>::handle(
                raw_plugin_ptr,
                |wrapper| Ok(wrapper.shared() as *const crate::clap::plugin::NamClapShared),
            )
            .expect("Falha ao obter wrapper do plugin")
        };
        let shared = unsafe { &*shared_ptr };

        // Check model name basename was updated
        {
            let name_guard = shared.ui_model_name.lock().unwrap();
            assert_eq!(*name_guard, "BossWN-nano.nam");
        }

        // 2. Activate and process to check peak and clipping telemetry
        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        // Let's reset the peaks
        shared.ui_peak_l.store(0.0f32.to_bits(), Ordering::Relaxed);
        shared.ui_peak_r.store(0.0f32.to_bits(), Ordering::Relaxed);
        shared.ui_clipped.store(false, Ordering::Relaxed);

        let n = 512;
        // Signal with peaks at 0.5 (left) and 1.5 (right - clipping)
        let mut in_l = vec![0.5f32; n];
        let mut in_r = vec![1.5f32; n];
        let mut out_l = vec![0.0f32; n];
        let mut out_r = vec![0.0f32; n];

        let mut input_ports = AudioPorts::with_capacity(2, 1);
        let mut output_ports = AudioPorts::with_capacity(2, 1);
        let mut output_events_buffer = EventBuffer::new();

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
            .unwrap();

        // Verify peaks are greater than 0.0
        let peak_l = f32::from_bits(shared.ui_peak_l.load(Ordering::Relaxed));
        let peak_r = f32::from_bits(shared.ui_peak_r.load(Ordering::Relaxed));
        assert!(peak_l > 0.0, "peak_l was: {}", peak_l);
        assert!(peak_r > 0.0, "peak_r was: {}", peak_r);

        // Verify that clipping occurred because input right was 1.5
        let clipped = shared.ui_clipped.load(Ordering::Relaxed);
        assert!(
            clipped,
            "ui_clipped should be true since input right was 1.5"
        );
    }
}
