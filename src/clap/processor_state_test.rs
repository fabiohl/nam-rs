// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::clap::test_util::{self, MonoTestBuffers, StereoTestBuffers};
    use clack_host::prelude::*;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

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
            oversample: crate::dsp::oversample::OversampleFactor::Off,
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
