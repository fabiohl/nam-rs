// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![cfg(feature = "clap-plugin")]

use clack_host::prelude::*;
use std::sync::atomic::Ordering;

struct MultiHostShared;
impl<'a> SharedHandler<'a> for MultiHostShared {
    fn request_restart(&self) {}
    fn request_process(&self) {}
    fn request_callback(&self) {}
}

struct MultiHost;
impl HostHandlers for MultiHost {
    type Shared<'a> = MultiHostShared;
    type MainThread<'a> = ();
    type AudioProcessor<'a> = ();
}

const INSTANCE_COUNT: usize = 10;
const BLOCK_SIZE: usize = 256;

fn process_block(
    started_processor: &mut StartedPluginAudioProcessor<MultiHost>,
    input_audio_buf: &mut [[f32; BLOCK_SIZE]; 2],
    output_audio_buf: &mut [[f32; BLOCK_SIZE]; 2],
    events_buf: &mut EventBuffer,
) {
    let mut input_ports = AudioPorts::with_capacity(2, 1);
    let mut output_ports = AudioPorts::with_capacity(2, 1);

    let input_events = InputEvents::empty();
    let mut output_events = OutputEvents::from_buffer(events_buf);

    let mut input_audio_buffers = [
        input_audio_buf[0],
        input_audio_buf[1],
    ];
    let mut output_audio_buffers = [
        output_audio_buf[0],
        output_audio_buf[1],
    ];

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
        .expect("process() failed");

    assert!(
        status == ProcessStatus::Continue || status == ProcessStatus::Sleep,
        "Unexpected process status: {:?}",
        status
    );
}

#[test]
#[ignore = "long-running stress test"]
fn test_multi_instance_rt_priority() {
    let entry = PluginEntry::load_from_clack::<
        clack_plugin::entry::SinglePluginEntry<nam_rs::clap::plugin::NamClapPlugin>,
    >(c"/test")
    .expect("Failed to load PluginEntry");

    let host_info = HostInfo::new(
        "NAM-rs Multi-Instance Stress Test",
        "NAM-rs",
        "https://github.com/fabiohl/nam-rs",
        "0.1.0",
    )
    .expect("Failed to create HostInfo");

    let audio_config = PluginAudioConfiguration {
        sample_rate: 48000.0,
        min_frames_count: BLOCK_SIZE as u32,
        max_frames_count: BLOCK_SIZE as u32,
    };

    let mut instances: Vec<PluginInstance<MultiHost>> = Vec::with_capacity(INSTANCE_COUNT);
    let mut shared_refs: Vec<*const nam_rs::clap::plugin::NamClapShared> =
        Vec::with_capacity(INSTANCE_COUNT);

    for _i in 0..INSTANCE_COUNT {
        let mut plugin_instance = PluginInstance::<MultiHost>::new(
            |_| MultiHostShared,
            |_| (),
            &entry,
            c"br.eti.fabiolima.nam-rs",
            &host_info,
        )
        .expect("Failed to create plugin instance");

        let raw_plugin_ptr = plugin_instance.plugin_handle().as_raw_ptr();
        let shared_ptr = unsafe {
            clack_plugin::extensions::wrapper::PluginWrapper::<
                nam_rs::clap::plugin::NamClapPlugin,
            >::handle(
                raw_plugin_ptr,
                |wrapper| Ok(wrapper.shared() as *const nam_rs::clap::plugin::NamClapShared),
            )
            .expect("Failed to obtain plugin wrapper")
        };
        shared_refs.push(shared_ptr);

        let stopped_processor = plugin_instance
            .activate(|_, _| (), audio_config)
            .expect("Failed to activate plugin");

        let mut started_processor = stopped_processor
            .start_processing()
            .expect("Failed to start processing");

        let mut input_buf = [[0.1f32; BLOCK_SIZE]; 2];
        let mut output_buf = [[0.0f32; BLOCK_SIZE]; 2];
        let mut events_buf = EventBuffer::with_capacity(10);

        // Process 3 blocks to ensure the instance is fully exercised
        for _block in 0..3 {
            process_block(
                &mut started_processor,
                &mut input_buf,
                &mut output_buf,
                &mut events_buf,
            );
        }

        let stopped_processor = started_processor.stop_processing();
        plugin_instance.deactivate(stopped_processor);

        instances.push(plugin_instance);
    }

    // Verify that rt_priority is correctly set for each instance (not the sentinel -1).
    // The ONCE_PRIO flag is per-instance, so each instance should detect its own rt priority.
    for (i, shared_ptr) in shared_refs.iter().enumerate() {
        let shared = unsafe { &**shared_ptr };
        let priority = shared.rt_status.rt_priority.load(Ordering::Relaxed);
        assert_ne!(
            priority,
            -1,
            "Instance {}: rt_priority is still at sentinel value (-1). \
             Each instance must independently detect its thread priority.",
            i
        );
    }

    // Drop all instances and verify no panics on cleanup
    drop(instances);
}
