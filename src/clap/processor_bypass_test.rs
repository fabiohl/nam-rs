// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::clap::test_util::{self, StereoTestBuffers};
    use clack_host::prelude::*;

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
}
