// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Processador de áudio CLAP.

use crate::clap::plugin::{NamClapMainThread, NamClapShared};
use clack_plugin::prelude::*;

/// Processador de áudio RT-safe. Executa na audio thread do host.
/// Bypass: copia cada amostra de input para output sem processar.
pub struct NamClapProcessor;

impl<'a> PluginAudioProcessor<'a, NamClapShared, NamClapMainThread<'a>> for NamClapProcessor {
    fn activate(
        _host: HostAudioProcessorHandle<'a>,
        _main_thread: &mut NamClapMainThread<'a>,
        _shared: &'a NamClapShared,
        _audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        Ok(Self)
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        _events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        for mut port_pair in &mut audio {
            let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
                continue;
            };
            for channel_pair in channel_pairs {
                match channel_pair {
                    ChannelPair::InputOnly(_) => {}
                    ChannelPair::OutputOnly(buf) => buf.fill(0.0),
                    ChannelPair::InputOutput(input, output) => {
                        output.copy_from_slice(input);
                    }
                    ChannelPair::InPlace(_) => {
                        // Comportamento correto para bypass in-place:
                        // O buffer de entrada é fisicamente o mesmo buffer de saída.
                        // Como estamos em bypass, não fazemos nada (noop).
                    }
                }
            }
        }
        Ok(ProcessStatus::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clap::NamClapPlugin;
    use crate::dsp::pipeline::test_util::infra::{ALLOC_COUNT, TrackingGuard};
    use clack_host::prelude::*;
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
        // 1. Setup do Plugin via clack-host
        // No clack-host 0.1.0, PluginEntry::load_from_clack é o caminho para plugins built-in
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

        // No clack-host 0.1, PluginInstance::new é o construtor
        let mut plugin_instance = PluginInstance::<TestHost>::new(
            |_| TestHostShared,
            |_| (),
            &entry,
            c"br.eti.fabiolima.nam-rs",
            &host_info,
        )
        .expect("Falha ao instanciar plugin");

        // 2. Configuração e Ativação
        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

        // activate retorna o processador parado
        let stopped_processor = plugin_instance.activate(|_, _| (), audio_config).unwrap();
        let mut started_processor = stopped_processor.start_processing().unwrap();

        // 3. Preparação de Buffers (Simulação de Host)
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

        // 4. Execução e Validação de Alocações
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

        // 5. Verificação de Paridade (Bypass)
        for i in 0..512 {
            assert_eq!(
                output_l[i], input_l[i],
                "Falha no bypass Canal L amostra {}",
                i
            );
            assert_eq!(
                output_r[i], input_r[i],
                "Falha no bypass Canal R amostra {}",
                i
            );
        }
    }
}
