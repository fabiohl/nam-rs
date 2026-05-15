// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Processador de áudio CLAP.

use crate::clap::plugin::{ClapParamPayload, NamClapMainThread, NamClapShared};
use crate::common::params::NamPluginParams;
use crate::dsp::gate::DynamicHysteresis;
use crate::dsp::resampler::NamResampler;
use crate::models::DynamicModel;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};

/// Processador de áudio RT-safe. Executa na audio thread do host.
///
/// Detém os buffers pré-alocados e o estado mutable da inferência.
/// É criado no `activate()` e destruído no `deactivate()`.
#[allow(dead_code)]
pub struct NamClapProcessor {
    /// Modelo ativo para o canal esquerdo (None = bypass).
    #[allow(dead_code)]
    model_l: Option<Box<DynamicModel>>,
    /// Modelo ativo para o canal direito (None = bypass).
    #[allow(dead_code)]
    model_r: Option<Box<DynamicModel>>,
    /// Resampler sinc polifásico (bypass quando sample_rate == 48000).
    #[allow(dead_code)]
    resampler: NamResampler,
    /// Parâmetros atuais na audio thread (snapshottados do SPSC a cada process()).
    #[allow(dead_code)]
    params: NamPluginParams,

    /// Buffers intermediários pré-alocados no activate() — ZERO alloc no process().
    /// pós-resampler input (f32 @ 48kHz)
    #[allow(dead_code)]
    buf_mid_l: Vec<f32>,
    #[allow(dead_code)]
    buf_mid_r: Vec<f32>,
    /// pós-modelo, pré-resampler output (f32 @ 48kHz)
    #[allow(dead_code)]
    buf_out_l: Vec<f32>,
    #[allow(dead_code)]
    buf_out_r: Vec<f32>,

    /// Gate FSM com histerese temporal (Noise Gate principal).
    #[allow(dead_code)]
    gate: DynamicHysteresis,
    /// Histerese para detecção de silêncio absoluto.
    #[allow(dead_code)]
    silence_hyst: DynamicHysteresis,
    /// Histerese para detecção de sinal mono estável.
    #[allow(dead_code)]
    mono_hyst: DynamicHysteresis,
    /// Flag indicando se estamos processando em mono (para otimização).
    #[allow(dead_code)]
    process_mono: bool,

    /// Canal SPSC: Main Thread -> Audio Thread (Consumidor).
    #[allow(dead_code)]
    param_rx: Consumer<ClapParamPayload>,
    /// Canal GC: Audio Thread -> Main Thread (Produtor).
    #[allow(dead_code)]
    gc_tx: Producer<Box<DynamicModel>>,
}

impl<'a> PluginAudioProcessor<'a, NamClapShared, NamClapMainThread<'a>> for NamClapProcessor {
    fn activate(
        _host: HostAudioProcessorHandle<'a>,
        _main_thread: &mut NamClapMainThread<'a>,
        shared: &'a NamClapShared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        // 1. Extração dos canais SPSC do Shared (ownership transfer)
        // O uso de expect() é seguro aqui pois estas instâncias DEVEM estar presentes
        // e são inicializadas no new_shared().
        let param_rx = shared
            .param_rx
            .lock()
            .expect("Falha ao travar o Mutex do param_rx")
            .take()
            .expect("Consumidor param_rx já foi extraído");

        let gc_tx = shared
            .gc_tx
            .lock()
            .expect("Falha ao travar o Mutex do gc_tx")
            .take()
            .expect("Produtor gc_tx já foi extraído");

        // 2. Pré-alocação de buffers intermediários
        // Dimensionados para max_frames_count * 2 para dar margem ao resampler se necessário.
        let buf_capacity = (audio_config.max_frames_count as usize) * 2;
        let buf_mid_l = vec![0.0f32; buf_capacity];
        let buf_mid_r = vec![0.0f32; buf_capacity];
        let buf_out_l = vec![0.0f32; buf_capacity];
        let buf_out_r = vec![0.0f32; buf_capacity];

        // 3. Inicialização de componentes DSP

        // NamResampler: fonte sempre 48k (padrão NAM), alvo SR do host.
        let resampler = NamResampler::new(48000, audio_config.sample_rate as u32, buf_capacity)
            .expect("Falha ao criar NamResampler");

        // Histerese com valores padrão
        let gate = DynamicHysteresis::new();
        let silence_hyst = DynamicHysteresis::new();
        let mono_hyst = DynamicHysteresis::new();

        Ok(Self {
            model_l: None,
            model_r: None,
            resampler,
            params: NamPluginParams::default(),
            buf_mid_l,
            buf_mid_r,
            buf_out_l,
            buf_out_r,
            gate,
            silence_hyst,
            mono_hyst,
            process_mono: false,
            param_rx,
            gc_tx,
        })
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        _events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        // Implementação temporária de bypass para manter funcionalidade básica
        // enquanto o pipeline DSP completo não é portado para a Tarefa 2.1.
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
                    ChannelPair::InPlace(_) => {}
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

        // 2. Configuração e Ativação
        let audio_config = PluginAudioConfiguration {
            sample_rate: 48000.0,
            min_frames_count: 512,
            max_frames_count: 512,
        };

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
