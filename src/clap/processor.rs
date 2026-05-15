// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Processador de áudio CLAP.

use crate::clap::plugin::{ClapParamPayload, NamClapMainThread, NamClapShared};
use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RT_STATUS_GC_OVERFLOW, RtStatusFlags};
use crate::dsp::gate::{DynamicHysteresis, GateParams, GateState};
use crate::dsp::pipeline::{
    DspPipelineContext, apply_input_stage, apply_output_stage, run_inference,
};
use crate::dsp::resampler::NamResampler;
use crate::models::DynamicModel;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::sync::Arc;

/// Processador de áudio RT-safe. Executa na audio thread do host.
///
/// Detém os buffers pré-alocados e o estado mutable da inferência.
/// É criado no `activate()` e destruído no `deactivate()`.
#[allow(dead_code)]
pub struct NamClapProcessor<'a> {
    /// Modelo ativo para o canal esquerdo (None = bypass).
    model_l: Option<Box<DynamicModel>>,
    /// Modelo ativo para o canal direito (None = bypass).
    model_r: Option<Box<DynamicModel>>,
    /// Resampler sinc polifásico (bypass quando sample_rate == 48000).
    resampler: NamResampler,
    /// Parâmetros atuais na audio thread (snapshottados do SPSC a cada process()).
    pub(crate) params: NamPluginParams,

    /// Buffers intermediários pré-alocados no activate() — ZERO alloc no process().
    /// 1. Cópia do input do host (sample_rate variável)
    buf_host_l: Box<[f32]>,
    buf_host_r: Box<[f32]>,
    /// 2. Pós-resampler input / Pré-modelo (f32 @ 48kHz)
    buf_mid_l: Box<[f32]>,
    buf_mid_r: Box<[f32]>,
    /// 3. Pós-modelo / Pré-resampler output (f32 @ 48kHz)
    buf_model_l: Box<[f32]>,
    buf_model_r: Box<[f32]>,
    /// 4. Pós-resampler output / Final (sample_rate variável)
    buf_out_l: Box<[f32]>,
    buf_out_r: Box<[f32]>,

    /// Gate FSM com histerese temporal (Noise Gate principal).
    gate: DynamicHysteresis,
    /// Histerese para detecção de silêncio absoluto.
    silence_hyst: DynamicHysteresis,
    /// Histerese para detecção de sinal mono estável.
    mono_hyst: DynamicHysteresis,
    /// Flag indicando se estamos processando em mono (para otimização).
    process_mono: bool,

    /// Flags de status para telemetria RT.
    rt_status: Arc<RtStatusFlags>,
    /// Referência ao estado compartilhado (para devolver os canais no deactivate).
    shared: &'a NamClapShared,
    /// Canal SPSC: Main Thread -> Audio Thread (Consumidor).
    param_rx: Consumer<ClapParamPayload>,
    /// Canal GC: Audio Thread -> Main Thread (Produtor).
    gc_tx: Producer<GcItem>,
    /// Buffer de fallback para overflow de GC (overwrite).
    gc_overflow: Arc<GcOverflowBuffer>,
}

impl<'a> NamClapProcessor<'a> {
    /// Tenta enviar um item para descarte seguro (GC).
    /// Se o canal principal estiver cheio, usa o buffer de overflow.
    fn push_to_gc(&mut self, model: Box<DynamicModel>) {
        if let Err(rtrb::PushError::Full(item)) = self.gc_tx.push(GcItem::Model(model)) {
            self.rt_status.set_flag(RT_STATUS_GC_OVERFLOW);
            let ptr = Box::into_raw(Box::new(item));
            // SAFETY: O buffer de overflow assume a propriedade do ponteiro raw.
            if let Some(leaked_ptr) = self.gc_overflow.push_raw(ptr) {
                unsafe { drop(Box::from_raw(leaked_ptr)) };
            }
        }
    }
}

impl<'a> PluginAudioProcessor<'a, NamClapShared, NamClapMainThread<'a>> for NamClapProcessor<'a> {
    fn activate(
        _host: HostAudioProcessorHandle<'a>,
        _main_thread: &mut NamClapMainThread<'a>,
        shared: &'a NamClapShared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        // 1. Extração dos canais SPSC do Shared (ownership transfer)
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

        // 2. Pré-alocação de buffers intermediários (Disjoint Stages)
        let buf_capacity = (audio_config.max_frames_count as usize).max(1024) * 2;
        let buf_host_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_host_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_mid_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_mid_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_model_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_model_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_out_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_out_r = vec![0.0f32; buf_capacity].into_boxed_slice();

        // 3. Inicialização de componentes DSP
        let resampler = NamResampler::new(audio_config.sample_rate as u32, 48000, buf_capacity)
            .expect("Falha ao criar NamResampler");

        let gate = DynamicHysteresis::new();
        let silence_hyst = DynamicHysteresis::new();
        let mono_hyst = DynamicHysteresis::new();

        Ok(Self {
            model_l: None,
            model_r: None,
            resampler,
            params: NamPluginParams::default(),
            buf_host_l,
            buf_host_r,
            buf_mid_l,
            buf_mid_r,
            buf_model_l,
            buf_model_r,
            buf_out_l,
            buf_out_r,
            gate,
            silence_hyst,
            mono_hyst,
            process_mono: false,
            rt_status: Arc::clone(&shared.rt_status),
            shared,
            param_rx,
            gc_tx,
            gc_overflow: Arc::clone(&shared.gc_overflow),
        })
    }

    fn deactivate(self, _main_thread: &mut NamClapMainThread<'a>) {
        if let Ok(mut guard) = self.shared.param_rx.lock() {
            *guard = Some(self.param_rx);
        }
        if let Ok(mut guard) = self.shared.gc_tx.lock() {
            *guard = Some(self.gc_tx);
        }
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        _events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        while let Ok(payload) = self.param_rx.pop() {
            match payload {
                ClapParamPayload::Params(new_params) => {
                    self.params = new_params;
                }
                ClapParamPayload::LoadModel(model_pair) => {
                    if let Some(old_l) = std::mem::replace(&mut self.model_l, model_pair.model_l) {
                        self.push_to_gc(old_l);
                    }
                    if let Some(old_r) = std::mem::replace(&mut self.model_r, model_pair.model_r) {
                        self.push_to_gc(old_r);
                    }
                }
            }
        }

        for mut port_pair in &mut audio {
            let n_samples = port_pair.frames_count() as usize;
            if n_samples == 0 {
                continue;
            }

            let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
                continue;
            };

            let mut channel_iter = channel_pairs.into_iter();
            let pair_l = channel_iter.next();
            let pair_r = channel_iter.next();

            let mut out_l: Option<&mut [f32]> = None;
            let mut out_r: Option<&mut [f32]> = None;

            if let Some(pair) = pair_l {
                match pair {
                    ChannelPair::InputOutput(i, o) => {
                        self.buf_host_l[..n_samples].copy_from_slice(&i[..n_samples]);
                        out_l = Some(o);
                    }
                    ChannelPair::InPlace(io) => {
                        self.buf_host_l[..n_samples].copy_from_slice(&io[..n_samples]);
                        out_l = Some(io);
                    }
                    ChannelPair::InputOnly(i) => {
                        self.buf_host_l[..n_samples].copy_from_slice(&i[..n_samples]);
                    }
                    ChannelPair::OutputOnly(o) => {
                        self.buf_host_l[..n_samples].fill(0.0);
                        out_l = Some(o);
                    }
                }
            } else {
                self.buf_host_l[..n_samples].fill(0.0);
            }

            if let Some(pair) = pair_r {
                match pair {
                    ChannelPair::InputOutput(i, o) => {
                        self.buf_host_r[..n_samples].copy_from_slice(&i[..n_samples]);
                        out_r = Some(o);
                    }
                    ChannelPair::InPlace(io) => {
                        self.buf_host_r[..n_samples].copy_from_slice(&io[..n_samples]);
                        out_r = Some(io);
                    }
                    ChannelPair::InputOnly(i) => {
                        self.buf_host_r[..n_samples].copy_from_slice(&i[..n_samples]);
                    }
                    ChannelPair::OutputOnly(o) => {
                        self.buf_host_r[..n_samples].fill(0.0);
                        out_r = Some(o);
                    }
                }
            } else {
                self.buf_host_r[..n_samples].fill(0.0);
            }

            let lut = crate::math::dsp::gain_lut::get_gain_lut();
            let gate_params = GateParams {
                threshold_open_db: self.params.gate_threshold_db,
                threshold_close_db: self.params.gate_threshold_db - 6.0,
                ..Default::default()
            };

            let mut ctx = DspPipelineContext {
                resampler: &mut self.resampler,
                active_model_l: &mut self.model_l,
                active_model_r: &mut self.model_r,
                input_gain_mult: lut.db_to_linear(self.params.input_gain_db),
                output_gain_mult: lut.db_to_linear(self.params.output_gain_db),
                gate_params: &gate_params,
                silence_hysteresis: &mut self.silence_hyst,
                mono_hysteresis: &mut self.mono_hyst,
                threshold_open_sq: lut.db_to_linear(self.params.gate_threshold_db).powi(2),
                threshold_close_sq: lut
                    .db_to_linear(self.params.gate_threshold_db - 6.0)
                    .powi(2),
                process_mono: &mut self.process_mono,
                rt_status: &self.rt_status,
                bridge_ptr: crate::dsp::pipeline::BridgeRef::null(),
            };

            let gate_state = apply_input_stage(
                &mut self.buf_host_l[..n_samples],
                &mut self.buf_host_r[..n_samples],
                n_samples,
                &mut ctx,
            );

            // Reporta estado do gate via flags atômicas (RT-Safe logging)
            match gate_state {
                GateState::Closed => {
                    self.rt_status
                        .set_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
                    self.rt_status
                        .clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);
                }
                GateState::FadingIn | GateState::FadingOut => {
                    self.rt_status
                        .clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
                    self.rt_status
                        .set_flag(crate::common::spsc::RT_STATUS_IS_FADING);
                }
                GateState::Open => {
                    self.rt_status
                        .clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
                    self.rt_status
                        .clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);
                }
            }

            if gate_state == GateState::Closed {
                if let Some(out) = out_l {
                    out.fill(0.0);
                }
                if let Some(out) = out_r {
                    out.fill(0.0);
                }
                continue;
            }

            // Reporta falha de modelo se bypass estiver desligado mas nenhum modelo carregado
            if ctx.active_model_l.is_none() && !self.params.bypass {
                self.rt_status
                    .set_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED);
            } else {
                self.rt_status
                    .clear_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED);
            }

            let n_out = run_inference(
                &mut self.buf_host_l[..n_samples],
                &mut self.buf_host_r[..n_samples],
                n_samples,
                &mut ctx,
                &mut self.buf_mid_l,
                &mut self.buf_mid_r,
                &mut self.buf_out_l,
                &mut self.buf_out_r,
                &mut self.buf_model_l,
                &mut self.buf_model_r,
            );

            apply_output_stage(
                &mut self.buf_out_l[..n_out],
                &mut self.buf_out_r[..n_out],
                n_out,
                ctx.output_gain_mult,
                ctx.silence_hysteresis,
                ctx.rt_status,
            );

            if let Some(o_l) = out_l {
                let n = n_out.min(o_l.len());
                o_l[..n].copy_from_slice(&self.buf_out_l[..n]);
            }
            if let Some(o_r) = out_r {
                let n = n_out.min(o_r.len());
                o_r[..n].copy_from_slice(&self.buf_out_r[..n]);
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
    #[cfg(any(feature = "standalone", test))]
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
}
