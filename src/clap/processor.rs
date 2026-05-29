// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Processador de áudio CLAP.

use crate::clap::param_smoother::ParamSmoother;
use crate::clap::plugin::{ClapParamPayload, NamClapMainThread, NamClapShared};
use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::gate::{DynamicHysteresis, GateParams, GateState};
use crate::dsp::pipeline::{
    DspPipelineContext, apply_input_stage, apply_output_stage, run_inference,
};
use crate::dsp::resampler::NamResampler;
use crate::models::DynamicModel;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Processador de áudio RT-safe. Executa na audio thread do host.
///
/// Detém os buffers pré-alocados e o estado mutable da inferência.
/// É criado no `activate()` e destruído no `deactivate()`.
pub struct NamClapProcessor<'a> {
    /// Modelo ativo para o canal esquerdo (None = bypass).
    model_l: Option<Box<DynamicModel>>,
    /// Modelo ativo para o canal direito (None = bypass).
    model_r: Option<Box<DynamicModel>>,
    /// Resampler sinc polifásico (bypass quando sample_rate == 48000).
    /// Mantido em Box para descarte RT-safe sem alocação.
    resampler: Box<NamResampler>,
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

    /// Histerese para detecção de silêncio absoluto.
    silence_hyst: DynamicHysteresis,
    /// Histerese para detecção de sinal mono estável.
    mono_hyst: DynamicHysteresis,
    /// Flag indicando se estamos processando em mono (para otimização).
    process_mono: bool,

    /// Flags de status para telemetria RT.
    rt_status: Arc<RtStatusFlags>,
    /// Referência ao estado compartilhado (para devolver os canais no deactivate).
    pub(crate) shared: &'a NamClapShared,
    /// Smoothers para ganhos de entrada e saída.
    smoother_in: ParamSmoother,
    /// Smoothers para ganhos de entrada e saída.
    smoother_out: ParamSmoother,
    /// Parking lot para descarte de modelos/resamplers se o canal GC estiver cheio.
    parking_lot: [Option<GcItem>; 16],
    /// Canal SPSC: Main Thread -> Audio Thread (Consumidor).
    param_rx: Consumer<ClapParamPayload>,
    /// Canal GC: Audio Thread -> Main Thread (Produtor).
    gc_tx: Producer<GcItem>,
    /// Buffer de fallback para overflow de GC (overwrite).
    gc_overflow: Arc<GcOverflowBuffer>,
    /// Offsets de modulação (CLAP Parameter Modulation).
    mod_input_gain: f32,
    /// Offsets de modulação (CLAP Parameter Modulation).
    mod_output_gain: f32,
    /// Offsets de modulação (CLAP Parameter Modulation).
    mod_gate_thresh: f32,
    /// Handle do host para chamadas na thread de áudio.
    host: HostAudioProcessorHandle<'a>,
}

impl<'a> NamClapProcessor<'a> {
    /// Tenta enviar um item para descarte seguro (GC).
    /// Se o canal principal estiver cheio, usa o parking lot e então o overflow buffer.
    fn push_to_gc(&mut self, item: GcItem) {
        let mut item = Some(item);

        // 1. Tenta o canal principal (SPSC)
        if let Some(i) = item.take() {
            if let Err(rtrb::PushError::Full(returned)) = self.gc_tx.push(i) {
                item = Some(returned);
            } else {
                return; // Sucesso!
            }
        }

        // 2. Se falhou, tenta o Parking Lot (Array stack-based)
        if let Some(i) = item.take() {
            let mut i_opt = Some(i);
            for slot in self.parking_lot.iter_mut() {
                if slot.is_none() {
                    *slot = i_opt.take();
                    return; // Estacionado com sucesso!
                }
            }
            item = i_opt;
        }

        // 3. Se até o Parking Lot falhou, usa o Overflow Buffer (sobrescrita/leak controlado)
        if let Some(i) = item.take() {
            self.rt_status
                .set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
            self.gc_overflow.push(i);
        }
    }
}

impl<'a> PluginAudioProcessor<'a, NamClapShared, NamClapMainThread<'a>> for NamClapProcessor<'a> {
    fn activate(
        host: HostAudioProcessorHandle<'a>,
        _main_thread: &mut NamClapMainThread<'a>,
        shared: &'a NamClapShared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        #[cfg(feature = "heap-audit")]
        {
            let audit = std::env::var("NAM_HEAP_AUDIT").is_ok();
            crate::clap::heap_audit::AUDIT_ENABLED.store(audit, Ordering::Relaxed);
        }
        // 1. Extração dos canais SPSC do Shared (ownership transfer)
        let param_rx = shared
            .param_rx
            .lock()
            .expect("Failed to lock param_rx Mutex")
            .take()
            .expect("param_rx consumer has already been extracted");

        let gc_tx = shared
            .gc_tx
            .lock()
            .expect("Failed to lock gc_tx Mutex")
            .take()
            .expect("gc_tx producer has already been extracted");

        // 2. Pré-alocação de buffers intermediários (Disjoint Stages)
        let buf_capacity = (audio_config.max_frames_count as usize)
            .max(crate::dsp::pipeline::MAX_RESAMP_BUF)
            .max(1024)
            * 2;
        let buf_host_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_host_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_mid_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_mid_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_model_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_model_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_out_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_out_r = vec![0.0f32; buf_capacity].into_boxed_slice();

        // 3. Inicialização de componentes DSP
        let model_rate = shared.model_sample_rate.load(Ordering::Relaxed);
        let model_rate = if model_rate == 0 { 48000 } else { model_rate };
        let resampler = Box::new(
            NamResampler::new(audio_config.sample_rate as u32, model_rate, buf_capacity)
                .expect("Failed to create NamResampler"),
        );

        let silence_hyst = DynamicHysteresis::new();
        let mono_hyst = DynamicHysteresis::new();

        // 4. Inicialização de Smoothers (Sample-Accurate)
        // Começamos em 1.0 (ganho unitário) para evitar silêncio no primeiro bloco.
        let smoother_in = ParamSmoother::new(1.0, audio_config.sample_rate as f32, 20.0);
        let smoother_out = ParamSmoother::new(1.0, audio_config.sample_rate as f32, 20.0);

        // 5. Reporta a latência inicial ao estado compartilhado
        shared.current_latency.store(
            resampler.latency_samples(audio_config.sample_rate as u32),
            Ordering::Relaxed,
        );
        shared
            .sample_rate
            .store(audio_config.sample_rate as u32, Ordering::Relaxed);

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
            silence_hyst,
            mono_hyst,
            process_mono: false,
            rt_status: Arc::clone(&shared.rt_status),
            shared,
            smoother_in,
            smoother_out,
            param_rx,
            gc_tx,
            gc_overflow: Arc::clone(&shared.gc_overflow),
            parking_lot: Default::default(),
            mod_input_gain: 0.0,
            mod_output_gain: 0.0,
            mod_gate_thresh: 0.0,
            host,
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
        events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        #[cfg(feature = "heap-audit")]
        let _guard = if crate::clap::heap_audit::AUDIT_ENABLED.load(Ordering::Relaxed) {
            Some(crate::clap::heap_audit::TrackingGuard::new())
        } else {
            None
        };

        let start_time = minstant::Instant::now();

        // Consulta única da prioridade da thread no primeiro bloco processado
        static ONCE_PRIO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !ONCE_PRIO.load(Ordering::Relaxed) {
            unsafe {
                let thread_id = libc::pthread_self();
                let mut policy = 0i32;
                let mut param: libc::sched_param = std::mem::zeroed();
                if libc::pthread_getschedparam(thread_id, &mut policy, &mut param) == 0 {
                    self.rt_status
                        .rt_priority
                        .store(param.sched_priority, Ordering::Relaxed);
                    if policy == libc::SCHED_FIFO || policy == libc::SCHED_RR {
                        self.rt_status
                            .set_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
                    }
                }
            }
            ONCE_PRIO.store(true, Ordering::Relaxed);
        }

        // Envia quaisquer atualizações de parâmetros/gestos pendentes originados da GUI para o host.
        self.shared.write_gui_events(events.output);

        // 1. Processamento de Eventos (Main Thread SPSC)
        let lut = crate::math::dsp::gain_lut::get_gain_lut();

        while let Ok(payload) = self.param_rx.pop() {
            match payload {
                ClapParamPayload::Params(new_params) => {
                    self.params = new_params;
                    // Usa GainLut ao invés de powf() para consistência RT (~2-3 ciclos vs ~20-60).
                    self.smoother_in.set_target(
                        lut.db_to_linear(self.params.input_gain_db + self.mod_input_gain),
                    );
                    self.smoother_out.set_target(
                        lut.db_to_linear(self.params.output_gain_db + self.mod_output_gain),
                    );
                }
                ClapParamPayload::LoadModel(model_pair, new_resampler) => {
                    if let Some(old_l) = std::mem::replace(&mut self.model_l, model_pair.model_l) {
                        self.push_to_gc(GcItem::Model(old_l));
                    }
                    if let Some(old_r) = std::mem::replace(&mut self.model_r, model_pair.model_r) {
                        self.push_to_gc(GcItem::Model(old_r));
                    }
                    let old_resampler = std::mem::replace(&mut self.resampler, new_resampler);
                    self.push_to_gc(GcItem::Resampler(old_resampler));
                }
            }
        }

        // 2. Processamento de Eventos (Host Events Queue - Sample Accurate)
        use crate::clap::extensions::params::{
            PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
        };
        use clack_plugin::events::event_types::{ParamModEvent, ParamValueEvent};

        for event in events.input {
            if let Some(param_event) = event.as_event::<ParamValueEvent>() {
                let Some(clap_id) = param_event.param_id() else {
                    continue;
                };
                let val = param_event.value() as f32;
                match clap_id.get() {
                    PARAM_INPUT_GAIN => {
                        self.params.input_gain_db = val;
                        self.shared
                            .param_input_gain
                            .store(val.to_bits(), Ordering::Relaxed);
                        self.smoother_in
                            .set_target(lut.db_to_linear(val + self.mod_input_gain));
                    }
                    PARAM_OUTPUT_GAIN => {
                        self.params.output_gain_db = val;
                        self.shared
                            .param_output_gain
                            .store(val.to_bits(), Ordering::Relaxed);
                        self.smoother_out
                            .set_target(lut.db_to_linear(val + self.mod_output_gain));
                    }
                    PARAM_GATE_THRESH => {
                        self.params.gate_threshold_db = val;
                        self.shared
                            .param_gate_thresh
                            .store(val.to_bits(), Ordering::Relaxed);
                    }
                    PARAM_BYPASS => {
                        self.params.bypass = val > 0.5;
                        self.shared
                            .param_bypass
                            .store(if val > 0.5 { 1 } else { 0 }, Ordering::Relaxed);
                    }
                    _ => {}
                }
            } else if let Some(mod_event) = event.as_event::<ParamModEvent>() {
                let Some(clap_id) = mod_event.param_id() else {
                    continue;
                };
                let amount = mod_event.amount() as f32;
                match clap_id.get() {
                    PARAM_INPUT_GAIN => {
                        self.mod_input_gain = amount;
                        self.smoother_in
                            .set_target(lut.db_to_linear(self.params.input_gain_db + amount));
                    }
                    PARAM_OUTPUT_GAIN => {
                        self.mod_output_gain = amount;
                        self.smoother_out
                            .set_target(lut.db_to_linear(self.params.output_gain_db + amount));
                    }
                    PARAM_GATE_THRESH => {
                        self.mod_gate_thresh = amount;
                    }
                    _ => {}
                }
            }
        }

        // Sincroniza parâmetros alterados via GUI que não foram ecoados como eventos de entrada pelo host.
        let shared_in_db = f32::from_bits(self.shared.param_input_gain.load(Ordering::Relaxed));
        if shared_in_db != self.params.input_gain_db {
            self.params.input_gain_db = shared_in_db;
            self.smoother_in
                .set_target(lut.db_to_linear(shared_in_db + self.mod_input_gain));
        }

        let shared_out_db = f32::from_bits(self.shared.param_output_gain.load(Ordering::Relaxed));
        if shared_out_db != self.params.output_gain_db {
            self.params.output_gain_db = shared_out_db;
            self.smoother_out
                .set_target(lut.db_to_linear(shared_out_db + self.mod_output_gain));
        }

        let shared_gate_db = f32::from_bits(self.shared.param_gate_thresh.load(Ordering::Relaxed));
        if shared_gate_db != self.params.gate_threshold_db {
            self.params.gate_threshold_db = shared_gate_db;
        }

        let shared_bypass = self.shared.param_bypass.load(Ordering::Relaxed) != 0;
        if shared_bypass != self.params.bypass {
            self.params.bypass = shared_bypass;
        }

        // Monitoramento dinâmico de latência na Audio Thread
        let host_rate = self.shared.sample_rate.load(Ordering::Relaxed);
        let host_rate = if host_rate == 0 { 48000 } else { host_rate };
        let effective_latency = self.resampler.latency_samples(host_rate);
        if effective_latency != self.shared.current_latency.load(Ordering::Relaxed) {
            self.shared
                .current_latency
                .store(effective_latency, Ordering::Relaxed);
            self.host.request_callback();
        }

        for mut port_pair in &mut audio {
            let n_samples = port_pair.frames_count() as usize;
            if n_samples == 0 {
                continue;
            }
            self.rt_status
                .last_n_samples
                .store(n_samples as u32, Ordering::Relaxed);

            // Bypass explícito: copia input → output sem processamento.
            // Implementado aqui (não apenas delegado ao host) para conformidade
            // com o flag IS_BYPASS declarado no parâmetro PARAM_BYPASS.
            if self.params.bypass {
                let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
                    continue;
                };
                let mut peak_l = 0.0f32;
                let mut peak_r = 0.0f32;
                let mut channel_iter = channel_pairs.into_iter();

                if let Some(pair) = channel_iter.next() {
                    match pair {
                        ChannelPair::InputOutput(i, o) => {
                            let n = n_samples.min(o.len());
                            o[..n].copy_from_slice(&i[..n]);
                            for &sample in &o[..n] {
                                let abs_val = sample.abs();
                                if abs_val > peak_l {
                                    peak_l = abs_val;
                                }
                            }
                        }
                        ChannelPair::InPlace(io) => {
                            let n = n_samples.min(io.len());
                            for &sample in &io[..n] {
                                let abs_val = sample.abs();
                                if abs_val > peak_l {
                                    peak_l = abs_val;
                                }
                            }
                        }
                        ChannelPair::InputOnly(_) | ChannelPair::OutputOnly(_) => {}
                    }
                }
                if let Some(pair) = channel_iter.next() {
                    match pair {
                        ChannelPair::InputOutput(i, o) => {
                            let n = n_samples.min(o.len());
                            o[..n].copy_from_slice(&i[..n]);
                            for &sample in &o[..n] {
                                let abs_val = sample.abs();
                                if abs_val > peak_r {
                                    peak_r = abs_val;
                                }
                            }
                        }
                        ChannelPair::InPlace(io) => {
                            let n = n_samples.min(io.len());
                            for &sample in &io[..n] {
                                let abs_val = sample.abs();
                                if abs_val > peak_r {
                                    peak_r = abs_val;
                                }
                            }
                        }
                        ChannelPair::InputOnly(_) | ChannelPair::OutputOnly(_) => {}
                    }
                }

                let current_peak_l = f32::from_bits(self.shared.ui_peak_l.load(Ordering::Relaxed));
                if peak_l > current_peak_l {
                    self.shared
                        .ui_peak_l
                        .store(peak_l.to_bits(), Ordering::Relaxed);
                }
                let current_peak_r = f32::from_bits(self.shared.ui_peak_r.load(Ordering::Relaxed));
                if peak_r > current_peak_r {
                    self.shared
                        .ui_peak_r
                        .store(peak_r.to_bits(), Ordering::Relaxed);
                }
                if peak_l > 1.0 || peak_r > 1.0 {
                    self.shared.ui_clipped.store(true, Ordering::Relaxed);
                }
                continue;
            }

            let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
                continue;
            };

            let mut channel_iter = channel_pairs.into_iter();
            let pair_l = channel_iter.next();
            let pair_r = channel_iter.next();

            // Detecta o modo de canal e sinaliza para a GUI.
            // Quando pair_r é None, o host forneceu apenas 1 canal (modo mono).
            // Quando pair_r existe, operamos em stereo (2 canais).
            let is_stereo = pair_r.is_some();
            self.shared
                .active_channel_count
                .store(if is_stereo { 2 } else { 1 }, Ordering::Relaxed);

            // Modo mono explícito: força process_mono imediatamente, sem histerese.
            // Não há canal R para comparar, portanto a decisão é determinista.
            if !is_stereo {
                self.process_mono = true;
            }

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

            // 2. Aplicação do Ganho de Entrada (Sample-Accurate Smoothing)
            let mut input_has_clipped = false;
            for i in 0..n_samples {
                let g = self.smoother_in.tick();
                self.buf_host_l[i] *= g;
                self.buf_host_r[i] *= g;
                if self.buf_host_l[i].abs() > 1.0 || self.buf_host_r[i].abs() > 1.0 {
                    input_has_clipped = true;
                }
            }
            if input_has_clipped {
                self.shared.ui_clipped.store(true, Ordering::Relaxed);
            }

            // A LUT já foi obtida acima do loop de ports (linha ~178).
            let modulated_gate_db = self.params.gate_threshold_db + self.mod_gate_thresh;
            let gate_params = GateParams {
                threshold_open_db: modulated_gate_db,
                threshold_close_db: modulated_gate_db - 6.0,
                ..Default::default()
            };

            let mut ctx = DspPipelineContext {
                resampler: &mut self.resampler,
                active_model_l: &mut self.model_l,
                active_model_r: &mut self.model_r,
                input_gain_mult: 1.0, // Aplicado manualmente via smoother abaixo
                output_gain_mult: 1.0, // Aplicado manualmente via smoother abaixo
                gate_params: &gate_params,
                silence_hysteresis: &mut self.silence_hyst,
                mono_hysteresis: &mut self.mono_hyst,
                // Decisão técnica: powi(2) é otimizado pelo compilador como
                // uma simples multiplicação (x * x) — overhead zero. Mantido por clareza semântica
                // ("quadrado do threshold") ao invés de manual `let x = ...; x * x`.
                threshold_open_sq: lut.db_to_linear(modulated_gate_db).powi(2),
                threshold_close_sq: lut.db_to_linear(modulated_gate_db - 6.0).powi(2),
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
                1.0, // Aplicado manualmente via smoother abaixo
                ctx.silence_hysteresis,
                ctx.rt_status,
            );

            // 5. Aplicação do Ganho de Saída (Sample-Accurate Smoothing)
            for i in 0..n_out {
                let g = self.smoother_out.tick();
                self.buf_out_l[i] *= g;
                self.buf_out_r[i] *= g;
            }

            let mut peak_l = 0.0f32;
            if let Some(o_l) = out_l {
                let n = n_out.min(o_l.len());
                o_l[..n].copy_from_slice(&self.buf_out_l[..n]);
                for &sample in &self.buf_out_l[..n] {
                    let abs_val = sample.abs();
                    if abs_val > peak_l {
                        peak_l = abs_val;
                    }
                }
            }
            let mut peak_r = 0.0f32;
            if let Some(o_r) = out_r {
                let n = n_out.min(o_r.len());
                o_r[..n].copy_from_slice(&self.buf_out_r[..n]);
                for &sample in &self.buf_out_r[..n] {
                    let abs_val = sample.abs();
                    if abs_val > peak_r {
                        peak_r = abs_val;
                    }
                }
            }

            let current_peak_l = f32::from_bits(self.shared.ui_peak_l.load(Ordering::Relaxed));
            if peak_l > current_peak_l {
                self.shared
                    .ui_peak_l
                    .store(peak_l.to_bits(), Ordering::Relaxed);
            }
            let current_peak_r = f32::from_bits(self.shared.ui_peak_r.load(Ordering::Relaxed));
            if peak_r > current_peak_r {
                self.shared
                    .ui_peak_r
                    .store(peak_r.to_bits(), Ordering::Relaxed);
            }
            if peak_l > 1.0 || peak_r > 1.0 {
                self.shared.ui_clipped.store(true, Ordering::Relaxed);
            }
        }

        let elapsed_nanos = start_time.elapsed().as_nanos() as u64;
        self.rt_status
            .dsp_cycle_time
            .store(elapsed_nanos, Ordering::Relaxed);
        self.rt_status.latency_hist.record(elapsed_nanos);

        // Se o processamento excedeu 85% do tempo limite (budget) do bloco, incrementa dsp_overloads
        let sample_rate = self.shared.sample_rate.load(Ordering::Relaxed);
        let last_n_samples = self.rt_status.last_n_samples.load(Ordering::Relaxed);
        if sample_rate > 0 && last_n_samples > 0 {
            let budget_ns = (last_n_samples as u64 * 1_000_000_000) / sample_rate as u64;
            let threshold_ns = (budget_ns * 85) / 100;
            if elapsed_nanos > threshold_ns {
                self.rt_status.dsp_overloads.fetch_add(1, Ordering::Relaxed);
            }
        }

        #[cfg(feature = "heap-audit")]
        if crate::clap::heap_audit::AUDIT_ENABLED.load(Ordering::Relaxed) {
            let allocs = crate::clap::heap_audit::ALLOC_COUNT.load(Ordering::Relaxed);
            if allocs > 0 {
                eprintln!(
                    "[NAM-rs Heap Audit] ERROR: {} heap allocation(s) detected in audio thread during process()!",
                    allocs
                );
                panic!("Heap allocation detected in RT thread! Count: {}", allocs);
            }
        }

        Ok(ProcessStatus::Continue)
    }
}

#[cfg(test)]
#[path = "processor_test.rs"]
mod processor_test;
