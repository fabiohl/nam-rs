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
}

impl<'a> NamClapProcessor<'a> {
    /// Tenta enviar um item para descarte seguro (GC).
    /// Se o canal principal estiver cheio, usa o parking lot e então o overflow buffer.
    fn push_to_gc(&mut self, model: Box<DynamicModel>) {
        let mut item = Some(GcItem::Model(model));

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
        let resampler = Box::new(
            NamResampler::new(audio_config.sample_rate as u32, 48000, buf_capacity)
                .expect("Falha ao criar NamResampler"),
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
        shared.sample_rate.store(audio_config.sample_rate as u32, Ordering::Relaxed);

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
        // 1. Processamento de Eventos (Main Thread SPSC)
        let lut = crate::math::dsp::gain_lut::get_gain_lut();

        while let Ok(payload) = self.param_rx.pop() {
            match payload {
                ClapParamPayload::Params(new_params) => {
                    self.params = new_params;
                    // Usa GainLut ao invés de powf() para consistência RT (~2-3 ciclos vs ~20-60).
                    self.smoother_in
                        .set_target(lut.db_to_linear(self.params.input_gain_db));
                    self.smoother_out
                        .set_target(lut.db_to_linear(self.params.output_gain_db));
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

        // 2. Processamento de Eventos (Host Events Queue - Sample Accurate)
        use crate::clap::extensions::params::{
            PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
        };
        use clack_plugin::events::event_types::ParamValueEvent;

        for event in events.input {
            let Some(param_event) = event.as_event::<ParamValueEvent>() else {
                continue;
            };
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
                    self.smoother_in.set_target(lut.db_to_linear(val));
                }
                PARAM_OUTPUT_GAIN => {
                    self.params.output_gain_db = val;
                    self.shared
                        .param_output_gain
                        .store(val.to_bits(), Ordering::Relaxed);
                    self.smoother_out.set_target(lut.db_to_linear(val));
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
        }

        for mut port_pair in &mut audio {
            let n_samples = port_pair.frames_count() as usize;
            if n_samples == 0 {
                continue;
            }

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
            let gate_params = GateParams {
                threshold_open_db: self.params.gate_threshold_db,
                threshold_close_db: self.params.gate_threshold_db - 6.0,
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

        Ok(ProcessStatus::Continue)
    }
}

#[cfg(test)]
#[path = "processor_test.rs"]
mod processor_test;
