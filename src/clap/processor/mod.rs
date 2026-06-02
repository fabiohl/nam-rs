// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Processador de áudio CLAP.
//!
//! Submódulos:
//! - `events`: drenagem de eventos SPSC (Main Thread → Audio Thread) e eventos do host.
//! - `dsp`: bloco DSP propriamente dito (gate, inferência, resampling, output).

mod dsp;
mod events;

use crate::clap::param_smoother::ParamSmoother;
use crate::clap::plugin::{ClapParamPayload, NamClapMainThread, NamClapShared};
use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::gate::DynamicHysteresis;
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
    pub(crate) buf_mid_l: Box<[f32]>,
    pub(crate) buf_mid_r: Box<[f32]>,
    /// 3. Pós-modelo / Pré-resampler output (f32 @ 48kHz)
    pub(crate) buf_model_l: Box<[f32]>,
    pub(crate) buf_model_r: Box<[f32]>,
    /// 4. Pós-resampler output / Final (sample_rate variável)
    pub(crate) buf_out_l: Box<[f32]>,
    pub(crate) buf_out_r: Box<[f32]>,

    /// Histerese para detecção de silêncio absoluto.
    silence_hyst: DynamicHysteresis,
    /// Modelo ativo para o canal direito (None = processa como mono ou bypass).
    active_model_r: Option<Box<DynamicModel>>,
    /// Histerese para detecção de sinal mono. Campo persistente para evitar
    /// re-inicialização a cada iteração do port_pair.
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
    /// Thresholds pré-calculados (linear²) — invalidados apenas quando
    /// gate_threshold_db ou mod_gate_thresh muda (Exemplo: S6.T04).
    /// ALGORITMO COMPARTILHADO: Toda alteração na lógica de cache/invalidação
    /// aqui deve ser refletida em src/standalone/pw_host.rs (threshold_open_sq
    /// e threshold_close_sq), e vice-versa. Ambos pré-calculam thresholds em
    /// linear² via LUT para evitar lookups no hotpath RT.
    cached_threshold_open_sq: f32,
    cached_threshold_close_sq: f32,
    gate_dirty: bool,
    /// Decimação de telemetria: 1-em-16. Contador de ciclos desde a última medição.
    /// ALGORITMO COMPARTILHADO: Mesma estratégia de decimação de `src/standalone/pw_host.rs` (frame_count & 0xF).
    /// Toda alteração na lógica de decimação aqui deve ser refletida em pw_host.rs, e vice-versa.
    cycles_since_telemetry: u32,
    /// Handle do host para chamadas na thread de áudio.
    host: HostAudioProcessorHandle<'a>,
    /// Flag per-instância para consulta única da prioridade RT no primeiro bloco.
    prio_checked: bool,
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
            if std::env::var("NAM_HEAP_AUDIT").is_ok() {
                crate::clap::heap_audit::AUDIT_ENABLED.store(true, Ordering::Relaxed);
            }
        }
        // 1. Extração dos canais SPSC do Shared (ownership transfer)
        let param_rx = shared
            .param_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| PluginError::Message("param_rx consumer has already been extracted"))?;

        let gc_tx = shared
            .gc_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| PluginError::Message("gc_tx producer has already been extracted"))?;

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
            NamResampler::new(audio_config.sample_rate as u32, model_rate, buf_capacity).map_err(
                |e| {
                    PluginError::Message(Box::leak(
                        format!("Failed to create NamResampler: {:?}", e).into_boxed_str(),
                    ))
                },
            )?,
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
            active_model_r: None,
            mono_hyst,
            process_mono: true,
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
            cached_threshold_open_sq: 0.0,
            cached_threshold_close_sq: 0.0,
            gate_dirty: true,
            cycles_since_telemetry: 0,
            host,
            prio_checked: false,
        })
    }

    fn deactivate(self, _main_thread: &mut NamClapMainThread<'a>) {
        let mut param_rx_guard = self
            .shared
            .param_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *param_rx_guard = Some(self.param_rx);

        let mut gc_tx_guard = self.shared.gc_tx.lock().unwrap_or_else(|e| e.into_inner());
        *gc_tx_guard = Some(self.gc_tx);
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        #[cfg(feature = "heap-audit")]
        let _guard = if crate::clap::heap_audit::AUDIT_ENABLED.load(Ordering::Relaxed) {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            let audit_thread = crate::clap::heap_audit::AUDIT_THREAD.load(Ordering::Relaxed);
            if audit_thread == 0 || audit_thread == tid {
                Some(crate::clap::heap_audit::TrackingGuard::new())
            } else {
                None
            }
        } else {
            None
        };

        let start_time = minstant::Instant::now();

        // Consulta única da prioridade da thread no primeiro bloco processado
        if !self.prio_checked {
            self.prio_checked = true;
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
        }

        // Drenagem de eventos (SPSC + Host + GUI sync + Latência)
        self.process_events(events);

        // Bloco DSP (gate, inferência, resampling, output, telemetria)
        self.process_dsp_audio(&mut audio, start_time)
    }
}

#[cfg(test)]
#[path = "../processor_test.rs"]
mod processor_test;
