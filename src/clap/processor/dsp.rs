// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bloco DSP propriamente dito: extração de canais, gate, inferência,
//! resampling, ganho de saída, picos e telemetria.

use super::NamClapProcessor;
use crate::dsp::gate::{GateParams, GateState};
use crate::dsp::pipeline::{
    DspPipelineContext, apply_input_stage, apply_output_stage, run_inference,
};
use crate::math::dsp::gain_lut::get_gain_lut;
use clack_plugin::prelude::*;
use minstant::Instant;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    /// Processa o bloco de áudio: extração de canais, bypass, gate,
    /// inferência neural, resampling, ganho de saída, picos e telemetria.
    pub(super) fn process_dsp_audio(
        &mut self,
        audio: &mut Audio,
        start_time: Instant,
    ) -> Result<ProcessStatus, PluginError> {
        for mut port_pair in audio {
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

            // Como o plugin funciona estritamente em mono, definimos a contagem de canais como 1
            // e process_mono como true. Não executamos nenhuma detecção de estéreo ativa.
            self.shared.active_channel_count.store(1, Ordering::Relaxed);
            self.process_mono = true;

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

            // Copia o input do canal esquerdo para o direito para garantir que o pipeline DSP
            // (que espera buffers válidos em ambos os lados L/R) processe o mesmo sinal mono.
            self.buf_host_r[..n_samples].copy_from_slice(&self.buf_host_l[..n_samples]);

            if let Some(pair) = pair_r {
                match pair {
                    ChannelPair::InputOutput(_, o) | ChannelPair::OutputOnly(o) => {
                        out_r = Some(o);
                    }
                    ChannelPair::InPlace(io) => {
                        out_r = Some(io);
                    }
                    ChannelPair::InputOnly(_) => {}
                }
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

            let lut = get_gain_lut();
            let modulated_gate_db = self.params.gate_threshold_db + self.mod_gate_thresh;
            let close_db = modulated_gate_db - 6.0;

            if self.gate_dirty {
                // Pré-cálculo cold-path (equivalente a pw_host.rs:855-863).
                // Ambos usam a mesma LUT db_to_linear + quadratura; mudanças
                // aqui devem ensejar revisão no standalone.
                self.cached_threshold_open_sq = lut.db_to_linear(modulated_gate_db).powi(2);
                self.cached_threshold_close_sq = lut.db_to_linear(close_db).powi(2);
                self.gate_dirty = false;
            }

            let gate_params = GateParams {
                threshold_open_db: modulated_gate_db,
                threshold_close_db: close_db,
                ..Default::default()
            };

            let mut ctx = DspPipelineContext {
                resampler: &mut self.resampler,
                active_model_l: &mut self.model_l,
                active_model_r: &mut self.active_model_r,
                input_gain_mult: 1.0, // Aplicado manualmente via smoother abaixo
                output_gain_mult: 1.0, // Aplicado manualmente via smoother abaixo
                gate_params: &gate_params,
                silence_hysteresis: &mut self.silence_hyst,
                mono_hysteresis: &mut self.mono_hyst,
                threshold_open_sq: self.cached_threshold_open_sq,
                threshold_close_sq: self.cached_threshold_close_sq,
                process_mono: &mut self.process_mono,
                rt_status: &self.rt_status,
                bridge_writer: None,
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

        // Decimação de Telemetria: Economiza ciclos medindo apenas 1 de cada 16 quadros.
        // ALGORITMO COMPARTILHADO: Mesma lógica de decimação de src/standalone/pw_host.rs (linha ~978).
        // Toda alteração aqui deve ensejar revisão também em pw_host.rs.
        let should_measure = self.cycles_since_telemetry & 0xF == 0;
        self.cycles_since_telemetry = self.cycles_since_telemetry.wrapping_add(1);

        if should_measure {
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
        }

        #[cfg(feature = "heap-audit")]
        if crate::clap::heap_audit::AUDIT_ENABLED.load(Ordering::Relaxed) {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            let audit_thread = crate::clap::heap_audit::AUDIT_THREAD.load(Ordering::Relaxed);
            if audit_thread == 0 || audit_thread == tid {
                let allocs = crate::clap::heap_audit::ALLOC_COUNT.load(Ordering::Relaxed);
                if allocs > 0 {
                    self.rt_status
                        .set_flag(crate::common::spsc::RT_STATUS_HEAP_ALLOC);
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "[NAM-rs Heap Audit] ERROR: {} heap allocation(s) detected in audio thread during process()!",
                        allocs
                    );
                    return Ok(ProcessStatus::Sleep);
                }
            }
        }

        Ok(ProcessStatus::Continue)
    }
}
