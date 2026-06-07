// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP block proper: channel extraction, gate, inference,
//! resampling, output gain, peaks, and telemetry.

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
    /// Processes the audio block: channel extraction, bypass, gate,
    /// neural inference, resampling, output gain, peaks, and telemetry.
    pub(super) fn process_dsp_audio(
        &mut self,
        audio: &mut Audio,
        start_time: Option<Instant>,
    ) -> Result<ProcessStatus, PluginError> {
        for mut port_pair in audio {
            let n_samples = port_pair.frames_count() as usize;
            if n_samples == 0 {
                continue;
            }
            self.rt_status
                .last_n_samples
                .store(n_samples as u32, Ordering::Relaxed);

            // Explicit bypass: copy input → output without processing.
            // Implemented here (not merely delegated to the host) for compliance
            // with the IS_BYPASS flag declared on the PARAM_BYPASS parameter.
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

                let current_peak_l =
                    f32::from_bits(self.shared.rt_to_ui.ui_peak_l.load(Ordering::Relaxed));
                if peak_l > current_peak_l {
                    self.shared
                        .rt_to_ui
                        .ui_peak_l
                        .store(peak_l.to_bits(), Ordering::Relaxed);
                }
                let current_peak_r =
                    f32::from_bits(self.shared.rt_to_ui.ui_peak_r.load(Ordering::Relaxed));
                if peak_r > current_peak_r {
                    self.shared
                        .rt_to_ui
                        .ui_peak_r
                        .store(peak_r.to_bits(), Ordering::Relaxed);
                }
                if peak_l > 1.0 || peak_r > 1.0 {
                    self.shared
                        .rt_to_ui
                        .ui_clipped
                        .store(true, Ordering::Relaxed);
                }
                continue;
            }

            let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
                continue;
            };

            let mut channel_iter = channel_pairs.into_iter();
            let pair_l = channel_iter.next();
            let pair_r = channel_iter.next();

            // Since the plugin operates strictly in mono, we set the channel count to 1
            // and process_mono to true. We do not run any active stereo detection.
            self.shared
                .rt_to_ui
                .active_channel_count
                .store(1, Ordering::Relaxed);
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

            // Copy the left channel input to the right channel to ensure the DSP pipeline
            // (which expects valid buffers on both L/R sides) processes the same mono signal.
            #[cfg(feature = "stereo")]
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

            // 2. Input Gain Application (Sample-Accurate Smoothing)
            let mut input_has_clipped = false;
            #[cfg(feature = "stereo")]
            {
                let start = self.smoother_in.peek();
                let target = self.smoother_in.target_value();
                if (start - target).abs() < 1e-9 {
                    input_has_clipped = unsafe {
                        crate::math::dsp::gain::apply_gain_and_detect_clipping_stereo(
                            &mut self.buf_host_l[..n_samples],
                            &mut self.buf_host_r[..n_samples],
                            start,
                        )
                    };
                } else {
                    let step = (target - start) / n_samples as f32;
                    unsafe {
                        crate::math::dsp::gain::apply_ramp_stereo(
                            &mut self.buf_host_l[..n_samples],
                            &mut self.buf_host_r[..n_samples],
                            start,
                            step,
                        );
                    }
                    self.smoother_in.set(target);
                    let (peak_l, peak_r) = unsafe {
                        crate::math::dsp::stereo::compute_peak_abs_stereo(
                            &self.buf_host_l[..n_samples],
                            &self.buf_host_r[..n_samples],
                        )
                    };
                    if peak_l > 1.0 || peak_r > 1.0 {
                        input_has_clipped = true;
                    }
                }
            }
            #[cfg(not(feature = "stereo"))]
            {
                let start = self.smoother_in.peek();
                let target = self.smoother_in.target_value();
                if (start - target).abs() < 1e-9 {
                    crate::math::dsp::gain::apply_gain_simd(
                        &mut self.buf_host_l[..n_samples],
                        start,
                    );
                } else {
                    let step = (target - start) / n_samples as f32;
                    crate::math::dsp::gain::apply_ramp_simd(
                        &mut self.buf_host_l[..n_samples],
                        start,
                        step,
                    );
                    self.smoother_in.set(target);
                }
                for &sample in &self.buf_host_l[..n_samples] {
                    if sample.abs() > 1.0 {
                        input_has_clipped = true;
                        break;
                    }
                }
            }
            if input_has_clipped {
                self.shared
                    .rt_to_ui
                    .ui_clipped
                    .store(true, Ordering::Relaxed);
            }

            let lut = get_gain_lut();
            let modulated_gate_db = self.params.gate_threshold_db + self.mod_gate_thresh;
            let close_db = modulated_gate_db - 6.0;

            if self.gate_dirty {
                // Cold-path pre-calculation (equivalent to pw_host.rs:855-863).
                // Both use the same db_to_linear LUT + squaring; changes
                // here must trigger a review in standalone.
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
                input_gain_mult: 1.0,  // Applied manually via smoother below
                output_gain_mult: 1.0, // Applied manually via smoother below
                gate_params: &gate_params,
                silence_hysteresis: &mut self.silence_hyst,
                mono_hysteresis: &mut self.mono_hyst,
                threshold_open_sq: self.cached_threshold_open_sq,
                threshold_close_sq: self.cached_threshold_close_sq,
                process_mono: &mut self.process_mono,
                rt_status: &self.rt_status,
                adaptive: &mut self.adaptive_compute,
                bridge_writer: None,
            };

            let gate_state = apply_input_stage(
                &mut self.buf_host_l[..n_samples],
                &mut self.buf_host_r[..n_samples],
                n_samples,
                &mut ctx,
            );

            // Report gate state via atomic flags (RT-Safe logging)
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

            // Report model failure if bypass is off but no model is loaded
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
                1.0, // Applied manually via smoother below
                ctx.silence_hysteresis,
                ctx.rt_status,
                *ctx.process_mono,
                ctx.adaptive,
                self.shared.cold.sample_rate.load(Ordering::Relaxed),
            );

            // 5. Output Gain Application (Sample-Accurate Smoothing)
            #[cfg(feature = "stereo")]
            {
                let start = self.smoother_out.peek();
                let target = self.smoother_out.target_value();
                if (start - target).abs() < 1e-9 {
                    unsafe {
                        crate::math::dsp::gain::apply_gain_stereo(
                            &mut self.buf_out_l[..n_out],
                            &mut self.buf_out_r[..n_out],
                            start,
                        );
                    }
                } else {
                    let step = (target - start) / n_out as f32;
                    unsafe {
                        crate::math::dsp::gain::apply_ramp_stereo(
                            &mut self.buf_out_l[..n_out],
                            &mut self.buf_out_r[..n_out],
                            start,
                            step,
                        );
                    }
                    self.smoother_out.set(target);
                }
            }
            #[cfg(not(feature = "stereo"))]
            {
                let start = self.smoother_out.peek();
                let target = self.smoother_out.target_value();
                if (start - target).abs() < 1e-9 {
                    crate::math::dsp::gain::apply_gain_simd(&mut self.buf_out_l[..n_out], start);
                } else {
                    let step = (target - start) / n_out as f32;
                    crate::math::dsp::gain::apply_ramp_simd(
                        &mut self.buf_out_l[..n_out],
                        start,
                        step,
                    );
                    self.smoother_out.set(target);
                }
            }

            let mut peak_l = 0.0f32;
            let mut peak_r = 0.0f32;
            let mut len_l = 0;
            if let Some(o_l) = out_l {
                let n = n_out.min(o_l.len());
                o_l[..n].copy_from_slice(&self.buf_out_l[..n]);
                len_l = n;
            }
            let mut len_r = 0;
            if let Some(o_r) = out_r {
                let n = n_out.min(o_r.len());
                #[cfg(feature = "stereo")]
                o_r[..n].copy_from_slice(&self.buf_out_r[..n]);
                #[cfg(not(feature = "stereo"))]
                o_r[..n].copy_from_slice(&self.buf_out_l[..n]);
                len_r = n;
            }

            #[cfg(feature = "stereo")]
            {
                if len_l > 0 && len_r > 0 {
                    let n = len_l.min(len_r);
                    let (p_l, p_r) = unsafe {
                        crate::math::dsp::stereo::compute_peak_abs_stereo(
                            &self.buf_out_l[..n],
                            &self.buf_out_r[..n],
                        )
                    };
                    peak_l = p_l;
                    peak_r = p_r;
                } else if len_l > 0 {
                    let (p_l, _) = unsafe {
                        crate::math::dsp::stereo::compute_peak_abs_stereo(
                            &self.buf_out_l[..len_l],
                            &self.buf_out_l[..len_l],
                        )
                    };
                    peak_l = p_l;
                } else if len_r > 0 {
                    let (_, p_r) = unsafe {
                        crate::math::dsp::stereo::compute_peak_abs_stereo(
                            &self.buf_out_r[..len_r],
                            &self.buf_out_r[..len_r],
                        )
                    };
                    peak_r = p_r;
                }
            }
            #[cfg(not(feature = "stereo"))]
            {
                if len_l > 0 {
                    let (p_l, _) = unsafe {
                        crate::math::dsp::stereo::compute_peak_abs_stereo(
                            &self.buf_out_l[..len_l],
                            &self.buf_out_l[..len_l],
                        )
                    };
                    peak_l = p_l;
                    peak_r = p_l;
                } else if len_r > 0 {
                    let (p_l, _) = unsafe {
                        crate::math::dsp::stereo::compute_peak_abs_stereo(
                            &self.buf_out_l[..len_r],
                            &self.buf_out_l[..len_r],
                        )
                    };
                    peak_l = p_l;
                    peak_r = p_l;
                }
            }

            let current_peak_l =
                f32::from_bits(self.shared.rt_to_ui.ui_peak_l.load(Ordering::Relaxed));
            if peak_l > current_peak_l {
                self.shared
                    .rt_to_ui
                    .ui_peak_l
                    .store(peak_l.to_bits(), Ordering::Relaxed);
            }
            let current_peak_r =
                f32::from_bits(self.shared.rt_to_ui.ui_peak_r.load(Ordering::Relaxed));
            if peak_r > current_peak_r {
                self.shared
                    .rt_to_ui
                    .ui_peak_r
                    .store(peak_r.to_bits(), Ordering::Relaxed);
            }
            if peak_l > 1.0 || peak_r > 1.0 {
                self.shared
                    .rt_to_ui
                    .ui_clipped
                    .store(true, Ordering::Relaxed);
            }
        }

        // Telemetry: measures only when start_time is captured (1-in-16 decimation,
        // controlled by cycles_since_telemetry in process()).
        if let Some(start_time) = start_time {
            let elapsed_nanos = start_time.elapsed().as_nanos() as u64;
            self.rt_status
                .dsp_cycle_time
                .store(elapsed_nanos, Ordering::Relaxed);
            self.rt_status.latency_hist.record(elapsed_nanos);

            // If processing exceeded 85% of the block time budget, increment dsp_overloads
            let sample_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);
            let last_n_samples = self.rt_status.last_n_samples.load(Ordering::Relaxed);
            if sample_rate > 0 && last_n_samples > 0 {
                let budget_ns = (last_n_samples as u64 * 1_000_000_000) / sample_rate as u64;
                let threshold_ns = (budget_ns * 85) / 100;
                if elapsed_nanos > threshold_ns {
                    self.rt_status.dsp_overloads.fetch_add(1, Ordering::Relaxed);
                }

                // Feed adaptive compute FSM
                let latency_us = elapsed_nanos / 1000;
                let budget_us = budget_ns / 1000;
                self.adaptive_compute
                    .update(latency_us, budget_us, sample_rate, &self.rt_status);
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
