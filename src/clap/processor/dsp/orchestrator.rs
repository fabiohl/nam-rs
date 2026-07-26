// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::NamClapProcessor;
use crate::clap::extensions::params::{
    PARAM_ACTIVATION, PARAM_ADAPTIVE_COMPUTE, PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN,
    PARAM_OUTPUT_GAIN, PARAM_OVERSAMPLE, PARAM_SLIM_OVERRIDE,
};
use crate::clap::processor::dsp::{channels, peaks};
use crate::clap::processor::state::BypassCrossfader;
use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;
use crate::dsp::gate::GateState;
use crate::dsp::gate_flags;
use crate::dsp::pipeline::{
    DspPipelineContext, apply_input_stage, apply_output_stage, run_inference,
};
use crate::math::dsp::stereo::compute_peak_abs_stereo;
use clack_plugin::events::event_types::{ParamModEvent, ParamValueEvent};
use clack_plugin::prelude::*;
use std::sync::atomic::Ordering;

const MAX_SCHEDULED_EVENTS: usize = 4096;

#[derive(Clone, Copy)]
pub(crate) struct ScheduledEvent {
    pub(crate) time: usize,
    pub(crate) param_id: u32,
    pub(crate) value: f32,
    pub(crate) is_mod: bool,
}

impl<'a> NamClapProcessor<'a> {
    #[inline(always)]
    pub(crate) fn process_dsp_audio(
        &mut self,
        audio: &mut Audio,
        input_events: &InputEvents,
        start_nanos: u64,
    ) -> Result<ProcessStatus, PluginError> {
        {
            let events = &mut self.scheduled_events;
            events.clear();

            for event in input_events {
                if events.len() >= MAX_SCHEDULED_EVENTS {
                    debug_assert!(
                        false,
                        "CLAP-F007: event flood > {MAX_SCHEDULED_EVENTS} in one block; truncating"
                    );
                    break;
                }
                let time = event.header().time() as usize;
                if let Some(param_event) = event.as_event::<ParamValueEvent>() {
                    let Some(clap_id) = param_event.param_id() else {
                        continue;
                    };
                    events.push(ScheduledEvent {
                        time,
                        param_id: clap_id.get(),
                        value: param_event.value() as f32,
                        is_mod: false,
                    });
                } else if let Some(mod_event) = event.as_event::<ParamModEvent>() {
                    let Some(clap_id) = mod_event.param_id() else {
                        continue;
                    };
                    events.push(ScheduledEvent {
                        time,
                        param_id: clap_id.get(),
                        value: mod_event.amount() as f32,
                        is_mod: true,
                    });
                }
            }
        }

        let event_count = self.scheduled_events.len();
        let mut event_idx = 0;

        for mut port_pair in audio {
            let n_samples_raw = port_pair.frames_count() as usize;
            if n_samples_raw > self.max_frames_count {
                debug_assert!(
                    false,
                    "Host contract violation: n_samples={n_samples_raw} > max_frames_count={}",
                    self.max_frames_count
                );
                self.rt_status.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION);
            }
            let n_samples = n_samples_raw.min(self.max_frames_count);
            if n_samples == 0 {
                continue;
            }
            let n = n_samples as u32;
            if self.rt_status.last_n_samples.load(Ordering::Relaxed) != n {
                self.rt_status.last_n_samples.store(n, Ordering::Relaxed);
            }

            let Some((mut out_l, mut out_r)) = channels::extract_channels(
                &mut port_pair,
                &mut self.buf_host_l,
                &mut self.buf_host_r,
                &self.shared.rt_to_ui.active_channel_count,
                &mut self.process_mono,
                n_samples,
            )?
            else {
                continue;
            };

            if self
                .shared
                .ui_to_rt
                .host_r_deactivated
                .load(Ordering::Acquire)
            {
                self.process_mono = true;
            }

            if self.gate_dirty {
                let modulated_gate_db = self.params.gate_threshold_db + self.mod_gate_thresh;
                let close_db = modulated_gate_db - 6.0;
                self.cached_threshold_open_sq =
                    self.gain_lut.db_to_linear(modulated_gate_db).powi(2);
                self.cached_threshold_close_sq = self.gain_lut.db_to_linear(close_db).powi(2);
                self.cached_gate_params.threshold_open_db = modulated_gate_db;
                self.cached_gate_params.threshold_close_db = close_db;
                self.gate_dirty = false;
            }

            let model_load_fail = self.model_l.is_none() && !self.params.bypass;
            if model_load_fail {
                self.rt_status
                    .set_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED);
            } else {
                self.rt_status
                    .clear_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED);
            }

            let mut block_offset = 0usize;
            let mut output_offset = 0usize;
            let mut peak_l = 0.0f32;
            let mut peak_r = 0.0f32;
            let mut last_gate_state = GateState::Open;
            let mut any_active = false;
            let model_output_mult_adj = self.model_output_mult_adj;
            let shared_sample_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);

            let mut input_clipped = false;

            // Sync crossfader with current bypass state. If bypass was changed
            // by SPSC event sync (process_events) before this block, trigger
            // the crossfade to avoid click artifacts.
            self.bypass_xfade.trigger(self.params.bypass);

            while block_offset < n_samples {
                while event_idx < event_count
                    && self.scheduled_events[event_idx].time < block_offset
                {
                    event_idx += 1;
                }

                let sub_end = if event_idx < event_count {
                    let et = self.scheduled_events[event_idx].time;
                    if et < n_samples { et } else { n_samples }
                } else {
                    n_samples
                };

                let sub_n = sub_end - block_offset;
                if sub_n > 0 {
                    let bypass = self.params.bypass;
                    let process_mono = self.process_mono;

                    let (n_out, gate_state) = {
                        let mut ctx = DspPipelineContext {
                            resampler: &mut self.resampler,
                            os_l: &mut self.os_l,
                            os_r: &mut self.os_r,
                            active_model_l: &mut self.model_l,
                            active_model_r: &mut None,
                            input_gain_mult: self.model_input_mult_adj,
                            output_gain_mult: model_output_mult_adj,
                            gate_params: &self.cached_gate_params,
                            silence_hysteresis: &mut self.silence_hyst,
                            mono_hysteresis: &mut self.mono_hyst,
                            threshold_open_sq: self.cached_threshold_open_sq,
                            threshold_close_sq: self.cached_threshold_close_sq,
                            process_mono: &mut self.process_mono,
                            rt_status: &self.rt_status,
                            adaptive: &mut self.adaptive_compute,
                            bridge_writer: None,
                            conv: self.cabsim_adapter.as_mut(),
                        };

                        process_sub_block(
                            block_offset,
                            sub_n,
                            &mut out_l,
                            &mut out_r,
                            output_offset,
                            &mut ctx,
                            bypass,
                            process_mono,
                            &mut self.bypass_xfade,
                            &mut self.buf_xfade_dry_l,
                            &mut self.buf_xfade_dry_r,
                            &mut input_clipped,
                            &mut self.smoother_in,
                            &mut self.smoother_out,
                            &mut self.buf_host_l,
                            &mut self.buf_host_r,
                            &mut self.buf_mid_l,
                            &mut self.buf_mid_r,
                            &mut self.buf_out_l,
                            &mut self.buf_out_r,
                            &mut self.buf_model_l,
                            &mut self.buf_model_r,
                            &mut self.buf_os_in_l,
                            &mut self.buf_os_in_r,
                            &mut self.buf_os_model_l,
                            &mut self.buf_os_model_r,
                            model_output_mult_adj,
                            shared_sample_rate,
                            self.gain_lut,
                        )
                    };

                    if input_clipped {
                        self.shared
                            .rt_to_ui
                            .ui_clipped
                            .store(true, Ordering::Relaxed);
                    }

                    output_offset += n_out;

                    if let (Some(o_l), Some(o_r)) = (&out_l, &out_r) {
                        let o_start = output_offset - n_out;
                        let o_end = o_start + n_out;
                        let avail_l = o_l.len().min(o_end).saturating_sub(o_start);
                        let avail_r = o_r.len().min(o_end).saturating_sub(o_start);
                        let n = avail_l.min(avail_r);
                        if n > 0 {
                            let (pl, pr) = unsafe {
                                compute_peak_abs_stereo(
                                    &o_l[o_start..o_start + n],
                                    &o_r[o_start..o_start + n],
                                )
                            };
                            peak_l = peak_l.max(pl);
                            peak_r = peak_r.max(pr);
                        }
                    } else if let Some(o_l) = &out_l {
                        let o_start = output_offset - n_out;
                        let o_end = o_start + n_out;
                        let avail = o_l.len().min(o_end).saturating_sub(o_start);
                        if avail > 0 {
                            let (pl, _) = unsafe {
                                compute_peak_abs_stereo(
                                    &o_l[o_start..o_start + avail],
                                    &o_l[o_start..o_start + avail],
                                )
                            };
                            peak_l = peak_l.max(pl);
                            peak_r = peak_r.max(pl);
                        }
                    }

                    if gate_state != GateState::Closed {
                        last_gate_state = gate_state;
                        any_active = true;
                    }
                }

                while event_idx < event_count && self.scheduled_events[event_idx].time == sub_end {
                    let evt = &self.scheduled_events[event_idx];
                    apply_scheduled_event(
                        evt.param_id,
                        evt.value,
                        evt.is_mod,
                        &mut self.params,
                        &mut self.smoother_in,
                        &mut self.smoother_out,
                        &mut self.gate_dirty,
                        &mut self.mod_input_gain,
                        &mut self.mod_output_gain,
                        &mut self.mod_gate_thresh,
                        &mut self.adaptive_compute,
                        &self.rt_status,
                        &self.shared.ui_to_rt,
                        self.gain_lut,
                    );
                    event_idx += 1;
                }

                if self.gate_dirty {
                    let modulated_gate_db = self.params.gate_threshold_db + self.mod_gate_thresh;
                    let close_db = modulated_gate_db - 6.0;
                    self.cached_threshold_open_sq =
                        self.gain_lut.db_to_linear(modulated_gate_db).powi(2);
                    self.cached_threshold_close_sq = self.gain_lut.db_to_linear(close_db).powi(2);
                    self.cached_gate_params.threshold_open_db = modulated_gate_db;
                    self.cached_gate_params.threshold_close_db = close_db;
                    self.gate_dirty = false;
                }

                // Trigger bypass crossfade if the bypass state changed via
                // a host event at this sub-block boundary.
                self.bypass_xfade.trigger(self.params.bypass);

                block_offset = sub_end;
            }

            if any_active {
                gate_flags::report_gate_flags(&self.rt_status, last_gate_state);
            } else {
                gate_flags::report_gate_flags(&self.rt_status, GateState::Closed);
            }

            peaks::store_peaks(self.shared, peak_l, peak_r);
        }

        self.process_telemetry(start_nanos);

        #[cfg(feature = "heap-audit")]
        if crate::common::alloc_audit::AUDIT_ENABLED.load(Ordering::Relaxed) {
            let allocs = crate::common::alloc_audit::get_alloc_count();
            if allocs > 0 {
                self.rt_status
                    .set_flag(crate::common::spsc::RT_STATUS_HEAP_ALLOC);
                return Ok(ProcessStatus::Sleep);
            }
        }

        Ok(ProcessStatus::Continue)
    }
}

#[inline(always)]
#[expect(clippy::too_many_arguments)]
fn process_sub_block(
    offset: usize,
    n_samples: usize,
    out_l: &mut Option<&mut [f32]>,
    out_r: &mut Option<&mut [f32]>,
    output_offset: usize,
    ctx: &mut DspPipelineContext<'_>,
    bypass: bool,
    process_mono: bool,
    crossfader: &mut BypassCrossfader,
    buf_xfade_dry_l: &mut [f32],
    buf_xfade_dry_r: &mut [f32],
    input_clipped: &mut bool,
    smoother_in: &mut crate::dsp::smoother::ParamSmoother,
    smoother_out: &mut crate::dsp::smoother::ParamSmoother,
    buf_host_l: &mut [f32],
    buf_host_r: &mut [f32],
    buf_mid_l: &mut [f32],
    buf_mid_r: &mut [f32],
    buf_out_l: &mut [f32],
    buf_out_r: &mut [f32],
    buf_model_l: &mut [f32],
    buf_model_r: &mut [f32],
    buf_os_in_l: &mut [f32],
    buf_os_in_r: &mut [f32],
    buf_os_model_l: &mut [f32],
    buf_os_model_r: &mut [f32],
    model_output_mult_adj: f32,
    shared_sample_rate: u32,
    gain_lut: &crate::math::dsp::gain_lut::GainLUT,
) -> (usize, GateState) {
    if crossfader.active {
        return process_crossfade_sub_block(
            offset,
            n_samples,
            out_l,
            out_r,
            output_offset,
            ctx,
            process_mono,
            crossfader,
            buf_xfade_dry_l,
            buf_xfade_dry_r,
            input_clipped,
            smoother_in,
            smoother_out,
            buf_host_l,
            buf_host_r,
            buf_mid_l,
            buf_mid_r,
            buf_out_l,
            buf_out_r,
            buf_model_l,
            buf_model_r,
            buf_os_in_l,
            buf_os_in_r,
            buf_os_model_l,
            buf_os_model_r,
            model_output_mult_adj,
            shared_sample_rate,
            gain_lut,
        );
    }

    if bypass {
        copy_bypass_to_output(
            out_l,
            out_r,
            &buf_host_l[offset..offset + n_samples],
            &buf_host_r[offset..offset + n_samples],
            output_offset,
            process_mono,
        );
        return (n_samples, GateState::Open);
    }

    apply_iir_gain_ramp_sub_block(
        smoother_in,
        buf_host_l,
        buf_host_r,
        offset,
        n_samples,
        true,
        input_clipped,
    );

    let gate_state = apply_input_stage(
        &mut buf_host_l[offset..offset + n_samples],
        &mut buf_host_r[offset..offset + n_samples],
        n_samples,
        ctx,
    );

    if gate_state == GateState::Closed {
        copy_silence_to_output(out_l, out_r, output_offset, n_samples, process_mono);
        return (n_samples, GateState::Closed);
    }

    let n_out = run_inference(
        &mut buf_host_l[offset..offset + n_samples],
        &mut buf_host_r[offset..offset + n_samples],
        n_samples,
        ctx,
        buf_mid_l,
        buf_mid_r,
        buf_out_l,
        buf_out_r,
        buf_model_l,
        buf_model_r,
        buf_os_in_l,
        buf_os_in_r,
        buf_os_model_l,
        buf_os_model_r,
    );

    if let Some(ref mut conv) = ctx.conv {
        if !conv.is_passthrough() {
            conv.process_variable(
                &buf_out_l[..n_out],
                &mut buf_model_l[..n_out],
            );
            unsafe {
                core::ptr::copy_nonoverlapping(
                    buf_model_l.as_ptr(),
                    buf_out_l.as_mut_ptr(),
                    n_out,
                );
            }
            unsafe {
                core::ptr::copy_nonoverlapping(
                    buf_out_l.as_ptr(),
                    buf_out_r.as_mut_ptr(),
                    n_out,
                );
            }
        }
    }

    apply_output_stage(
        &mut buf_out_l[..n_out],
        &mut buf_out_r[..n_out],
        n_out,
        model_output_mult_adj,
        ctx.silence_hysteresis,
        ctx.rt_status,
        *ctx.process_mono,
        ctx.adaptive,
        shared_sample_rate,
    );

    apply_iir_gain_ramp_sub_block(
        smoother_out,
        buf_out_l,
        buf_out_r,
        0,
        n_out,
        false,
        &mut false,
    );

    copy_output_from_sub_block(
        out_l,
        out_r,
        buf_out_l,
        buf_out_r,
        n_out,
        output_offset,
        process_mono,
    );

    (n_out, gate_state)
}

#[inline(always)]
#[expect(clippy::too_many_arguments)]
fn process_crossfade_sub_block(
    offset: usize,
    n_samples: usize,
    out_l: &mut Option<&mut [f32]>,
    out_r: &mut Option<&mut [f32]>,
    output_offset: usize,
    ctx: &mut DspPipelineContext<'_>,
    process_mono: bool,
    crossfader: &mut BypassCrossfader,
    buf_xfade_dry_l: &mut [f32],
    buf_xfade_dry_r: &mut [f32],
    input_clipped: &mut bool,
    smoother_in: &mut crate::dsp::smoother::ParamSmoother,
    smoother_out: &mut crate::dsp::smoother::ParamSmoother,
    buf_host_l: &mut [f32],
    buf_host_r: &mut [f32],
    buf_mid_l: &mut [f32],
    buf_mid_r: &mut [f32],
    buf_out_l: &mut [f32],
    buf_out_r: &mut [f32],
    buf_model_l: &mut [f32],
    buf_model_r: &mut [f32],
    buf_os_in_l: &mut [f32],
    buf_os_in_r: &mut [f32],
    buf_os_model_l: &mut [f32],
    buf_os_model_r: &mut [f32],
    model_output_mult_adj: f32,
    shared_sample_rate: u32,
    _gain_lut: &crate::math::dsp::gain_lut::GainLUT,
) -> (usize, GateState) {
    // 1. Save dry input before pipeline modifies buf_host in place
    let dry_n = n_samples.min(buf_xfade_dry_l.len());
    buf_xfade_dry_l[..dry_n].copy_from_slice(&buf_host_l[offset..offset + dry_n]);
    #[cfg(feature = "stereo")]
    buf_xfade_dry_r[..dry_n].copy_from_slice(&buf_host_r[offset..offset + dry_n]);
    #[cfg(not(feature = "stereo"))]
    buf_xfade_dry_r[..dry_n].copy_from_slice(&buf_xfade_dry_l[..dry_n]);

    // 2. Run full wet pipeline
    apply_iir_gain_ramp_sub_block(
        smoother_in,
        buf_host_l,
        buf_host_r,
        offset,
        n_samples,
        true,
        input_clipped,
    );

    let gate_state = apply_input_stage(
        &mut buf_host_l[offset..offset + n_samples],
        &mut buf_host_r[offset..offset + n_samples],
        n_samples,
        ctx,
    );

    let n_out = if gate_state == GateState::Closed {
        // Gate closed: wet = silence, dry = original input.
        // Crossfade still blends dry→silence smoothly.
        buf_out_l[..n_samples].fill(0.0);
        buf_out_r[..n_samples].fill(0.0);
        n_samples
    } else {
        let n_o = run_inference(
            &mut buf_host_l[offset..offset + n_samples],
            &mut buf_host_r[offset..offset + n_samples],
            n_samples,
            ctx,
            buf_mid_l,
            buf_mid_r,
            buf_out_l,
            buf_out_r,
            buf_model_l,
            buf_model_r,
            buf_os_in_l,
            buf_os_in_r,
            buf_os_model_l,
            buf_os_model_r,
        );

        if let Some(ref mut conv) = ctx.conv {
            if !conv.is_passthrough() {
                conv.process_variable(
                    &buf_out_l[..n_o],
                    &mut buf_model_l[..n_o],
                );
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        buf_model_l.as_ptr(),
                        buf_out_l.as_mut_ptr(),
                        n_o,
                    );
                }
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        buf_out_l.as_ptr(),
                        buf_out_r.as_mut_ptr(),
                        n_o,
                    );
                }
            }
        }

        apply_output_stage(
            &mut buf_out_l[..n_o],
            &mut buf_out_r[..n_o],
            n_o,
            model_output_mult_adj,
            ctx.silence_hysteresis,
            ctx.rt_status,
            *ctx.process_mono,
            ctx.adaptive,
            shared_sample_rate,
        );

        apply_iir_gain_ramp_sub_block(
            smoother_out,
            buf_out_l,
            buf_out_r,
            0,
            n_o,
            false,
            &mut false,
        );

        n_o
    };

    // 3. Crossfade blend: output = dry * (1 - mix_i) + wet * mix_i
    let n_xfade = n_out.min(crossfader.remaining);
    let step = crossfader.step;
    let mut mix = crossfader.mix;

    // Blended portion (first n_xfade samples): ramp from current mix towards target
    for i in 0..n_xfade {
        let dry_l = buf_xfade_dry_l[i];
        let dry_r = buf_xfade_dry_r[i];
        buf_out_l[i] = dry_l + (buf_out_l[i] - dry_l) * mix;
        buf_out_r[i] = dry_r + (buf_out_r[i] - dry_r) * mix;
        mix += step;
    }

    // Pure portion (remaining n_out - n_xfade samples): final mix value
    let final_mix = if crossfader.target { 0.0 } else { 1.0 };
    if (final_mix - 1.0f32).abs() > f32::EPSILON {
        // final_mix is 0.0 (dry target): copy dry to output
        for i in n_xfade..n_out {
            let di = i.min(dry_n.saturating_sub(1));
            buf_out_l[i] = buf_xfade_dry_l[di];
            buf_out_r[i] = buf_xfade_dry_r[di];
        }
    }
    // If final_mix is 1.0 (wet target): buf_out already has wet, nothing to do

    crossfader.mix = mix;
    crossfader.remaining = crossfader.remaining.saturating_sub(n_xfade);
    if crossfader.remaining == 0 {
        crossfader.active = false;
        crossfader.mix = final_mix;
    }

    // 4. Copy blended result to output
    copy_output_from_sub_block(
        out_l,
        out_r,
        buf_out_l,
        buf_out_r,
        n_out,
        output_offset,
        process_mono,
    );

    (n_out, gate_state)
}

#[inline(always)]
fn apply_iir_gain_ramp_sub_block(
    smoother: &mut crate::dsp::smoother::ParamSmoother,
    buf_l: &mut [f32],
    buf_r: &mut [f32],
    offset: usize,
    n: usize,
    detect_clip: bool,
    input_clipped: &mut bool,
) {
    let start = smoother.peek();
    let target = smoother.target_value();

    // Fast path: gain is stable — single SIMD multiply.
    if (start - target).abs() < 1e-9 {
        #[cfg(feature = "stereo")]
        {
            if detect_clip {
                let clipped = unsafe {
                    crate::math::dsp::gain::apply_gain_and_detect_clipping_stereo(
                        &mut buf_l[offset..offset + n],
                        &mut buf_r[offset..offset + n],
                        start,
                    )
                };
                if clipped {
                    *input_clipped = true;
                }
            } else {
                unsafe {
                    crate::math::dsp::gain::apply_gain_stereo(
                        &mut buf_l[offset..offset + n],
                        &mut buf_r[offset..offset + n],
                        start,
                    );
                }
            }
        }
        #[cfg(not(feature = "stereo"))]
        {
            let _ = buf_r;
            if detect_clip {
                let clipped = unsafe {
                    crate::math::dsp::gain::apply_gain_and_detect_clipping_mono(
                        &mut buf_l[offset..offset + n],
                        start,
                    )
                };
                if clipped {
                    *input_clipped = true;
                }
            } else {
                crate::math::dsp::gain::apply_gain_simd(&mut buf_l[offset..offset + n], start);
            }
        }
        return;
    }

    // IIR exponential ramp: exactly matches tick() output for all block sizes.
    // y[i] = target + (1-α)^(i+1) * (start - target)
    // Single branchless loop replaces the old small-block (< 8 tick path)
    // and large-block (linear ramp + snap) paths.
    let alpha = smoother.alpha();
    let beta = 1.0 - alpha;
    let diff = start - target;
    let mut bp = beta;

    let slice_l = &mut buf_l[offset..offset + n];
    let slice_r = &mut buf_r[offset..offset + n];

    #[cfg(feature = "stereo")]
    {
        for i in 0..n {
            let gain = target + bp * diff;
            unsafe {
                *slice_l.get_unchecked_mut(i) *= gain;
                *slice_r.get_unchecked_mut(i) *= gain;
            }
            bp *= beta;
            if detect_clip && (slice_l[i].abs() > 1.0 || slice_r[i].abs() > 1.0) {
                *input_clipped = true;
            }
        }
    }
    #[cfg(not(feature = "stereo"))]
    {
        let _ = buf_r;
        for i in 0..n {
            let gain = target + bp * diff;
            unsafe {
                *slice_l.get_unchecked_mut(i) *= gain;
            }
            bp *= beta;
            if detect_clip && slice_l[i].abs() > 1.0 {
                *input_clipped = true;
            }
        }
    }

    // After n iterations, bp = beta^(n+1).
    // The last smoother state is y[n-1] = target + beta^n * diff = target + (bp / beta) * diff.
    let final_val = target + (bp / beta) * diff;
    smoother.set(final_val);
}

#[expect(clippy::too_many_arguments)]
fn apply_scheduled_event(
    param_id: u32,
    value: f32,
    is_mod: bool,
    params: &mut crate::common::params::RtPluginParams,
    smoother_in: &mut crate::dsp::smoother::ParamSmoother,
    smoother_out: &mut crate::dsp::smoother::ParamSmoother,
    gate_dirty: &mut bool,
    mod_input_gain: &mut f32,
    mod_output_gain: &mut f32,
    mod_gate_thresh: &mut f32,
    adaptive_compute: &mut crate::dsp::adaptive::AdaptiveCompute,
    rt_status: &crate::common::spsc::RtStatusFlags,
    ui_to_rt: &crate::clap::plugin::UiToRt,
    gain_lut: &crate::math::dsp::gain_lut::GainLUT,
) {
    use crate::clap::extensions::params::{bypass_bool_to_u32, bypass_f32_to_bool};
    use std::sync::atomic::Ordering;

    if is_mod {
        let amount = value;
        match param_id {
            PARAM_INPUT_GAIN => {
                *mod_input_gain = amount;
                smoother_in.set_target(gain_lut.db_to_linear(params.input_gain_db + amount));
            }
            PARAM_OUTPUT_GAIN => {
                *mod_output_gain = amount;
                smoother_out.set_target(gain_lut.db_to_linear(params.output_gain_db + amount));
            }
            PARAM_GATE_THRESH => {
                *mod_gate_thresh = amount;
                *gate_dirty = true;
            }
            _ => {}
        }
    } else {
        let val = value;
        match param_id {
            PARAM_INPUT_GAIN => {
                params.input_gain_db = val;
                ui_to_rt
                    .param_input_gain
                    .store(val.to_bits(), Ordering::Relaxed);
                smoother_in.set_target(gain_lut.db_to_linear(val + *mod_input_gain));
            }
            PARAM_OUTPUT_GAIN => {
                params.output_gain_db = val;
                ui_to_rt
                    .param_output_gain
                    .store(val.to_bits(), Ordering::Relaxed);
                smoother_out.set_target(gain_lut.db_to_linear(val + *mod_output_gain));
            }
            PARAM_GATE_THRESH => {
                params.gate_threshold_db = val;
                ui_to_rt
                    .param_gate_thresh
                    .store(val.to_bits(), Ordering::Relaxed);
                *gate_dirty = true;
            }
            PARAM_BYPASS => {
                let bypass = bypass_f32_to_bool(val);
                params.bypass = bypass;
                ui_to_rt
                    .param_bypass
                    .store(bypass_bool_to_u32(bypass), Ordering::Relaxed);
            }
            PARAM_ADAPTIVE_COMPUTE => {
                let mode = crate::common::params::AdaptiveComputeMode::from_f32(val);
                params.adaptive_compute = mode;
                ui_to_rt
                    .param_adaptive_compute
                    .store(mode as u32, Ordering::Relaxed);
                adaptive_compute.set_mode(mode, rt_status);
            }
            PARAM_SLIM_OVERRIDE => {
                let ov = crate::dsp::adaptive::SlimOverride::from_f32(val);
                params.slim_override = ov;
                ui_to_rt
                    .param_slim_override
                    .store(ov as u32, Ordering::Relaxed);
                adaptive_compute.set_slim_override(ov);
            }
            PARAM_OVERSAMPLE => {
                let factor = crate::dsp::oversample::OversampleFactor::from_f32(val);
                if factor != params.oversample {
                    params.oversample = factor;
                    ui_to_rt
                        .param_oversample
                        .store(factor.to_f32() as u32, Ordering::Relaxed);
                    rt_status
                        .requested_os_factor
                        .store(factor.to_f32() as u32, Ordering::Relaxed);
                    rt_status.set_flag_release(crate::common::spsc::RT_STATUS_NEEDS_OS_REBUILD);
                }
            }
            PARAM_ACTIVATION => {
                let mode = crate::common::params::ActivationPrecision::from_f32(val);
                if mode != params.activation_precision {
                    params.activation_precision = mode;
                    ui_to_rt
                        .param_activation
                        .store(mode as u32, Ordering::Relaxed);
                    crate::math::activations::set_activation_precision(mode);
                }
            }
            _ => {}
        }
    }
}

#[inline(always)]
fn copy_silence_to_output(
    out_l: &mut Option<&mut [f32]>,
    out_r: &mut Option<&mut [f32]>,
    output_offset: usize,
    n_samples: usize,
    _process_mono: bool,
) {
    if let Some(o_l) = out_l {
        let end = (output_offset + n_samples).min(o_l.len());
        o_l[output_offset..end].fill(0.0);
    }
    if let Some(o_r) = out_r {
        let end = (output_offset + n_samples).min(o_r.len());
        o_r[output_offset..end].fill(0.0);
    }
}

#[inline(always)]
fn copy_output_from_sub_block(
    out_l: &mut Option<&mut [f32]>,
    out_r: &mut Option<&mut [f32]>,
    buf_out_l: &[f32],
    buf_out_r: &[f32],
    n_out: usize,
    output_offset: usize,
    process_mono: bool,
) {
    if let Some(o_l) = out_l {
        let n = n_out.min(o_l.len().saturating_sub(output_offset));
        o_l[output_offset..output_offset + n].copy_from_slice(&buf_out_l[..n]);
    }
    if let Some(o_r) = out_r {
        let n = n_out.min(o_r.len().saturating_sub(output_offset));
        if process_mono {
            o_r[output_offset..output_offset + n].copy_from_slice(&buf_out_l[..n]);
        } else {
            o_r[output_offset..output_offset + n].copy_from_slice(&buf_out_r[..n]);
        }
    }
}

#[inline(always)]
fn copy_bypass_to_output(
    out_l: &mut Option<&mut [f32]>,
    out_r: &mut Option<&mut [f32]>,
    buf_host_l: &[f32],
    buf_host_r: &[f32],
    output_offset: usize,
    process_mono: bool,
) {
    if let Some(o_l) = out_l {
        let n = buf_host_l
            .len()
            .min(o_l.len().saturating_sub(output_offset));
        o_l[output_offset..output_offset + n].copy_from_slice(&buf_host_l[..n]);
    }
    if let Some(o_r) = out_r {
        let n = buf_host_l
            .len()
            .min(o_r.len().saturating_sub(output_offset));
        if process_mono {
            o_r[output_offset..output_offset + n].copy_from_slice(&buf_host_l[..n]);
        } else {
            o_r[output_offset..output_offset + n].copy_from_slice(&buf_host_r[..n]);
        }
    }
}
