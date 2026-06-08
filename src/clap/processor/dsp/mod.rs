// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP block proper: channel extraction, gate, inference,
//! resampling, output gain, peaks, and telemetry.

mod bypass;
mod gain;
mod gate_flags;
mod peaks;
mod telemetry;

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

            if self.process_bypass(&mut port_pair, n_samples)? {
                continue;
            }

            let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
                continue;
            };

            let mut channel_iter = channel_pairs.into_iter();
            let pair_l = channel_iter.next();
            let pair_r = channel_iter.next();

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

            self.apply_input_gain(n_samples);

            let lut = get_gain_lut();
            let modulated_gate_db = self.params.gate_threshold_db + self.mod_gate_thresh;
            let close_db = modulated_gate_db - 6.0;

            if self.gate_dirty {
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
                input_gain_mult: 1.0,
                output_gain_mult: 1.0,
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

            gate_flags::report_gate_flags(&self.rt_status, gate_state);

            if gate_state == GateState::Closed {
                if let Some(out) = out_l {
                    out.fill(0.0);
                }
                if let Some(out) = out_r {
                    out.fill(0.0);
                }
                continue;
            }

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
                1.0,
                ctx.silence_hysteresis,
                ctx.rt_status,
                *ctx.process_mono,
                ctx.adaptive,
                self.shared.cold.sample_rate.load(Ordering::Relaxed),
            );

            self.apply_output_gain(n_out);

            let (peak_l, peak_r) = self.compute_output_peaks(&mut out_l, &mut out_r, n_out);
            peaks::store_peaks(self.shared, peak_l, peak_r);
        }

        self.process_telemetry(start_time);

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
