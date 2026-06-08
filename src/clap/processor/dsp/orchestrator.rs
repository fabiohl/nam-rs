// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::NamClapProcessor;
use crate::clap::processor::dsp::{channels, gate_flags, peaks};
use crate::dsp::gate::GateState;
use crate::dsp::pipeline::{
    DspPipelineContext, apply_input_stage, apply_output_stage, run_inference,
};
use clack_plugin::prelude::*;
use minstant::Instant;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    #[inline(always)]
    pub(crate) fn process_dsp_audio(
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

            self.apply_input_gain(n_samples);

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

            // In the CLAP plugin, input/output gain is applied separately via
            // `apply_input_gain`/`apply_output_gain` (which use smoothed
            // user-configured gains from `smoother_in`/`smoother_out`), so the
            // pipeline-level multipliers are always 1.0.
            let mut ctx = DspPipelineContext {
                resampler: &mut self.resampler,
                active_model_l: &mut self.model_l,
                active_model_r: &mut self.active_model_r,
                input_gain_mult: 1.0,
                output_gain_mult: 1.0,
                gate_params: &self.cached_gate_params,
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
        if crate::common::alloc_audit::AUDIT_ENABLED.load(Ordering::Relaxed) {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            let audit_thread = crate::common::alloc_audit::AUDIT_THREAD.load(Ordering::Relaxed);
            if audit_thread == 0 || audit_thread == tid {
                let allocs = crate::common::alloc_audit::ALLOC_COUNT.load(Ordering::Relaxed);
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
