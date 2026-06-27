// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::NamClapProcessor;
use crate::clap::processor::dsp::{channels, peaks};
use crate::dsp::gate::GateState;
use crate::dsp::gate_flags;
use crate::dsp::pipeline::{
    DspPipelineContext, apply_input_stage, apply_output_stage, run_inference,
};
use clack_plugin::prelude::*;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    #[inline(always)]
    pub(crate) fn process_dsp_audio(
        &mut self,
        audio: &mut Audio,
        start_nanos: u64,
    ) -> Result<ProcessStatus, PluginError> {
        for mut port_pair in audio {
            let n_samples = port_pair.frames_count() as usize;
            if n_samples == 0 {
                continue;
            }
            let n = n_samples as u32;
            if self.rt_status.last_n_samples.load(Ordering::Relaxed) != n {
                self.rt_status.last_n_samples.store(n, Ordering::Relaxed);
            }

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

            // In the CLAP plugin, user input/output gain is applied separately via
            // `apply_input_gain`/`apply_output_gain` (which use smoothed
            // user-configured gains from `smoother_in`/`smoother_out`). Model
            // calibration multipliers (`input_mult_adj`/`output_mult_adj`), derived
            // from `input_level_dbu`/`loudness` metadata, are applied through the
            // pipeline context — enforcing parity with the standalone.
            let mut ctx = DspPipelineContext {
                resampler: &mut self.resampler,
                os_l: &mut self.os_l,
                os_r: &mut self.os_r,
                active_model_l: &mut self.model_l,
                active_model_r: &mut None,
                input_gain_mult: self.model_input_mult_adj,
                output_gain_mult: self.model_output_mult_adj,
                gate_params: &self.cached_gate_params,
                silence_hysteresis: &mut self.silence_hyst,
                mono_hysteresis: &mut self.mono_hyst,
                threshold_open_sq: self.cached_threshold_open_sq,
                threshold_close_sq: self.cached_threshold_close_sq,
                process_mono: &mut self.process_mono,
                rt_status: &self.rt_status,
                adaptive: &mut self.adaptive_compute,
                bridge_writer: None,
                conv: self.conv_engine.as_mut().map(|e| e.as_mut()),
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
                &mut self.buf_os_in_l,
                &mut self.buf_os_in_r,
                &mut self.buf_os_model_l,
                &mut self.buf_os_model_r,
            );

            apply_output_stage(
                &mut self.buf_out_l[..n_out],
                &mut self.buf_out_r[..n_out],
                n_out,
                self.model_output_mult_adj,
                ctx.silence_hysteresis,
                ctx.rt_status,
                *ctx.process_mono,
                ctx.adaptive,
                self.shared.cold.sample_rate.load(Ordering::Relaxed),
            );

            self.apply_output_gain(n_out);

            let (peak_l, peak_r) =
                self.compute_output_peaks(&mut out_l, &mut out_r, n_out, self.process_mono);
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
