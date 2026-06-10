// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Parameter synchronization helpers for `NamClapProcessor`.
//!
//! Extracts the triplicated logic of updating `self.params.*`,
//! `self.shared.ui_to_rt.*`, `smoother`, and `gate_dirty` from the
//! three event-processing paths (SPSC, Host Events, GUI sync).

use super::NamClapProcessor;
use crate::clap::extensions::params::{bypass_bool_to_u32, bypass_f32_to_bool, bypass_u32_to_bool};
use crate::common::params::RtPluginParams;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    // ── Write helpers (Host Events path) ──────────────────────────

    pub(super) fn set_input_gain(&mut self, db: f32) {
        self.params.input_gain_db = db;
        self.shared
            .ui_to_rt
            .param_input_gain
            .store(db.to_bits(), Ordering::Relaxed);
        self.smoother_in
            .set_target(self.gain_lut.db_to_linear(db + self.mod_input_gain));
    }

    pub(super) fn set_output_gain(&mut self, db: f32) {
        self.params.output_gain_db = db;
        self.shared
            .ui_to_rt
            .param_output_gain
            .store(db.to_bits(), Ordering::Relaxed);
        self.smoother_out
            .set_target(self.gain_lut.db_to_linear(db + self.mod_output_gain));
    }

    pub(super) fn set_gate_threshold(&mut self, db: f32) {
        self.params.gate_threshold_db = db;
        self.shared
            .ui_to_rt
            .param_gate_thresh
            .store(db.to_bits(), Ordering::Relaxed);
        self.gate_dirty = true;
    }

    pub(super) fn set_bypass(&mut self, val: f32) {
        let bypass = bypass_f32_to_bool(val);
        self.params.bypass = bypass;
        self.shared
            .ui_to_rt
            .param_bypass
            .store(bypass_bool_to_u32(bypass), Ordering::Relaxed);
    }

    pub(super) fn set_adaptive_compute(&mut self, val: f32) {
        let mode = crate::common::params::AdaptiveComputeMode::from_f32(val);
        self.params.adaptive_compute = mode;
        self.shared
            .ui_to_rt
            .param_adaptive_compute
            .store(mode as u32, Ordering::Relaxed);
        self.adaptive_compute.set_mode(mode, &self.rt_status);
    }

    pub(super) fn set_slim_override(&mut self, val: f32) {
        let ov = crate::dsp::adaptive::SlimOverride::from_f32(val);
        self.params.slim_override = ov;
        self.shared
            .ui_to_rt
            .param_slim_override
            .store(ov as u32, Ordering::Relaxed);
        self.adaptive_compute.set_slim_override(ov);
    }

    // ── Modulation helpers ────────────────────────────────────────

    pub(super) fn set_mod_input_gain(&mut self, amount: f32) {
        self.mod_input_gain = amount;
        self.smoother_in.set_target(
            self.gain_lut
                .db_to_linear(self.params.input_gain_db + amount),
        );
    }

    pub(super) fn set_mod_output_gain(&mut self, amount: f32) {
        self.mod_output_gain = amount;
        self.smoother_out.set_target(
            self.gain_lut
                .db_to_linear(self.params.output_gain_db + amount),
        );
    }

    pub(super) fn set_mod_gate_thresh(&mut self, amount: f32) {
        self.mod_gate_thresh = amount;
        self.gate_dirty = true;
    }

    // ── SPSC full-apply ───────────────────────────────────────────

    pub(super) fn apply_params_from_spsc(&mut self, new_params: RtPluginParams) {
        let adaptive_changed = self.params.adaptive_compute != new_params.adaptive_compute;
        let slim_override_changed = self.params.slim_override != new_params.slim_override;
        self.params = new_params;
        self.smoother_in.set_target(
            self.gain_lut
                .db_to_linear(self.params.input_gain_db + self.mod_input_gain),
        );
        self.smoother_out.set_target(
            self.gain_lut
                .db_to_linear(self.params.output_gain_db + self.mod_output_gain),
        );
        if adaptive_changed {
            self.adaptive_compute
                .set_mode(self.params.adaptive_compute, &self.rt_status);
        }
        if slim_override_changed {
            self.adaptive_compute
                .set_slim_override(self.params.slim_override);
        }
    }

    // ── GUI sync helpers ──────────────────────────────────────────

    pub(super) fn sync_input_gain_from_gui(&mut self) {
        let shared_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_input_gain
                .load(Ordering::Relaxed),
        );
        if shared_db != self.params.input_gain_db {
            self.params.input_gain_db = shared_db;
            self.smoother_in
                .set_target(self.gain_lut.db_to_linear(shared_db + self.mod_input_gain));
        }
    }

    pub(super) fn sync_output_gain_from_gui(&mut self) {
        let shared_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_output_gain
                .load(Ordering::Relaxed),
        );
        if shared_db != self.params.output_gain_db {
            self.params.output_gain_db = shared_db;
            self.smoother_out
                .set_target(self.gain_lut.db_to_linear(shared_db + self.mod_output_gain));
        }
    }

    pub(super) fn sync_gate_thresh_from_gui(&mut self) {
        let shared_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_gate_thresh
                .load(Ordering::Relaxed),
        );
        if shared_db != self.params.gate_threshold_db {
            self.params.gate_threshold_db = shared_db;
            self.gate_dirty = true;
        }
    }

    pub(super) fn sync_bypass_from_gui(&mut self) {
        let shared_bypass =
            bypass_u32_to_bool(self.shared.ui_to_rt.param_bypass.load(Ordering::Relaxed));
        if shared_bypass != self.params.bypass {
            self.params.bypass = shared_bypass;
        }
    }

    pub(super) fn sync_adaptive_compute_from_gui(&mut self) {
        let shared_adaptive = crate::common::params::AdaptiveComputeMode::from_f32(
            self.shared
                .ui_to_rt
                .param_adaptive_compute
                .load(Ordering::Relaxed) as f32,
        );
        if shared_adaptive != self.params.adaptive_compute {
            self.params.adaptive_compute = shared_adaptive;
            self.adaptive_compute
                .set_mode(shared_adaptive, &self.rt_status);
        }
    }

    pub(super) fn sync_slim_override_from_gui(&mut self) {
        let shared_override = crate::dsp::adaptive::SlimOverride::from_f32(
            self.shared
                .ui_to_rt
                .param_slim_override
                .load(Ordering::Relaxed) as f32,
        );
        if shared_override != self.params.slim_override {
            self.params.slim_override = shared_override;
            self.adaptive_compute.set_slim_override(shared_override);
        }
    }
}
