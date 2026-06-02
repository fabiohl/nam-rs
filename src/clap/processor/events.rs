// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Event draining: SPSC (Main Thread → Audio Thread), host events,
//! GUI parameter sync and latency monitoring.

use super::NamClapProcessor;
use crate::clap::extensions::params::{
    PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
};
use crate::clap::plugin::ClapParamPayload;
use crate::common::spsc::GcItem;
use crate::math::dsp::gain_lut::get_gain_lut;
use clack_plugin::events::event_types::{ParamModEvent, ParamValueEvent};
use clack_plugin::prelude::Events;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    /// Processes all input events: GUI gestures → host, SPSC payloads,
    /// sample-accurate host events, GUI parameter sync and latency.
    pub(super) fn process_events(&mut self, events: Events) {
        self.shared.write_gui_events(events.output);

        // 1. Event Processing (Main Thread SPSC)
        let lut = get_gain_lut();

        while let Ok(payload) = self.param_rx.pop() {
            match payload {
                ClapParamPayload::Params(new_params) => {
                    self.params = new_params;
                    self.smoother_in.set_target(
                        lut.db_to_linear(self.params.input_gain_db + self.mod_input_gain),
                    );
                    self.smoother_out.set_target(
                        lut.db_to_linear(self.params.output_gain_db + self.mod_output_gain),
                    );
                }
                ClapParamPayload::LoadModel(model_pair, new_resampler) => {
                    self.shared
                        .rt_status
                        .clear_flag(crate::common::spsc::RT_STATUS_A2_PLACEHOLDER);
                    if let Some(old_l) = std::mem::replace(&mut self.model_l, model_pair.model_l) {
                        self.push_to_gc(GcItem::Model(old_l));
                    }
                    if let Some(ref mut model) = self.model_l {
                        model.inject_rt_status(std::sync::Arc::clone(&self.shared.rt_status));
                    }
                    if let Some(model_r) = model_pair.model_r {
                        self.push_to_gc(GcItem::Model(model_r));
                    }
                    let old_resampler = std::mem::replace(&mut self.resampler, new_resampler);
                    self.push_to_gc(GcItem::Resampler(old_resampler));
                }
            }
        }

        // 2. Event Processing (Host Events Queue - Sample Accurate)
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
                        self.gate_dirty = true;
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
                        self.gate_dirty = true;
                    }
                    _ => {}
                }
            }
        }

        // Sync parameters changed via GUI that were not echoed as input events by the host.
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
            self.gate_dirty = true;
        }

        let shared_bypass = self.shared.param_bypass.load(Ordering::Relaxed) != 0;
        if shared_bypass != self.params.bypass {
            self.params.bypass = shared_bypass;
        }

        // Dynamic latency monitoring on the Audio Thread
        let host_rate = self.shared.sample_rate.load(Ordering::Relaxed);
        let host_rate = if host_rate == 0 { 48000 } else { host_rate };
        let effective_latency = self.resampler.latency_samples(host_rate);
        if effective_latency != self.shared.current_latency.load(Ordering::Relaxed) {
            self.shared
                .current_latency
                .store(effective_latency, Ordering::Relaxed);
            self.host.request_callback();
        }
    }
}
