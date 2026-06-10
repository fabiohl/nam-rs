// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Event draining: SPSC (Main Thread → Audio Thread), host events,
//! GUI parameter sync and latency monitoring.

use super::NamClapProcessor;
use crate::clap::extensions::params::{
    PARAM_ADAPTIVE_COMPUTE, PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
};
use crate::clap::plugin::ClapParamPayload;
use crate::common::spsc::GcItem;
use crate::models::NamModel;
use clack_plugin::events::event_types::{ParamModEvent, ParamValueEvent};
use clack_plugin::prelude::Events;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    /// Processes all input events: GUI gestures → host, SPSC payloads,
    /// sample-accurate host events, GUI parameter sync and latency.
    pub(super) fn process_events(&mut self, events: Events) {
        self.shared.write_gui_events(events.output);

        // 0. Drain parking lot: re-try items parked during previous swaps
        //    when the GC SPSC channel was full.
        self.drain_parking_lot();

        // 1. Event Processing (Main Thread SPSC)

        while let Ok(payload) = self.param_rx.pop() {
            match payload {
                ClapParamPayload::Params(new_params) => {
                    self.apply_params_from_spsc(new_params);
                }
                ClapParamPayload::LoadModel {
                    model_l,
                    model_r,
                    new_resampler,
                } => self.cold_load_model(model_l, model_r, new_resampler),
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
                    PARAM_INPUT_GAIN => self.set_input_gain(val),
                    PARAM_OUTPUT_GAIN => self.set_output_gain(val),
                    PARAM_GATE_THRESH => self.set_gate_threshold(val),
                    PARAM_BYPASS => self.set_bypass(val),
                    PARAM_ADAPTIVE_COMPUTE => self.set_adaptive_compute(val),
                    _ => {}
                }
            } else if let Some(mod_event) = event.as_event::<ParamModEvent>() {
                let Some(clap_id) = mod_event.param_id() else {
                    continue;
                };
                let amount = mod_event.amount() as f32;
                match clap_id.get() {
                    PARAM_INPUT_GAIN => self.set_mod_input_gain(amount),
                    PARAM_OUTPUT_GAIN => self.set_mod_output_gain(amount),
                    PARAM_GATE_THRESH => self.set_mod_gate_thresh(amount),
                    _ => {}
                }
            }
        }

        // Sync parameters changed via GUI that were not echoed as input events by the host.
        // Single Acquire load of the generation counter avoids 5 Relaxed loads per block
        // in the common case where no GUI change occurred.
        let generation = self
            .shared
            .ui_to_rt
            .gui_param_generation
            .load(Ordering::Acquire);
        if generation != self.last_seen_generation {
            self.last_seen_generation = generation;

            self.sync_input_gain_from_gui();
            self.sync_output_gain_from_gui();
            self.sync_gate_thresh_from_gui();
            self.sync_bypass_from_gui();
            self.sync_adaptive_compute_from_gui();
        } // generation guard

        // Dynamic latency monitoring on the Audio Thread
        let host_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);
        let host_rate = if host_rate == 0 { 48000 } else { host_rate };
        let effective_latency = self.resampler.latency_samples(host_rate);
        if effective_latency != self.shared.rt_to_ui.current_latency.load(Ordering::Relaxed) {
            self.shared
                .rt_to_ui
                .current_latency
                .store(effective_latency, Ordering::Relaxed);
            self.host.request_callback();
        }

        // Honor render mode override: in offline mode, force adaptive compute to Off
        // for deterministic maximum-quality output. The Main Thread writes render_mode
        // with Release ordering via `clap.render.set()`.
        let render_mode = self.shared.cold.render_mode.load(Ordering::Acquire);
        if render_mode != self.last_render_mode {
            self.last_render_mode = render_mode;
            if render_mode == crate::clap::plugin::RENDER_MODE_OFFLINE {
                self.adaptive_compute.set_mode(
                    crate::common::params::AdaptiveComputeMode::Off,
                    &self.rt_status,
                );
            } else {
                let user_mode = crate::common::params::AdaptiveComputeMode::from_f32(
                    self.shared
                        .ui_to_rt
                        .param_adaptive_compute
                        .load(Ordering::Relaxed) as f32,
                );
                self.adaptive_compute.set_mode(user_mode, &self.rt_status);
            }
        }
        // Also guard against user changing adaptive compute while offline (via host events
        // or SPSC, which may have bypassed the offline constraint in this same block).
        if render_mode == crate::clap::plugin::RENDER_MODE_OFFLINE
            && self.adaptive_compute.mode() != crate::common::params::AdaptiveComputeMode::Off
        {
            self.adaptive_compute.set_mode(
                crate::common::params::AdaptiveComputeMode::Off,
                &self.rt_status,
            );
        }
    }

    #[cold]
    fn cold_load_model(
        &mut self,
        model_l: Option<Box<crate::models::StaticModel>>,
        model_r: Option<Box<crate::models::StaticModel>>,
        new_resampler: Box<crate::dsp::resampler::NamResampler>,
    ) {
        if let Some(old_l) = std::mem::replace(&mut self.model_l, model_l) {
            self.push_to_gc(GcItem::Model(old_l));
        }
        if let Some(ref mut model) = self.model_l {
            model.inject_rt_status(std::sync::Arc::clone(&self.shared.cold.rt_status));
            model.set_max_buffer_size(self.max_frames_count);
        }
        if let Some(model_r) = model_r {
            self.push_to_gc(GcItem::Model(model_r));
        }
        let old_resampler = std::mem::replace(&mut self.resampler, new_resampler);
        self.push_to_gc(GcItem::Resampler(old_resampler));
    }
}
