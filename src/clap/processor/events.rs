// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Event draining: SPSC (Main Thread → Audio Thread), host events,
//! GUI parameter sync and latency monitoring.

use super::NamClapProcessor;
use crate::clap::extensions::params::{
    PARAM_ADAPTIVE_COMPUTE, PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
    PARAM_SLIM_OVERRIDE,
};
use crate::clap::plugin::ClapParamPayload;
use crate::common::spsc::{GcItem, gc_cascade};
use crate::models::slimmable::try_slimmable_rebuild_single;
use crate::models::{NamModel, StaticModel};
use clack_plugin::events::event_types::{ParamModEvent, ParamValueEvent};
use clack_plugin::prelude::Events;
use std::sync::atomic::Ordering;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::cabsim::conv::ConvEngine;

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
                    new_resampler,
                    input_mult_adj,
                    output_mult_adj,
                } => self.cold_load_model(model_l, new_resampler, input_mult_adj, output_mult_adj),
                #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
                ClapParamPayload::LoadCabIr { engine } => {
                    self.cold_load_cabsim(engine);
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
                    PARAM_INPUT_GAIN => self.set_input_gain(val),
                    PARAM_OUTPUT_GAIN => self.set_output_gain(val),
                    PARAM_GATE_THRESH => self.set_gate_threshold(val),
                    PARAM_BYPASS => self.set_bypass(val),
                    PARAM_ADAPTIVE_COMPUTE => self.set_adaptive_compute(val),
                    PARAM_SLIM_OVERRIDE => self.set_slim_override(val),
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
            self.sync_slim_override_from_gui();
        } // generation guard

        // Dynamic latency monitoring on the Audio Thread
        let host_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);
        let host_rate = if host_rate == 0 { 48000 } else { host_rate };
        let mut effective_latency = self.resampler.latency_samples(host_rate);
        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        {
            if let Some(ref conv) = self.conv_engine {
                effective_latency += conv.latency_samples() as u32;
            }
        }
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

        // WaveNet slimmable rebuild: check if FSM demands a different channel
        // count and perform the allocation-intensive slice+swap before DSP.
        self.try_slimmable_rebuild();
    }

    #[cold]
    fn cold_load_model(
        &mut self,
        model_l: Option<Box<crate::models::StaticModel>>,
        new_resampler: Box<crate::dsp::resampler::NamResampler>,
        input_mult_adj: f32,
        output_mult_adj: f32,
    ) {
        if let Some(old_l) = std::mem::replace(&mut self.model_l, model_l) {
            self.push_to_gc(GcItem::Model(old_l));
        }
        if let Some(ref mut model) = self.model_l {
            model.inject_rt_status(std::sync::Arc::clone(&self.shared.cold.rt_status));
        }
        let resize_failed = if let Some(ref mut model) = self.model_l {
            model.set_max_buffer_size(self.max_frames_count).is_err()
        } else {
            false
        };
        if resize_failed {
            if let Some(failed) = self.model_l.take() {
                self.push_to_gc(GcItem::Model(failed));
            }
            self.rt_status
                .set_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED);
        }
        if self.model_l.is_none() {
            self.rt_status
                .set_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED);
        }

        let old_resampler = std::mem::replace(&mut self.resampler, new_resampler);
        self.push_to_gc(GcItem::Resampler(old_resampler));

        self.model_input_mult_adj = input_mult_adj;
        self.model_output_mult_adj = output_mult_adj;

        if let Some(ref model) = self.model_l
            && let StaticModel::WavenetDyn(w) = model.as_ref()
        {
            self.adaptive_compute.set_wavenet_full_ch(w.ch);
        }
    }

    #[cold]
    #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
    fn cold_load_cabsim(&mut self, engine: Option<Box<ConvEngine>>) {
        if let Some(old_engine) = std::mem::replace(&mut self.conv_engine, engine) {
            self.push_to_gc(GcItem::CabConvEngine(old_engine));
        }
    }

    /// Checks if the adaptive FSM demands a WaveNet channel count change
    /// and performs the allocation-intensive `slice_channels` + GC swap.
    ///
    /// Must be called **before** DSP (inside `process_events`) to keep the
    /// hot-path zero-alloc.
    fn try_slimmable_rebuild(&mut self) {
        let Some(target_ch) = self.adaptive_compute.take_slimmable_rebuild() else {
            return;
        };
        let max_frames = self.max_frames_count;
        let gc_tx = &mut self.gc_tx;
        let parking_lot = &mut self.parking_lot;
        let gc_overflow = &self.gc_overflow;
        let rt_status = &self.rt_status;

        let mut on_gc = |item: GcItem| {
            gc_cascade(Some(item), gc_tx, parking_lot, gc_overflow, rt_status);
        };

        try_slimmable_rebuild_single(
            &mut self.model_l,
            target_ch,
            Some(max_frames),
            &mut on_gc,
            &mut || {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_SLIMMABLE_SLICE_FAILED);
            },
        );
    }
}
