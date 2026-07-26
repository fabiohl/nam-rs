// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Event draining: SPSC (Main Thread → Audio Thread), host events,
//! GUI parameter sync and latency monitoring.

use super::NamClapProcessor;
use crate::clap::plugin::ClapParamPayload;
use crate::common::spsc::GcItem;
use crate::models::StaticModel;
use clack_plugin::prelude::OutputEvents;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    /// Processes SPSC payloads, GUI parameter sync, latency, and render mode.
    /// Host parameter events are handled later via block-splitting in
    /// `process_dsp_audio`.
    pub(super) fn process_events(&mut self, output: &mut OutputEvents) {
        self.shared.write_gui_events(output);

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
                ClapParamPayload::LoadCabIr { adapter } => {
                    self.cold_load_cabsim(adapter);
                }
                ClapParamPayload::SetOversample { os_l, os_r } => {
                    self.cold_load_os(os_l, os_r);
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
            .load(Ordering::Acquire); // pairs with Release fetch_add em plugin/shared.rs:313, gui/ui/bypass.rs:62, gui/ui/knob.rs:281
        if generation != self.last_seen_generation {
            self.last_seen_generation = generation;

            self.sync_input_gain_from_gui();
            self.sync_output_gain_from_gui();
            self.sync_gate_thresh_from_gui();
            self.sync_bypass_from_gui();
            self.sync_adaptive_compute_from_gui();
            self.sync_slim_override_from_gui();
            self.sync_oversample_from_gui();
            self.sync_activation_from_gui();
        } // generation guard

        // Dynamic latency monitoring on the Audio Thread.
        //
        // RT-safety: `host.request_callback()` is intentionally NOT called here
        // (it would write to eventfd/pipe, violating "Zero Blocking I/O").
        // The atomic store is sufficient — the main thread's housekeeping loop
        // polls `current_latency` and calls `latency_ext.changed()` on its
        // regular cycle. Worst case: one main-thread-period delay in reporting.
        // This is a cold path (model activation/swap only), not the hot path.
        let host_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);
        let host_rate = if host_rate == 0 { 48000 } else { host_rate };
        let mut effective_latency = self.resampler.latency_samples(host_rate);
        effective_latency += self.os_l.latency_samples() as u32;
        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        {
            if let Some(ref adapter) = self.cabsim_adapter {
                effective_latency += adapter.latency_samples() as u32;
            }
        }
        if effective_latency != self.shared.rt_to_ui.current_latency.load(Ordering::Relaxed) {
            self.shared
                .rt_to_ui
                .current_latency
                .store(effective_latency, Ordering::Relaxed);
        }

        // Honor render mode override: in offline mode, force adaptive compute to Off
        // for deterministic maximum-quality output. The Main Thread writes render_mode
        // with Release ordering via `clap.render.set()`.
        let render_mode = self.shared.cold.render_mode.load(Ordering::Acquire); // pairs with Release store em extensions/render.rs:30
        if render_mode != self.last_render_mode {
            self.last_render_mode = render_mode;
            if render_mode == crate::clap::plugin::RENDER_MODE_OFFLINE {
                // ── Entering Offline ──
                // Capture an immutable snapshot of the realtime state BEFORE
                // overwriting it. This snapshot is restored when returning to
                // Realtime (S0-E0-T04, CLAP-F009).
                self.realtime_activation = self.params.activation_precision;
                self.adaptive_compute.set_mode(
                    crate::common::params::AdaptiveComputeMode::Off,
                    &self.rt_status,
                );
                self.params.activation_precision =
                    crate::common::params::ActivationPrecision::Standard;
                crate::math::activations::set_activation_precision(
                    crate::common::params::ActivationPrecision::Standard,
                );
            } else {
                // ── Returning to Realtime ──
                // Restore from the immutable snapshot captured at offline entry.
                let user_mode = crate::common::params::AdaptiveComputeMode::from_f32(
                    self.shared
                        .ui_to_rt
                        .param_adaptive_compute
                        .load(Ordering::Relaxed) as f32,
                );
                self.adaptive_compute.set_mode(user_mode, &self.rt_status);
                self.params.activation_precision = self.realtime_activation;
                crate::math::activations::set_activation_precision(self.realtime_activation);
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
        // count and signal the main thread to perform the allocation-intensive
        // slice+prewarm+mmap off the audio thread.
        self.signal_slimmable_rebuild();

        // Drain any slimmable-rebuilt models delivered from the main thread.
        self.drain_slimmable_models();
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
            // F3: buffer sizing is guaranteed on the main thread before SPSC delivery
            // (load.rs when buffer_size > 0, or flush_pending_model() in activate/housekeeping
            // when buffer_size was 0). set_max_buffer_size is NEVER called here anymore.
            // The heap-audit CI lane catches any regression.
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
    fn cold_load_cabsim(&mut self, adapter: Option<crate::dsp::cabsim::adapter::CabSimAdapter>) {
        if let Some(old_adapter) = std::mem::replace(&mut self.cabsim_adapter, adapter) {
            self.push_to_gc(GcItem::CabConvAdapter(Box::new(old_adapter)));
        }
        let cabsim_tail = self
            .cabsim_adapter
            .as_ref()
            .map(|a| (a.num_partitions() * a.latency_samples()) as u32)
            .unwrap_or(0);
        self.shared
            .rt_to_ui
            .cabsim_tail_samples
            .store(cabsim_tail, Ordering::Relaxed);
    }

    #[cold]
    fn cold_load_os(
        &mut self,
        os_l: Box<crate::dsp::oversample::OversampleEngine>,
        os_r: Box<crate::dsp::oversample::OversampleEngine>,
    ) {
        let old_l = std::mem::replace(&mut self.os_l, os_l);
        let old_r = std::mem::replace(&mut self.os_r, os_r);
        self.push_to_gc(GcItem::Oversample(old_l));
        self.push_to_gc(GcItem::Oversample(old_r));
    }

    /// Checks if the adaptive FSM demands a WaveNet channel count change
    /// and signals the main thread to perform the rebuild off the audio thread.
    ///
    /// The audio thread ONLY sets the atomic flag and target channel count.
    /// All allocation, prewarm, and mmap happen on the main thread.
    fn signal_slimmable_rebuild(&mut self) {
        let Some(target_ch) = self.adaptive_compute.take_slimmable_rebuild() else {
            return;
        };
        self.rt_status
            .requested_slimmable_ch
            .store(target_ch as u32, Ordering::Relaxed);
        self.rt_status
            .set_flag_release(crate::common::spsc::RT_STATUS_NEEDS_SLIMMABLE_REBUILD);
    }

    /// Drains slimmable-rebuilt models delivered by the main thread via SPSC.
    /// The main thread has already done slice_channels, prewarm, and set_max_buffer_size.
    /// The audio thread only swaps the pointer and sends the old model to GC.
    fn drain_slimmable_models(&mut self) {
        while let Ok(Some(new_model)) = self.slimmable_rx.pop() {
            let old = self.model_l.replace(new_model);
            if let Some(old) = old {
                self.push_to_gc(GcItem::Model(old));
            }
        }
    }
}
