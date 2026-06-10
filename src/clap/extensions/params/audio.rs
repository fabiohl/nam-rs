// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `PluginAudioProcessorParams` implementation for the Audio Thread.

use super::{
    PARAM_ADAPTIVE_COMPUTE, PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
    PARAM_SLIM_OVERRIDE, bypass_bool_to_u32, bypass_f32_to_bool, bypass_u32_to_bool,
};
use crate::clap::processor::NamClapProcessor;
use clack_extensions::params::PluginAudioProcessorParams;
use clack_plugin::events::event_types::ParamValueEvent;
use clack_plugin::prelude::{InputEvents, OutputEvents};

/// Implementation of `PluginAudioProcessorParams` for the Audio Thread.
///
/// # Design: Intentional Duplication
///
/// The parameter event parsing here is structurally identical to the one in
/// `PluginMainThreadParams::flush()` above. This duplication is **intentional**:
///
/// - **Main Thread flush()**: Updates `self.params` + atomics + sends snapshot via SPSC
///   (necessary because the Audio Thread may not be active).
/// - **Audio Thread flush()**: Updates `self.params` + atomics + syncs parameters
///   coming from the GUI via direct atomic reads (no SPSC, since we're already on the Audio Thread).
///
/// Extracting a common helper would be possible but would add unnecessary indirection
/// and complicate the `self` mutable lifetimes in each context.
impl PluginAudioProcessorParams for NamClapProcessor<'_> {
    /// Processes parameter events on the Audio Thread when `process()` is not being called.
    ///
    /// In addition to applying host events, this method also syncs parameters
    /// that were changed by the GUI directly in the shared atomics.
    fn flush(&mut self, input: &InputEvents, output: &mut OutputEvents) {
        self.shared.write_gui_events(output);

        for event in input {
            let Some(param_event) = event.as_event::<ParamValueEvent>() else {
                continue;
            };
            let Some(clap_id) = param_event.param_id() else {
                continue;
            };
            let id = clap_id.get();
            let val = param_event.value() as f32;

            match id {
                PARAM_INPUT_GAIN => {
                    self.params.input_gain_db = val;
                    self.shared
                        .ui_to_rt
                        .param_input_gain
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_OUTPUT_GAIN => {
                    self.params.output_gain_db = val;
                    self.shared
                        .ui_to_rt
                        .param_output_gain
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_GATE_THRESH => {
                    self.params.gate_threshold_db = val;
                    self.shared
                        .ui_to_rt
                        .param_gate_thresh
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_BYPASS => {
                    self.params.bypass = bypass_f32_to_bool(val);
                    self.shared.ui_to_rt.param_bypass.store(
                        bypass_bool_to_u32(self.params.bypass),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                PARAM_ADAPTIVE_COMPUTE => {
                    let mode = crate::common::params::AdaptiveComputeMode::from_f32(val);
                    self.params.adaptive_compute = mode;
                    self.shared
                        .ui_to_rt
                        .param_adaptive_compute
                        .store(mode as u32, std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_SLIM_OVERRIDE => {
                    let ov = crate::dsp::adaptive::SlimOverride::from_f32(val);
                    self.params.slim_override = ov;
                    self.shared
                        .ui_to_rt
                        .param_slim_override
                        .store(ov as u32, std::sync::atomic::Ordering::Relaxed);
                }
                _ => continue,
            }
        }

        let generation = self
            .shared
            .ui_to_rt
            .gui_param_generation
            .load(std::sync::atomic::Ordering::Acquire);
        if generation != self.last_seen_generation {
            self.last_seen_generation = generation;

            let shared_in_db = f32::from_bits(
                self.shared
                    .ui_to_rt
                    .param_input_gain
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            if shared_in_db != self.params.input_gain_db {
                self.params.input_gain_db = shared_in_db;
            }

            let shared_out_db = f32::from_bits(
                self.shared
                    .ui_to_rt
                    .param_output_gain
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            if shared_out_db != self.params.output_gain_db {
                self.params.output_gain_db = shared_out_db;
            }

            let shared_gate_db = f32::from_bits(
                self.shared
                    .ui_to_rt
                    .param_gate_thresh
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            if shared_gate_db != self.params.gate_threshold_db {
                self.params.gate_threshold_db = shared_gate_db;
            }

            let shared_bypass = bypass_u32_to_bool(
                self.shared
                    .ui_to_rt
                    .param_bypass
                    .load(std::sync::atomic::Ordering::Relaxed),
            );
            if shared_bypass != self.params.bypass {
                self.params.bypass = shared_bypass;
            }

            let shared_adaptive = crate::common::params::AdaptiveComputeMode::from_f32(
                self.shared
                    .ui_to_rt
                    .param_adaptive_compute
                    .load(std::sync::atomic::Ordering::Relaxed) as f32,
            );
            if shared_adaptive != self.params.adaptive_compute {
                self.params.adaptive_compute = shared_adaptive;
            }

            let shared_slim_override = crate::dsp::adaptive::SlimOverride::from_f32(
                self.shared
                    .ui_to_rt
                    .param_slim_override
                    .load(std::sync::atomic::Ordering::Relaxed) as f32,
            );
            if shared_slim_override != self.params.slim_override {
                self.params.slim_override = shared_slim_override;
            }
        }
    }
}
