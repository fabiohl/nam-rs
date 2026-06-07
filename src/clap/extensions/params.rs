// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the CLAP parameters extension for NAM-rs.

use crate::clap::plugin::{ClapParamPayload, NamClapMainThread};
use crate::clap::processor::NamClapProcessor;
use crate::common::params::RtPluginParams;
use crate::math::constants::{GAIN_MAX_DB, GAIN_MIN_DB};
use clack_extensions::params::{
    ParamDisplayWriter, ParamInfo, ParamInfoFlags, ParamInfoWriter, PluginAudioProcessorParams,
    PluginMainThreadParams,
};
use clack_plugin::events::event_types::ParamValueEvent;
use clack_plugin::prelude::{ClapId, InputEvents, OutputEvents};
use std::ffi::CStr;

/// Constants for CLAP parameter IDs.
/// Input gain parameter ID.
pub const PARAM_INPUT_GAIN: u32 = 0;
/// Output gain parameter ID.
pub const PARAM_OUTPUT_GAIN: u32 = 1;
/// Noise gate threshold parameter ID.
pub const PARAM_GATE_THRESH: u32 = 2;
/// Plugin bypass parameter ID.
pub const PARAM_BYPASS: u32 = 3;
/// Loaded model name parameter ID (read-only).
pub const PARAM_ACTIVE_MODEL: u32 = 4;
/// Adaptive compute mode parameter ID.
pub const PARAM_ADAPTIVE_COMPUTE: u32 = 5;

impl PluginMainThreadParams for NamClapMainThread<'_> {
    fn count(&mut self) -> u32 {
        6
    }

    fn get_info(&mut self, param_index: u32, info: &mut ParamInfoWriter) {
        match param_index {
            PARAM_INPUT_GAIN => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_INPUT_GAIN),
                    flags: ParamInfoFlags::IS_AUTOMATABLE | ParamInfoFlags::IS_MODULATABLE,
                    cookie: clack_plugin::utils::Cookie::empty(),
                    name: b"Input Gain",
                    module: b"",
                    min_value: GAIN_MIN_DB as f64,
                    max_value: GAIN_MAX_DB as f64,
                    default_value: 0.0,
                });
            }
            PARAM_OUTPUT_GAIN => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_OUTPUT_GAIN),
                    flags: ParamInfoFlags::IS_AUTOMATABLE | ParamInfoFlags::IS_MODULATABLE,
                    cookie: clack_plugin::utils::Cookie::empty(),
                    name: b"Output Gain",
                    module: b"",
                    min_value: GAIN_MIN_DB as f64,
                    max_value: GAIN_MAX_DB as f64,
                    default_value: 0.0,
                });
            }
            PARAM_GATE_THRESH => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_GATE_THRESH),
                    flags: ParamInfoFlags::IS_AUTOMATABLE | ParamInfoFlags::IS_MODULATABLE,
                    cookie: clack_plugin::utils::Cookie::empty(),
                    name: b"Gate Threshold",
                    module: b"",
                    min_value: -90.0,
                    max_value: -40.0,
                    default_value: -70.0,
                });
            }
            PARAM_BYPASS => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_BYPASS),
                    flags: ParamInfoFlags::IS_AUTOMATABLE
                        | ParamInfoFlags::IS_STEPPED
                        | ParamInfoFlags::IS_BYPASS,
                    cookie: clack_plugin::utils::Cookie::empty(),
                    name: b"Bypass",
                    module: b"",
                    min_value: 0.0,
                    max_value: 1.0,
                    default_value: 0.0,
                });
            }
            PARAM_ACTIVE_MODEL => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_ACTIVE_MODEL),
                    flags: ParamInfoFlags::IS_READONLY,
                    cookie: clack_plugin::utils::Cookie::empty(),
                    name: b"Active Model",
                    module: b"",
                    min_value: 0.0,
                    max_value: 1000.0,
                    default_value: 0.0,
                });
            }
            PARAM_ADAPTIVE_COMPUTE => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_ADAPTIVE_COMPUTE),
                    flags: ParamInfoFlags::IS_AUTOMATABLE | ParamInfoFlags::IS_STEPPED,
                    cookie: clack_plugin::utils::Cookie::empty(),
                    name: b"Adaptive Compute",
                    module: b"",
                    min_value: 0.0,
                    max_value: 2.0,
                    default_value: 1.0, // Conservative (default for CLAP plugin)
                });
            }
            _ => {}
        }
    }

    fn get_value(&mut self, id: ClapId) -> Option<f64> {
        match id.get() {
            PARAM_INPUT_GAIN => Some(f32::from_bits(
                self.shared
                    .ui_to_rt
                    .param_input_gain
                    .load(std::sync::atomic::Ordering::Relaxed),
            ) as f64),
            PARAM_OUTPUT_GAIN => Some(f32::from_bits(
                self.shared
                    .ui_to_rt
                    .param_output_gain
                    .load(std::sync::atomic::Ordering::Relaxed),
            ) as f64),
            PARAM_GATE_THRESH => Some(f32::from_bits(
                self.shared
                    .ui_to_rt
                    .param_gate_thresh
                    .load(std::sync::atomic::Ordering::Relaxed),
            ) as f64),
            PARAM_BYPASS => Some(
                if self
                    .shared
                    .ui_to_rt
                    .param_bypass
                    .load(std::sync::atomic::Ordering::Relaxed)
                    != 0
                {
                    1.0
                } else {
                    0.0
                },
            ),
            PARAM_ACTIVE_MODEL => Some(
                self.shared
                    .cold
                    .model_load_counter
                    .load(std::sync::atomic::Ordering::Relaxed) as f64,
            ),
            PARAM_ADAPTIVE_COMPUTE => Some(
                self.shared
                    .ui_to_rt
                    .param_adaptive_compute
                    .load(std::sync::atomic::Ordering::Relaxed) as f64,
            ),
            _ => None,
        }
    }

    fn value_to_text(
        &mut self,
        id: ClapId,
        value: f64,
        writer: &mut ParamDisplayWriter,
    ) -> core::fmt::Result {
        use core::fmt::Write;
        match id.get() {
            PARAM_INPUT_GAIN | PARAM_OUTPUT_GAIN | PARAM_GATE_THRESH => {
                writer.write_fmt(format_args!("{:.1} dB", value))
            }
            PARAM_BYPASS => {
                if value > 0.5 {
                    writer.write_str("Bypassed")
                } else {
                    writer.write_str("Active")
                }
            }
            PARAM_ACTIVE_MODEL => {
                let name = if let Ok(guard) = self.shared.cold.ui_model_name.lock() {
                    if guard.is_empty() {
                        "None".to_string()
                    } else {
                        guard.clone()
                    }
                } else {
                    "None".to_string()
                };
                writer.write_str(&name)
            }
            PARAM_ADAPTIVE_COMPUTE => match value.round() as i32 {
                0 => writer.write_str("Off"),
                1 => writer.write_str("Conservative"),
                2 => writer.write_str("Aggressive"),
                _ => writer.write_str("Off"),
            },
            _ => Ok(()),
        }
    }

    fn text_to_value(&mut self, id: ClapId, text: &CStr) -> Option<f64> {
        let text_str = text.to_str().ok()?;
        match id.get() {
            PARAM_INPUT_GAIN | PARAM_OUTPUT_GAIN | PARAM_GATE_THRESH => {
                // Remove " dB" se presente
                let clean_text = text_str.trim_end_matches(" dB").trim();
                clean_text.parse::<f64>().ok()
            }
            PARAM_BYPASS => match text_str.to_lowercase().as_str() {
                "active" | "0" | "false" | "off" => Some(0.0),
                "bypassed" | "1" | "true" | "on" => Some(1.0),
                _ => None,
            },
            PARAM_ACTIVE_MODEL => {
                let current_name = if let Ok(guard) = self.shared.cold.ui_model_name.lock() {
                    if guard.is_empty() {
                        "None".to_string()
                    } else {
                        guard.clone()
                    }
                } else {
                    "None".to_string()
                };

                if text_str == current_name {
                    Some(
                        self.shared
                            .cold
                            .model_load_counter
                            .load(std::sync::atomic::Ordering::Relaxed)
                            as f64,
                    )
                } else if let Ok(val) = text_str.parse::<f64>() {
                    Some(val)
                } else {
                    Some(0.0)
                }
            }
            PARAM_ADAPTIVE_COMPUTE => match text_str.to_lowercase().as_str() {
                "off" | "0" => Some(0.0),
                "conservative" | "1" => Some(1.0),
                "aggressive" | "2" => Some(2.0),
                _ => text_str.parse::<f64>().ok(),
            },
            _ => None,
        }
    }

    /// Processes parameter events received from the host while processing is inactive.
    ///
    /// This method is called by the Main Thread when no `AudioProcessor` is active.
    /// Updates both local parameters and shared atomics, and
    /// sends the updated snapshot to the Audio Thread via the SPSC channel.
    fn flush(&mut self, input: &InputEvents, output: &mut OutputEvents) {
        // Send any pending parameter/gesture updates originating from the GUI to the host.
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
                    self.params.bypass = val > 0.5;
                    self.shared.ui_to_rt.param_bypass.store(
                        if val > 0.5 { 1 } else { 0 },
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
                _ => continue,
            }

            // Sync with the RT thread (only if called on the offline main thread, but harmless)
            let _ = self.param_tx.push(ClapParamPayload::Params(
                RtPluginParams::from_plugin_params(&self.params),
            ));
        }
    }
}

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
        // Send any pending parameter/gesture updates originating from the GUI to the host.
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
                    self.params.bypass = val > 0.5;
                    self.shared.ui_to_rt.param_bypass.store(
                        if val > 0.5 { 1 } else { 0 },
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
                _ => continue,
            }
        }

        // ── GUI → Audio Thread Synchronization ───────────────────────────────────
        // The GUI writes directly to `NamClapShared` atomics (e.g., `param_input_gain`).
        // The host may not echo these changes as input events in this cycle.
        // A single Acquire load of the generation counter avoids 5 Relaxed loads
        // when no GUI change occurred since the last reconciliation.
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

            let shared_bypass = self
                .shared
                .ui_to_rt
                .param_bypass
                .load(std::sync::atomic::Ordering::Relaxed)
                != 0;
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
        }
    }
}
