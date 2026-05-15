// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão de parâmetros CLAP para o NAM-rs.

use crate::clap::plugin::{ClapParamPayload, NamClapMainThread};
use crate::clap::processor::NamClapProcessor;
use clack_extensions::params::{
    ParamDisplayWriter, ParamInfo, ParamInfoFlags, ParamInfoWriter, PluginAudioProcessorParams,
    PluginMainThreadParams,
};
use clack_plugin::events::event_types::ParamValueEvent;
use clack_plugin::prelude::{ClapId, InputEvents, OutputEvents};
use std::ffi::CStr;

/// Constantes de ID para os parâmetros CLAP.
/// ID do parâmetro de ganho de entrada.
pub const PARAM_INPUT_GAIN: u32 = 0;
/// ID do parâmetro de ganho de saída.
pub const PARAM_OUTPUT_GAIN: u32 = 1;
/// ID do parâmetro de threshold do noise gate.
pub const PARAM_GATE_THRESH: u32 = 2;
/// ID do parâmetro de bypass do plugin.
pub const PARAM_BYPASS: u32 = 3;

impl PluginMainThreadParams for NamClapMainThread<'_> {
    fn count(&mut self) -> u32 {
        4
    }

    fn get_info(&mut self, param_index: u32, info: &mut ParamInfoWriter) {
        match param_index {
            PARAM_INPUT_GAIN => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_INPUT_GAIN),
                    flags: ParamInfoFlags::IS_AUTOMATABLE,
                    cookie: clack_plugin::utils::Cookie::empty(),
                    name: b"Input Gain",
                    module: b"",
                    min_value: -24.0,
                    max_value: 24.0,
                    default_value: 0.0,
                });
            }
            PARAM_OUTPUT_GAIN => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_OUTPUT_GAIN),
                    flags: ParamInfoFlags::IS_AUTOMATABLE,
                    cookie: clack_plugin::utils::Cookie::empty(),
                    name: b"Output Gain",
                    module: b"",
                    min_value: -24.0,
                    max_value: 24.0,
                    default_value: 0.0,
                });
            }
            PARAM_GATE_THRESH => {
                info.set(&ParamInfo {
                    id: ClapId::new(PARAM_GATE_THRESH),
                    flags: ParamInfoFlags::IS_AUTOMATABLE,
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
            _ => {}
        }
    }

    fn get_value(&mut self, id: ClapId) -> Option<f64> {
        match id.get() {
            PARAM_INPUT_GAIN => Some(self.params.input_gain_db as f64),
            PARAM_OUTPUT_GAIN => Some(self.params.output_gain_db as f64),
            PARAM_GATE_THRESH => Some(self.params.gate_threshold_db as f64),
            PARAM_BYPASS => Some(if self.params.bypass { 1.0 } else { 0.0 }),
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
            _ => None,
        }
    }

    fn flush(&mut self, input: &InputEvents, _output: &mut OutputEvents) {
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
                PARAM_INPUT_GAIN => self.params.input_gain_db = val,
                PARAM_OUTPUT_GAIN => self.params.output_gain_db = val,
                PARAM_GATE_THRESH => self.params.gate_threshold_db = val,
                PARAM_BYPASS => self.params.bypass = val > 0.5,
                _ => continue,
            }

            // Sincroniza com a thread RT
            let _ = self
                .param_tx
                .push(ClapParamPayload::Params(self.params.clone()));
        }
    }
}

impl PluginAudioProcessorParams for NamClapProcessor<'_> {
    fn flush(&mut self, input: &InputEvents, _output: &mut OutputEvents) {
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
                PARAM_INPUT_GAIN => self.params.input_gain_db = val,
                PARAM_OUTPUT_GAIN => self.params.output_gain_db = val,
                PARAM_GATE_THRESH => self.params.gate_threshold_db = val,
                PARAM_BYPASS => self.params.bypass = val > 0.5,
                _ => continue,
            }
        }
    }
}
