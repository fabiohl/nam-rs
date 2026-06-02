// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão de parâmetros CLAP para o NAM-rs.

use crate::clap::plugin::{ClapParamPayload, NamClapMainThread};
use crate::clap::processor::NamClapProcessor;
use crate::math::constants::{GAIN_MAX_DB, GAIN_MIN_DB};
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
/// ID do parâmetro do nome do modelo carregado (somente leitura).
pub const PARAM_ACTIVE_MODEL: u32 = 4;

impl PluginMainThreadParams for NamClapMainThread<'_> {
    fn count(&mut self) -> u32 {
        5
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
            _ => {}
        }
    }

    fn get_value(&mut self, id: ClapId) -> Option<f64> {
        match id.get() {
            PARAM_INPUT_GAIN => Some(f32::from_bits(
                self.shared
                    .param_input_gain
                    .load(std::sync::atomic::Ordering::Relaxed),
            ) as f64),
            PARAM_OUTPUT_GAIN => Some(f32::from_bits(
                self.shared
                    .param_output_gain
                    .load(std::sync::atomic::Ordering::Relaxed),
            ) as f64),
            PARAM_GATE_THRESH => Some(f32::from_bits(
                self.shared
                    .param_gate_thresh
                    .load(std::sync::atomic::Ordering::Relaxed),
            ) as f64),
            PARAM_BYPASS => Some(
                if self
                    .shared
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
                    .model_load_counter
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
                let name = if let Ok(guard) = self.shared.ui_model_name.lock() {
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
                let current_name = if let Ok(guard) = self.shared.ui_model_name.lock() {
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
            _ => None,
        }
    }

    /// Processa eventos de parâmetros recebidos do host enquanto o processamento está inativo.
    ///
    /// Este método é chamado pela Main Thread quando não há `AudioProcessor` ativo.
    /// Atualiza tanto os parâmetros locais quanto os atômicos compartilhados, e
    /// envia o snapshot atualizado para a Audio Thread via canal SPSC.
    fn flush(&mut self, input: &InputEvents, output: &mut OutputEvents) {
        // Envia quaisquer atualizações de parâmetros/gestos pendentes originados da GUI para o host.
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
                        .param_input_gain
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_OUTPUT_GAIN => {
                    self.params.output_gain_db = val;
                    self.shared
                        .param_output_gain
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_GATE_THRESH => {
                    self.params.gate_threshold_db = val;
                    self.shared
                        .param_gate_thresh
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_BYPASS => {
                    self.params.bypass = val > 0.5;
                    self.shared.param_bypass.store(
                        if val > 0.5 { 1 } else { 0 },
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                _ => continue,
            }

            // Sincroniza com a thread RT (apenas se for chamado na main thread offline, mas não faz mal)
            let _ = self
                .param_tx
                .push(ClapParamPayload::Params(self.params.clone()));
        }
    }
}

/// Implementação de `PluginAudioProcessorParams` para a Audio Thread.
///
/// # Design: Duplicação Intencional
///
/// O parsing de eventos de parâmetros aqui é estruturalmente idêntico ao de
/// `PluginMainThreadParams::flush()` acima. Esta duplicação é **intencional**:
///
/// - **Main Thread flush()**: Atualiza `self.params` + atômicos + envia snapshot via SPSC
///   (necessário porque a Audio Thread pode não estar ativa).
/// - **Audio Thread flush()**: Atualiza `self.params` + atômicos + sincroniza parâmetros
///   vindos da GUI via leitura direta dos atômicos (sem SPSC, pois já estamos na Audio Thread).
///
/// Extrair uma helper comum seria possível mas adicionaria indireção desnecessária
/// e complicaria os lifetimes do `self` mutável em cada contexto.
impl PluginAudioProcessorParams for NamClapProcessor<'_> {
    /// Processa eventos de parâmetros na Audio Thread quando `process()` não está sendo chamado.
    ///
    /// Além de aplicar eventos do host, este método também sincroniza parâmetros
    /// que foram alterados pela GUI diretamente nos atômicos compartilhados.
    fn flush(&mut self, input: &InputEvents, output: &mut OutputEvents) {
        // Envia quaisquer atualizações de parâmetros/gestos pendentes originados da GUI para o host.
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
                        .param_input_gain
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_OUTPUT_GAIN => {
                    self.params.output_gain_db = val;
                    self.shared
                        .param_output_gain
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_GATE_THRESH => {
                    self.params.gate_threshold_db = val;
                    self.shared
                        .param_gate_thresh
                        .store(val.to_bits(), std::sync::atomic::Ordering::Relaxed);
                }
                PARAM_BYPASS => {
                    self.params.bypass = val > 0.5;
                    self.shared.param_bypass.store(
                        if val > 0.5 { 1 } else { 0 },
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                _ => continue,
            }
        }

        // ── Sincronização GUI → Audio Thread ───────────────────────────────────
        // A GUI escreve diretamente nos atômicos de `NamClapShared` (ex: `param_input_gain`).
        // O host pode não ecoar essas mudanças como eventos de entrada neste ciclo.
        // Aqui, lemos os atômicos e atualizamos `self.params` se houver divergência.
        // Isso garante que o próximo `process()` use valores atualizados.
        let shared_in_db = f32::from_bits(
            self.shared
                .param_input_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        if shared_in_db != self.params.input_gain_db {
            self.params.input_gain_db = shared_in_db;
        }

        let shared_out_db = f32::from_bits(
            self.shared
                .param_output_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        if shared_out_db != self.params.output_gain_db {
            self.params.output_gain_db = shared_out_db;
        }

        let shared_gate_db = f32::from_bits(
            self.shared
                .param_gate_thresh
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        if shared_gate_db != self.params.gate_threshold_db {
            self.params.gate_threshold_db = shared_gate_db;
        }

        let shared_bypass = self
            .shared
            .param_bypass
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0;
        if shared_bypass != self.params.bypass {
            self.params.bypass = shared_bypass;
        }
    }
}
