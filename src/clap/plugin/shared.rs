// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Estado compartilhado lock-free entre a audio thread e a main thread.

use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::resampler::NamResampler;
use crate::loader::LoadedModelPair;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Payload de comunicação Main -> RT para o plugin CLAP.
pub enum ClapParamPayload {
    /// Atualização de parâmetros (ganho, gate, bypass).
    Params(NamPluginParams),
    /// Carregamento de um novo par de modelos com metadados de calibração e seu resampler.
    LoadModel(Box<LoadedModelPair>, Box<NamResampler>),
}

/// Metadados do modelo para exibição na GUI.
#[derive(Clone, Debug, Default)]
pub struct NamModelMetadata {
    /// Arquitetura do modelo (ex: "LSTM", "WaveNet").
    pub architecture: String,
    /// Topologia do modelo (ex: "Standard", "1x64").
    pub topology: String,
    /// Autor / Modeled by.
    pub modeled_by: Option<String>,
    /// Fabricante do equipamento original.
    pub gear_make: Option<String>,
    /// O modelo do equipamento original.
    pub gear_model: Option<String>,
    /// Tipo de equipamento.
    pub gear_type: Option<String>,
    /// Estilo/Tipo de tom do equipamento.
    pub tone_type: Option<String>,
    /// Data formatada YYYY-MM-DD.
    pub date: Option<String>,
}

/// Wrapper seguro para ponteiro de NamClapShared repassado à thread de GUI.
#[derive(Clone, Copy)]
pub struct NamClapSharedRef(pub *const NamClapShared);

unsafe impl Send for NamClapSharedRef {}
unsafe impl Sync for NamClapSharedRef {}

/// Estado compartilhado entre a audio thread e a main thread (lock-free).
///
/// Adota alinhamento a 128 bytes para mitigar False Sharing.
/// Os canais SPSC são envolvidos em Mutex<Option<...>> apenas para permitir
/// que sejam "extraídos" pelas respectivas threads durante a inicialização,
/// satisfazendo o requisito `Sync` da trait `PluginShared`.
#[repr(align(128))]
pub struct NamClapShared {
    /// Canal SPSC: Main Thread -> Audio Thread (Novos parâmetros/modelos).
    pub param_tx: Mutex<Option<Producer<ClapParamPayload>>>,
    /// Canal SPSC: Main Thread -> Audio Thread (Consumidor).
    pub param_rx: Mutex<Option<Consumer<ClapParamPayload>>>,
    /// Canal GC: Audio Thread -> Main Thread (Modelos obsoletos para descarte).
    pub gc_tx: Mutex<Option<Producer<GcItem>>>,
    /// Canal GC: Audio Thread -> Main Thread (Consumidor).
    pub gc_rx: Mutex<Option<Consumer<GcItem>>>,
    /// Buffer de fallback para overflow de GC (overwrite).
    pub gc_overflow: Arc<GcOverflowBuffer>,
    /// Flags atômicas de status (telemetria RT->Main).
    pub rt_status: Arc<RtStatusFlags>,
    /// Latência atual reportada ao host (em samples).
    pub current_latency: AtomicU32,
    /// Taxa de amostragem nativa exigida pelo modelo carregado ativamente.
    pub model_sample_rate: AtomicU32,
    /// Último valor do parâmetro Input Gain (f32 como bits).
    pub param_input_gain: AtomicU32,
    /// Último valor do parâmetro Output Gain (f32 como bits).
    pub param_output_gain: AtomicU32,
    /// Último valor do parâmetro Gate Threshold (f32 como bits).
    pub param_gate_thresh: AtomicU32,
    /// Último valor do parâmetro Bypass (0 = false, 1 = true).
    pub param_bypass: AtomicU32,
    /// Nível True Peak L setado pela audio thread (bits de f32 via f32::to_bits()). Lido pela UI thread.
    pub ui_peak_l: AtomicU32,
    /// Nível True Peak R setado pela audio thread (bits de f32 via f32::to_bits()). Lido pela UI thread.
    pub ui_peak_r: AtomicU32,
    /// Flag indicando se ocorreu clipping desde o último frame da UI. Lido/resetado pela UI thread.
    pub ui_clipped: std::sync::atomic::AtomicBool,
    /// Nome do modelo carregado (path basename). Escrito pela main thread, lido pela UI thread.
    pub ui_model_name: Mutex<String>,
    /// Metadados do modelo carregado para exibição na UI.
    pub ui_model_metadata: Mutex<Option<NamModelMetadata>>,
    /// Caminho do modelo pendente para ser carregado pela Main Thread. Escrito pela UI thread.
    pub ui_pending_model: Mutex<Option<std::path::PathBuf>>,
    /// Indica se a GUI está no meio de um carregamento assíncrono de modelo.
    pub ui_loading: std::sync::atomic::AtomicBool,
    /// Flag sinalizando que ocorreu um erro ao carregar o modelo.
    pub ui_load_error: std::sync::atomic::AtomicBool,
    /// Mensagem detalhada de erro para a GUI.
    pub ui_load_error_msg: Mutex<String>,
    /// Sample rate detectado do host.
    pub sample_rate: AtomicU32,
    /// Número de canais ativos: 1 = mono, 2 = stereo.
    pub active_channel_count: AtomicU32,
    /// Cor de accent dinâmica baseada na cor da track do DAW (ARGB compactado).
    pub track_accent_color: AtomicU32,
    /// Indicação de parâmetros (mapeamento, automação e override) para os 5 parâmetros.
    /// Bit 0: Mapeado, Bit 1: Automatizando, Bit 2: Override.
    pub param_indication: [std::sync::atomic::AtomicU8; 5],
    /// Cores dos parâmetros indicados/mapeados (ARGB compactado).
    pub param_indication_color: [std::sync::atomic::AtomicU32; 5],
    /// Contador de carregamentos de modelo (incrementado a cada modelo carregado com sucesso).
    pub model_load_counter: AtomicU32,
    /// Fence de vida útil: true enquanto o plugin existe. Verificado pela thread do File Picker.
    pub alive_fence: Arc<std::sync::atomic::AtomicBool>,

    // Flags de controle de modificação por parâmetro (GUI -> Host/Processor)
    /// Indica que o ganho de entrada foi alterado pela GUI.
    pub gui_input_gain_changed: std::sync::atomic::AtomicBool,
    /// Indica o início de um gesto de alteração do ganho de entrada.
    pub gesture_begin_input_gain: std::sync::atomic::AtomicBool,
    /// Indica o término de um gesto de alteração do ganho de entrada.
    pub gesture_end_input_gain: std::sync::atomic::AtomicBool,

    /// Indica que o ganho de saída foi alterado pela GUI.
    pub gui_output_gain_changed: std::sync::atomic::AtomicBool,
    /// Indica o início de um gesto de alteração do ganho de saída.
    pub gesture_begin_output_gain: std::sync::atomic::AtomicBool,
    /// Indica o término de um gesto de alteração do ganho de saída.
    pub gesture_end_output_gain: std::sync::atomic::AtomicBool,

    /// Indica que o threshold do noise gate foi alterado pela GUI.
    pub gui_gate_thresh_changed: std::sync::atomic::AtomicBool,
    /// Indica o início de um gesto de alteração do noise gate threshold.
    pub gesture_begin_gate_thresh: std::sync::atomic::AtomicBool,
    /// Indica o término de um gesto de alteração do noise gate threshold.
    pub gesture_end_gate_thresh: std::sync::atomic::AtomicBool,

    /// Indica que o bypass foi alterado pela GUI.
    pub gui_bypass_changed: std::sync::atomic::AtomicBool,
    /// Indica o início de um gesto de alteração do bypass.
    pub gesture_begin_bypass: std::sync::atomic::AtomicBool,
    /// Indica o término de um gesto de alteração do bypass.
    pub gesture_end_bypass: std::sync::atomic::AtomicBool,
}

impl<'a> PluginShared<'a> for NamClapShared {}

impl Drop for NamClapShared {
    fn drop(&mut self) {
        self.alive_fence.store(false, Ordering::Relaxed);
    }
}

impl NamClapShared {
    /// Descarrega os gestos e atualizações de parâmetros iniciados pela GUI
    /// na fila de eventos de saída do host.
    pub fn write_gui_events(&self, output: &mut OutputEvents) {
        use crate::clap::extensions::params::{
            PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
        };
        use clack_plugin::events::event_types::{
            ParamGestureBeginEvent, ParamGestureEndEvent, ParamValueEvent,
        };

        let mut handle_param =
            |param_id: u32,
             changed_flag: &std::sync::atomic::AtomicBool,
             begin_flag: &std::sync::atomic::AtomicBool,
             end_flag: &std::sync::atomic::AtomicBool,
             value_atomic: &std::sync::atomic::AtomicU32| {
                if begin_flag.swap(false, Ordering::Relaxed) {
                    let ev = ParamGestureBeginEvent::new(0, ClapId::new(param_id));
                    let _ = output.try_push(ev);
                }
                if changed_flag.swap(false, Ordering::Relaxed) {
                    let val = f32::from_bits(value_atomic.load(Ordering::Relaxed)) as f64;
                    let ev = ParamValueEvent::new(
                        0,
                        ClapId::new(param_id),
                        clack_plugin::events::Pckn::new(0u8, 0u8, 0u8, 0u8),
                        val,
                        clack_plugin::utils::Cookie::empty(),
                    );
                    let _ = output.try_push(ev);
                }
                if end_flag.swap(false, Ordering::Relaxed) {
                    let ev = ParamGestureEndEvent::new(0, ClapId::new(param_id));
                    let _ = output.try_push(ev);
                }
            };

        handle_param(
            PARAM_INPUT_GAIN,
            &self.gui_input_gain_changed,
            &self.gesture_begin_input_gain,
            &self.gesture_end_input_gain,
            &self.param_input_gain,
        );
        handle_param(
            PARAM_OUTPUT_GAIN,
            &self.gui_output_gain_changed,
            &self.gesture_begin_output_gain,
            &self.gesture_end_output_gain,
            &self.param_output_gain,
        );
        handle_param(
            PARAM_GATE_THRESH,
            &self.gui_gate_thresh_changed,
            &self.gesture_begin_gate_thresh,
            &self.gesture_end_gate_thresh,
            &self.param_gate_thresh,
        );
        handle_param(
            PARAM_BYPASS,
            &self.gui_bypass_changed,
            &self.gesture_begin_bypass,
            &self.gesture_end_bypass,
            &self.param_bypass,
        );
    }
}
