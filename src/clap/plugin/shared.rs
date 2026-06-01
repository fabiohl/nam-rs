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

    /// Bitmap de flags de gesto e modificação por parâmetro (GUI -> Host/Processor).
    /// Layout: para cada parâmetro (0=input_gain, 1=output_gain, 2=gate_thresh, 3=bypass):
    ///   bit (param_index * 3 + 0) = Changed (gui_*_changed)
    ///   bit (param_index * 3 + 1) = Gesture Begin
    ///   bit (param_index * 3 + 2) = Gesture End
    pub gesture_flags: AtomicU32,
}

impl<'a> PluginShared<'a> for NamClapShared {}

impl Drop for NamClapShared {
    fn drop(&mut self) {
        self.alive_fence.store(false, Ordering::Relaxed);
    }
}

impl NamClapShared {
    /// Bitmask do parâmetro no campo `gesture_flags`.
    /// 3 flags por parâmetro: Changed, GestureBegin, GestureEnd.
    const GESTURE_CHANGED_SHIFT: u32 = 0;
    const GESTURE_BEGIN_SHIFT: u32 = 1;
    const GESTURE_END_SHIFT: u32 = 2;
    const GESTURE_BITS_PER_PARAM: u32 = 3;

    /// Mapeia um CLAP param_id (0..3) para índice interno 0..3.
    const fn param_index(param_id: u32) -> usize {
        param_id as usize
    }

    /// Seta uma flag de gesto para o parâmetro (store = true).
    pub fn set_gesture(&self, param_index: usize, flag_shift: u32) {
        let bit = 1u32 << (param_index as u32 * Self::GESTURE_BITS_PER_PARAM + flag_shift);
        self.gesture_flags.fetch_or(bit, Ordering::Relaxed);
    }

    /// Lê e limpa uma flag de gesto (swap to false), retorna o valor anterior.
    pub fn take_gesture(&self, param_index: usize, flag_shift: u32) -> bool {
        let bit = 1u32 << (param_index as u32 * Self::GESTURE_BITS_PER_PARAM + flag_shift);
        (self.gesture_flags.fetch_and(!bit, Ordering::Relaxed) & bit) != 0
    }

    /// Zera todas as flags de gesto.
    pub fn clear_gestures(&self) {
        self.gesture_flags.store(0, Ordering::Relaxed);
    }

    /// Descarrega os gestos e atualizações de parâmetros iniciados pela GUI
    /// na fila de eventos de saída do host.
    pub fn write_gui_events(&self, output: &mut OutputEvents) {
        use crate::clap::extensions::params::{
            PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
        };
        use clack_plugin::events::event_types::{
            ParamGestureBeginEvent, ParamGestureEndEvent, ParamValueEvent,
        };

        let params: [(u32, u32, &AtomicU32); 4] = [
            (PARAM_INPUT_GAIN, Self::param_index(PARAM_INPUT_GAIN) as u32, &self.param_input_gain),
            (PARAM_OUTPUT_GAIN, Self::param_index(PARAM_OUTPUT_GAIN) as u32, &self.param_output_gain),
            (PARAM_GATE_THRESH, Self::param_index(PARAM_GATE_THRESH) as u32, &self.param_gate_thresh),
            (PARAM_BYPASS, Self::param_index(PARAM_BYPASS) as u32, &self.param_bypass),
        ];

        for (param_id, param_idx, value_atomic) in &params {
            let pi = *param_idx as usize;
            if self.take_gesture(pi, Self::GESTURE_BEGIN_SHIFT) {
                let ev = ParamGestureBeginEvent::new(0, ClapId::new(*param_id));
                let _ = output.try_push(ev);
            }
            if self.take_gesture(pi, Self::GESTURE_CHANGED_SHIFT) {
                let val = f32::from_bits(value_atomic.load(Ordering::Relaxed)) as f64;
                let ev = ParamValueEvent::new(
                    0,
                    ClapId::new(*param_id),
                    clack_plugin::events::Pckn::new(0u8, 0u8, 0u8, 0u8),
                    val,
                    clack_plugin::utils::Cookie::empty(),
                );
                let _ = output.try_push(ev);
            }
            if self.take_gesture(pi, Self::GESTURE_END_SHIFT) {
                let ev = ParamGestureEndEvent::new(0, ClapId::new(*param_id));
                let _ = output.try_push(ev);
            }
        }
    }
}
