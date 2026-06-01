// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Estado persistente da interface gráfica: `UiState`, `VuMeterSharedState`, `VuUniforms`.

use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

/// Cache de locais de variáveis uniformes no shader GLSL do VU meter.
#[derive(Debug, Clone)]
pub struct VuUniforms {
    /// Localização do uniform de viewport.
    pub loc_viewport: Option<glow::UniformLocation>,
    /// Localização do uniform do retângulo do medidor.
    pub loc_meter_rect: Option<glow::UniformLocation>,
    /// Localização do uniform do valor de pico.
    pub loc_peak_frac: Option<glow::UniformLocation>,
    /// Localização do uniform do valor hold.
    pub loc_hold_frac: Option<glow::UniformLocation>,
    /// Localização do uniform da cor do hold.
    pub loc_hold_color_type: Option<glow::UniformLocation>,
}

/// Estado compartilhado do VU meter para renderização em callback do egui sem alocações no heap por frame.
#[derive(Debug)]
pub struct VuMeterSharedState {
    /// Fração atual do nível de pico do medidor (0.0 a 1.0, codificado como bits u32).
    pub peak_frac: std::sync::atomic::AtomicU32,
    /// Fração atual do nível do indicador peak hold (0.0 a 1.0, codificado como bits u32).
    pub hold_frac: std::sync::atomic::AtomicU32,
    /// Tipo de cor a aplicar no indicador peak hold (representando verde, amarelo ou vermelho).
    pub hold_color_type: std::sync::atomic::AtomicI32,
    /// Coordenada mínima X do retângulo de desenho lógica do medidor (codificada como bits u32).
    pub rect_min_x: std::sync::atomic::AtomicU32,
    /// Coordenada mínima Y do retângulo de desenho lógica do medidor (codificada como bits u32).
    pub rect_min_y: std::sync::atomic::AtomicU32,
    /// Coordenada máxima X do retângulo de desenho lógica do medidor (codificada como bits u32).
    pub rect_max_x: std::sync::atomic::AtomicU32,
    /// Coordenada máxima Y do retângulo de desenho lógica do medidor (codificada como bits u32).
    pub rect_max_y: std::sync::atomic::AtomicU32,
}

impl Default for VuMeterSharedState {
    fn default() -> Self {
        Self {
            peak_frac: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            hold_frac: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            hold_color_type: std::sync::atomic::AtomicI32::new(0),
            rect_min_x: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            rect_min_y: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            rect_max_x: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
            rect_max_y: std::sync::atomic::AtomicU32::new(0.0f32.to_bits()),
        }
    }
}

/// Estado persistente da interface gráfica entre frames.
#[derive(Clone)]
pub struct UiState {
    /// Último valor de pico do canal esquerdo retido.
    pub peak_l_hold: f32,
    /// Último valor de pico do canal direito retido.
    pub peak_r_hold: f32,
    /// Instante em que o valor de pico esquerdo foi retido.
    pub peak_l_hold_time: Instant,
    /// Instante em que o valor de pico direito foi retido.
    pub peak_r_hold_time: Instant,
    /// LED de clipping L persistente — limpa com clique no medidor.
    pub clip_l: bool,
    /// LED de clipping R persistente — limpa com clique no medidor.
    pub clip_r: bool,
    /// Indica se o painel expandido de telemetria deve ser exibido.
    pub show_telemetry: bool,
    /// Último instante em que as informações de telemetria foram atualizadas na GUI.
    pub last_telem_update: Instant,
    /// Cópia amortecida do ciclo de DSP para exibição.
    pub telem_cycles: u64,
    /// Cópia amortecida do último tamanho de bloco para exibição.
    pub telem_last_n: u32,
    /// Cópia amortecida da prioridade RT da thread DSP.
    pub telem_prio: i32,
    /// Cópia amortecida do número de overloads.
    pub telem_overloads: u32,
    /// Program do shader OpenGL compilado para desenhar o VU meter.
    pub vu_program: Option<glow::Program>,
    /// VAO vazio para draw calls sem buffers.
    pub vu_vao: Option<glow::VertexArray>,
    /// Cache de locations dos uniforms do shader GLSL do VU meter.
    pub vu_uniforms: Arc<OnceLock<VuUniforms>>,
    /// Estado compartilhado do medidor esquerdo.
    pub vu_l_state: Option<Arc<VuMeterSharedState>>,
    /// Estado compartilhado do medidor direito.
    pub vu_r_state: Option<Arc<VuMeterSharedState>>,
    /// Callback de desenho OpenGL do medidor esquerdo.
    pub vu_l_callback: Option<Arc<egui_glow::CallbackFn>>,
    /// Callback de desenho OpenGL do medidor direito.
    pub vu_r_callback: Option<Arc<egui_glow::CallbackFn>>,
    /// Indica se um arquivo de modelo válido está sendo arrastado sobre a janela.
    pub drag_active: bool,
    /// Carga da CPU/DSP amortecida em porcentagem.
    pub telem_load_pct: f64,
    /// Tempo de ciclo amortecido em nanosegundos.
    pub telem_cycle_ns: u64,
    /// Buffer de tempo máximo em nanosegundos.
    pub telem_budget_ns: u64,
    /// Expiração do banner visual de erro de carregamento (None se não estiver ativo).
    pub error_expiration: Option<Instant>,
    /// Mensagem curta/resumida do erro.
    pub error_msg: String,
}

impl std::fmt::Debug for UiState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiState")
            .field("peak_l_hold", &self.peak_l_hold)
            .field("peak_r_hold", &self.peak_r_hold)
            .field("peak_l_hold_time", &self.peak_l_hold_time)
            .field("peak_r_hold_time", &self.peak_r_hold_time)
            .field("clip_l", &self.clip_l)
            .field("clip_r", &self.clip_r)
            .field("show_telemetry", &self.show_telemetry)
            .field("last_telem_update", &self.last_telem_update)
            .field("telem_cycles", &self.telem_cycles)
            .field("telem_last_n", &self.telem_last_n)
            .field("telem_prio", &self.telem_prio)
            .field("telem_overloads", &self.telem_overloads)
            .field("vu_program", &self.vu_program)
            .field("vu_vao", &self.vu_vao)
            .field("vu_uniforms", &self.vu_uniforms)
            .field("vu_l_state", &self.vu_l_state)
            .field("vu_r_state", &self.vu_r_state)
            .field(
                "vu_l_callback",
                &self.vu_l_callback.as_ref().map(|_| "<callback>"),
            )
            .field(
                "vu_r_callback",
                &self.vu_r_callback.as_ref().map(|_| "<callback>"),
            )
            .field("telem_load_pct", &self.telem_load_pct)
            .field("telem_cycle_ns", &self.telem_cycle_ns)
            .field("telem_budget_ns", &self.telem_budget_ns)
            .finish()
    }
}

impl Default for UiState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            peak_l_hold: 0.0,
            peak_r_hold: 0.0,
            peak_l_hold_time: now,
            peak_r_hold_time: now,
            clip_l: false,
            clip_r: false,
            show_telemetry: false,
            last_telem_update: now - std::time::Duration::from_secs(2),
            telem_cycles: 0,
            telem_last_n: 0,
            telem_prio: 0,
            telem_overloads: 0,
            vu_program: None,
            vu_vao: None,
            vu_uniforms: Arc::new(OnceLock::new()),
            vu_l_state: None,
            vu_r_state: None,
            vu_l_callback: None,
            vu_r_callback: None,
            drag_active: false,
            telem_load_pct: 0.0,
            telem_cycle_ns: 0,
            telem_budget_ns: 0,
            error_expiration: None,
            error_msg: String::new(),
        }
    }
}
