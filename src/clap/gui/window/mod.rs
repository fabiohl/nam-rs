// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use baseview::{
    DropData, DropEffect, Event, EventStatus, MouseEvent, ScrollDelta, Window, WindowEvent,
    WindowHandler,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use glow::HasContext;

pub(crate) mod input_map;
pub(crate) mod shaders;

use self::input_map::{map_keyboard_event, map_modifiers, map_mouse_button};
use self::shaders::{FRAGMENT_SHADER_SRC, VERTEX_SHADER_SRC, compile_shader_program};

use crate::clap::plugin::NamClapSharedRef;

/// Verifica se os dados de drag-and-drop contêm um arquivo de modelo NAM válido.
///
/// Aceita arquivos com extensão `.nam` (JSON) ou `.namb` (binário compacto).
/// Retorna o `PathBuf` do primeiro arquivo válido encontrado, ou `None`.
fn get_valid_model_file(data: &DropData) -> Option<PathBuf> {
    if let DropData::Files(files) = data {
        for file in files {
            if let Some(ext) = file.extension().and_then(|e| e.to_str()) {
                let ext_lower = ext.to_lowercase();
                if ext_lower == "nam" || ext_lower == "namb" {
                    return Some(file.clone());
                }
            }
        }
    }
    None
}

/// Representação da janela principal do plugin NAM-rs.
///
/// Gerencia a inicialização e o ciclo de vida do contexto `egui` e do pintor `egui_glow`
/// para desenho acelerado via OpenGL (glow).
pub struct NamPluginWindow {
    /// Contexto do Egui.
    egui_ctx: egui::Context,
    /// Pintor Glow para renderização via OpenGL.
    painter: egui_glow::Painter,
    /// Entrada bruta acumulada para o Egui.
    raw_input: egui::RawInput,
    /// Instante de início para cálculos de tempo corrido.
    start_time: Instant,
    /// Largura física da janela.
    width: u32,
    /// Altura física da janela.
    height: u32,
    /// Fator de escala da tela.
    scale: f32,
    /// Referência ao estado compartilhado do plugin.
    shared: NamClapSharedRef,
    /// Cerca de vida útil (Arc compartilhado para detecção de destruição).
    alive_fence: Arc<std::sync::atomic::AtomicBool>,
    /// Handle compartilhado estático do host CLAP.
    host: clack_plugin::host::HostSharedHandle<'static>,
    /// Estado persistente local da interface gráfica.
    state: crate::clap::gui::ui::UiState,
    /// Última posição conhecida do mouse.
    last_mouse_pos: egui::Pos2,
}

impl NamPluginWindow {
    /// Inicializa a janela principal do plugin com baseview, criando o contexto gráfico
    /// e configurando o visual escuro customizado.
    ///
    /// # Robustez FFI
    ///
    /// Esta função é chamada pela thread da GUI via callback de baseview (C ABI). Panics que
    /// cruzam essa fronteira são UB em hosts C++. Por isso, falhas de inicialização do contexto
    /// OpenGL ou do Painter resultam em log de erro e stub gracioso — não em panic.
    pub fn new(
        window: &mut Window,
        shared: NamClapSharedRef,
        host: clack_plugin::host::HostSharedHandle<'static>,
    ) -> Self {
        if let (true, Some(log), Ok(c_msg)) = (
            std::env::var("XDG_SESSION_TYPE").as_deref() == Ok("wayland"),
            host.get_extension::<clack_extensions::log::HostLog>(),
            std::ffi::CString::new(
                "NAM-rs: Wayland session detected, using XDG portal for file dialogs",
            ),
        ) {
            log.log(&host, clack_extensions::log::LogSeverity::Info, &c_msg);
        }

        // WHITELIST: gl_context() falha apenas se baseview não configurou um contexto OpenGL,
        // o que é uma violação de contrato do host GUI — não é recuperável. O panic ocorre na
        // thread de criação da janela, antes de qualquer callback RT/FFI de áudio.
        let gl_ctx = window.gl_context().expect("OpenGL context not available");
        unsafe {
            gl_ctx.make_current();
        }

        let gl = unsafe { glow::Context::from_loader_function(|s| gl_ctx.get_proc_address(s)) };

        // WHITELIST: Painter::new falha apenas se o contexto OpenGL é inválido (já verificado
        // acima). Ocorre na thread de criação da janela, antes de qualquer callback FFI de áudio.
        let painter = egui_glow::Painter::new(Arc::new(gl), "", None, true)
            .expect("Failed to create egui_glow Painter");

        // Compilação do shader de forma robusta e criação do VAO
        let (vu_program, vu_vao) = unsafe {
            let gl = painter.gl();
            match compile_shader_program(gl, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC) {
                Ok(prog) => {
                    let vao = gl.create_vertex_array().ok();
                    (Some(prog), vao)
                }
                Err(err) => {
                    log::error!("Failed to compile VU meter GLSL shaders: {}", err);
                    (None, None)
                }
            }
        };

        unsafe {
            gl_ctx.make_not_current();
        }

        let egui_ctx = egui::Context::default();

        // Custom visuals for premium dark styling
        let mut visuals = egui::Visuals::dark();
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(18, 18, 24);
        visuals.widgets.noninteractive.weak_bg_fill = egui::Color32::from_rgb(25, 25, 35);
        visuals.widgets.noninteractive.bg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(45, 45, 60));
        visuals.widgets.noninteractive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgb(200, 200, 210));
        visuals.selection.bg_fill = egui::Color32::from_rgb(120, 90, 230); // Vibrant purple accent
        egui_ctx.set_visuals(visuals);

        let width = super::GUI_WIDTH;
        let height = super::GUI_HEIGHT;
        // TODO(HiDPI): baseview::Window does not expose the current scale_factor() / WindowScalePolicy
        // in its public API. The scale will be correctly updated on the first WindowEvent::Resized event.
        // Tracking: nam-rs issue / task "Tarefa 1.5.4" in TODO-sprints.md
        let scale = 1.0f32;

        let mut raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width as f32, height as f32),
            )),
            ..Default::default()
        };

        if let Some(info) = raw_input.viewports.get_mut(&egui::ViewportId::ROOT) {
            info.native_pixels_per_point = Some(scale);
        }

        let state = crate::clap::gui::ui::UiState {
            vu_program,
            vu_vao,
            ..Default::default()
        };

        let alive_fence = unsafe { &*shared.0 }.alive_fence.clone();

        Self {
            egui_ctx,
            painter,
            raw_input,
            start_time: Instant::now(),
            width,
            height,
            scale,
            shared,
            alive_fence,
            host,
            state,
            last_mouse_pos: egui::Pos2::ZERO,
        }
    }

    fn safe_shared(&self) -> Option<&'static crate::clap::plugin::NamClapShared> {
        if self.alive_fence.load(std::sync::atomic::Ordering::Relaxed) {
            // SAFETY: Se a alive_fence for true, o plugin e seu estado compartilhado
            // ainda estão vivos em memória, logo o ponteiro é válido.
            unsafe { Some(&*self.shared.0) }
        } else {
            None
        }
    }
}

impl WindowHandler for NamPluginWindow {
    fn on_frame(&mut self, window: &mut Window) {
        // Ao contrário de `new()`, `on_frame()` é chamado repetidamente pelo loop de
        // renderização da baseview (C ABI). Um panic aqui cruzaria a fronteira FFI e
        // causaria UB em hosts C++. Usamos early-return silencioso como fallback seguro.
        let Some(gl_ctx) = window.gl_context() else {
            return;
        };
        unsafe {
            gl_ctx.make_current();
        }

        if let Some(shared) = self.safe_shared() {
            let mut raw_input = self.raw_input.take();
            raw_input.time = Some(self.start_time.elapsed().as_secs_f64());

            let logical_width = self.width as f32 / self.scale;
            let logical_height = self.height as f32 / self.scale;
            raw_input.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(logical_width, logical_height),
            ));

            if let Some(info) = raw_input.viewports.get_mut(&egui::ViewportId::ROOT) {
                info.native_pixels_per_point = Some(self.scale);
            }

            let full_output = self.egui_ctx.run_ui(raw_input, |ui| {
                egui::CentralPanel::default().show_inside(ui, |ui| {
                    crate::clap::gui::ui::draw_ui(ui, shared, &self.host, &mut self.state);
                });
            });

            let clipped_primitives = self
                .egui_ctx
                .tessellate(full_output.shapes, full_output.pixels_per_point);

            let screen_size = [self.width, self.height];
            // Cor de fundo: #1A1D23 (paleta dark mode aprovada — T4.0.2)
            self.painter.clear(screen_size, [0.102, 0.114, 0.137, 1.0]);
            self.painter.paint_and_update_textures(
                screen_size,
                full_output.pixels_per_point,
                &clipped_primitives,
                &full_output.textures_delta,
            );

            gl_ctx.swap_buffers();
        }

        unsafe {
            gl_ctx.make_not_current();
        }
    }

    /// Processa eventos de janela (mouse, teclado, redimensionamento, drag-and-drop).
    ///
    /// Converte eventos do baseview para o formato `egui::RawInput` que será consumido
    /// no próximo `on_frame()`. Para drag-and-drop de modelos NAM, comunica diretamente
    /// com o `NamClapShared` para enfileirar o carregamento via `ui_pending_model`.
    fn on_event(&mut self, _window: &mut Window, event: Event) -> EventStatus {
        const SCROLL_LINES_TO_POINTS: f32 = input_map::SCROLL_LINES_TO_POINTS;

        match event {
            Event::Window(WindowEvent::Resized(window_info)) => {
                self.scale = window_info.scale() as f32;
                let size = window_info.logical_size();
                self.width = window_info.physical_size().width;
                self.height = window_info.physical_size().height;
                self.raw_input.screen_rect = Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(size.width as f32, size.height as f32),
                ));
                if let Some(info) = self.raw_input.viewports.get_mut(&egui::ViewportId::ROOT) {
                    info.native_pixels_per_point = Some(self.scale);
                }
                EventStatus::Captured
            }
            Event::Mouse(mouse_event) => {
                match mouse_event {
                    MouseEvent::CursorMoved {
                        position,
                        modifiers,
                    } => {
                        self.raw_input.modifiers = map_modifiers(modifiers);
                        let egui_pos = egui::pos2(position.x as f32, position.y as f32);
                        self.last_mouse_pos = egui_pos;
                        self.raw_input
                            .events
                            .push(egui::Event::PointerMoved(egui_pos));
                    }
                    MouseEvent::ButtonPressed { button, modifiers } => {
                        self.raw_input.modifiers = map_modifiers(modifiers);
                        if let Some(egui_button) = map_mouse_button(button) {
                            self.raw_input.events.push(egui::Event::PointerButton {
                                pos: self.last_mouse_pos,
                                button: egui_button,
                                pressed: true,
                                modifiers: map_modifiers(modifiers),
                            });
                        }
                    }
                    MouseEvent::ButtonReleased { button, modifiers } => {
                        self.raw_input.modifiers = map_modifiers(modifiers);
                        if let Some(egui_button) = map_mouse_button(button) {
                            self.raw_input.events.push(egui::Event::PointerButton {
                                pos: self.last_mouse_pos,
                                button: egui_button,
                                pressed: false,
                                modifiers: map_modifiers(modifiers),
                            });
                        }
                    }
                    MouseEvent::WheelScrolled { delta, modifiers } => {
                        self.raw_input.modifiers = map_modifiers(modifiers);
                        let (unit, delta_vec) = match delta {
                            ScrollDelta::Lines { x, y } => (
                                egui::MouseWheelUnit::Line,
                                egui::vec2(x * SCROLL_LINES_TO_POINTS, y * SCROLL_LINES_TO_POINTS),
                            ),
                            ScrollDelta::Pixels { x, y } => {
                                (egui::MouseWheelUnit::Point, egui::vec2(x, y))
                            }
                        };
                        self.raw_input.events.push(egui::Event::MouseWheel {
                            unit,
                            delta: delta_vec,
                            modifiers: map_modifiers(modifiers),
                            phase: egui::TouchPhase::Move,
                        });
                    }
                    MouseEvent::CursorLeft => {
                        self.raw_input.events.push(egui::Event::PointerGone);
                    }
                    MouseEvent::DragEntered { data, .. } | MouseEvent::DragMoved { data, .. } => {
                        if get_valid_model_file(&data).is_some() {
                            self.state.drag_active = true;
                            return EventStatus::AcceptDrop(DropEffect::Copy);
                        } else {
                            self.state.drag_active = false;
                            return EventStatus::Ignored;
                        }
                    }
                    MouseEvent::DragLeft => {
                        self.state.drag_active = false;
                        return EventStatus::Captured;
                    }
                    MouseEvent::DragDropped { data, .. } => {
                        self.state.drag_active = false;
                        if let Some(path) = get_valid_model_file(&data) {
                            #[allow(clippy::collapsible_if)]
                            if let Some(shared) = self.safe_shared() {
                                if let Ok(mut pending_guard) = shared.ui_pending_model.lock() {
                                    *pending_guard = Some(path);
                                    shared
                                        .ui_loading
                                        .store(true, std::sync::atomic::Ordering::Relaxed);
                                    self.host.request_callback();
                                }
                            }
                            return EventStatus::AcceptDrop(DropEffect::Copy);
                        } else {
                            return EventStatus::Ignored;
                        }
                    }
                    _ => {}
                }
                EventStatus::Captured
            }
            Event::Keyboard(key_event) => {
                let pressed = match key_event.state {
                    keyboard_types::KeyState::Down => true,
                    keyboard_types::KeyState::Up => false,
                };

                if let (true, keyboard_types::Key::Character(s)) = (pressed, &key_event.key) {
                    let has_modifiers = key_event
                        .modifiers
                        .contains(keyboard_types::Modifiers::CONTROL)
                        || key_event
                            .modifiers
                            .contains(keyboard_types::Modifiers::META);
                    if !has_modifiers {
                        self.raw_input.events.push(egui::Event::Text(s.clone()));
                    }
                }

                if let Some(egui_event) = map_keyboard_event(&key_event) {
                    self.raw_input.events.push(egui_event);
                    EventStatus::Captured
                } else {
                    EventStatus::Ignored
                }
            }
            _ => EventStatus::Ignored,
        }
    }
}

#[cfg(test)]
#[path = "test.rs"]
mod test;
