// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use baseview::{
    Event, EventStatus, MouseButton, MouseEvent, ScrollDelta, Window, WindowEvent, WindowHandler,
};
use std::sync::Arc;
use std::time::Instant;

use crate::clap::plugin::NamClapSharedRef;

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
    pub fn new(
        window: &mut Window,
        shared: NamClapSharedRef,
        host: clack_plugin::host::HostSharedHandle<'static>,
    ) -> Self {
        let gl_ctx = window.gl_context().expect("OpenGL context not available");
        unsafe {
            gl_ctx.make_current();
        }

        let gl = unsafe { glow::Context::from_loader_function(|s| gl_ctx.get_proc_address(s)) };

        let painter = egui_glow::Painter::new(Arc::new(gl), "", None, true)
            .expect("Failed to create egui_glow Painter");

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

        let width = 600;
        let height = 210;
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

        Self {
            egui_ctx,
            painter,
            raw_input,
            start_time: Instant::now(),
            width,
            height,
            scale,
            shared,
            host,
            state: crate::clap::gui::ui::UiState::default(),
            last_mouse_pos: egui::Pos2::ZERO,
        }
    }
}

impl WindowHandler for NamPluginWindow {
    fn on_frame(&mut self, window: &mut Window) {
        let gl_ctx = window.gl_context().expect("OpenGL context not available");
        unsafe {
            gl_ctx.make_current();
        }

        let shared = unsafe { &*self.shared.0 };
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

        let host_ref: &clack_plugin::host::HostSharedHandle = unsafe {
            std::mem::transmute(&self.host)
        };

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                crate::clap::gui::ui::draw_ui(ui, shared, host_ref, &mut self.state);
            });
        });

        let clipped_primitives = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        let screen_size = [self.width, self.height];
        self.painter.clear(screen_size, [0.07, 0.07, 0.09, 1.0]);
        self.painter.paint_and_update_textures(
            screen_size,
            full_output.pixels_per_point,
            &clipped_primitives,
            &full_output.textures_delta,
        );

        gl_ctx.swap_buffers();

        unsafe {
            gl_ctx.make_not_current();
        }
    }

    fn on_event(&mut self, _window: &mut Window, event: Event) -> EventStatus {
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
                        modifiers: _,
                    } => {
                        let egui_pos = egui::pos2(position.x as f32, position.y as f32);
                        self.last_mouse_pos = egui_pos;
                        self.raw_input
                            .events
                            .push(egui::Event::PointerMoved(egui_pos));
                    }
                    MouseEvent::ButtonPressed { button, modifiers } => {
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
                        let (unit, delta_vec) = match delta {
                            ScrollDelta::Lines { x, y } => {
                                (egui::MouseWheelUnit::Line, egui::vec2(x * 10.0, y * 10.0))
                            }
                            ScrollDelta::Pixels { x, y } => {
                                (egui::MouseWheelUnit::Point, egui::vec2(x, y))
                            }
                        };
                        self.raw_input.events.push(egui::Event::MouseWheel {
                            unit,
                            delta: delta_vec,
                            modifiers: map_modifiers(modifiers),
                        });
                    }
                    MouseEvent::CursorLeft => {
                        self.raw_input.events.push(egui::Event::PointerGone);
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

fn map_mouse_button(button: MouseButton) -> Option<egui::PointerButton> {
    match button {
        MouseButton::Left => Some(egui::PointerButton::Primary),
        MouseButton::Right => Some(egui::PointerButton::Secondary),
        MouseButton::Middle => Some(egui::PointerButton::Middle),
        _ => None,
    }
}

fn map_modifiers(mods: keyboard_types::Modifiers) -> egui::Modifiers {
    egui::Modifiers {
        alt: mods.contains(keyboard_types::Modifiers::ALT),
        ctrl: mods.contains(keyboard_types::Modifiers::CONTROL),
        shift: mods.contains(keyboard_types::Modifiers::SHIFT),
        mac_cmd: mods.contains(keyboard_types::Modifiers::META),
        command: mods.contains(keyboard_types::Modifiers::META)
            || mods.contains(keyboard_types::Modifiers::CONTROL),
    }
}

fn map_keyboard_event(key_event: &keyboard_types::KeyboardEvent) -> Option<egui::Event> {
    let pressed = match key_event.state {
        keyboard_types::KeyState::Down => true,
        keyboard_types::KeyState::Up => false,
    };

    let modifiers = map_modifiers(key_event.modifiers);

    let egui_key = match &key_event.key {
        keyboard_types::Key::ArrowDown => Some(egui::Key::ArrowDown),
        keyboard_types::Key::ArrowLeft => Some(egui::Key::ArrowLeft),
        keyboard_types::Key::ArrowRight => Some(egui::Key::ArrowRight),
        keyboard_types::Key::ArrowUp => Some(egui::Key::ArrowUp),
        keyboard_types::Key::Escape => Some(egui::Key::Escape),
        keyboard_types::Key::Enter => Some(egui::Key::Enter),
        keyboard_types::Key::Tab => Some(egui::Key::Tab),
        keyboard_types::Key::Backspace => Some(egui::Key::Backspace),
        keyboard_types::Key::Delete => Some(egui::Key::Delete),
        keyboard_types::Key::Home => Some(egui::Key::Home),
        keyboard_types::Key::End => Some(egui::Key::End),
        keyboard_types::Key::PageUp => Some(egui::Key::PageUp),
        keyboard_types::Key::PageDown => Some(egui::Key::PageDown),
        keyboard_types::Key::Character(s) => {
            if s.len() == 1 {
                let c = s.chars().next().unwrap().to_ascii_lowercase();
                match c {
                    'a' => Some(egui::Key::A),
                    'b' => Some(egui::Key::B),
                    'c' => Some(egui::Key::C),
                    'd' => Some(egui::Key::D),
                    'e' => Some(egui::Key::E),
                    'f' => Some(egui::Key::F),
                    'g' => Some(egui::Key::G),
                    'h' => Some(egui::Key::H),
                    'i' => Some(egui::Key::I),
                    'j' => Some(egui::Key::J),
                    'k' => Some(egui::Key::K),
                    'l' => Some(egui::Key::L),
                    'm' => Some(egui::Key::M),
                    'n' => Some(egui::Key::N),
                    'o' => Some(egui::Key::O),
                    'p' => Some(egui::Key::P),
                    'q' => Some(egui::Key::Q),
                    'r' => Some(egui::Key::R),
                    's' => Some(egui::Key::S),
                    't' => Some(egui::Key::T),
                    'u' => Some(egui::Key::U),
                    'v' => Some(egui::Key::V),
                    'w' => Some(egui::Key::W),
                    'x' => Some(egui::Key::X),
                    'y' => Some(egui::Key::Y),
                    'z' => Some(egui::Key::Z),
                    '0' => Some(egui::Key::Num0),
                    '1' => Some(egui::Key::Num1),
                    '2' => Some(egui::Key::Num2),
                    '3' => Some(egui::Key::Num3),
                    '4' => Some(egui::Key::Num4),
                    '5' => Some(egui::Key::Num5),
                    '6' => Some(egui::Key::Num6),
                    '7' => Some(egui::Key::Num7),
                    '8' => Some(egui::Key::Num8),
                    '9' => Some(egui::Key::Num9),
                    ' ' => Some(egui::Key::Space),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    };

    egui_key.map(|key| egui::Event::Key {
        key,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers,
    })
}
