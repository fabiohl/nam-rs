// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use baseview::{
    Event, EventStatus, MouseButton, MouseEvent, ScrollDelta, Window, WindowEvent, WindowHandler,
};
use std::sync::Arc;
use std::time::Instant;

use glow::HasContext;

const VERTEX_SHADER_SRC: &str = r#"#version 330 core
uniform vec4 u_viewport;
uniform vec4 u_meter_rect;
out vec2 v_uv;

void main() {
    int id = gl_VertexID;
    vec2 pos = vec2(0.0);
    vec2 uv = vec2(0.0);
    
    float left = u_meter_rect.x - 2.0;
    float right = u_meter_rect.z + 2.0;
    float top = u_meter_rect.y;
    float bottom = u_meter_rect.w;
    
    if (id == 0) {
        pos = vec2(left, bottom);
        uv = vec2(0.0, 0.0);
    } else if (id == 1) {
        pos = vec2(right, bottom);
        uv = vec2(1.0, 0.0);
    } else if (id == 2) {
        pos = vec2(left, top);
        uv = vec2(0.0, 1.0);
    } else if (id == 3) {
        pos = vec2(left, top);
        uv = vec2(0.0, 1.0);
    } else if (id == 4) {
        pos = vec2(right, bottom);
        uv = vec2(1.0, 0.0);
    } else if (id == 5) {
        pos = vec2(right, top);
        uv = vec2(1.0, 1.0);
    }
    
    float ndc_x = (pos.x - u_viewport.x) / (u_viewport.z - u_viewport.x) * 2.0 - 1.0;
    float ndc_y = 1.0 - (pos.y - u_viewport.y) / (u_viewport.w - u_viewport.y) * 2.0;
    
    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    v_uv = uv;
}
"#;

const FRAGMENT_SHADER_SRC: &str = r#"#version 330 core
precision mediump float;
in vec2 v_uv;
out vec4 f_color;

uniform float u_peak_frac;
uniform float u_hold_frac;
uniform int u_hold_color_type;

const vec4 COL_BG = vec4(0.10196, 0.11373, 0.13725, 1.0);
const vec4 COL_VU_GREEN = vec4(0.26275, 0.91373, 0.48235, 1.0);
const vec4 COL_VU_YELLOW = vec4(0.96078, 0.80784, 0.38431, 1.0);
const vec4 COL_VU_RED = vec4(0.96863, 0.30588, 0.30588, 1.0);

const float green_frac = 48.0 / 66.0;
const float yellow_frac = 57.0 / 66.0;

void main() {
    float line_half_h = 0.75 / 130.0;
    if (u_hold_frac > 0.0 && abs(v_uv.y - u_hold_frac) <= line_half_h) {
        vec4 hold_color = COL_VU_GREEN;
        if (u_hold_color_type == 1) {
            hold_color = COL_VU_YELLOW;
        } else if (u_hold_color_type == 2) {
            hold_color = COL_VU_RED;
        }
        f_color = hold_color;
        return;
    }
    
    if (v_uv.x >= 0.1 && v_uv.x <= 0.9) {
        vec2 size = vec2(16.0, 130.0);
        vec2 pos = vec2((v_uv.x - 0.1) / 0.8 * size.x, v_uv.y * size.y);
        float r = 1.5;
        
        bool in_corner = false;
        vec2 corner_center = vec2(0.0);
        if (pos.x < r && pos.y < r) {
            in_corner = true;
            corner_center = vec2(r, r);
        } else if (pos.x > size.x - r && pos.y < r) {
            in_corner = true;
            corner_center = vec2(size.x - r, r);
        } else if (pos.x < r && pos.y > size.y - r) {
            in_corner = true;
            corner_center = vec2(r, size.y - r);
        } else if (pos.x > size.x - r && pos.y > size.y - r) {
            in_corner = true;
            corner_center = vec2(size.x - r, size.y - r);
        }
        
        if (in_corner && distance(pos, corner_center) > r) {
            discard;
        }
        
        if (v_uv.y <= u_peak_frac) {
            if (v_uv.y <= green_frac) {
                f_color = COL_VU_GREEN;
            } else if (v_uv.y <= yellow_frac) {
                f_color = COL_VU_YELLOW;
            } else {
                f_color = COL_VU_RED;
            }
        } else {
            f_color = COL_BG;
        }
    } else {
        discard;
    }
}
"#;

fn compile_shader_program(
    gl: &glow::Context,
    vertex_source: &str,
    fragment_source: &str,
) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program()?;

        let shader_sources = [
            (glow::VERTEX_SHADER, vertex_source),
            (glow::FRAGMENT_SHADER, fragment_source),
        ];

        let mut shaders = Vec::new();
        for (shader_type, shader_source) in shader_sources {
            let shader = gl.create_shader(shader_type)?;
            gl.shader_source(shader, shader_source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                let log = gl.get_shader_info_log(shader);
                for s in &shaders {
                    gl.detach_shader(program, *s);
                    gl.delete_shader(*s);
                }
                gl.delete_program(program);
                return Err(format!("Shader compilation failed: {log}"));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            for s in shaders {
                gl.detach_shader(program, s);
                gl.delete_shader(s);
            }
            gl.delete_program(program);
            return Err(format!("Program link failed: {log}"));
        }

        for s in shaders {
            gl.detach_shader(program, s);
            gl.delete_shader(s);
        }

        Ok(program)
    }
}

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
        if let (true, Some(log), Ok(c_msg)) = (
            std::env::var("XDG_SESSION_TYPE").as_deref() == Ok("wayland"),
            host.get_extension::<clack_extensions::log::HostLog>(),
            std::ffi::CString::new(
                "NAM-rs: Wayland session detected, using XDG portal for file dialogs",
            ),
        ) {
            log.log(&host, clack_extensions::log::LogSeverity::Info, &c_msg);
        }

        let gl_ctx = window.gl_context().expect("OpenGL context not available");
        unsafe {
            gl_ctx.make_current();
        }

        let gl = unsafe { glow::Context::from_loader_function(|s| gl_ctx.get_proc_address(s)) };

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

        let width = 600;
        let height = 260;
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
            state,
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

        let host_ref: &clack_plugin::host::HostSharedHandle =
            unsafe { std::mem::transmute(&self.host) };

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                crate::clap::gui::ui::draw_ui(ui, shared, host_ref, &mut self.state);
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
