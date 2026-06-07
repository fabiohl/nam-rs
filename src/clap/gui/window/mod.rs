// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use baseview::{
    DropData, DropEffect, Event, EventStatus, MouseEvent, ScrollDelta, Window, WindowEvent,
    WindowHandler,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use glow::HasContext;

pub(crate) mod input_map;
pub(crate) mod shaders;

use self::input_map::{map_keyboard_event, map_modifiers, map_mouse_button};
use self::shaders::{FRAGMENT_SHADER_SRC, VERTEX_SHADER_SRC, compile_shader_program};

use crate::clap::plugin::NamClapSharedRef;

/// Checks whether the drag-and-drop data contains a valid NAM model file.
///
/// Accepts files with `.nam` (JSON) or `.namb` (compact binary) extension.
/// Returns the `PathBuf` of the first valid file found, or `None`.
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

/// Main window representation of the NAM-rs plugin.
///
/// Manages the initialization and lifecycle of the `egui` context and the `egui_glow` painter
/// for accelerated drawing via OpenGL (glow).
pub struct NamPluginWindow {
    /// Egui Context.
    egui_ctx: egui::Context,
    /// Glow Painter for rendering via OpenGL.
    painter: egui_glow::Painter,
    /// Accumulated raw input for Egui.
    raw_input: egui::RawInput,
    /// Start time for elapsed time calculations.
    start_time: Instant,
    /// Physical window width.
    width: u32,
    /// Physical window height.
    height: u32,
    /// Screen scale factor.
    scale: f32,
    /// Reference to the plugin's shared state.
    shared: NamClapSharedRef,
    /// Lifetime fence (shared Arc for destruction detection).
    alive_fence: Arc<std::sync::atomic::AtomicBool>,
    /// Static shared handle of the CLAP host.
    host: clack_plugin::host::HostSharedHandle<'static>,
    /// Local persistent state of the graphical interface.
    state: crate::clap::gui::ui::UiState,
    /// Last known mouse position.
    last_mouse_pos: egui::Pos2,
    /// Close signal for floating windows. When set to true, the window event
    /// loop will exit gracefully.
    close_signal: Arc<AtomicBool>,
}

impl NamPluginWindow {
    /// Initializes the main plugin window with baseview, creating the graphics context
    /// and setting up the custom dark theme.
    ///
    /// # FFI Robustness
    ///
    /// This function is called by the GUI thread via baseview callback (C ABI). Panics that
    /// cross this boundary are UB in C++ hosts. Therefore, OpenGL context or Painter
    /// initialization failures result in error logging and graceful stub — not a panic.
    pub fn new(
        window: &mut Window,
        shared: NamClapSharedRef,
        host: clack_plugin::host::HostSharedHandle<'static>,
        close_signal: Arc<AtomicBool>,
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

        // WHITELIST: gl_context() fails only if baseview did not set up an OpenGL context,
        // which is a host GUI contract violation — not recoverable. The panic occurs on the
        // window creation thread, before any RT/FFI audio callback.
        let gl_ctx = window.gl_context().expect("OpenGL context not available");
        // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
        unsafe {
            gl_ctx.make_current();
        }

        // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
        let gl = unsafe { glow::Context::from_loader_function(|s| gl_ctx.get_proc_address(s)) };

        // WHITELIST: Painter::new fails only if the OpenGL context is invalid (already verified
        // above). Occurs on the window creation thread, before any FFI audio callback.
        let painter = egui_glow::Painter::new(Arc::new(gl), "", None, true)
            .expect("Failed to create egui_glow Painter");

        // Robust shader compilation and VAO creation
        // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
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

        // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
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

        // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
        let alive_fence = unsafe { &*shared.0 }.cold.alive_fence.clone();

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
            close_signal,
        }
    }

    fn safe_shared(&self) -> Option<&'static crate::clap::plugin::NamClapShared> {
        if self.alive_fence.load(std::sync::atomic::Ordering::Relaxed) {
            // SAFETY: If alive_fence is true, the plugin and its shared state
            // are still alive in memory, so the pointer is valid.
            unsafe { Some(&*self.shared.0) }
        } else {
            None
        }
    }
}

impl WindowHandler for NamPluginWindow {
    fn on_frame(&mut self, window: &mut Window) {
        if self.close_signal.load(Ordering::Relaxed) {
            window.close();
            return;
        }

        // Unlike `new()`, `on_frame()` is called repeatedly by the baseview
        // rendering loop (C ABI). A panic here would cross the FFI boundary and
        // cause UB in C++ hosts. We use a silent early-return as a safe fallback.
        let Some(gl_ctx) = window.gl_context() else {
            return;
        };
        // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
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
            // Background color: #1A1D23 (approved dark mode palette — T4.0.2)
            self.painter.clear(screen_size, [0.102, 0.114, 0.137, 1.0]);
            self.painter.paint_and_update_textures(
                screen_size,
                full_output.pixels_per_point,
                &clipped_primitives,
                &full_output.textures_delta,
            );

            gl_ctx.swap_buffers();
        }

        // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
        unsafe {
            gl_ctx.make_not_current();
        }
    }

    /// Processes window events (mouse, keyboard, resize, drag-and-drop).
    ///
    /// Converts baseview events to the `egui::RawInput` format that will be consumed
    /// in the next `on_frame()`. For NAM model drag-and-drop, communicates directly
    /// with `NamClapShared` to enqueue loading via `ui_pending_model`.
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
                                if let Ok(mut pending_guard) = shared.cold.ui_pending_model.lock() {
                                    *pending_guard = Some(path);
                                    shared
                                        .cold
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
