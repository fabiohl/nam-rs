// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Zone 1 (left): Identity — logo, version, SIMD badge, model load button.

use crate::clap::plugin::NamClapShared;
use clack_plugin::host::HostSharedHandle;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::clap::gui::ui::{
    colors::{COL_BG, COL_BORDER, COL_MUTED, COL_PANEL, COL_TEXT, COL_VU_RED},
    simd::get_simd_badge,
    state::UiState,
};

fn spawn_file_dialog(
    shared_addr: usize,
    host_static: clack_plugin::host::HostSharedHandle<'static>,
    alive_fence: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let path_opt = rfd::FileDialog::new()
                .add_filter("NAM Model", &["nam", "namb"])
                .pick_file();
            let _ = tx.send(path_opt);
        });

        match rx.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(path_opt) => {
                if alive_fence.load(Ordering::Relaxed) {
                    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
                    let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                    if let Some(path) = path_opt {
                        if let Ok(mut pending_guard) = shared.cold.ui_pending_model.lock() {
                            *pending_guard = Some(path);
                            host_static.request_callback();
                        }
                    } else {
                        shared.cold.ui_loading.store(false, Ordering::Relaxed);
                    }
                }
            }
            Err(_) => {
                if alive_fence.load(Ordering::Relaxed) {
                    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
                    let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                    shared.cold.ui_loading.store(false, Ordering::Relaxed);

                    if let (Some(log), Ok(c_msg)) = (
                        host_static.get_extension::<clack_extensions::log::HostLog>(),
                        std::ffi::CString::new("NAM-rs: File dialog portal timed out after 120s"),
                    ) {
                        log.log(
                            &host_static,
                            clack_extensions::log::LogSeverity::Warning,
                            &c_msg,
                        );
                    }
                }
            }
        }
    });
}

pub(crate) fn draw_zone1_identity(
    ui: &mut egui::Ui,
    shared: &NamClapShared,
    host: &HostSharedHandle,
    state: &mut UiState,
    accent_color: egui::Color32,
) -> Option<egui::Id> {
    let mut load_btn_id = None;
    ui.allocate_ui(egui::vec2(135.0, 210.0), |ui| {
        ui.vertical(|ui| {
            ui.add_space(8.0);

            ui.label(
                egui::RichText::new("NAM-rs⚡")
                    .font(egui::FontId::proportional(24.0))
                    .strong()
                    .color(accent_color),
            );

            ui.label(
                egui::RichText::new("Neural Amp Modeler")
                    .font(egui::FontId::proportional(9.5))
                    .color(COL_MUTED),
            );

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                        .font(egui::FontId::proportional(9.0))
                        .color(COL_MUTED),
                );
                ui.add_space(4.0);
                let simd = get_simd_badge();
                let badge_color = if simd.contains("AVX") {
                    accent_color
                } else {
                    COL_MUTED
                };
                let response = ui.label(
                    egui::RichText::new(simd)
                        .font(egui::FontId::monospace(8.0))
                        .strong()
                        .color(badge_color),
                );
                response.on_hover_text(format!(
                    "Active runtime SIMD engine backend: {}.",
                    crate::math::common::SIMD_MATH.name
                ));
            });

            ui.add_space(14.0);

            ui.label(
                egui::RichText::new("MODEL")
                    .font(egui::FontId::proportional(9.0))
                    .strong()
                    .color(COL_MUTED),
            );
            ui.add_space(2.0);

            // "Load Model" button
            let load_btn = ui.add(
                egui::Button::new(
                    egui::RichText::new("📂 Load Model")
                        .font(egui::FontId::proportional(11.5))
                        .strong()
                        .color(COL_TEXT),
                )
                .fill(COL_PANEL)
                .stroke(egui::Stroke::new(1.0, COL_BORDER)),
            );
            let btn_id = load_btn.id;
            load_btn_id = Some(btn_id);
            ui.memory_mut(|mem| mem.interested_in_focus(btn_id, ui.layer_id()));

            if load_btn.has_focus() {
                ui.painter().rect_stroke(
                    load_btn.rect,
                    2.0,
                    egui::Stroke::new(2.0, accent_color),
                    egui::StrokeKind::Outside,
                );
            }

            let mut load_clicked = load_btn.clicked();
            if load_btn.has_focus() {
                ui.input(|i| {
                    if i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::Enter) {
                        load_clicked = true;
                    }
                });
            }

            if load_clicked && !shared.cold.ui_loading.load(Ordering::Relaxed) {
                shared.cold.ui_loading.store(true, Ordering::Relaxed);
                let shared_addr = shared as *const NamClapShared as usize;
                let host_static: clack_plugin::host::HostSharedHandle<'static> =
                    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
                    unsafe { crate::clap::gui::extend_host_lifetime(*host) };
                let alive_fence = Arc::clone(&shared.cold.alive_fence);
                spawn_file_dialog(shared_addr, host_static, alive_fence);
            }

            ui.add_space(6.0);

            let is_error_active = if let Some(expiration) = state.error_expiration {
                if Instant::now() < expiration {
                    ui.ctx().request_repaint();
                    true
                } else {
                    state.error_expiration = None;
                    false
                }
            } else {
                false
            };

            let model_name: &str = if is_error_active {
                if state.model_display_name != "⚠ Load failed" {
                    state.model_display_name.clear();
                    state.model_display_name.push_str("⚠ Load failed");
                }
                &state.model_display_name
            } else if shared.cold.ui_loading.load(Ordering::Relaxed) {
                ui.ctx().request_repaint_after(Duration::from_millis(100));
                let elapsed = ui.input(|i| i.time);
                let frames = ["Loading", "Loading.", "Loading..", "Loading..."];
                let idx = (elapsed * 4.0) as usize % frames.len();
                state.model_display_name.clear();
                state.model_display_name.push_str(frames[idx]);
                &state.model_display_name
            } else {
                let name_guard = shared
                    .cold
                    .ui_model_name
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if name_guard.is_empty() {
                    if state.model_display_name != "No model loaded" {
                        state.model_display_name.clear();
                        state.model_display_name.push_str("No model loaded");
                    }
                    &state.model_display_name
                } else {
                    if *name_guard != state.model_display_name {
                        state.model_display_name.clear();
                        state.model_display_name.push_str(&name_guard);
                    }
                    &state.model_display_name
                }
            };

            let text_color = if is_error_active {
                COL_VU_RED
            } else {
                COL_MUTED
            };

            let frame_res = egui::Frame::new()
                .fill(COL_BG)
                .stroke(egui::Stroke::new(
                    1.0,
                    if is_error_active {
                        COL_VU_RED
                    } else {
                        COL_BORDER
                    },
                ))
                .corner_radius(egui::CornerRadius::same(3))
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    ui.set_min_width(120.0);
                    ui.set_max_width(120.0);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(model_name)
                                .font(egui::FontId::proportional(9.5))
                                .color(text_color)
                                .italics(),
                        )
                        .wrap(),
                    );
                });

            if is_error_active {
                frame_res.response.on_hover_text(&state.error_msg);
            } else {
                frame_res.response.on_hover_text(model_name);
            }
        });
    });
    load_btn_id
}
