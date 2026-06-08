// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Zone 5 (footer): Status bar with sample rate, latency, DSP load and telemetry.

mod metadata;
mod telemetry;

use crate::clap::gui::ui::colors::{COL_AMBER, COL_BORDER, COL_PANEL, COL_TEXT};
use crate::clap::gui::ui::state::UiState;
use crate::clap::plugin::NamClapShared;
use crate::common::spsc;
use std::time::Instant;

use self::metadata::{draw_metadata_strings, update_metadata_cache};
use self::telemetry::{draw_telemetry_strings, update_telemetry_state};

pub(crate) fn draw_zone5_status_bar(
    ui: &mut egui::Ui,
    shared: &NamClapShared,
    state: &mut UiState,
    accent_color: egui::Color32,
) {
    ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
        ui.painter().line(
            vec![ui.max_rect().left_top(), ui.max_rect().right_top()],
            egui::Stroke::new(0.5, COL_BORDER),
        );

        let model_meta_opt = if let Ok(guard) = shared.cold.ui_model_metadata.lock() {
            guard.clone()
        } else {
            None
        };

        let a2_placeholder = shared
            .cold
            .rt_status
            .check_flag(spsc::RT_STATUS_A2_PLACEHOLDER);

        let is_toast_active = if let Some(expiration) = state.toast_expiration {
            if Instant::now() < expiration {
                ui.ctx().request_repaint();
                true
            } else {
                state.toast_expiration = None;
                false
            }
        } else {
            false
        };

        let available_h = ui.available_height();
        let has_meta = model_meta_opt.is_some();
        let line_height = 11.5;
        let spacing = 3.0;

        let mut lines = 1; // Telemetry is always present
        if a2_placeholder {
            lines += 1;
        }
        if has_meta {
            lines += 1;
        }
        if is_toast_active {
            lines += 1;
        }
        let content_height = lines as f32 * line_height + (lines - 1) as f32 * spacing;

        let extra_space = (available_h - content_height).max(0.0);
        let top_space = extra_space * 0.55;
        ui.add_space(top_space);

        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = spacing;
            ui.spacing_mut().interact_size.y = 10.0;

            update_telemetry_state(state, shared);
            draw_telemetry_strings(ui, state, shared, accent_color);

            if a2_placeholder {
                ui.horizontal(|ui| {
                    let warn_font = egui::FontId::proportional(9.0);
                    let galley = ui.painter().layout_no_wrap(
                        "⚠ A2 model not supported — bypass active".to_string(),
                        warn_font.clone(),
                        COL_AMBER,
                    );
                    let available_width = ui.available_width();
                    let x_center = (available_width - galley.rect.width()).max(0.0) / 2.0;
                    ui.add_space(x_center);
                    ui.label(
                        egui::RichText::new("⚠ A2 model not supported — bypass active")
                            .font(warn_font)
                            .color(COL_AMBER),
                    );
                });
            }

            if is_toast_active {
                ui.horizontal(|ui| {
                    let toast_font = egui::FontId::proportional(9.0);
                    ui.label(
                        egui::RichText::new("Diagnostic copied · file in ~/.cache/nam-rs/")
                            .font(toast_font.clone())
                            .color(accent_color)
                            .strong(),
                    );
                    #[cfg(target_os = "linux")]
                    {
                        ui.add_space(4.0);
                        let open_btn = ui.add(
                            egui::Button::new(
                                egui::RichText::new("Open Folder")
                                    .font(egui::FontId::proportional(8.5))
                                    .color(COL_TEXT),
                            )
                            .fill(COL_PANEL)
                            .stroke(egui::Stroke::new(0.5, COL_BORDER)),
                        );
                        if open_btn.clicked() {
                            let home_dir = std::env::var_os("HOME");
                            if let Some(home) = home_dir {
                                let mut cache_dir = std::path::PathBuf::from(home);
                                cache_dir.push(".cache/nam-rs");
                                let _ = std::process::Command::new("xdg-open")
                                    .arg(cache_dir)
                                    .spawn();
                            }
                        }
                    }
                });
            }

            if let Some(meta) = model_meta_opt {
                update_metadata_cache(state, &meta);
                draw_metadata_strings(ui, state, accent_color);
            }
        });
    });
}
