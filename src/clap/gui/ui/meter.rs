// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Medidor VU vertical com gradient tricolor, peak hold analógico e LED de clipping.

use glow::HasContext;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::colors::{COL_BG, COL_BYPASS_OFF, COL_MUTED, COL_VU_GREEN, COL_VU_RED, COL_VU_YELLOW};
use super::state::{VuMeterSharedState, VuUniforms};

/// Desenha um medidor VU vertical com gradiente tricolor, peak hold com fade analógico (E2),
/// LED de clipping persistente acima (E4) e labels L/R acima e abaixo (m4).
#[allow(clippy::too_many_arguments)]
pub fn draw_vertical_meter(
    ui: &mut egui::Ui,
    peak_val: f32,
    hold_val: &mut f32,
    hold_time: &mut Instant,
    clipped: &mut bool,
    label: &str,
    vu_program: Option<glow::Program>,
    vu_vao: Option<glow::VertexArray>,
    vu_shared_state: &mut Option<Arc<VuMeterSharedState>>,
    vu_callback: &mut Option<Arc<egui_glow::CallbackFn>>,
    vu_uniforms: Arc<OnceLock<VuUniforms>>,
) {
    let meter_h = 130.0f32;
    let meter_w = 16.0f32;

    ui.vertical(|ui| {
        if !label.is_empty() {
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new(label)
                        .font(egui::FontId::monospace(9.0))
                        .color(COL_MUTED),
                );
            });
            ui.add_space(2.0);
        }

        let desired_led = egui::vec2(meter_w, 6.0);
        let led_rect = ui
            .horizontal(|ui| {
                let avail = ui.available_width();
                let space = (avail - meter_w) / 2.0;
                if space > 0.0 {
                    ui.add_space(space);
                }
                let (r, _) = ui.allocate_exact_size(desired_led, egui::Sense::hover());
                r
            })
            .inner;
        {
            let enabled = ui.is_enabled();
            let painter = ui.painter();
            let clip_color = if *clipped {
                if enabled { COL_VU_RED } else { COL_BYPASS_OFF }
            } else {
                egui::Color32::from_rgb(50, 30, 30)
            };
            painter.rect_filled(led_rect, 1.0, clip_color);
        }
        let led_response = ui.interact(
            led_rect,
            ui.id().with((label, "clip_led")),
            egui::Sense::click(),
        );
        if led_response.clicked() {
            *clipped = false;
        }

        ui.add_space(2.0);

        let desired_size = egui::vec2(meter_w, meter_h);
        let meter_rect = ui
            .horizontal(|ui| {
                let avail = ui.available_width();
                let space = (avail - meter_w) / 2.0;
                if space > 0.0 {
                    ui.add_space(space);
                }
                let (r, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
                r
            })
            .inner;

        let meter_response = ui.interact(
            meter_rect,
            ui.id().with((label, "meter_bar")),
            egui::Sense::click(),
        );
        if meter_response.clicked() {
            *clipped = false;
        }

        let now = Instant::now();
        if peak_val >= *hold_val {
            *hold_val = peak_val;
            *hold_time = now;
        } else if now.duration_since(*hold_time) > Duration::from_secs(2) {
            *hold_val = (*hold_val * 0.95_f32).max(peak_val).max(0.0);
        }

        if peak_val >= 1.0 {
            *clipped = true;
        }

        let peak_db = if peak_val > 1e-5 {
            20.0 * peak_val.log10()
        } else {
            -60.0
        };
        let hold_db = if *hold_val > 1e-5 {
            20.0 * (*hold_val).log10()
        } else {
            -60.0
        };

        let min_db = -60.0f32;
        let max_db = 6.0f32;
        let db_to_frac = |db: f32| ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0);

        let peak_frac = db_to_frac(peak_db);
        let hold_frac = db_to_frac(hold_db);
        let green_frac = db_to_frac(-12.0);
        let yellow_frac = db_to_frac(-3.0);

        let hold_color_type = if hold_db >= -3.0 {
            2
        } else if hold_db >= -12.0 {
            1
        } else {
            0
        };

        let enabled = ui.is_enabled();
        let mut drawn_with_glow = false;
        if let (true, Some(program), Some(vao)) = (enabled, vu_program, vu_vao) {
            let shared_state =
                vu_shared_state.get_or_insert_with(|| Arc::new(VuMeterSharedState::default()));
            let callback = vu_callback.get_or_insert_with(|| {
                let shared_state_capture = Arc::clone(shared_state);
                let uniforms_capture = Arc::clone(&vu_uniforms);
                Arc::new(egui_glow::CallbackFn::new(move |info, painter| {
                    let gl = painter.gl();
                    let peak_frac = f32::from_bits(
                        shared_state_capture
                            .peak_frac
                            .load(std::sync::atomic::Ordering::Relaxed),
                    );
                    let hold_frac = f32::from_bits(
                        shared_state_capture
                            .hold_frac
                            .load(std::sync::atomic::Ordering::Relaxed),
                    );
                    let hold_color_type = shared_state_capture
                        .hold_color_type
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let rect_min_x = f32::from_bits(
                        shared_state_capture
                            .rect_min_x
                            .load(std::sync::atomic::Ordering::Relaxed),
                    );
                    let rect_min_y = f32::from_bits(
                        shared_state_capture
                            .rect_min_y
                            .load(std::sync::atomic::Ordering::Relaxed),
                    );
                    let rect_max_x = f32::from_bits(
                        shared_state_capture
                            .rect_max_x
                            .load(std::sync::atomic::Ordering::Relaxed),
                    );
                    let rect_max_y = f32::from_bits(
                        shared_state_capture
                            .rect_max_y
                            .load(std::sync::atomic::Ordering::Relaxed),
                    );

                    let ppp = info.pixels_per_point;
                    let p_left = rect_min_x * ppp;
                    let p_top = rect_min_y * ppp;
                    let p_right = rect_max_x * ppp;
                    let p_bottom = rect_max_y * ppp;

                    let vp_left = info.viewport.min.x;
                    let vp_top = info.viewport.min.y;
                    let vp_right = info.viewport.max.x;
                    let vp_bottom = info.viewport.max.y;

                    unsafe {
                        gl.enable(glow::BLEND);
                        gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

                        gl.use_program(Some(program));
                        gl.bind_vertex_array(Some(vao));

                        let uniforms = uniforms_capture.get_or_init(|| VuUniforms {
                            loc_viewport: gl.get_uniform_location(program, "u_viewport"),
                            loc_meter_rect: gl.get_uniform_location(program, "u_meter_rect"),
                            loc_peak_frac: gl.get_uniform_location(program, "u_peak_frac"),
                            loc_hold_frac: gl.get_uniform_location(program, "u_hold_frac"),
                            loc_hold_color_type: gl
                                .get_uniform_location(program, "u_hold_color_type"),
                        });

                        if let Some(ref loc) = uniforms.loc_viewport {
                            gl.uniform_4_f32(Some(loc), vp_left, vp_top, vp_right, vp_bottom);
                        }
                        if let Some(ref loc) = uniforms.loc_meter_rect {
                            gl.uniform_4_f32(Some(loc), p_left, p_top, p_right, p_bottom);
                        }
                        if let Some(ref loc) = uniforms.loc_peak_frac {
                            gl.uniform_1_f32(Some(loc), peak_frac);
                        }
                        if let Some(ref loc) = uniforms.loc_hold_frac {
                            gl.uniform_1_f32(Some(loc), hold_frac);
                        }
                        if let Some(ref loc) = uniforms.loc_hold_color_type {
                            gl.uniform_1_i32(Some(loc), hold_color_type);
                        }

                        gl.draw_arrays(glow::TRIANGLES, 0, 6);

                        gl.bind_vertex_array(None);
                        gl.use_program(None);
                        gl.disable(glow::BLEND);
                    }
                }))
            });

            shared_state
                .peak_frac
                .store(peak_frac.to_bits(), std::sync::atomic::Ordering::Relaxed);
            shared_state
                .hold_frac
                .store(hold_frac.to_bits(), std::sync::atomic::Ordering::Relaxed);
            shared_state
                .hold_color_type
                .store(hold_color_type, std::sync::atomic::Ordering::Relaxed);
            shared_state.rect_min_x.store(
                meter_rect.min.x.to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
            shared_state.rect_min_y.store(
                meter_rect.min.y.to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
            shared_state.rect_max_x.store(
                meter_rect.max.x.to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );
            shared_state.rect_max_y.store(
                meter_rect.max.y.to_bits(),
                std::sync::atomic::Ordering::Relaxed,
            );

            let callback_fn = Arc::clone(callback);
            let callback = egui::PaintCallback {
                rect: meter_rect,
                callback: callback_fn,
            };
            ui.painter().add(egui::Shape::Callback(callback));
            drawn_with_glow = true;
        }
        if !drawn_with_glow {
            let painter = ui.painter();
            painter.rect_filled(meter_rect, 1.5, COL_BG);

            let draw_segment = |from_frac: f32, to_frac: f32, color: egui::Color32| {
                if to_frac > from_frac {
                    let y_bottom = meter_rect.bottom() - from_frac * meter_h;
                    let y_top = meter_rect.bottom() - to_frac * meter_h;
                    let seg = egui::Rect::from_min_max(
                        egui::pos2(meter_rect.left(), y_top),
                        egui::pos2(meter_rect.right(), y_bottom),
                    );
                    painter.rect_filled(seg, 0.0, color);
                }
            };

            let (col_g, col_y, col_r) = if enabled {
                (COL_VU_GREEN, COL_VU_YELLOW, COL_VU_RED)
            } else {
                (COL_BYPASS_OFF, COL_BYPASS_OFF, COL_BYPASS_OFF)
            };

            draw_segment(0.0, peak_frac.min(green_frac), col_g);
            if peak_frac > green_frac {
                draw_segment(green_frac, peak_frac.min(yellow_frac), col_y);
            }
            if peak_frac > yellow_frac {
                draw_segment(yellow_frac, peak_frac, col_r);
            }

            if hold_frac > 0.0 {
                let y_hold = meter_rect.bottom() - hold_frac * meter_h;
                let hold_color = if enabled {
                    if hold_db >= -3.0 {
                        COL_VU_RED
                    } else if hold_db >= -12.0 {
                        COL_VU_YELLOW
                    } else {
                        COL_VU_GREEN
                    }
                } else {
                    COL_MUTED
                };
                painter.line(
                    vec![
                        egui::pos2(meter_rect.left() - 2.0, y_hold),
                        egui::pos2(meter_rect.right() + 2.0, y_hold),
                    ],
                    egui::Stroke::new(1.5, hold_color),
                );
            }
        }

        ui.add_space(2.0);
        ui.vertical_centered(|ui| {
            let peak_text = if !enabled || peak_db <= -59.9 {
                "-inf".to_string()
            } else {
                format!("{:.1}", peak_db)
            };

            let peak_color = if !enabled {
                COL_MUTED
            } else if peak_db >= -3.0 {
                COL_VU_RED
            } else if peak_db >= -12.0 {
                COL_VU_YELLOW
            } else {
                COL_VU_GREEN
            };

            ui.label(
                egui::RichText::new(peak_text)
                    .font(egui::FontId::monospace(9.5))
                    .strong()
                    .color(peak_color),
            );
        });

        ui.add_space(2.0);
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new(label)
                    .font(egui::FontId::monospace(9.0))
                    .color(COL_MUTED),
            );
        });
    });
}
