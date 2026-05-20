// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::clap::plugin::NamClapShared;
use clack_plugin::host::HostSharedHandle;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Estado persistente da interface gráfica entre frames.
#[derive(Clone, Debug)]
pub struct UiState {
    /// Último valor de pico do canal esquerdo retido.
    pub peak_l_hold: f32,
    /// Último valor de pico do canal direito retido.
    pub peak_r_hold: f32,
    /// Instante em que o valor de pico esquerdo foi retido.
    pub peak_l_hold_time: Instant,
    /// Instante em que o valor de pico direito foi retido.
    pub peak_r_hold_time: Instant,
    /// Indica se o painel expandido de telemetria deve ser exibido.
    pub show_telemetry: bool,
}

impl Default for UiState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            peak_l_hold: 0.0,
            peak_r_hold: 0.0,
            peak_l_hold_time: now,
            peak_r_hold_time: now,
            show_telemetry: false,
        }
    }
}

fn get_simd_badge() -> &'static str {
    #[cfg(target_feature = "avx512f")]
    {
        "AVX-512"
    }
    #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx2"))]
    {
        "AVX2"
    }
    #[cfg(all(not(target_feature = "avx512f"), not(target_feature = "avx2"), target_feature = "sse4.1"))]
    {
        "SSE4.1"
    }
    #[cfg(all(not(target_feature = "avx512f"), not(target_feature = "avx2"), not(target_feature = "sse4.1")))]
    {
        "GENERIC"
    }
}

fn knob_widget(
    ui: &mut egui::Ui,
    _id: egui::Id,
    value: f32,
    range: std::ops::RangeInclusive<f32>,
    color: egui::Color32,
) -> (egui::Response, f32) {
    let desired_size = egui::vec2(54.0, 54.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::drag());

    let mut current_value = value;

    // Drag handling
    if response.dragged() {
        let delta_y = response.drag_delta().y;
        let range_len = range.end() - range.start();
        let value_delta = -delta_y * (range_len / 200.0);
        current_value = (current_value + value_delta).clamp(*range.start(), *range.end());
    }

    // Scroll handling
    if response.hovered() {
        let scroll_y = ui.input(|i| i.raw_scroll_delta.y);
        if scroll_y != 0.0 {
            let range_len = range.end() - range.start();
            let value_delta = scroll_y * (range_len / 1000.0);
            current_value = (current_value + value_delta).clamp(*range.start(), *range.end());
        }
    }

    let painter = ui.painter();
    let center = rect.center();
    let radius = rect.width() / 2.0 - 4.0;

    // Outer dial track
    painter.circle_stroke(
        center,
        radius,
        egui::Stroke::new(3.0, egui::Color32::from_rgb(26, 29, 35)),
    );

    let frac = (current_value - range.start()) / (range.end() - range.start());
    let angle_start = -135.0f32.to_radians();
    let angle_end = 135.0f32.to_radians();
    let angle = angle_start + frac * (angle_end - angle_start);

    // Active value arc
    let num_segments = 24;
    let arc_radius = radius;
    let mut points = Vec::new();
    for i in 0..=num_segments {
        let f = i as f32 / num_segments as f32;
        if f <= frac {
            let a = angle_start + f * (angle_end - angle_start) - std::f32::consts::FRAC_PI_2;
            points.push(center + egui::vec2(a.cos() * arc_radius, a.sin() * arc_radius));
        }
    }
    if points.len() > 1 {
        painter.add(egui::Shape::line(
            points,
            egui::Stroke::new(3.0, color),
        ));
    }

    // Knob body
    let body_radius = radius - 4.0;
    painter.circle_filled(
        center,
        body_radius,
        egui::Color32::from_rgb(46, 52, 64),
    );
    painter.circle_stroke(
        center,
        body_radius,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(74, 79, 90)),
    );

    // Indicator pointer line
    let pointer_angle = angle - std::f32::consts::FRAC_PI_2;
    let p_start = center + egui::vec2(pointer_angle.cos() * (body_radius - 8.0), pointer_angle.sin() * (body_radius - 8.0));
    let p_end = center + egui::vec2(pointer_angle.cos() * body_radius, pointer_angle.sin() * body_radius);
    painter.line(
        vec![p_start, p_end],
        egui::Stroke::new(2.5, egui::Color32::from_rgb(229, 233, 240)),
    );

    (response, current_value)
}

fn handle_knob(
    ui: &mut egui::Ui,
    id: egui::Id,
    label: &str,
    range: std::ops::RangeInclusive<f32>,
    default_value: f32,
    atomic_val: &std::sync::atomic::AtomicU32,
    changed_flag: &std::sync::atomic::AtomicBool,
    begin_flag: &std::sync::atomic::AtomicBool,
    end_flag: &std::sync::atomic::AtomicBool,
    color: egui::Color32,
    host: &HostSharedHandle,
) {
    let current_val = f32::from_bits(atomic_val.load(Ordering::Relaxed));
    let (response, new_val) = knob_widget(ui, id, current_val, range, color);

    if response.drag_started() {
        begin_flag.store(true, Ordering::Relaxed);
        if let Some(params_ext) = host.get_extension::<clack_extensions::params::HostParams>() {
            params_ext.request_flush(host);
        }
    }

    let reset_clicked = response.double_clicked();
    let final_val = if reset_clicked { default_value } else { new_val };

    if final_val != current_val {
        if reset_clicked {
            begin_flag.store(true, Ordering::Relaxed);
        }
        atomic_val.store(final_val.to_bits(), Ordering::Relaxed);
        changed_flag.store(true, Ordering::Relaxed);
        if reset_clicked {
            end_flag.store(true, Ordering::Relaxed);
        }
        if let Some(params_ext) = host.get_extension::<clack_extensions::params::HostParams>() {
            params_ext.request_flush(host);
        }
    }

    if response.drag_stopped() {
        end_flag.store(true, Ordering::Relaxed);
        if let Some(params_ext) = host.get_extension::<clack_extensions::params::HostParams>() {
            params_ext.request_flush(host);
        }
    }

    ui.vertical_centered(|ui| {
        ui.label(
            egui::RichText::new(label)
                .font(egui::FontId::proportional(11.0))
                .strong()
                .color(egui::Color32::from_rgb(139, 149, 165)),
        );
        ui.label(
            egui::RichText::new(format!("{:.1} dB", final_val))
                .font(egui::FontId::monospace(10.0))
                .color(egui::Color32::from_rgb(229, 233, 240)),
        );
    });
}

fn draw_vertical_meter(
    ui: &mut egui::Ui,
    peak_val: f32,
    hold_val: &mut f32,
    hold_time: &mut Instant,
    label: &str,
) {
    let height = 140.0;
    let width = 10.0;
    let desired_size = egui::vec2(width, height);
    let (rect, _response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

    let now = Instant::now();
    if peak_val >= *hold_val {
        *hold_val = peak_val;
        *hold_time = now;
    } else if now.duration_since(*hold_time) > Duration::from_secs(2) {
        *hold_val = (*hold_val - 0.05f32).max(peak_val);
    }

    let painter = ui.painter();
    painter.rect_filled(rect, 1.5, egui::Color32::from_rgb(26, 29, 35));

    let peak_db = if peak_val > 1e-5 { 20.0 * peak_val.log10() } else { -60.0 };
    let hold_db = if *hold_val > 1e-5 { 20.0 * (*hold_val).log10() } else { -60.0 };

    let min_db = -60.0f32;
    let max_db = 6.0f32;
    let db_to_frac = |db: f32| {
        ((db - min_db) / (max_db - min_db)).clamp(0.0, 1.0)
    };

    let peak_frac = db_to_frac(peak_db);
    let hold_frac = db_to_frac(hold_db);

    let green_frac = db_to_frac(-12.0);
    let yellow_frac = db_to_frac(-3.0);

    let draw_segment = |from_frac: f32, to_frac: f32, color: egui::Color32| {
        if to_frac > from_frac {
            let y_bottom = rect.bottom() - from_frac * height;
            let y_top = rect.bottom() - to_frac * height;
            let segment_rect = egui::Rect::from_min_max(
                egui::pos2(rect.left(), y_top),
                egui::pos2(rect.right(), y_bottom),
            );
            painter.rect_filled(segment_rect, 0.0, color);
        }
    };

    // Green
    let g_end = peak_frac.min(green_frac);
    draw_segment(0.0, g_end, egui::Color32::from_rgb(67, 233, 123));

    // Yellow
    if peak_frac > green_frac {
        let y_end = peak_frac.min(yellow_frac);
        draw_segment(green_frac, y_end, egui::Color32::from_rgb(245, 206, 98));
    }

    // Red
    if peak_frac > yellow_frac {
        draw_segment(yellow_frac, peak_frac, egui::Color32::from_rgb(247, 78, 78));
    }

    // Peak Hold Line
    if hold_frac > 0.0 {
        let y_hold = rect.bottom() - hold_frac * height;
        let line_color = if hold_db >= -3.0 {
            egui::Color32::from_rgb(247, 78, 78)
        } else if hold_db >= -12.0 {
            egui::Color32::from_rgb(245, 206, 98)
        } else {
            egui::Color32::from_rgb(67, 233, 123)
        };
        painter.line(
            vec![
                egui::pos2(rect.left() - 2.0, y_hold),
                egui::pos2(rect.right() + 2.0, y_hold),
            ],
            egui::Stroke::new(1.5, line_color),
        );
    }

    ui.vertical_centered(|ui| {
        ui.label(
            egui::RichText::new(label)
                .font(egui::FontId::monospace(8.0))
                .color(egui::Color32::from_rgb(139, 149, 165)),
        );
    });
}

fn handle_bypass(
    ui: &mut egui::Ui,
    _id: egui::Id,
    atomic_val: &std::sync::atomic::AtomicU32,
    changed_flag: &std::sync::atomic::AtomicBool,
    begin_flag: &std::sync::atomic::AtomicBool,
    end_flag: &std::sync::atomic::AtomicBool,
    host: &HostSharedHandle,
) {
    let current_bypass = atomic_val.load(Ordering::Relaxed) != 0;

    let desired_size = egui::vec2(50.0, 50.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());

    if response.clicked() {
        let new_bypass = !current_bypass;
        begin_flag.store(true, Ordering::Relaxed);
        atomic_val.store(if new_bypass { 1 } else { 0 }, Ordering::Relaxed);
        changed_flag.store(true, Ordering::Relaxed);
        end_flag.store(true, Ordering::Relaxed);

        if let Some(params_ext) = host.get_extension::<clack_extensions::params::HostParams>() {
            params_ext.request_flush(host);
        }
    }

    let painter = ui.painter();
    let center = rect.center();
    let outer_radius = rect.width() / 2.0 - 4.0;

    painter.circle_filled(center, outer_radius, egui::Color32::from_rgb(35, 40, 48));
    painter.circle_stroke(
        center,
        outer_radius,
        egui::Stroke::new(1.5, egui::Color32::from_rgb(46, 52, 64)),
    );

    let led_color = if current_bypass {
        egui::Color32::from_rgb(74, 79, 90)
    } else {
        egui::Color32::from_rgb(0, 212, 170)
    };

    painter.circle_filled(center, outer_radius - 8.0, led_color);

    ui.vertical_centered(|ui| {
        ui.label(
            egui::RichText::new("BYPASS")
                .font(egui::FontId::proportional(11.0))
                .strong()
                .color(egui::Color32::from_rgb(139, 149, 165)),
        );
        let status_text = if current_bypass { "BYPASSED" } else { "ACTIVE" };
        ui.label(
            egui::RichText::new(status_text)
                .font(egui::FontId::monospace(9.0))
                .color(led_color),
        );
    });
}

/// Desenha os componentes e o layout de 5 zonas da interface gráfica do NAM-rs.
pub fn draw_ui(
    ui: &mut egui::Ui,
    shared: &NamClapShared,
    host: &HostSharedHandle,
    state: &mut UiState,
) {
    ui.spacing_mut().item_spacing = egui::vec2(8.0, 4.0);

    // Main layout: horizontal layout containing Zone 1 to Zone 4
    ui.horizontal(|ui| {
        // Zone 1: Identity (left)
        ui.allocate_ui(egui::vec2(130.0, 200.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(5.0);
                ui.label(
                    egui::RichText::new("NAM-rs⚡")
                        .font(egui::FontId::proportional(22.0))
                        .strong()
                        .color(egui::Color32::from_rgb(0, 212, 170)),
                );
                ui.label(
                    egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                        .font(egui::FontId::proportional(10.0))
                        .color(egui::Color32::from_rgb(139, 149, 165)),
                );

                ui.add_space(4.0);
                // SIMD Badge with subtle frame
                let simd = get_simd_badge();
                let badge_color = if simd.contains("AVX") {
                    egui::Color32::from_rgb(0, 212, 170)
                } else {
                    egui::Color32::from_rgb(139, 149, 165)
                };
                ui.horizontal(|ui| {
                    let text = egui::RichText::new(simd)
                        .font(egui::FontId::monospace(9.0))
                        .strong()
                        .color(badge_color);
                    ui.add(
                        egui::Label::new(text)
                    );
                });

                ui.add_space(15.0);

                let load_btn = ui.add(
                    egui::Button::new(
                        egui::RichText::new("📂 Load Model")
                             .font(egui::FontId::proportional(12.0))
                             .strong()
                             .color(egui::Color32::from_rgb(229, 233, 240))
                    )
                    .fill(egui::Color32::from_rgb(35, 40, 48))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(46, 52, 64)))
                );

                if load_btn.clicked() && !shared.ui_loading.load(Ordering::Relaxed) {
                    shared.ui_loading.store(true, Ordering::Relaxed);
                    let shared_addr = shared as *const NamClapShared as usize;
                    let host_static: clack_plugin::host::HostSharedHandle<'static> = unsafe {
                        std::mem::transmute(*host)
                    };
                    std::thread::spawn(move || {
                        let path_opt = rfd::FileDialog::new()
                            .add_filter("NAM Model", &["nam", "namb"])
                            .pick_file();

                        let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                        if let Some(path) = path_opt {
                            if let Ok(mut pending_guard) = shared.ui_pending_model.lock() {
                                *pending_guard = Some(path);
                                host_static.request_callback();
                            }
                        } else {
                            shared.ui_loading.store(false, Ordering::Relaxed);
                        }
                    });
                }

                ui.add_space(6.0);
                let model_name = if shared.ui_loading.load(Ordering::Relaxed) {
                    ui.ctx().request_repaint_after(Duration::from_millis(100));
                    let elapsed = ui.input(|i| i.time);
                    let frames = ["⏳ Loading", "⏳ Loading.", "⏳ Loading..", "⏳ Loading..."];
                    let idx = (elapsed * 4.0) as usize % frames.len();
                    frames[idx].to_string()
                } else {
                    let name_guard = shared.ui_model_name.lock().unwrap();
                    if name_guard.is_empty() {
                        "No model loaded".to_string()
                    } else {
                        name_guard.clone()
                    }
                };
                ui.label(
                    egui::RichText::new(&model_name)
                        .font(egui::FontId::proportional(10.0))
                        .color(egui::Color32::from_rgb(139, 149, 165))
                        .italics(),
                );
            });
        });

        // Vertical Separator
        ui.add(egui::Separator::default().vertical());

        // Zone 2: Controls (center)
        ui.allocate_ui(egui::vec2(220.0, 200.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        handle_knob(
                            ui,
                            ui.make_persistent_id("input_gain_knob"),
                            "INPUT",
                            crate::math::constants::GAIN_MIN_DB..=crate::math::constants::GAIN_MAX_DB,
                            0.0,
                            &shared.param_input_gain,
                            &shared.gui_input_gain_changed,
                            &shared.gesture_begin_input_gain,
                            &shared.gesture_end_input_gain,
                            egui::Color32::from_rgb(0, 212, 170),
                            host,
                        );
                    });
                    ui.add_space(6.0);
                    ui.vertical(|ui| {
                        handle_knob(
                            ui,
                            ui.make_persistent_id("gate_thresh_knob"),
                            "GATE",
                            -90.0..=-40.0,
                            -70.0,
                            &shared.param_gate_thresh,
                            &shared.gui_gate_thresh_changed,
                            &shared.gesture_begin_gate_thresh,
                            &shared.gesture_end_gate_thresh,
                            egui::Color32::from_rgb(245, 166, 35),
                            host,
                        );
                    });
                    ui.add_space(6.0);
                    ui.vertical(|ui| {
                        handle_knob(
                            ui,
                            ui.make_persistent_id("output_gain_knob"),
                            "OUTPUT",
                            crate::math::constants::GAIN_MIN_DB..=crate::math::constants::GAIN_MAX_DB,
                            0.0,
                            &shared.param_output_gain,
                            &shared.gui_output_gain_changed,
                            &shared.gesture_begin_output_gain,
                            &shared.gesture_end_output_gain,
                            egui::Color32::from_rgb(0, 212, 170),
                            host,
                        );
                    });
                });
            });
        });

        // Vertical Separator
        ui.add(egui::Separator::default().vertical());

        // Zone 3: VU Meters (right)
        ui.allocate_ui(egui::vec2(75.0, 200.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(5.0);
                ui.horizontal(|ui| {
                    let peak_l = f32::from_bits(shared.ui_peak_l.swap(0.0f32.to_bits(), Ordering::Relaxed));
                    let peak_r = f32::from_bits(shared.ui_peak_r.swap(0.0f32.to_bits(), Ordering::Relaxed));

                    draw_vertical_meter(
                        ui,
                        peak_l,
                        &mut state.peak_l_hold,
                        &mut state.peak_l_hold_time,
                        "L",
                    );
                    ui.add_space(4.0);
                    draw_vertical_meter(
                        ui,
                        peak_r,
                        &mut state.peak_r_hold,
                        &mut state.peak_r_hold_time,
                        "R",
                    );
                });
            });
        });

        // Vertical Separator
        ui.add(egui::Separator::default().vertical());

        // Zone 4: Bypass (far right)
        ui.allocate_ui(egui::vec2(65.0, 200.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(15.0);
                handle_bypass(
                    ui,
                    ui.make_persistent_id("bypass_btn"),
                    &shared.param_bypass,
                    &shared.gui_bypass_changed,
                    &shared.gesture_begin_bypass,
                    &shared.gesture_end_bypass,
                    host,
                );
            });
        });
    });

    // Horizontal Separator above Status Bar
    ui.add(egui::Separator::default());

    // Zone 5: Status Bar / Footer
    ui.horizontal(|ui| {
        let sr = shared.sample_rate.load(Ordering::Relaxed);
        let lat = shared.current_latency.load(Ordering::Relaxed);

        let sr_text = if sr == 0 {
            "Rate: Off".to_string()
        } else {
            format!("Rate: {} Hz", sr)
        };

        ui.label(
            egui::RichText::new(sr_text)
                .font(egui::FontId::proportional(11.0))
                .color(egui::Color32::from_rgb(139, 149, 165)),
        );

        ui.label(
            egui::RichText::new(format!("Latency: {} spls", lat))
                .font(egui::FontId::proportional(11.0))
                .color(egui::Color32::from_rgb(139, 149, 165)),
        );

        // Model Type badge
        let model_type = {
            let name_guard = shared.ui_model_name.lock().unwrap();
            if name_guard.is_empty() {
                "NONE".to_string()
            } else if name_guard.ends_with(".namb") {
                "NAMB".to_string()
            } else {
                "NAM".to_string()
            }
        };

        ui.label(
            egui::RichText::new(format!("Type: {}", model_type))
                .font(egui::FontId::proportional(11.0))
                .color(egui::Color32::from_rgb(139, 149, 165)),
        );

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let telem_btn = ui.add(
                egui::Button::new(
                    egui::RichText::new("📊")
                        .font(egui::FontId::proportional(11.0))
                        .color(if state.show_telemetry {
                            egui::Color32::from_rgb(0, 212, 170)
                        } else {
                            egui::Color32::from_rgb(139, 149, 165)
                        })
                )
                .fill(egui::Color32::from_rgb(35, 40, 48))
            );

            if telem_btn.clicked() {
                state.show_telemetry = !state.show_telemetry;
            }
        });
    });

    // Optional Telemetry Expanded Panel
    if state.show_telemetry {
        ui.add_space(4.0);
        ui.add(egui::Separator::default());
        ui.horizontal(|ui| {
            let cycles = shared.rt_status.dsp_cycle_time.load(Ordering::Relaxed);
            let last_n = shared.rt_status.last_n_samples.load(Ordering::Relaxed);
            let prio = shared.rt_status.rt_priority.load(Ordering::Relaxed);
            let overloads = shared.rt_status.dsp_overloads.load(Ordering::Relaxed);
            let status_bits = shared.rt_status.status_bits.load(Ordering::Relaxed);

            ui.label(
                egui::RichText::new(format!("Cycles: {}", cycles))
                    .font(egui::FontId::monospace(10.0))
                    .color(egui::Color32::from_rgb(139, 149, 165)),
            );
            ui.label(
                egui::RichText::new(format!("Last N: {}", last_n))
                    .font(egui::FontId::monospace(10.0))
                    .color(egui::Color32::from_rgb(139, 149, 165)),
            );
            ui.label(
                egui::RichText::new(format!("RT Prio: {}", prio))
                    .font(egui::FontId::monospace(10.0))
                    .color(egui::Color32::from_rgb(139, 149, 165)),
            );
            ui.label(
                egui::RichText::new(format!("Overloads: {}", overloads))
                    .font(egui::FontId::monospace(10.0))
                    .color(egui::Color32::from_rgb(139, 149, 165)),
            );
            ui.label(
                egui::RichText::new(format!("Flags: {:#X}", status_bits))
                    .font(egui::FontId::monospace(10.0))
                    .color(egui::Color32::from_rgb(139, 149, 165)),
            );
        });
    }
}
