// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Interface gráfica do plugin NAM-rs renderizada via `egui` + `egui_glow`.
//!
//! Este módulo implementa toda a renderização da GUI do plugin CLAP,
//! organizada em 5 zonas visuais:
//!
//! - **Zona 1** (esquerda): Identidade — logo, versão, badge SIMD, botão de carga de modelo
//! - **Zona 2** (centro): Controles — knobs de Input Gain, Output Gain e Gate Threshold
//! - **Zona 3** (direita): Medidores VU stereo com peak hold e LEDs de clipping
//! - **Zona 4** (extrema direita): Toggle de Bypass com LED de status
//! - **Zona 5** (footer): Status bar com sample rate, latência, carga DSP e telemetria
//!
//! A comunicação com o estado do plugin é feita inteiramente via atômicos
//! em `NamClapShared`, sem locks no caminho de renderização.

mod bypass;
pub mod colors;
mod knob;
mod meter;
mod simd;
pub mod state;
mod vsep;

pub use colors::{
    COL_ACCENT, COL_AMBER, COL_BG, COL_BORDER, COL_BYPASS_OFF, COL_MUTED, COL_PANEL, COL_TEXT,
    COL_VU_GREEN, COL_VU_RED, COL_VU_YELLOW,
};
pub use colors::{resolve_accent, resolve_color};
pub use state::{UiState, VuMeterSharedState, VuUniforms};

use crate::clap::plugin::NamClapShared;
use clack_plugin::host::HostSharedHandle;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::common::spsc;

use self::bypass::handle_bypass;
use self::knob::handle_knob;
#[allow(unused_imports)]
use self::knob::knob_widget;
use self::meter::draw_vertical_meter;
use self::simd::get_simd_badge;
use self::vsep::styled_vsep;

/// Desenha os componentes e o layout de 5 zonas da interface gráfica do NAM-rs.
///
/// Esta é a função principal de renderização da GUI, chamada a cada frame pelo
/// `NamPluginWindow::on_frame()`. Lê o estado compartilhado via atômicos e
/// escreve alterações de parâmetros de volta nos atômicos + solicita flush ao host.
///
/// # Layout
///
/// ```text
/// ┌─────────┬────────────────┬───────┬──────┐
/// │  ZONA 1  │    ZONA 2        │ ZONA 3  │ ZONA 4 │
/// │  Logo    │  Input  Out Gate │  VU L R │ Bypass │
/// │  Model   │  Knobs           │  Meters │ Switch │
/// ├─────────┴────────────────┴───────┴──────┤
/// │            ZONA 5: Status Bar              │
/// └─────────────────────────────────────────┘
/// ```
pub fn draw_ui(
    ui: &mut egui::Ui,
    shared: &NamClapShared,
    host: &HostSharedHandle,
    state: &mut UiState,
) {
    if shared.ui_load_error.swap(false, Ordering::Relaxed) {
        state.error_expiration = Some(Instant::now() + Duration::from_secs(3));
        if let Ok(msg_guard) = shared.ui_load_error_msg.lock() {
            state.error_msg = msg_guard.clone();
        } else {
            state.error_msg = "Load failed".to_string();
        }
    }

    let current_bypass = shared.param_bypass.load(Ordering::Relaxed) != 0;
    let accent_color = resolve_accent(shared);
    let mut load_btn_id = None;
    ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);

    let available_w = ui.available_width();

    // Layout principal: horizontal com Zonas 1–4 centralizadas
    ui.horizontal(|ui| {
        let total_content_width =
            135.0 + 240.0 + 80.0 + 60.0 + 3.0 * 13.0 + 6.0 * ui.spacing().item_spacing.x;
        let extra_width = (available_w - total_content_width).max(0.0);
        let left_space = (extra_width / 2.0) - ui.spacing().item_spacing.x;
        if left_space > 0.0 {
            ui.add_space(left_space);
        }

        // ── Zona 1: Identidade (esquerda) ─────────────────────
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

                // Botão "Load Model"
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

                if load_clicked && !shared.ui_loading.load(Ordering::Relaxed) {
                    shared.ui_loading.store(true, Ordering::Relaxed);
                    let shared_addr = shared as *const NamClapShared as usize;
                    let host_static: clack_plugin::host::HostSharedHandle<'static> =
                        unsafe { super::extend_host_lifetime(*host) };
                    let alive_fence = Arc::clone(&shared.alive_fence);
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
                                    let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                                    if let Some(path) = path_opt {
                                        if let Ok(mut pending_guard) =
                                            shared.ui_pending_model.lock()
                                        {
                                            *pending_guard = Some(path);
                                            host_static.request_callback();
                                        }
                                    } else {
                                        shared.ui_loading.store(false, Ordering::Relaxed);
                                    }
                                }
                            }
                            Err(_) => {
                                if alive_fence.load(Ordering::Relaxed) {
                                    let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                                    shared.ui_loading.store(false, Ordering::Relaxed);

                                    if let (Some(log), Ok(c_msg)) = (
                                        host_static
                                            .get_extension::<clack_extensions::log::HostLog>(),
                                        std::ffi::CString::new(
                                            "NAM-rs: File dialog portal timed out after 120s",
                                        ),
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

                let model_name = if is_error_active {
                    "⚠ Load failed".to_string()
                } else if shared.ui_loading.load(Ordering::Relaxed) {
                    ui.ctx().request_repaint_after(Duration::from_millis(100));
                    let elapsed = ui.input(|i| i.time);
                    let frames = ["Loading", "Loading.", "Loading..", "Loading..."];
                    let idx = (elapsed * 4.0) as usize % frames.len();
                    frames[idx].to_string()
                } else {
                    let name_guard = shared
                        .ui_model_name
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    if name_guard.is_empty() {
                        "No model loaded".to_string()
                    } else {
                        name_guard.clone()
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
                                egui::RichText::new(&model_name)
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
                    frame_res.response.on_hover_text(&model_name);
                }
            });
        });

        styled_vsep(ui);

        // ── Zona 2: Controles (centro) ─────────────────────
        ui.allocate_ui(egui::vec2(240.0, 210.0), |ui| {
            if current_bypass {
                ui.disable();
            }
            ui.vertical(|ui| {
                ui.add_space(12.0);
                let ind_input = shared.param_indication
                    [crate::clap::extensions::params::PARAM_INPUT_GAIN as usize]
                    .load(Ordering::Relaxed);
                let ind_input_color = resolve_color(
                    shared.param_indication_color
                        [crate::clap::extensions::params::PARAM_INPUT_GAIN as usize]
                        .load(Ordering::Relaxed),
                    egui::Color32::from_rgb(94, 129, 172),
                );

                let ind_output = shared.param_indication
                    [crate::clap::extensions::params::PARAM_OUTPUT_GAIN as usize]
                    .load(Ordering::Relaxed);
                let ind_output_color = resolve_color(
                    shared.param_indication_color
                        [crate::clap::extensions::params::PARAM_OUTPUT_GAIN as usize]
                        .load(Ordering::Relaxed),
                    egui::Color32::from_rgb(94, 129, 172),
                );

                let ind_gate = shared.param_indication
                    [crate::clap::extensions::params::PARAM_GATE_THRESH as usize]
                    .load(Ordering::Relaxed);
                let ind_gate_color = resolve_color(
                    shared.param_indication_color
                        [crate::clap::extensions::params::PARAM_GATE_THRESH as usize]
                        .load(Ordering::Relaxed),
                    egui::Color32::from_rgb(94, 129, 172),
                );

                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(78.0, 185.0), |ui| {
                        handle_knob(
                            ui,
                            ui.make_persistent_id("input_gain_knob"),
                            "INPUT",
                            crate::math::constants::GAIN_MIN_DB
                                ..=crate::math::constants::GAIN_MAX_DB,
                            0.0,
                            &shared.param_input_gain,
                            &shared.gui_input_gain_changed,
                            &shared.gesture_begin_input_gain,
                            &shared.gesture_end_input_gain,
                            accent_color,
                            accent_color,
                            host,
                            egui::vec2(70.0, 70.0),
                            ind_input,
                            ind_input_color,
                            " dB",
                        );
                    });
                    ui.add_space(2.0);
                    ui.allocate_ui(egui::vec2(78.0, 185.0), |ui| {
                        handle_knob(
                            ui,
                            ui.make_persistent_id("output_gain_knob"),
                            "OUTPUT",
                            crate::math::constants::GAIN_MIN_DB
                                ..=crate::math::constants::GAIN_MAX_DB,
                            0.0,
                            &shared.param_output_gain,
                            &shared.gui_output_gain_changed,
                            &shared.gesture_begin_output_gain,
                            &shared.gesture_end_output_gain,
                            accent_color,
                            accent_color,
                            host,
                            egui::vec2(70.0, 70.0),
                            ind_output,
                            ind_output_color,
                            " dB",
                        );
                    });
                    ui.add_space(2.0);
                    ui.allocate_ui(egui::vec2(70.0, 185.0), |ui| {
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
                            COL_AMBER,
                            accent_color,
                            host,
                            egui::vec2(42.0, 42.0),
                            ind_gate,
                            ind_gate_color,
                            " dB (Threshold)",
                        );
                    });
                });
            });
        });

        styled_vsep(ui);

        // ── Zona 3: Medidores VU (direita) ──────────────────
        ui.allocate_ui(egui::vec2(80.0, 210.0), |ui| {
            if current_bypass {
                ui.disable();
            }
            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                ui.add_space(8.0);

                let peak_l =
                    f32::from_bits(shared.ui_peak_l.swap(0.0f32.to_bits(), Ordering::Relaxed));
                let peak_r =
                    f32::from_bits(shared.ui_peak_r.swap(0.0f32.to_bits(), Ordering::Relaxed));

                let is_stereo = shared.active_channel_count.load(Ordering::Relaxed) >= 2;

                ui.horizontal(|ui| {
                    if is_stereo {
                        ui.allocate_ui(egui::vec2(36.0, 190.0), |ui| {
                            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                draw_vertical_meter(
                                    ui,
                                    peak_l,
                                    &mut state.peak_l_hold,
                                    &mut state.peak_l_hold_time,
                                    &mut state.clip_l,
                                    "L",
                                    state.vu_program,
                                    state.vu_vao,
                                    &mut state.vu_l_state,
                                    &mut state.vu_l_callback,
                                    Arc::clone(&state.vu_uniforms),
                                );
                            });
                        });
                        ui.add_space(4.0);
                        ui.allocate_ui(egui::vec2(36.0, 190.0), |ui| {
                            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                draw_vertical_meter(
                                    ui,
                                    peak_r,
                                    &mut state.peak_r_hold,
                                    &mut state.peak_r_hold_time,
                                    &mut state.clip_r,
                                    "R",
                                    state.vu_program,
                                    state.vu_vao,
                                    &mut state.vu_r_state,
                                    &mut state.vu_r_callback,
                                    Arc::clone(&state.vu_uniforms),
                                );
                            });
                        });
                    } else {
                        ui.allocate_ui(egui::vec2(76.0, 190.0), |ui| {
                            ui.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
                                draw_vertical_meter(
                                    ui,
                                    peak_l,
                                    &mut state.peak_l_hold,
                                    &mut state.peak_l_hold_time,
                                    &mut state.clip_l,
                                    "",
                                    state.vu_program,
                                    state.vu_vao,
                                    &mut state.vu_l_state,
                                    &mut state.vu_l_callback,
                                    Arc::clone(&state.vu_uniforms),
                                );
                            });
                        });
                    }
                });
            });
        });

        styled_vsep(ui);

        // ── Zona 4: Bypass (extrema direita) ─────────────────
        ui.allocate_ui(egui::vec2(60.0, 210.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(18.0);
                let ind_bypass = shared.param_indication
                    [crate::clap::extensions::params::PARAM_BYPASS as usize]
                    .load(Ordering::Relaxed);
                let ind_bypass_color = resolve_color(
                    shared.param_indication_color
                        [crate::clap::extensions::params::PARAM_BYPASS as usize]
                        .load(Ordering::Relaxed),
                    egui::Color32::from_rgb(94, 129, 172),
                );
                let bypass_id = ui.make_persistent_id("bypass_switch");
                handle_bypass(
                    ui,
                    bypass_id,
                    &shared.param_bypass,
                    &shared.gui_bypass_changed,
                    &shared.gesture_begin_bypass,
                    &shared.gesture_end_bypass,
                    accent_color,
                    host,
                    ind_bypass,
                    ind_bypass_color,
                );
            });
        });
    });

    // ── Zona 5: Status Bar / Footer ──────────────────────────────────────────
    ui.with_layout(
        egui::Layout::top_down(egui::Align::LEFT),
        |ui| {
            ui.painter().line(
                vec![ui.max_rect().left_top(), ui.max_rect().right_top()],
                egui::Stroke::new(0.5, COL_BORDER),
            );

            let model_meta_opt = if let Ok(guard) = shared.ui_model_metadata.lock() {
                guard.clone()
            } else {
                None
            };

            let a2_placeholder = shared.rt_status.check_flag(spsc::RT_STATUS_A2_PLACEHOLDER);

            let available_h = ui.available_height();
            let has_meta = model_meta_opt.is_some();
            let line_height = 11.5;
            let spacing = 3.0;
            let content_height = match (has_meta, a2_placeholder) {
                (true, true) => 3.0 * line_height + 2.0 * spacing,
                (true, false) | (false, true) => 2.0 * line_height + spacing,
                (false, false) => line_height,
            };

            let extra_space = (available_h - content_height).max(0.0);
            let top_space = extra_space * 0.55;
            ui.add_space(top_space);

            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = spacing;
                ui.spacing_mut().interact_size.y = 10.0;

                let now = Instant::now();
                if now.duration_since(state.last_telem_update) >= Duration::from_secs(1)
                    || (state.telem_cycles == 0 && state.telem_last_n == 0)
                {
                    state.telem_cycles = shared.rt_status.dsp_cycle_time.load(Ordering::Relaxed);
                    state.telem_last_n = shared.rt_status.last_n_samples.load(Ordering::Relaxed);
                    state.telem_prio = shared.rt_status.rt_priority.load(Ordering::Relaxed);
                    state.telem_overloads = shared.rt_status.dsp_overloads.load(Ordering::Relaxed);

                    let sr = shared.sample_rate.load(Ordering::Relaxed);
                    if sr > 0 && state.telem_last_n > 0 {
                        let budget_ns = (state.telem_last_n as u64 * 1_000_000_000) / sr as u64;
                        state.telem_budget_ns = budget_ns;
                        state.telem_cycle_ns = state.telem_cycles;
                        state.telem_load_pct =
                            ((state.telem_cycles as f64 / budget_ns as f64) * 100.0).min(999.0);
                    } else {
                        state.telem_budget_ns = 0;
                        state.telem_cycle_ns = 0;
                        state.telem_load_pct = 0.0;
                    }

                    state.last_telem_update = now;
                }

                ui.horizontal(|ui| {
                    let sr = shared.sample_rate.load(Ordering::Relaxed);
                    let lat = shared.current_latency.load(Ordering::Relaxed);

                    let sr_text = if sr == 0 {
                        "SR: Off".to_string()
                    } else {
                        format!("SR: {:.1} kHz", sr as f64 / 1000.0)
                    };

                    let lat_ms = if sr > 0 {
                        (lat as f64 * 1000.0) / sr as f64
                    } else {
                        0.0
                    };
                    let lat_text = if lat == 0 {
                        "Lat: 0 ms".to_string()
                    } else {
                        format!("Lat: {:.1} ms ({} spl)", lat_ms, lat)
                    };

                    let cycles = state.telem_cycles;
                    let last_n = state.telem_last_n;
                    let prio = state.telem_prio;
                    let overloads = state.telem_overloads;
                    let status_bits = shared.rt_status.status_bits.load(Ordering::Relaxed);

                    let telem_items = [
                        (
                            sr_text,
                            "Host DAW Sample Rate.\nThe neural model runs internally at 48 kHz. High quality resampling is automatically active if rates differ.",
                            None,
                        ),
                        (
                            lat_text,
                            "Latency introduced by the internal resampler to align host sample rate with neural model's native 48 kHz.\nBypassed (0 ms) when host sample rate is 48 kHz.",
                            None,
                        ),
                        (
                            format!("DSP: {:.1}%", state.telem_load_pct),
                            "DSP Thread Load: portion of real-time audio time budget used.",
                            Some(if state.telem_load_pct < 50.0 {
                                COL_VU_GREEN
                            } else if state.telem_load_pct <= 80.0 {
                                COL_AMBER
                            } else {
                                COL_VU_RED
                            }),
                        ),
                        (
                            format!("Cycles: {}", cycles),
                            "Average CPU clock cycles consumed per real-time processing block",
                            None,
                        ),
                        (
                            format!("Last N: {}", last_n),
                            "Number of audio samples processed in the last block",
                            None,
                        ),
                        (
                            format!("RT Prio: {}", prio),
                            "Real-time thread scheduling priority. Values > 0 indicate active RT thread scheduling",
                            None,
                        ),
                        (
                            format!("Overloads: {}", overloads),
                            "Number of real-time buffer deadline overruns (XRUNs) detected since start",
                            None,
                        ),
                        (
                            format!("Flags: {:#X}", status_bits),
                            "Real-time engine diagnostic status flags bitmask",
                            None,
                        ),
                    ];

                    let available_width = ui.available_width();
                    let baseline_s = 10.0;
                    let baseline_font = egui::FontId::proportional(baseline_s);
                    let mut sum_widths = 0.0;
                    for (text, _, _) in &telem_items {
                        let galley = ui.painter().layout_no_wrap(text.clone(), baseline_font.clone(), egui::Color32::WHITE);
                        sum_widths += galley.rect.width();
                    }
                    let separator_text = " | ".to_string();
                    let sep_galley = ui.painter().layout_no_wrap(separator_text, baseline_font.clone(), egui::Color32::WHITE);
                    sum_widths += sep_galley.rect.width() * (telem_items.len() - 1) as f32;

                    let num_gaps = (telem_items.len() * 2 - 2) as f32;
                    let total_gap_width = num_gaps * ui.spacing().item_spacing.x;

                    let target_width = available_width - total_gap_width - 8.0;
                    let calculated_font_size = if sum_widths > 0.0 && target_width > 0.0 {
                        let scale = target_width / sum_widths;
                        (baseline_s * scale).clamp(6.0, 14.0)
                    } else {
                        8.5
                    };
                    let font_size = egui::FontId::proportional(calculated_font_size);

                    let mut first = true;
                    for (text, tooltip, custom_color) in telem_items {
                        if !first {
                            ui.label(
                                egui::RichText::new("|")
                                    .font(font_size.clone())
                                    .color(COL_BORDER),
                            );
                        }
                        first = false;

                        let color = custom_color.unwrap_or(COL_MUTED);
                        let label = ui.label(
                            egui::RichText::new(text)
                                .font(font_size.clone())
                                .color(color),
                        );
                        label.on_hover_text(tooltip);
                    }
                });

                if a2_placeholder {
                    ui.horizontal(|ui| {
                        let warn_font = egui::FontId::proportional(9.0);
                        let galley = ui.painter().layout_no_wrap(
                            "⚠ Modelo A2 não suportado — bypass ativo".to_string(),
                            warn_font.clone(),
                            COL_AMBER,
                        );
                        let available_width = ui.available_width();
                        let x_center = (available_width - galley.rect.width()).max(0.0) / 2.0;
                        ui.add_space(x_center);
                        ui.label(
                            egui::RichText::new("⚠ Modelo A2 não suportado — bypass ativo")
                                .font(warn_font)
                                .color(COL_AMBER),
                        );
                    });
                }

                if let Some(meta) = model_meta_opt {
                    let mut metadata_items = Vec::new();

                    metadata_items.push((
                        format!("Model: {} ({})", meta.architecture, meta.topology),
                        "Neural network architecture type and topology geometry".to_string(),
                        Some(accent_color),
                    ));

                    if let Some(ref author) = meta.modeled_by {
                        let trimmed = author.trim();
                        if !trimmed.is_empty() {
                            metadata_items.push((
                                format!("Author: {}", trimmed),
                                "Creator / trainer of the model".to_string(),
                                None,
                            ));
                        }
                    }

                    let mut gear_parts = Vec::new();
                    if let Some(ref make) = meta.gear_make {
                        let trimmed = make.trim();
                        if !trimmed.is_empty() {
                            gear_parts.push(trimmed.to_string());
                        }
                    }
                    if let Some(ref model) = meta.gear_model {
                        let trimmed = model.trim();
                        if !trimmed.is_empty() {
                            gear_parts.push(trimmed.to_string());
                        }
                    }
                    let gear_name = if gear_parts.is_empty() {
                        None
                    } else {
                        let mut s = gear_parts.join(" ");
                        if let Some(ref gtype) = meta.gear_type {
                            let trimmed = gtype.trim();
                            if !trimmed.is_empty() {
                                s.push_str(&format!(" ({})", trimmed));
                            }
                        }
                        Some(s)
                    };
                    if let Some(gear) = gear_name {
                        metadata_items.push((
                            format!("Gear: {}", gear),
                            "Original hardware equipment modeled by this neural network".to_string(),
                            None,
                        ));
                    }

                    if let Some(ref tone) = meta.tone_type {
                        let trimmed = tone.trim();
                        if !trimmed.is_empty() {
                            metadata_items.push((
                                format!("Tone: {}", trimmed),
                                "Classification style of the tone".to_string(),
                                None,
                            ));
                        }
                    }

                    if let Some(ref date) = meta.date {
                        let trimmed = date.trim();
                        if !trimmed.is_empty() {
                            metadata_items.push((
                                format!("Date: {}", trimmed),
                                "Model creation / export date".to_string(),
                                None,
                            ));
                        }
                    }

                    let available_width = ui.available_width();
                    let baseline_s = 10.0;
                    let baseline_font = egui::FontId::proportional(baseline_s);
                    let mut sum_widths = 0.0;
                    for (text, _, _) in &metadata_items {
                        let galley = ui.painter().layout_no_wrap(text.clone(), baseline_font.clone(), egui::Color32::WHITE);
                        sum_widths += galley.rect.width();
                    }
                    let separator_text = " | ".to_string();
                    let sep_galley = ui.painter().layout_no_wrap(separator_text, baseline_font.clone(), egui::Color32::WHITE);
                    sum_widths += sep_galley.rect.width() * (metadata_items.len() - 1) as f32;

                    let num_gaps = (metadata_items.len() * 2 - 2) as f32;
                    let total_gap_width = num_gaps * ui.spacing().item_spacing.x;

                    let target_width = available_width - total_gap_width - 8.0;
                    let calculated_font_size = if sum_widths > 0.0 && target_width > 0.0 {
                        let scale = target_width / sum_widths;
                        (baseline_s * scale).clamp(9.0, 10.0)
                    } else {
                        9.0
                    };
                    let font_size = egui::FontId::proportional(calculated_font_size);

                    ui.horizontal(|ui| {
                        let mut first = true;
                        for (text, tooltip, custom_color) in metadata_items {
                            if !first {
                                ui.label(
                                    egui::RichText::new("|")
                                        .font(font_size.clone())
                                        .color(COL_BORDER),
                                );
                            }
                            first = false;

                            let color = custom_color.unwrap_or(COL_MUTED);
                            let label = ui.label(
                                egui::RichText::new(text)
                                    .font(font_size.clone())
                                    .color(color),
                            );
                            label.on_hover_text(tooltip);
                        }
                    });
                }
            });
        }
    );

    if state.drag_active {
        egui::Area::new(egui::Id::new("drop_overlay"))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .order(egui::Order::Foreground)
            .interactable(false)
            .show(ui.ctx(), |ui| {
                let screen_rect = ui.ctx().content_rect();
                let painter = ui.painter();
                painter.rect_filled(
                    screen_rect,
                    0.0,
                    egui::Color32::from_rgba_unmultiplied(26, 29, 35, 217),
                );
                ui.vertical_centered(|ui| {
                    ui.add_space(screen_rect.height() / 2.0 - 20.0);
                    ui.label(
                        egui::RichText::new("Drop NAM Model Here ⬇️")
                            .font(egui::FontId::proportional(20.0))
                            .strong()
                            .color(accent_color),
                    );
                });
            });
    }

    let load_btn_id = load_btn_id.unwrap_or_else(|| ui.make_persistent_id("load_model_button"));
    let controls = [
        ui.make_persistent_id("input_gain_knob"),
        ui.make_persistent_id("output_gain_knob"),
        ui.make_persistent_id("gate_thresh_knob"),
        ui.make_persistent_id("bypass_switch"),
        load_btn_id,
    ];

    let mut tab_pressed = false;
    let mut shift_pressed = false;
    ui.input_mut(|i| {
        i.events.retain(|e| {
            if let egui::Event::Key {
                key: egui::Key::Tab,
                pressed: true,
                modifiers,
                ..
            } = e
            {
                tab_pressed = true;
                shift_pressed = modifiers.shift;
                false
            } else {
                true
            }
        });
    });

    if tab_pressed {
        let focused = ui.memory(|mem| mem.focused());
        let next_focus = if let Some(curr) = focused {
            if let Some(idx) = controls.iter().position(|&id| id == curr) {
                if shift_pressed {
                    if idx == 0 {
                        controls[4]
                    } else {
                        controls[idx - 1]
                    }
                } else {
                    if idx == 4 {
                        controls[0]
                    } else {
                        controls[idx + 1]
                    }
                }
            } else if shift_pressed {
                controls[4]
            } else {
                controls[0]
            }
        } else if shift_pressed {
            controls[4]
        } else {
            controls[0]
        };
        ui.memory_mut(|mem| mem.request_focus(next_focus));
    }

    ui.ctx().request_repaint_after(Duration::from_millis(30));
}

#[cfg(test)]
#[path = "test.rs"]
mod test;
