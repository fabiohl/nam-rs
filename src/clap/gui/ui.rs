// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::clap::plugin::NamClapShared;
use clack_plugin::host::HostSharedHandle;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

// ── Paleta de cores aprovada (T4.0.2) ─────────────────────────────────────────
const COL_BG: egui::Color32 = egui::Color32::from_rgb(26, 29, 35); // #1A1D23
const COL_PANEL: egui::Color32 = egui::Color32::from_rgb(35, 40, 48); // #232830
const COL_BORDER: egui::Color32 = egui::Color32::from_rgb(46, 52, 64); // #2E3440
const COL_TEXT: egui::Color32 = egui::Color32::from_rgb(229, 233, 240); // #E5E9F0
const COL_MUTED: egui::Color32 = egui::Color32::from_rgb(139, 149, 165); // #8B95A5
const COL_ACCENT: egui::Color32 = egui::Color32::from_rgb(0, 212, 170); // #00D4AA
const COL_AMBER: egui::Color32 = egui::Color32::from_rgb(245, 166, 35); // #F5A623
const COL_VU_GREEN: egui::Color32 = egui::Color32::from_rgb(67, 233, 123); // #43E97B
const COL_VU_YELLOW: egui::Color32 = egui::Color32::from_rgb(245, 206, 98); // #F5CE62
const COL_VU_RED: egui::Color32 = egui::Color32::from_rgb(247, 78, 78); // #F74E4E
const COL_BYPASS_OFF: egui::Color32 = egui::Color32::from_rgb(74, 79, 90); // #4A4F5A

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
            last_telem_update: now - Duration::from_secs(2), // Garante atualização imediata na primeira exibição
            telem_cycles: 0,
            telem_last_n: 0,
            telem_prio: 0,
            telem_overloads: 0,
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
    #[cfg(all(
        not(target_feature = "avx512f"),
        not(target_feature = "avx2"),
        target_feature = "sse4.1"
    ))]
    {
        "SSE4.1"
    }
    #[cfg(all(
        not(target_feature = "avx512f"),
        not(target_feature = "avx2"),
        not(target_feature = "sse4.1")
    ))]
    {
        "GENERIC"
    }
}

/// Renderiza um knob customizado com:
/// - 48 segmentos (arco suave)
/// - Glow no arco durante drag (E1)
/// - Tooltip com valor exato no hover (E3)
/// - Ctrl+Drag para fine-tune ÷10 (E5)
///
/// Retorna `(response, new_value)`.
fn knob_widget(
    ui: &mut egui::Ui,
    _id: egui::Id,
    value: f32,
    range: std::ops::RangeInclusive<f32>,
    size: egui::Vec2,
    color: egui::Color32,
) -> (egui::Response, f32) {
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());

    let mut current_value = value;
    let range_len = range.end() - range.start();

    // Drag handling com Ctrl+fine-tune (E5)
    if response.dragged() {
        let delta_y = response.drag_delta().y;
        let ctrl_held = ui.input(|i| i.modifiers.ctrl);
        let sensitivity = if ctrl_held {
            range_len / 2000.0
        } else {
            range_len / 200.0
        };
        current_value = (current_value - delta_y * sensitivity).clamp(*range.start(), *range.end());
    }

    // Scroll handling com Ctrl+fine-tune (E5)
    if response.hovered() {
        let scroll_y = ui.input(|i| i.raw_scroll_delta.y);
        if scroll_y != 0.0 {
            let ctrl_held = ui.input(|i| i.modifiers.ctrl);
            let sensitivity = if ctrl_held {
                range_len / 10000.0
            } else {
                range_len / 1000.0
            };
            current_value =
                (current_value + scroll_y * sensitivity).clamp(*range.start(), *range.end());
        }
    }

    // Tooltip com valor exato (E3)
    let response = response.on_hover_text(format!("{:.2} dB", current_value));

    let painter = ui.painter();
    let center = rect.center();
    let radius = rect.width() / 2.0 - 2.0;
    let frac = (current_value - range.start()) / range_len;
    let angle_start = -135.0f32.to_radians();
    let angle_end = 135.0f32.to_radians();
    let angle = angle_start + frac * (angle_end - angle_start);

    // Track externo escuro
    painter.circle_stroke(center, radius, egui::Stroke::new(3.0, COL_BG));

    // Arco de valor ativo (48 segmentos para suavidade — m2)
    let num_segments = 48;
    let arc_radius = radius;
    let mut points = Vec::with_capacity(num_segments + 1);
    for i in 0..=num_segments {
        let f = i as f32 / num_segments as f32;
        if f <= frac {
            let a = angle_start + f * (angle_end - angle_start) - std::f32::consts::FRAC_PI_2;
            points.push(center + egui::vec2(a.cos() * arc_radius, a.sin() * arc_radius));
        }
    }
    if points.len() > 1 {
        painter.add(egui::Shape::line(
            points.clone(),
            egui::Stroke::new(3.5, color),
        ));
        // Glow durante drag (E1): halo externo semi-transparente
        if response.dragged() {
            let glow_color =
                egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 60);
            painter.add(egui::Shape::line(
                points,
                egui::Stroke::new(7.0, glow_color),
            ));
        }
    }

    // Corpo do knob
    let body_radius = radius - 3.0;
    painter.circle_filled(center, body_radius, egui::Color32::from_rgb(46, 52, 64));
    painter.circle_stroke(center, body_radius, egui::Stroke::new(1.0, COL_BORDER));

    // Ponteiro de posição
    let pointer_angle = angle - std::f32::consts::FRAC_PI_2;
    let pointer_len_start = body_radius * 0.4;
    let pointer_len_end = body_radius * 0.92;
    let p_start = center
        + egui::vec2(
            pointer_angle.cos() * pointer_len_start,
            pointer_angle.sin() * pointer_len_start,
        );
    let p_end = center
        + egui::vec2(
            pointer_angle.cos() * pointer_len_end,
            pointer_angle.sin() * pointer_len_end,
        );
    painter.line(
        vec![p_start, p_end],
        egui::Stroke::new(rect.width() * 0.05, COL_TEXT),
    );

    (response, current_value)
}

/// Gerencia um knob com label, valor em dB, gestos de automação e double-click para reset.
#[allow(clippy::too_many_arguments)]
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
    knob_size: egui::Vec2,
) {
    ui.vertical(|ui| {
        let current_val = f32::from_bits(atomic_val.load(Ordering::Relaxed));

        let (response, new_val) = ui
            .vertical(|ui| {
                ui.horizontal(|ui| {
                    let available_width = ui.available_width();
                    let space = (available_width - knob_size.x) / 2.0;
                    if space > 0.0 {
                        ui.add_space(space);
                    }
                    knob_widget(ui, id, current_val, range, knob_size, color)
                })
                .inner
            })
            .inner;

        if response.drag_started() {
            begin_flag.store(true, Ordering::Relaxed);
            if let Some(params_ext) = host.get_extension::<clack_extensions::params::HostParams>() {
                params_ext.request_flush(host);
            }
        }

        let reset_clicked = response.double_clicked();
        let final_val = if reset_clicked {
            default_value
        } else {
            new_val
        };

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

        ui.add_space(5.0);
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new(label)
                    .font(egui::FontId::proportional(11.0))
                    .strong()
                    .color(COL_MUTED),
            );
            ui.label(
                egui::RichText::new(format!("{:.1} dB", final_val))
                    .font(egui::FontId::monospace(10.0))
                    .color(COL_TEXT),
            );
        });
    });
}

/// Desenha um medidor VU vertical com gradiente tricolor, peak hold com fade analógico (E2),
/// LED de clipping persistente acima (E4) e labels L/R acima e abaixo (m4).
fn draw_vertical_meter(
    ui: &mut egui::Ui,
    peak_val: f32,
    hold_val: &mut f32,
    hold_time: &mut Instant,
    clipped: &mut bool,
    label: &str,
) {
    // Parâmetros visuais — m3: largura aumentada para 16px
    let meter_h = 130.0f32;
    let meter_w = 16.0f32;

    ui.vertical(|ui| {
        // Label ACIMA do medidor (m4)
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new(label)
                    .font(egui::FontId::monospace(9.0))
                    .color(COL_MUTED),
            );
        });
        ui.add_space(2.0);

        // LED de clipping (E4) — pisca em vermelho; clique para limpar
        let desired_led = egui::vec2(meter_w, 6.0);
        let led_rect = ui
            .horizontal(|ui| {
                let avail = ui.available_width();
                let space = (avail - meter_w) / 2.0;
                if space > 0.0 {
                    ui.add_space(space);
                }
                let (r, _) = ui.allocate_exact_size(desired_led, egui::Sense::click());
                r
            })
            .inner;
        {
            let painter = ui.painter();
            let clip_color = if *clipped {
                COL_VU_RED
            } else {
                egui::Color32::from_rgb(50, 30, 30)
            };
            painter.rect_filled(led_rect, 1.0, clip_color);
        }
        // Limpa o LED de clipping ao clicar no medidor (será verificado via response abaixo)
        let led_response = ui.interact(
            led_rect,
            ui.id().with((label, "clip_led")),
            egui::Sense::click(),
        );
        if led_response.clicked() {
            *clipped = false;
        }

        ui.add_space(2.0);

        // Barra do medidor
        let desired_size = egui::vec2(meter_w, meter_h);
        let meter_rect = ui
            .horizontal(|ui| {
                let avail = ui.available_width();
                let space = (avail - meter_w) / 2.0;
                if space > 0.0 {
                    ui.add_space(space);
                }
                let (r, _) = ui.allocate_exact_size(desired_size, egui::Sense::click());
                r
            })
            .inner;

        // Clique no meter também limpa o clipping
        let meter_response = ui.interact(
            meter_rect,
            ui.id().with((label, "meter_bar")),
            egui::Sense::click(),
        );
        if meter_response.clicked() {
            *clipped = false;
        }

        // Actualiza peak hold e clipping
        let now = Instant::now();
        if peak_val >= *hold_val {
            *hold_val = peak_val;
            *hold_time = now;
        } else if now.duration_since(*hold_time) > Duration::from_secs(2) {
            // E2: decaimento suave (fade analógico ~0.95x por frame @ 30ms ≈ -1.5dB/s)
            *hold_val = (*hold_val * 0.95_f32).max(peak_val).max(0.0);
        }

        // Detecta clipping (>= 0 dBFS)
        if peak_val >= 1.0 {
            *clipped = true;
        }

        let painter = ui.painter();
        painter.rect_filled(meter_rect, 1.5, COL_BG);

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

        // Verde
        draw_segment(0.0, peak_frac.min(green_frac), COL_VU_GREEN);
        // Amarelo
        if peak_frac > green_frac {
            draw_segment(green_frac, peak_frac.min(yellow_frac), COL_VU_YELLOW);
        }
        // Vermelho
        if peak_frac > yellow_frac {
            draw_segment(yellow_frac, peak_frac, COL_VU_RED);
        }

        // Linha de Peak Hold
        if hold_frac > 0.0 {
            let y_hold = meter_rect.bottom() - hold_frac * meter_h;
            let hold_color = if hold_db >= -3.0 {
                COL_VU_RED
            } else if hold_db >= -12.0 {
                COL_VU_YELLOW
            } else {
                COL_VU_GREEN
            };
            painter.line(
                vec![
                    egui::pos2(meter_rect.left() - 2.0, y_hold),
                    egui::pos2(meter_rect.right() + 2.0, y_hold),
                ],
                egui::Stroke::new(1.5, hold_color),
            );
        }

        // Label ABAIXO do medidor (m4)
        ui.add_space(4.0);
        ui.vertical_centered(|ui| {
            ui.label(
                egui::RichText::new(label)
                    .font(egui::FontId::monospace(9.0))
                    .color(COL_MUTED),
            );
        });
    });
}

/// Desenha o toggle de bypass com LED e labels "BYPASS" / "ACTIVE".
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
    let button_size = egui::vec2(32.0, 56.0);

    let response = ui
        .horizontal(|ui| {
            let available_width = ui.available_width();
            let space = (available_width - button_size.x) / 2.0;
            if space > 0.0 {
                ui.add_space(space);
            }
            let (_, resp) = ui.allocate_exact_size(button_size, egui::Sense::click());
            resp
        })
        .inner;

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
    let rect = response.rect;

    // Corpo do rocker switch
    painter.rect_filled(rect, 5.0, COL_PANEL);
    painter.rect_stroke(
        rect,
        5.0,
        egui::Stroke::new(1.5, COL_BORDER),
        egui::StrokeKind::Inside,
    );

    let led_color = if current_bypass {
        COL_BYPASS_OFF
    } else {
        COL_ACCENT
    };

    // LED retangular vertical no centro
    let led_rect = egui::Rect::from_center_size(rect.center(), egui::vec2(14.0, 8.0));
    painter.rect_filled(led_rect, 2.0, led_color);

    // Halo de glow quando ativo
    if !current_bypass {
        let glow = egui::Color32::from_rgba_unmultiplied(0, 212, 170, 40);
        painter.rect_filled(led_rect.expand(3.0), 4.0, glow);
    }

    ui.add_space(8.0);
    ui.vertical_centered(|ui| {
        ui.label(
            egui::RichText::new("BYPASS")
                .font(egui::FontId::proportional(11.0))
                .strong()
                .color(COL_MUTED),
        );
        let status_text = if current_bypass { "BYPASSED" } else { "ACTIVE" };
        ui.label(
            egui::RichText::new(status_text)
                .font(egui::FontId::monospace(9.0))
                .color(led_color),
        );
    });
}

/// Renderiza um separador vertical estilizado (m5) — linha fina #2E3440 ao invés do Separator padrão do egui.
fn styled_vsep(ui: &mut egui::Ui) {
    let space = 6.0;
    ui.add_space(space);
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(1.0, ui.available_height()), egui::Sense::hover());
    ui.painter().line(
        vec![rect.center_top(), rect.center_bottom()],
        egui::Stroke::new(0.5, COL_BORDER),
    );
    ui.add_space(space);
}

/// Desenha os componentes e o layout de 5 zonas da interface gráfica do NAM-rs.
pub fn draw_ui(
    ui: &mut egui::Ui,
    shared: &NamClapShared,
    host: &HostSharedHandle,
    state: &mut UiState,
) {
    ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);

    // Layout principal: horizontal com Zonas 1–4
    ui.horizontal(|ui| {
        // ── Zona 1: Identidade (esquerda) ─────────────────────
        ui.allocate_ui(egui::vec2(135.0, 210.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(8.0);

                // Logo "NAM-rs⚡" (grande, accent turquesa)
                ui.label(
                    egui::RichText::new("NAM-rs⚡")
                        .font(egui::FontId::proportional(24.0))
                        .strong()
                        .color(COL_ACCENT),
                );

                // M3: Subtítulo "Neural Amp Modeler"
                ui.label(
                    egui::RichText::new("Neural Amp Modeler")
                        .font(egui::FontId::proportional(9.5))
                        .color(COL_MUTED),
                );

                // Versão e badge SIMD
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                            .font(egui::FontId::proportional(9.0))
                            .color(COL_MUTED),
                    );
                    ui.add_space(4.0);
                    let simd = get_simd_badge();
                    let badge_color = if simd.contains("AVX") {
                        COL_ACCENT
                    } else {
                        COL_MUTED
                    };
                    ui.label(
                        egui::RichText::new(simd)
                            .font(egui::FontId::monospace(8.0))
                            .strong()
                            .color(badge_color),
                    );
                });

                ui.add_space(14.0);

                // M1: Label "MODEL" acima do botão de pasta
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

                // A1+A2 FIX: usar NamClapSharedRef (já Send+Sync) como endereço usize é a única
                // forma viável dado que NamClapShared tem lifetime gerenciado pelo framework CLAP.
                // A flag `ui_loading` garante que a janela espera o resultado antes de fechar —
                // o host não destrói o plugin enquanto houver um modal picker ativo (contrato CLAP).
                // O `transmute` do host é necessário pois `HostSharedHandle` é `!Send`; o host
                // handle permanece válido enquanto o plugin existir (garante o runtime CLAP).
                if load_btn.clicked() && !shared.ui_loading.load(Ordering::Relaxed) {
                    shared.ui_loading.store(true, Ordering::Relaxed);
                    let shared_addr = shared as *const NamClapShared as usize;
                    let host_static: clack_plugin::host::HostSharedHandle<'static> =
                        unsafe { std::mem::transmute(*host) };
                    std::thread::spawn(move || {
                        let path_opt = rfd::FileDialog::new()
                            .add_filter("NAM Model", &["nam", "namb"])
                            .pick_file();
                        // SAFETY: shared_addr aponta para NamClapShared que vive enquanto o
                        // plugin estiver instanciado. O flag ui_loading evita que o host feche
                        // a janela enquanto o picker está aberto.
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

                // M2: Nome do modelo em frame estilizado com borda (visual de "input field")
                let model_name = if shared.ui_loading.load(Ordering::Relaxed) {
                    ui.ctx().request_repaint_after(Duration::from_millis(100));
                    let elapsed = ui.input(|i| i.time);
                    // ASCII puro — sem emoji para evitar quadradinhos de fallback de glifo
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

                // Caixa estilizada do nome do modelo (M2).
                // Label::truncate() delega ao egui o corte visual automático —
                // sem precisar adivinhar o número de chars, funciona para qualquer fonte/escala.
                egui::Frame::new()
                    .fill(COL_BG)
                    .stroke(egui::Stroke::new(1.0, COL_BORDER))
                    .corner_radius(egui::CornerRadius::same(3))
                    .inner_margin(egui::Margin::symmetric(6, 4))
                    .show(ui, |ui| {
                        ui.set_min_width(120.0);
                        ui.set_max_width(120.0);
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(&model_name)
                                    .font(egui::FontId::proportional(9.5))
                                    .color(COL_MUTED)
                                    .italics(),
                            )
                            .truncate(),
                        );
                    });
            });
        });

        // Separador estilizado (m5)
        styled_vsep(ui);

        // ── Zona 2: Controles (centro) ─────────────────────
        ui.allocate_ui(egui::vec2(240.0, 210.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    // INPUT — knob grande, C2: 70×70
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
                            COL_ACCENT,
                            host,
                            egui::vec2(70.0, 70.0),
                        );
                    });
                    ui.add_space(2.0);
                    // OUTPUT — knob grande, C2: 70×70
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
                            COL_ACCENT,
                            host,
                            egui::vec2(70.0, 70.0),
                        );
                    });
                    ui.add_space(2.0);
                    // GATE — knob menor, âmbar
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
                            host,
                            egui::vec2(42.0, 42.0),
                        );
                    });
                });
            });
        });

        // Separador estilizado (m5)
        styled_vsep(ui);

        // ── Zona 3: Medidores VU (direita) ──────────────────
        ui.allocate_ui(egui::vec2(80.0, 210.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let peak_l =
                        f32::from_bits(shared.ui_peak_l.swap(0.0f32.to_bits(), Ordering::Relaxed));
                    let peak_r =
                        f32::from_bits(shared.ui_peak_r.swap(0.0f32.to_bits(), Ordering::Relaxed));

                    ui.allocate_ui(egui::vec2(36.0, 190.0), |ui| {
                        draw_vertical_meter(
                            ui,
                            peak_l,
                            &mut state.peak_l_hold,
                            &mut state.peak_l_hold_time,
                            &mut state.clip_l,
                            "L",
                        );
                    });
                    ui.add_space(4.0);
                    ui.allocate_ui(egui::vec2(36.0, 190.0), |ui| {
                        draw_vertical_meter(
                            ui,
                            peak_r,
                            &mut state.peak_r_hold,
                            &mut state.peak_r_hold_time,
                            &mut state.clip_r,
                            "R",
                        );
                    });
                });
            });
        });

        // Separador estilizado (m5)
        styled_vsep(ui);

        // ── Zona 4: Bypass (extrema direita) ─────────────────
        ui.allocate_ui(egui::vec2(60.0, 210.0), |ui| {
            ui.vertical(|ui| {
                ui.add_space(18.0);
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

    // ── Zona 5: Status Bar / Footer ──────────────────────────────────────────
    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
        ui.add_space(4.0);

        // Painel de telemetria expandido (colapsado por default)
        if state.show_telemetry {
            let now = Instant::now();
            if now.duration_since(state.last_telem_update) >= Duration::from_secs(1)
                || (state.telem_cycles == 0 && state.telem_last_n == 0)
            {
                state.telem_cycles = shared.rt_status.dsp_cycle_time.load(Ordering::Relaxed);
                state.telem_last_n = shared.rt_status.last_n_samples.load(Ordering::Relaxed);
                state.telem_prio = shared.rt_status.rt_priority.load(Ordering::Relaxed);
                state.telem_overloads = shared.rt_status.dsp_overloads.load(Ordering::Relaxed);
                state.last_telem_update = now;
            }

            ui.horizontal(|ui| {
                let cycles = state.telem_cycles;
                let last_n = state.telem_last_n;
                let prio = state.telem_prio;
                let overloads = state.telem_overloads;
                let status_bits = shared.rt_status.status_bits.load(Ordering::Relaxed);

                for text in [
                    format!("Cycles: {}", cycles),
                    format!("Last N: {}", last_n),
                    format!("RT Prio: {}", prio),
                    format!("Overloads: {}", overloads),
                    format!("Flags: {:#X}", status_bits),
                ] {
                    ui.label(
                        egui::RichText::new(text)
                            .font(egui::FontId::monospace(10.0))
                            .color(COL_MUTED),
                    );
                }
            });
            ui.painter().line(
                vec![
                    ui.min_rect().left_top() + egui::vec2(0.0, -2.0),
                    ui.min_rect().right_top() + egui::vec2(0.0, -2.0),
                ],
                egui::Stroke::new(0.5, COL_BORDER),
            );
        }

        // Linha de status bar
        ui.horizontal(|ui| {
            let sr = shared.sample_rate.load(Ordering::Relaxed);
            let lat = shared.current_latency.load(Ordering::Relaxed);

            let sr_text = if sr == 0 {
                "Rate: Off".to_string()
            } else {
                format!("{}kHz", sr / 1000)
            };

            // M5: Nome do modelo à esquerda na status bar
            let model_in_bar = {
                let name_guard = shared
                    .ui_model_name
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if name_guard.is_empty() {
                    "-".to_string()
                } else {
                    let s = name_guard.as_str();
                    if s.len() > 35 {
                        format!("{}...", &s[..32])
                    } else {
                        s.to_string()
                    }
                }
            };

            ui.label(
                egui::RichText::new(model_in_bar)
                    .font(egui::FontId::proportional(10.0))
                    .color(COL_MUTED),
            );

            // Separador de pipe
            ui.label(
                egui::RichText::new("|")
                    .font(egui::FontId::proportional(10.0))
                    .color(COL_BORDER),
            );

            ui.label(
                egui::RichText::new(sr_text)
                    .font(egui::FontId::proportional(10.0))
                    .color(COL_MUTED),
            );

            ui.label(
                egui::RichText::new("|")
                    .font(egui::FontId::proportional(10.0))
                    .color(COL_BORDER),
            );

            ui.label(
                egui::RichText::new(format!("{} samples", lat))
                    .font(egui::FontId::proportional(10.0))
                    .color(COL_MUTED),
            );

            // Botão de telemetria alinhado à direita
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let telem_btn = ui.add(
                    egui::Button::new(
                        egui::RichText::new("RT")
                            .font(egui::FontId::monospace(9.0))
                            .strong()
                            .color(if state.show_telemetry {
                                COL_ACCENT
                            } else {
                                COL_MUTED
                            }),
                    )
                    .fill(COL_PANEL),
                );
                if telem_btn.clicked() {
                    state.show_telemetry = !state.show_telemetry;
                }
            });
        });

        // Linha separadora acima da status bar
        ui.painter().line(
            vec![
                ui.min_rect().left_top() + egui::vec2(0.0, -1.0),
                ui.min_rect().right_top() + egui::vec2(0.0, -1.0),
            ],
            egui::Stroke::new(0.5, COL_BORDER),
        );
    });

    // Repaint a 30ms para manter VU meters e sincronização de parâmetros do host
    ui.ctx().request_repaint_after(Duration::from_millis(30));
}
