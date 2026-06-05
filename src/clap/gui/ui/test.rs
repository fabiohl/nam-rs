// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::spsc::{GcOverflowBuffer, RtStatusFlags};
use rtrb::RingBuffer;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU32;

fn make_test_shared(track_color: u32) -> NamClapShared {
    let (param_tx, _) = RingBuffer::new(8);
    let (gc_tx, _) = RingBuffer::new(32);

    NamClapShared {
        param_tx: Mutex::new(Some(param_tx)),
        param_rx: Mutex::new(None),
        gc_tx: Mutex::new(Some(gc_tx)),
        gc_rx: Mutex::new(None),
        gc_overflow: Arc::new(GcOverflowBuffer::new(64)),
        rt_status: Arc::new(RtStatusFlags::new()),
        current_latency: AtomicU32::new(0),
        model_sample_rate: AtomicU32::new(48000),
        param_input_gain: AtomicU32::new(0.0f32.to_bits()),
        param_output_gain: AtomicU32::new(0.0f32.to_bits()),
        param_gate_thresh: AtomicU32::new((-70.0f32).to_bits()),
        param_bypass: AtomicU32::new(0),
        param_adaptive_compute: AtomicU32::new(1),
        ui_peak_l: AtomicU32::new(0.0f32.to_bits()),
        ui_peak_r: AtomicU32::new(0.0f32.to_bits()),
        ui_clipped: std::sync::atomic::AtomicBool::new(false),
        ui_model_name: Mutex::new(String::new()),
        ui_model_metadata: Mutex::new(None),
        ui_pending_model: Mutex::new(None),
        ui_loading: std::sync::atomic::AtomicBool::new(false),
        ui_load_error: std::sync::atomic::AtomicBool::new(false),
        ui_load_error_msg: Mutex::new(String::new()),
        alive_fence: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        sample_rate: AtomicU32::new(44100),
        active_channel_count: AtomicU32::new(1),
        track_accent_color: AtomicU32::new(track_color),
        param_indication: [
            std::sync::atomic::AtomicU8::new(0),
            std::sync::atomic::AtomicU8::new(0),
            std::sync::atomic::AtomicU8::new(0),
            std::sync::atomic::AtomicU8::new(0),
            std::sync::atomic::AtomicU8::new(0),
            std::sync::atomic::AtomicU8::new(0),
        ],
        param_indication_color: [
            std::sync::atomic::AtomicU32::new(0),
            std::sync::atomic::AtomicU32::new(0),
            std::sync::atomic::AtomicU32::new(0),
            std::sync::atomic::AtomicU32::new(0),
            std::sync::atomic::AtomicU32::new(0),
            std::sync::atomic::AtomicU32::new(0),
        ],
        model_load_counter: AtomicU32::new(0),
        gesture_flags: AtomicU32::new(0),
    }
}

#[test]
fn test_track_color_conversion_and_fallback() {
    // 1. Fallback (alpha == 0)
    let shared_fallback = make_test_shared(0);
    let resolved = resolve_accent(&shared_fallback);
    assert_eq!(resolved, COL_ACCENT);

    // 2. White (0xFFFFFFFF)
    let shared_white = make_test_shared(0xFFFFFFFF);
    assert_eq!(
        resolve_accent(&shared_white),
        egui::Color32::from_rgb(255, 255, 255)
    );

    // 3. Black (0xFF000000)
    let shared_black = make_test_shared(0xFF000000);
    assert_eq!(
        resolve_accent(&shared_black),
        egui::Color32::from_rgb(0, 0, 0)
    );

    // 4. Pure Red (0xFFFF0000)
    let shared_red = make_test_shared(0xFFFF0000);
    assert_eq!(
        resolve_accent(&shared_red),
        egui::Color32::from_rgb(255, 0, 0)
    );

    // 5. Pure Green (0xFF00FF00)
    let shared_green = make_test_shared(0xFF00FF00);
    assert_eq!(
        resolve_accent(&shared_green),
        egui::Color32::from_rgb(0, 255, 0)
    );

    // 6. Bitwig Blue #5e81ac (0xFF5E81AC)
    let shared_bitwig = make_test_shared(0xFF5E81AC);
    assert_eq!(
        resolve_accent(&shared_bitwig),
        egui::Color32::from_rgb(0x5E, 0x81, 0xAC)
    );
}

#[test]
fn test_resolve_color() {
    let fallback = egui::Color32::BLUE;

    // 1. Fallback (alpha == 0)
    assert_eq!(resolve_color(0, fallback), fallback);

    // 2. Parsed color (alpha != 0)
    let packed = crate::clap::extensions::track_info::pack_argb(0xFF, 0xFF, 0x00, 0x00); // Red
    assert_eq!(
        resolve_color(packed, fallback),
        egui::Color32::from_rgb(255, 0, 0)
    );
}

#[test]
fn test_ui_load_error_visual_feedback() {
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    let shared = make_test_shared(0);
    shared.ui_load_error.store(true, Ordering::Relaxed);
    *shared.ui_load_error_msg.lock().unwrap() = "Invalid JSON format".to_string();

    let mut state = UiState::default();
    assert!(state.error_expiration.is_none());
    assert!(state.error_msg.is_empty());

    let ctx = egui::Context::default();
    let dummy = 42i32;
    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
    let host: HostSharedHandle = unsafe { std::mem::transmute(&dummy as *const i32) };

    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            draw_ui(ui, &shared, &host, &mut state);
        });
    });

    // 1. After draw_ui, state.error_expiration should be set.
    assert!(state.error_expiration.is_some());
    assert_eq!(state.error_msg, "Invalid JSON format");

    // 2. The flag ui_load_error should have been swapped to false.
    assert!(!shared.ui_load_error.load(Ordering::Relaxed));

    // 3. If we set error_expiration to the past, the next draw_ui should reset/clear it.
    state.error_expiration = Some(Instant::now() - Duration::from_secs(1));
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            draw_ui(ui, &shared, &host, &mut state);
        });
    });

    assert!(state.error_expiration.is_none());
}

#[test]
fn test_knob_tooltip_suffixes() {
    let ctx = egui::Context::default();
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let value = 0.0;
            let range = -10.0..=10.0;
            let size = egui::vec2(50.0, 50.0);
            let color = egui::Color32::RED;
            let indication = 0;

            // Render with " dB" suffix
            let (_response_db, new_val_db) = knob_widget(
                ui,
                ui.make_persistent_id("test_knob_db"),
                value,
                range.clone(),
                size,
                color,
                color,
                indication,
                egui::Color32::from_rgb(94, 129, 172),
                " dB",
            );
            assert_eq!(new_val_db, value);

            // Render with " dB (Threshold)" suffix
            let (_response_threshold, new_val_threshold) = knob_widget(
                ui,
                ui.make_persistent_id("test_knob_threshold"),
                value,
                range,
                size,
                color,
                color,
                indication,
                egui::Color32::from_rgb(94, 129, 172),
                " dB (Threshold)",
            );
            assert_eq!(new_val_threshold, value);
        });
    });
}

fn relative_luminance(color: egui::Color32) -> f32 {
    let r = color.r() as f32 / 255.0;
    let g = color.g() as f32 / 255.0;
    let b = color.b() as f32 / 255.0;

    let r_lin = if r <= 0.03928 {
        r / 12.92
    } else {
        ((r + 0.055) / 1.055).powf(2.4)
    };
    let g_lin = if g <= 0.03928 {
        g / 12.92
    } else {
        ((g + 0.055) / 1.055).powf(2.4)
    };
    let b_lin = if b <= 0.03928 {
        b / 12.92
    } else {
        ((b + 0.055) / 1.055).powf(2.4)
    };

    0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
}

fn contrast_ratio(c1: egui::Color32, c2: egui::Color32) -> f32 {
    let l1 = relative_luminance(c1);
    let l2 = relative_luminance(c2);
    if l1 > l2 {
        (l1 + 0.05) / (l2 + 0.05)
    } else {
        (l2 + 0.05) / (l1 + 0.05)
    }
}

#[test]
fn test_contrast_ratios() {
    let ratio_muted_panel = contrast_ratio(COL_MUTED, COL_PANEL);
    let ratio_vured_bg = contrast_ratio(COL_VU_RED, COL_BG);
    let ratio_muted_bg = contrast_ratio(COL_MUTED, COL_BG);

    // Muted on Panel should be >= 4.5 (typically ~4.88)
    assert!(
        ratio_muted_panel >= 4.5,
        "Muted on Panel: {}",
        ratio_muted_panel
    );
    // VU Red on BG should be >= 4.5 (typically ~4.99)
    assert!(ratio_vured_bg >= 4.5, "VU Red on BG: {}", ratio_vured_bg);
    // Muted on BG (Bypassed status text) should be >= 4.5 (typically ~5.63)
    assert!(ratio_muted_bg >= 4.5, "Muted on BG: {}", ratio_muted_bg);
}

#[test]
fn test_knob_keyboard_navigation() {
    let ctx = egui::Context::default();
    let id = egui::Id::new("test_knob");
    let mut state = 0.0f32;
    let range = -10.0..=10.0;
    let size = egui::vec2(50.0, 50.0);
    let color = egui::Color32::RED;
    let accent_color = egui::Color32::GREEN;

    // Frame 1: Render and request focus
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let (_, val) = knob_widget(
                ui,
                id,
                state,
                range.clone(),
                size,
                color,
                accent_color,
                0,
                egui::Color32::from_rgb(94, 129, 172),
                " dB",
            );
            state = val;
            ui.memory_mut(|mem| mem.request_focus(id));
        });
    });

    // Frame 2: ArrowUp should increment value by 1.0
    let mut input_up = egui::RawInput::default();
    input_up.events.push(egui::Event::Key {
        key: egui::Key::ArrowUp,
        physical_key: None,
        pressed: true,
        modifiers: egui::Modifiers::default(),
        repeat: false,
    });
    let _ = ctx.run_ui(input_up, |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let (_, val) = knob_widget(
                ui,
                id,
                state,
                range.clone(),
                size,
                color,
                accent_color,
                0,
                egui::Color32::from_rgb(94, 129, 172),
                " dB",
            );
            state = val;
        });
    });
    assert_eq!(state, 1.0);

    // Frame 3: Ctrl + ArrowDown should decrement value by 0.1
    let mut input_down_ctrl = egui::RawInput::default();
    input_down_ctrl.modifiers.ctrl = true;
    input_down_ctrl.events.push(egui::Event::Key {
        key: egui::Key::ArrowDown,
        physical_key: None,
        pressed: true,
        modifiers: egui::Modifiers {
            ctrl: true,
            ..Default::default()
        },
        repeat: false,
    });
    let _ = ctx.run_ui(input_down_ctrl, |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let (_, val) = knob_widget(
                ui,
                id,
                state,
                range.clone(),
                size,
                color,
                accent_color,
                0,
                egui::Color32::from_rgb(94, 129, 172),
                " dB",
            );
            state = val;
        });
    });
    assert!((state - 0.9).abs() < 1e-5, "Expected 0.9, got {}", state);
}

#[test]
fn test_bypass_keyboard_trigger() {
    use std::sync::atomic::Ordering;

    let ctx = egui::Context::default();
    let id = egui::Id::new("test_bypass");
    let atomic_val = std::sync::atomic::AtomicU32::new(0); // initial: bypass off
    let gesture_flags = std::sync::atomic::AtomicU32::new(0);
    let dummy = 42i32;
    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
    let host: HostSharedHandle = unsafe { std::mem::transmute(&dummy as *const i32) };

    const BYPASS_INDEX: usize = 3; // PARAM_BYPASS = 3
    const BITS_PER_PARAM: u32 = 3;
    const CHANGED_SHIFT: u32 = 0;
    const BEGIN_SHIFT: u32 = 1;
    const END_SHIFT: u32 = 2;
    let offset = BYPASS_INDEX as u32 * BITS_PER_PARAM;

    // Frame 1: Render and request focus
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            handle_bypass(
                ui,
                id,
                &atomic_val,
                &gesture_flags,
                BYPASS_INDEX,
                egui::Color32::GREEN,
                &host,
                0,
                egui::Color32::from_rgb(94, 129, 172),
            );
            ui.memory_mut(|mem| mem.request_focus(id));
        });
    });

    // Frame 2: Space key event to toggle bypass
    let mut input_space = egui::RawInput::default();
    input_space.events.push(egui::Event::Key {
        key: egui::Key::Space,
        physical_key: None,
        pressed: true,
        modifiers: egui::Modifiers::default(),
        repeat: false,
    });
    let _ = ctx.run_ui(input_space, |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            handle_bypass(
                ui,
                id,
                &atomic_val,
                &gesture_flags,
                BYPASS_INDEX,
                egui::Color32::GREEN,
                &host,
                0,
                egui::Color32::from_rgb(94, 129, 172),
            );
        });
    });
    assert_eq!(atomic_val.load(Ordering::Relaxed), 1); // Should be Bypassed (1)

    let flags = gesture_flags.load(Ordering::Relaxed);
    assert!(flags & (1 << (offset + CHANGED_SHIFT)) != 0);
    assert!(flags & (1 << (offset + BEGIN_SHIFT)) != 0);
    assert!(flags & (1 << (offset + END_SHIFT)) != 0);
}

#[test]
fn test_tab_order_navigation() {
    let ctx = egui::Context::default();
    let shared = make_test_shared(0);
    let mut state = UiState::default();
    let dummy = 42i32;
    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
    let host: HostSharedHandle = unsafe { std::mem::transmute(&dummy as *const i32) };

    // Frame 1: Initial render, no focus
    let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            draw_ui(ui, &shared, &host, &mut state);
        });
    });
    assert!(ctx.memory(|mem| mem.focused()).is_none());

    // Frame 2: Send Tab key -> should focus INPUT knob (controls[0])
    let mut tab_input = egui::RawInput::default();
    tab_input.events.push(egui::Event::Key {
        key: egui::Key::Tab,
        physical_key: None,
        pressed: true,
        modifiers: egui::Modifiers::default(),
        repeat: false,
    });
    let _ = ctx.run_ui(tab_input, |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            draw_ui(ui, &shared, &host, &mut state);
        });
    });
    let focused_1 = ctx.memory(|mem| mem.focused());
    assert!(focused_1.is_some());

    // Frame 3: Send Tab again -> should focus next widget (controls[1])
    let mut tab_input2 = egui::RawInput::default();
    tab_input2.events.push(egui::Event::Key {
        key: egui::Key::Tab,
        physical_key: None,
        pressed: true,
        modifiers: egui::Modifiers::default(),
        repeat: false,
    });
    let _ = ctx.run_ui(tab_input2, |ui| {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            draw_ui(ui, &shared, &host, &mut state);
        });
    });
    let focused_2 = ctx.memory(|mem| mem.focused());
    assert!(focused_2.is_some());
    assert_ne!(focused_1, focused_2);
}
