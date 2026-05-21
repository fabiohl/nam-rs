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
        param_input_gain: AtomicU32::new(0.0f32.to_bits()),
        param_output_gain: AtomicU32::new(0.0f32.to_bits()),
        param_gate_thresh: AtomicU32::new((-70.0f32).to_bits()),
        param_bypass: AtomicU32::new(0),
        ui_peak_l: AtomicU32::new(0.0f32.to_bits()),
        ui_peak_r: AtomicU32::new(0.0f32.to_bits()),
        ui_clipped: std::sync::atomic::AtomicBool::new(false),
        ui_model_name: Mutex::new(String::new()),
        ui_pending_model: Mutex::new(None),
        ui_loading: std::sync::atomic::AtomicBool::new(false),
        sample_rate: AtomicU32::new(44100),
        track_accent_color: AtomicU32::new(track_color),
        gui_input_gain_changed: std::sync::atomic::AtomicBool::new(false),
        gesture_begin_input_gain: std::sync::atomic::AtomicBool::new(false),
        gesture_end_input_gain: std::sync::atomic::AtomicBool::new(false),
        gui_output_gain_changed: std::sync::atomic::AtomicBool::new(false),
        gesture_begin_output_gain: std::sync::atomic::AtomicBool::new(false),
        gesture_end_output_gain: std::sync::atomic::AtomicBool::new(false),
        gui_gate_thresh_changed: std::sync::atomic::AtomicBool::new(false),
        gesture_begin_gate_thresh: std::sync::atomic::AtomicBool::new(false),
        gesture_end_gate_thresh: std::sync::atomic::AtomicBool::new(false),
        gui_bypass_changed: std::sync::atomic::AtomicBool::new(false),
        gesture_begin_bypass: std::sync::atomic::AtomicBool::new(false),
        gesture_end_bypass: std::sync::atomic::AtomicBool::new(false),
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
