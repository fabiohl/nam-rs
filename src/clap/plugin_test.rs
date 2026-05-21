// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::clap::extensions::params::PARAM_INPUT_GAIN;
use clack_host::prelude::EventBuffer;
use clack_plugin::events::event_types::{
    ParamGestureBeginEvent, ParamGestureEndEvent, ParamValueEvent,
};
use clack_plugin::prelude::InputEvents;
use std::sync::atomic::Ordering;

#[test]
fn test_gui_gestures_and_parameter_flow() {
    let (param_tx, _) = RingBuffer::new(8);
    let (gc_tx, _) = RingBuffer::new(32);

    let shared = NamClapShared {
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
    };

    // Simula início de gesto, mudança de valor e término de gesto do ganho de entrada
    shared
        .gesture_begin_input_gain
        .store(true, Ordering::Relaxed);
    shared.gui_input_gain_changed.store(true, Ordering::Relaxed);
    shared
        .param_input_gain
        .store(1.5f32.to_bits(), Ordering::Relaxed);
    shared.gesture_end_input_gain.store(true, Ordering::Relaxed);

    let mut output_events_buffer = EventBuffer::new();
    {
        let mut output_events = OutputEvents::from_buffer(&mut output_events_buffer);
        shared.write_gui_events(&mut output_events);
    }

    let input_view = InputEvents::from_buffer(&output_events_buffer);
    let mut begin_received = false;
    let mut value_received = false;
    let mut end_received = false;

    for event in &input_view {
        if let Some(begin) = event.as_event::<ParamGestureBeginEvent>()
            && begin.param_id().unwrap().get() == PARAM_INPUT_GAIN
        {
            begin_received = true;
        } else if let Some(val_ev) = event.as_event::<ParamValueEvent>() {
            if val_ev.param_id().unwrap().get() == PARAM_INPUT_GAIN {
                assert_eq!(val_ev.value(), 1.5);
                value_received = true;
            }
        } else if let Some(end) = event.as_event::<ParamGestureEndEvent>()
            && end.param_id().unwrap().get() == PARAM_INPUT_GAIN
        {
            end_received = true;
        }
    }

    assert!(begin_received, "Deveria receber ParamGestureBeginEvent");
    assert!(value_received, "Deveria receber ParamValueEvent");
    assert!(end_received, "Deveria receber ParamGestureEndEvent");

    // Verifica que as flags foram limpas
    assert!(!shared.gesture_begin_input_gain.load(Ordering::Relaxed));
    assert!(!shared.gui_input_gain_changed.load(Ordering::Relaxed));
    assert!(!shared.gesture_end_input_gain.load(Ordering::Relaxed));
}
