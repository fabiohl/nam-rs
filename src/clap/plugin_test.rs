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
        model_sample_rate: AtomicU32::new(48000),
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
        ui_load_error: std::sync::atomic::AtomicBool::new(false),
        ui_load_error_msg: Mutex::new(String::new()),
        alive_fence: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        sample_rate: AtomicU32::new(44100),
        active_channel_count: AtomicU32::new(1),
        track_accent_color: AtomicU32::new(0),
        param_indication: [
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
        ],
        model_load_counter: AtomicU32::new(0),
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

#[test]
fn test_file_picker_alive_fence_and_timeout() {
    let (param_tx, _) = RingBuffer::new(8);
    let (gc_tx, _) = RingBuffer::new(32);

    let shared = Arc::new(NamClapShared {
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
        ui_peak_l: AtomicU32::new(0.0f32.to_bits()),
        ui_peak_r: AtomicU32::new(0.0f32.to_bits()),
        ui_clipped: std::sync::atomic::AtomicBool::new(false),
        ui_model_name: Mutex::new(String::new()),
        ui_pending_model: Mutex::new(None),
        ui_loading: std::sync::atomic::AtomicBool::new(false),
        ui_load_error: std::sync::atomic::AtomicBool::new(false),
        ui_load_error_msg: Mutex::new(String::new()),
        alive_fence: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        sample_rate: AtomicU32::new(44100),
        active_channel_count: AtomicU32::new(1),
        track_accent_color: AtomicU32::new(0),
        param_indication: [
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
        ],
        model_load_counter: AtomicU32::new(0),
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
    });

    // 1. Caso de Sucesso: alive_fence é true e recebemos o caminho antes do timeout
    {
        shared.ui_loading.store(true, Ordering::Relaxed);
        let alive_fence = Arc::clone(&shared.alive_fence);
        let shared_addr = &*shared as *const NamClapShared as usize;

        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Some(std::path::PathBuf::from("/tmp/model.nam")))
            .unwrap();

        // Simula o comportamento do Picker Manager Thread
        match rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(path_opt) => {
                if alive_fence.load(Ordering::Relaxed) {
                    let shared_ref = unsafe { &*(shared_addr as *const NamClapShared) };
                    if let Some(path) = path_opt
                        && let Ok(mut pending_guard) = shared_ref.ui_pending_model.lock()
                    {
                        *pending_guard = Some(path);
                    }
                }
            }
            Err(_) => panic!("Deveria ter recebido o caminho"),
        }

        assert_eq!(
            *shared.ui_pending_model.lock().unwrap(),
            Some(std::path::PathBuf::from("/tmp/model.nam"))
        );
    }

    // Limpa o estado
    *shared.ui_pending_model.lock().unwrap() = None;

    // 2. Caso alive_fence seja false: o plugin foi destruído antes do picker retornar
    {
        shared.ui_loading.store(true, Ordering::Relaxed);
        let alive_fence = Arc::clone(&shared.alive_fence);
        let shared_addr = &*shared as *const NamClapShared as usize;

        // Modifica a cerca de vida útil para false (simulando a destruição do plugin/GUI)
        alive_fence.store(false, Ordering::Relaxed);

        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Some(std::path::PathBuf::from("/tmp/model2.nam")))
            .unwrap();

        match rx.recv_timeout(std::time::Duration::from_millis(50)) {
            Ok(path_opt) => {
                if alive_fence.load(Ordering::Relaxed) {
                    // Se entrasse aqui, seria um erro (acesso ilegal ao endereço liberado)
                    let shared_ref = unsafe { &*(shared_addr as *const NamClapShared) };
                    if let Some(path) = path_opt
                        && let Ok(mut pending_guard) = shared_ref.ui_pending_model.lock()
                    {
                        *pending_guard = Some(path);
                    }
                }
            }
            Err(_) => panic!("Deveria ter recebido o caminho"),
        }

        // Deve continuar None porque alive_fence era false e evitou acessar o shared_addr
        assert_eq!(*shared.ui_pending_model.lock().unwrap(), None);
    }

    // 3. Caso de Timeout: rx.recv_timeout falha por timeout, reiniciando ui_loading
    {
        // Restaura alive_fence para true
        shared.alive_fence.store(true, Ordering::Relaxed);
        shared.ui_loading.store(true, Ordering::Relaxed);
        let alive_fence = Arc::clone(&shared.alive_fence);
        let shared_addr = &*shared as *const NamClapShared as usize;

        let (_tx, rx) = std::sync::mpsc::channel::<Option<std::path::PathBuf>>();
        // Não enviamos nada pelo canal para forçar o timeout

        match rx.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(_) => panic!("Não deveria ter recebido nada"),
            Err(_) => {
                // Timeout ocorreu
                if alive_fence.load(Ordering::Relaxed) {
                    let shared_ref = unsafe { &*(shared_addr as *const NamClapShared) };
                    shared_ref.ui_loading.store(false, Ordering::Relaxed);
                }
            }
        }

        // ui_loading deve ser redefinido para false
        assert!(!shared.ui_loading.load(Ordering::Relaxed));
    }
}
