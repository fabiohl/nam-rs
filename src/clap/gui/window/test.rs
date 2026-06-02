// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use baseview::DropData;
use std::path::PathBuf;

#[test]
fn test_get_valid_model_file_none() {
    let data = DropData::None;
    assert_eq!(get_valid_model_file(&data), None);
}

#[test]
fn test_get_valid_model_file_invalid_extensions() {
    let files = vec![
        PathBuf::from("model.wav"),
        PathBuf::from("config.json"),
        PathBuf::from("readme.txt"),
    ];
    let data = DropData::Files(files);
    assert_eq!(get_valid_model_file(&data), None);
}

#[test]
fn test_get_valid_model_file_valid_nam() {
    let files = vec![PathBuf::from("my_amp_model.nam")];
    let data = DropData::Files(files);
    assert_eq!(
        get_valid_model_file(&data),
        Some(PathBuf::from("my_amp_model.nam"))
    );
}

#[test]
fn test_get_valid_model_file_valid_namb() {
    let files = vec![PathBuf::from("my_amp_model.namb")];
    let data = DropData::Files(files);
    assert_eq!(
        get_valid_model_file(&data),
        Some(PathBuf::from("my_amp_model.namb"))
    );
}

#[test]
fn test_get_valid_model_file_case_insensitive() {
    let files = vec![PathBuf::from("MY_AMP_MODEL.NAM")];
    let data = DropData::Files(files);
    assert_eq!(
        get_valid_model_file(&data),
        Some(PathBuf::from("MY_AMP_MODEL.NAM"))
    );

    let files_namb = vec![PathBuf::from("another_model.Namb")];
    let data_namb = DropData::Files(files_namb);
    assert_eq!(
        get_valid_model_file(&data_namb),
        Some(PathBuf::from("another_model.Namb"))
    );
}

#[test]
fn test_get_valid_model_file_multiple_mixed() {
    let files = vec![
        PathBuf::from("invalid.wav"),
        PathBuf::from("sweet_tone.nam"),
        PathBuf::from("other.namb"),
    ];
    let data = DropData::Files(files);
    // Should skip the first invalid file and return the first valid model file
    assert_eq!(
        get_valid_model_file(&data),
        Some(PathBuf::from("sweet_tone.nam"))
    );
}

#[test]
fn test_gui_drag_drop_fuzz() {
    use crate::clap::plugin::NamClapShared;
    use crate::common::spsc::{GcOverflowBuffer, RtStatusFlags};
    use rtrb::RingBuffer;
    use std::sync::atomic::AtomicU32;
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};

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
        ui_model_metadata: Mutex::new(None),
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
        gesture_flags: AtomicU32::new(0),
    });

    let shared_ref = NamClapSharedRef(&*shared as *const NamClapShared);
    let alive_fence = Arc::clone(&shared.alive_fence);

    // Simulates the safe_shared helper logic for drag-drop:
    let check_and_drop =
        |alive: &Arc<std::sync::atomic::AtomicBool>, s_ref: NamClapSharedRef, path: PathBuf| {
            if alive.load(Ordering::Relaxed) {
                let s = unsafe { &*s_ref.0 };
                if let Ok(mut pending_guard) = s.ui_pending_model.lock() {
                    *pending_guard = Some(path);
                    s.ui_loading.store(true, Ordering::Relaxed);
                }
                true
            } else {
                false
            }
        };

    // 1. Alive case: should set the pending model
    let path = PathBuf::from("model.nam");
    assert!(check_and_drop(&alive_fence, shared_ref, path.clone()));
    assert_eq!(*shared.ui_pending_model.lock().unwrap(), Some(path));

    // Reset
    *shared.ui_pending_model.lock().unwrap() = None;
    shared.ui_loading.store(false, Ordering::Relaxed);

    // 2. Dead case (fence false): should not access or change anything
    alive_fence.store(false, Ordering::Relaxed);
    assert!(!check_and_drop(
        &alive_fence,
        shared_ref,
        PathBuf::from("another.nam")
    ));
    assert_eq!(*shared.ui_pending_model.lock().unwrap(), None);
    assert!(!shared.ui_loading.load(Ordering::Relaxed));
}
