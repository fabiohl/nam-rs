// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::clap::plugin::shared::{
    ColdShared, NamClapShared, RENDER_MODE_REALTIME, RtToUi, UiToRt,
};

pub(crate) fn make_test_shared() -> NamClapShared {
    use crate::common::spsc::{GcOverflowBuffer, RtStatusFlags};
    use rtrb::RingBuffer;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32};
    use std::sync::{Arc, Mutex};

    let (param_tx, param_rx) = RingBuffer::new(8);
    let (gc_tx, gc_rx) = RingBuffer::new(32);

    NamClapShared {
        rt_to_ui: RtToUi {
            ui_peak_l: AtomicU32::new(0.0f32.to_bits()),
            ui_peak_r: AtomicU32::new(0.0f32.to_bits()),
            ui_clipped: AtomicBool::new(false),
            current_latency: AtomicU32::new(0),
            active_channel_count: AtomicU32::new(1),
        },
        ui_to_rt: UiToRt {
            param_input_gain: AtomicU32::new(0.0f32.to_bits()),
            param_output_gain: AtomicU32::new(0.0f32.to_bits()),
            param_gate_thresh: AtomicU32::new((-70.0f32).to_bits()),
            param_bypass: AtomicU32::new(0),
            param_adaptive_compute: AtomicU32::new(1),
            gesture_flags: AtomicU32::new(0),
            gui_param_generation: AtomicU32::new(0),
        },
        cold: ColdShared {
            param_tx: Mutex::new(Some(param_tx)),
            param_rx: Mutex::new(Some(param_rx)),
            gc_tx: Mutex::new(Some(gc_tx)),
            gc_rx: Mutex::new(Some(gc_rx)),
            gc_overflow: Arc::new(GcOverflowBuffer::new(64)),
            rt_status: Arc::new(RtStatusFlags::new()),
            model_sample_rate: AtomicU32::new(48000),
            sample_rate: AtomicU32::new(44100),
            buffer_size: AtomicU32::new(0),
            track_accent_color: AtomicU32::new(0),
            param_indication: [
                AtomicU8::new(0),
                AtomicU8::new(0),
                AtomicU8::new(0),
                AtomicU8::new(0),
                AtomicU8::new(0),
                AtomicU8::new(0),
            ],
            param_indication_color: [
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
                AtomicU32::new(0),
            ],
            model_load_counter: AtomicU32::new(0),
            ui_model_name: Mutex::new(String::new()),
            ui_model_metadata: Mutex::new(None),
            ui_pending_model: Mutex::new(None),
            ui_loading: AtomicBool::new(false),
            ui_load_error: AtomicBool::new(false),
            ui_load_error_msg: Mutex::new(String::new()),
            ui_model_info: Mutex::new(None),
            alive_fence: Arc::new(AtomicBool::new(true)),
            render_mode: AtomicU32::new(RENDER_MODE_REALTIME),
            gui_scale_factor: AtomicU32::new(0),
        },
    }
}

#[cfg(test)]
mod layout_tests {
    use crate::clap::plugin::shared::NamClapShared;

    #[test]
    fn rt_to_ui_and_ui_to_rt_in_separate_cache_lines() {
        let off_rt = std::mem::offset_of!(NamClapShared, rt_to_ui);
        let off_ui = std::mem::offset_of!(NamClapShared, ui_to_rt);
        let distance = off_ui.wrapping_sub(off_rt);
        assert!(
            distance >= 128,
            "RtToUi and UiToRt must not share a 128-byte cache line: offset(RtToUi)={off_rt}, offset(UiToRt)={off_ui}, distance={distance}"
        );
    }

    #[test]
    fn ui_to_rt_and_cold_in_separate_cache_lines() {
        let off_ui = std::mem::offset_of!(NamClapShared, ui_to_rt);
        let off_cold = std::mem::offset_of!(NamClapShared, cold);
        let distance = off_cold.wrapping_sub(off_ui);
        assert!(
            distance >= 128,
            "UiToRt and Cold must not share a 128-byte cache line: offset(UiToRt)={off_ui}, offset(Cold)={off_cold}, distance={distance}"
        );
    }
}
