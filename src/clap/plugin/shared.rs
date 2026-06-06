// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Lock-free shared state between the audio thread and the main thread.

use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::resampler::NamResampler;
use crate::loader::LoadedModelPair;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Main -> RT communication payload for the CLAP plugin.
pub enum ClapParamPayload {
    /// Parameter update (gain, gate, bypass).
    Params(NamPluginParams),
    /// Loading of a new model pair with calibration metadata and its resampler.
    LoadModel(Box<LoadedModelPair>, Box<NamResampler>),
}

/// Model metadata for display in the GUI.
#[derive(Clone, Debug, Default)]
pub struct NamModelMetadata {
    /// Model architecture (e.g. "LSTM", "WaveNet").
    pub architecture: String,
    /// Model topology (e.g. "Standard", "1x64").
    pub topology: String,
    /// Author / Modeled by.
    pub modeled_by: Option<String>,
    /// Original equipment manufacturer.
    pub gear_make: Option<String>,
    /// Original equipment model.
    pub gear_model: Option<String>,
    /// Equipment type.
    pub gear_type: Option<String>,
    /// Style/Tone type of the equipment.
    pub tone_type: Option<String>,
    /// Date formatted as YYYY-MM-DD.
    pub date: Option<String>,
}

/// Safe wrapper for a NamClapShared pointer passed to the GUI thread.
#[derive(Clone, Copy)]
pub struct NamClapSharedRef(pub *const NamClapShared);

unsafe impl Send for NamClapSharedRef {}
unsafe impl Sync for NamClapSharedRef {}

/// Lock-free shared state between the audio thread and the main thread.
///
/// Uses 128-byte alignment to mitigate False Sharing.
/// SPSC channels are wrapped in Mutex<Option<...>> only to allow
/// them to be "extracted" by their respective threads during initialization,
/// satisfying the `Sync` requirement of the `PluginShared` trait.
#[repr(align(128))]
pub struct NamClapShared {
    /// SPSC channel: Main Thread -> Audio Thread (New parameters/models).
    pub param_tx: Mutex<Option<Producer<ClapParamPayload>>>,
    /// SPSC channel: Main Thread -> Audio Thread (Consumer).
    pub param_rx: Mutex<Option<Consumer<ClapParamPayload>>>,
    /// GC channel: Audio Thread -> Main Thread (Obsolete models for disposal).
    pub gc_tx: Mutex<Option<Producer<GcItem>>>,
    /// GC channel: Audio Thread -> Main Thread (Consumer).
    pub gc_rx: Mutex<Option<Consumer<GcItem>>>,
    /// Fallback buffer for GC overflow (overwrite).
    pub gc_overflow: Arc<GcOverflowBuffer>,
    /// Atomic status flags (RT->Main telemetry).
    pub rt_status: Arc<RtStatusFlags>,
    /// Current latency reported to the host (in samples).
    pub current_latency: AtomicU32,
    /// Native sample rate required by the actively loaded model.
    pub model_sample_rate: AtomicU32,
    /// Latest Input Gain parameter value (f32 as bits).
    pub param_input_gain: AtomicU32,
    /// Latest Output Gain parameter value (f32 as bits).
    pub param_output_gain: AtomicU32,
    /// Latest Gate Threshold parameter value (f32 as bits).
    pub param_gate_thresh: AtomicU32,
    /// Latest Bypass parameter value (0 = false, 1 = true).
    pub param_bypass: AtomicU32,
    /// Latest Adaptive Compute mode parameter value (0=Off, 1=Conservative, 2=Aggressive).
    pub param_adaptive_compute: AtomicU32,
    /// True Peak L level set by the audio thread (f32 bits via f32::to_bits()). Read by the UI thread.
    pub ui_peak_l: AtomicU32,
    /// True Peak R level set by the audio thread (f32 bits via f32::to_bits()). Read by the UI thread.
    pub ui_peak_r: AtomicU32,
    /// Flag indicating whether clipping has occurred since the last UI frame. Read/reset by the UI thread.
    pub ui_clipped: std::sync::atomic::AtomicBool,
    /// Loaded model name (path basename). Written by the main thread, read by the UI thread.
    pub ui_model_name: Mutex<String>,
    /// Loaded model metadata for UI display.
    pub ui_model_metadata: Mutex<Option<NamModelMetadata>>,
    /// Pending model path to be loaded by the Main Thread. Written by the UI thread.
    pub ui_pending_model: Mutex<Option<std::path::PathBuf>>,
    /// Indicates whether the GUI is in the middle of an asynchronous model load.
    pub ui_loading: std::sync::atomic::AtomicBool,
    /// Flag signaling that a model loading error occurred.
    pub ui_load_error: std::sync::atomic::AtomicBool,
    /// Detailed error message for the GUI.
    pub ui_load_error_msg: Mutex<String>,
    /// Detected host sample rate.
    pub sample_rate: AtomicU32,
    /// Number of active channels: 1 = mono, 2 = stereo.
    pub active_channel_count: AtomicU32,
    /// Dynamic accent color based on DAW track color (packed ARGB).
    pub track_accent_color: AtomicU32,
    /// Parameter indication (mapping, automation, and override) for the 6 parameters.
    /// Bit 0: Mapped, Bit 1: Automating, Bit 2: Override.
    pub param_indication: [std::sync::atomic::AtomicU8; 6],
    /// Indicated/mapped parameter colors (packed ARGB).
    pub param_indication_color: [std::sync::atomic::AtomicU32; 6],
    /// Model load counter (incremented on each successful model load).
    pub model_load_counter: AtomicU32,
    /// Host buffer size.
    pub buffer_size: AtomicU32,
    /// Dynamic model info for diagnostics.
    pub ui_model_info: Mutex<Option<crate::common::diagnostics::ModelInfo>>,
    /// Lifetime fence: true while the plugin exists. Checked by the File Picker thread.
    pub alive_fence: Arc<std::sync::atomic::AtomicBool>,

    /// Gesture and modification flag bitmap per parameter (GUI -> Host/Processor).
    /// Layout: for each parameter (0=input_gain, 1=output_gain, 2=gate_thresh, 3=bypass):
    ///   bit (param_index * 3 + 0) = Changed (gui_*_changed)
    ///   bit (param_index * 3 + 1) = Gesture Begin
    ///   bit (param_index * 3 + 2) = Gesture End
    pub gesture_flags: AtomicU32,
}

impl<'a> PluginShared<'a> for NamClapShared {}

impl Drop for NamClapShared {
    fn drop(&mut self) {
        self.alive_fence.store(false, Ordering::Relaxed);
    }
}

impl NamClapShared {
    /// Bitmask for the parameter in the `gesture_flags` field.
    /// 3 flags per parameter: Changed, GestureBegin, GestureEnd.
    const GESTURE_CHANGED_SHIFT: u32 = 0;
    const GESTURE_BEGIN_SHIFT: u32 = 1;
    const GESTURE_END_SHIFT: u32 = 2;
    const GESTURE_BITS_PER_PARAM: u32 = 3;

    /// Maps a CLAP param_id (0..3) to internal index 0..3.
    const fn param_index(param_id: u32) -> usize {
        param_id as usize
    }

    /// Sets a gesture flag for the parameter (store = true).
    pub fn set_gesture(&self, param_index: usize, flag_shift: u32) {
        let bit = 1u32 << (param_index as u32 * Self::GESTURE_BITS_PER_PARAM + flag_shift);
        self.gesture_flags.fetch_or(bit, Ordering::Relaxed);
    }

    /// Reads and clears a gesture flag (swap to false), returns the previous value.
    pub fn take_gesture(&self, param_index: usize, flag_shift: u32) -> bool {
        let bit = 1u32 << (param_index as u32 * Self::GESTURE_BITS_PER_PARAM + flag_shift);
        (self.gesture_flags.fetch_and(!bit, Ordering::Relaxed) & bit) != 0
    }

    /// Zeros out all gesture flags.
    pub fn clear_gestures(&self) {
        self.gesture_flags.store(0, Ordering::Relaxed);
    }

    /// Flushes gestures and parameter updates initiated by the GUI
    /// into the host's output event queue.
    pub fn write_gui_events(&self, output: &mut OutputEvents) {
        use crate::clap::extensions::params::{
            PARAM_ADAPTIVE_COMPUTE, PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN,
            PARAM_OUTPUT_GAIN,
        };
        use clack_plugin::events::event_types::{
            ParamGestureBeginEvent, ParamGestureEndEvent, ParamValueEvent,
        };

        let params: [(u32, u32, &AtomicU32); 5] = [
            (
                PARAM_INPUT_GAIN,
                Self::param_index(PARAM_INPUT_GAIN) as u32,
                &self.param_input_gain,
            ),
            (
                PARAM_OUTPUT_GAIN,
                Self::param_index(PARAM_OUTPUT_GAIN) as u32,
                &self.param_output_gain,
            ),
            (
                PARAM_GATE_THRESH,
                Self::param_index(PARAM_GATE_THRESH) as u32,
                &self.param_gate_thresh,
            ),
            (
                PARAM_BYPASS,
                Self::param_index(PARAM_BYPASS) as u32,
                &self.param_bypass,
            ),
            (
                PARAM_ADAPTIVE_COMPUTE,
                Self::param_index(PARAM_ADAPTIVE_COMPUTE) as u32,
                &self.param_adaptive_compute,
            ),
        ];

        for (param_id, param_idx, value_atomic) in &params {
            let pi = *param_idx as usize;
            if self.take_gesture(pi, Self::GESTURE_BEGIN_SHIFT) {
                let ev = ParamGestureBeginEvent::new(0, ClapId::new(*param_id));
                let _ = output.try_push(ev);
            }
            if self.take_gesture(pi, Self::GESTURE_CHANGED_SHIFT) {
                let val = f32::from_bits(value_atomic.load(Ordering::Relaxed)) as f64;
                let ev = ParamValueEvent::new(
                    0,
                    ClapId::new(*param_id),
                    clack_plugin::events::Pckn::new(0u8, 0u8, 0u8, 0u8),
                    val,
                    clack_plugin::utils::Cookie::empty(),
                );
                let _ = output.try_push(ev);
            }
            if self.take_gesture(pi, Self::GESTURE_END_SHIFT) {
                let ev = ParamGestureEndEvent::new(0, ClapId::new(*param_id));
                let _ = output.try_push(ev);
            }
        }
    }
}

impl crate::common::diagnostics::HasRuntimeSnapshot for NamClapShared {
    fn model_info(&self) -> Option<crate::common::diagnostics::ModelInfo> {
        if let Ok(info_guard) = self.ui_model_info.lock() {
            info_guard.clone()
        } else {
            None
        }
    }

    fn audio_info(&self) -> crate::common::diagnostics::AudioInfo {
        let sr = self.sample_rate.load(Ordering::Relaxed);
        let buffer_size = self.buffer_size.load(Ordering::Relaxed) as usize;
        let channel_count = self.active_channel_count.load(Ordering::Relaxed) as usize;
        crate::common::diagnostics::AudioInfo {
            sample_rate: sr,
            buffer_size,
            channel_count,
            host_name: "CLAP".to_string(),
        }
    }

    fn rt_info(&self) -> crate::common::diagnostics::RtInfo {
        self.rt_status.rt_info()
    }

    fn telemetry_snapshot(&self) -> crate::common::diagnostics::TelemetrySnapshot {
        self.rt_status.telemetry_snapshot()
    }

    fn flags_seen(&self) -> u64 {
        self.rt_status.flags_seen()
    }
}
