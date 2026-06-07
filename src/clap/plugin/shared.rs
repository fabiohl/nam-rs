// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Lock-free shared state between the audio thread and the main thread.

use crate::common::params::RtPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::resampler::NamResampler;
use crate::models::DynamicModel;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Main -> RT communication payload for the CLAP plugin.
pub enum ClapParamPayload {
    /// Parameter update (gain, gate, bypass).
    Params(RtPluginParams),
    /// Loading of a new model pair (transferred/constructed outside RT) and its resampler.
    LoadModel {
        /// The encapsulated model for neural inference (Left Channel)
        model_l: Option<Box<DynamicModel>>,
        /// The encapsulated model for neural inference (Right Channel)
        model_r: Option<Box<DynamicModel>>,
        /// Polyphase sinc resampler
        new_resampler: Box<NamResampler>,
    },
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

// ---------------------------------------------------------------------------
// Cache-line-isolated sub-structs grouped by access pattern
// ---------------------------------------------------------------------------

/// Fields written every block by the RT thread, read by the UI thread.
#[repr(align(128))]
pub struct RtToUi {
    /// True Peak L level set by the audio thread (f32 bits via f32::to_bits()). Read by the UI thread.
    pub ui_peak_l: AtomicU32,
    /// True Peak R level set by the audio thread (f32 bits via f32::to_bits()). Read by the UI thread.
    pub ui_peak_r: AtomicU32,
    /// Flag indicating whether clipping has occurred since the last UI frame. Read/reset by the UI thread.
    pub ui_clipped: AtomicBool,
    /// Current latency reported to the host (in samples).
    pub current_latency: AtomicU32,
    /// Number of active channels: 1 = mono, 2 = stereo.
    pub active_channel_count: AtomicU32,
}

/// Fields written by the UI/Main thread, read every block by the RT thread.
#[repr(align(128))]
pub struct UiToRt {
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
    /// Gesture and modification flag bitmap per parameter (GUI -> Host/Processor).
    /// Layout: for each parameter (0=input_gain, 1=output_gain, 2=gate_thresh, 3=bypass):
    ///   bit (param_index * 3 + 0) = Changed (gui_*_changed)
    ///   bit (param_index * 3 + 1) = Gesture Begin
    ///   bit (param_index * 3 + 2) = Gesture End
    pub gesture_flags: AtomicU32,
    /// Monotonic generation counter bumped (Release) by GUI on any param write.
    /// Read (Acquire) by RT to detect un-echoed GUI changes in a single load.
    pub gui_param_generation: AtomicU32,
}

/// Fields accessed at low frequency by both threads (init, shutdown, rare events).
#[repr(align(128))]
pub struct ColdShared {
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
    /// Native sample rate required by the actively loaded model.
    pub model_sample_rate: AtomicU32,
    /// Detected host sample rate.
    pub sample_rate: AtomicU32,
    /// Host buffer size.
    pub buffer_size: AtomicU32,
    /// Dynamic accent color based on DAW track color (packed ARGB).
    pub track_accent_color: AtomicU32,
    /// Parameter indication (mapping, automation, and override) for the 6 parameters.
    /// Bit 0: Mapped, Bit 1: Automating, Bit 2: Override.
    pub param_indication: [AtomicU8; 6],
    /// Indicated/mapped parameter colors (packed ARGB).
    pub param_indication_color: [AtomicU32; 6],
    /// Model load counter (incremented on each successful model load).
    pub model_load_counter: AtomicU32,
    /// Loaded model name (path basename). Written by the main thread, read by the UI thread.
    pub ui_model_name: Mutex<String>,
    /// Loaded model metadata for UI display.
    pub ui_model_metadata: Mutex<Option<NamModelMetadata>>,
    /// Pending model path to be loaded by the Main Thread. Written by the UI thread.
    pub ui_pending_model: Mutex<Option<PathBuf>>,
    /// Indicates whether the GUI is in the middle of an asynchronous model load.
    pub ui_loading: AtomicBool,
    /// Flag signaling that a model loading error occurred.
    pub ui_load_error: AtomicBool,
    /// Detailed error message for the GUI.
    pub ui_load_error_msg: Mutex<String>,
    /// Dynamic model info for diagnostics.
    pub ui_model_info: Mutex<Option<crate::common::diagnostics::ModelInfo>>,
    /// Lifetime fence: true while the plugin exists. Checked by the File Picker thread.
    pub alive_fence: Arc<AtomicBool>,
    /// Render mode as set by the host via `clap.render`: 0 = Realtime, 1 = Offline.
    /// Written by the Main Thread, read by the RT thread at low frequency (transitions only).
    pub render_mode: AtomicU32,
    /// Current GUI scale factor (f32 bits). Written by `gui.set_scale` on the Main Thread,
    /// read by the window handler for HiDPI rendering before the first Resized event.
    pub gui_scale_factor: AtomicU32,
}

// ---------------------------------------------------------------------------
// Outer shared struct
// ---------------------------------------------------------------------------

/// Lock-free shared state between the audio thread and the main thread.
///
/// Fields are segregated into cache-line-isolated sub-structs grouped by
/// access pattern to eliminate False Sharing.  Each sub-struct has its own
/// `#[repr(align(128))]` so that no two sub-structs share a 128-byte cache
/// line, preventing cache-line bouncing between RT↔UI hotpath writes/reads.
///
/// SPSC channels are wrapped in Mutex<Option<...>> only to allow
/// them to be "extracted" by their respective threads during initialization,
/// satisfying the `Sync` requirement of the `PluginShared` trait.
///
/// - `rt_to_ui`: written every block by RT, read by UI.
/// - `ui_to_rt`: written by UI/Main, read every block by RT.
/// - `cold`: low-frequency access by both threads.
pub struct NamClapShared {
    /// Cache-line-isolated sub-struct: written every block by RT, read by UI.
    pub rt_to_ui: RtToUi,
    /// Cache-line-isolated sub-struct: written by UI/Main, read every block by RT.
    pub ui_to_rt: UiToRt,
    /// Cache-line-isolated sub-struct: low-frequency access by both threads.
    pub cold: ColdShared,
}

impl<'a> PluginShared<'a> for NamClapShared {}

impl Drop for NamClapShared {
    fn drop(&mut self) {
        self.cold.alive_fence.store(false, Ordering::Relaxed);
        crate::common::panic_hook::set_shutdown_in_progress();
    }
}

/// Render mode constants for `ColdShared::render_mode`.
/// Render mode: realtime (normal processing).
pub const RENDER_MODE_REALTIME: u32 = 0;
/// Render mode: offline (export/bounce, max quality, no soft-degrade).
pub const RENDER_MODE_OFFLINE: u32 = 1;

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
        self.ui_to_rt.gesture_flags.fetch_or(bit, Ordering::Relaxed);
    }

    /// Reads and clears a gesture flag (swap to false), returns the previous value.
    pub fn take_gesture(&self, param_index: usize, flag_shift: u32) -> bool {
        let bit = 1u32 << (param_index as u32 * Self::GESTURE_BITS_PER_PARAM + flag_shift);
        (self
            .ui_to_rt
            .gesture_flags
            .fetch_and(!bit, Ordering::Relaxed)
            & bit)
            != 0
    }

    /// Zeros out all gesture flags.
    pub fn clear_gestures(&self) {
        self.ui_to_rt.gesture_flags.store(0, Ordering::Relaxed);
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
                &self.ui_to_rt.param_input_gain,
            ),
            (
                PARAM_OUTPUT_GAIN,
                Self::param_index(PARAM_OUTPUT_GAIN) as u32,
                &self.ui_to_rt.param_output_gain,
            ),
            (
                PARAM_GATE_THRESH,
                Self::param_index(PARAM_GATE_THRESH) as u32,
                &self.ui_to_rt.param_gate_thresh,
            ),
            (
                PARAM_BYPASS,
                Self::param_index(PARAM_BYPASS) as u32,
                &self.ui_to_rt.param_bypass,
            ),
            (
                PARAM_ADAPTIVE_COMPUTE,
                Self::param_index(PARAM_ADAPTIVE_COMPUTE) as u32,
                &self.ui_to_rt.param_adaptive_compute,
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
        if let Ok(info_guard) = self.cold.ui_model_info.lock() {
            info_guard.clone()
        } else {
            None
        }
    }

    fn audio_info(&self) -> crate::common::diagnostics::AudioInfo {
        let sr = self.cold.sample_rate.load(Ordering::Relaxed);
        let buffer_size = self.cold.buffer_size.load(Ordering::Relaxed) as usize;
        let channel_count = self.rt_to_ui.active_channel_count.load(Ordering::Relaxed) as usize;
        crate::common::diagnostics::AudioInfo {
            sample_rate: sr,
            buffer_size,
            channel_count,
            host_name: "CLAP".to_string(),
        }
    }

    fn rt_info(&self) -> crate::common::diagnostics::RtInfo {
        self.cold.rt_status.rt_info()
    }

    fn telemetry_snapshot(&self) -> crate::common::diagnostics::TelemetrySnapshot {
        self.cold.rt_status.telemetry_snapshot()
    }

    fn flags_seen(&self) -> u64 {
        self.cold.rt_status.flags_seen()
    }
}

#[cfg(test)]
#[path = "shared_test.rs"]
mod shared_test;

#[cfg(test)]
pub(crate) use shared_test::make_test_shared;
