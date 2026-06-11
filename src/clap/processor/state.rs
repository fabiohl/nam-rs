// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Processor state (struct definition).

use crate::clap::plugin::{ClapParamPayload, NamClapShared};
use crate::common::params::RtPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::cabsim::conv::ConvEngine;
use crate::dsp::gate::{DynamicHysteresis, GateParams};
use crate::dsp::resampler::NamResampler;
use crate::dsp::smoother::ParamSmoother;
use crate::math::common::AlignedVec;
use crate::math::dsp::gain_lut::GainLUT;
use crate::models::StaticModel;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::sync::Arc;

/// RT-safe audio processor. Runs on the host's audio thread.
///
/// Holds pre-allocated buffers and mutable inference state.
/// Created in `activate()` and destroyed in `deactivate()`.
pub struct NamClapProcessor<'a> {
    /// Active model for the left channel (None = bypass).
    pub(crate) model_l: Option<Box<StaticModel>>,
    /// Active cab-sim convolution engine (None = bypass, zero cost).
    pub(crate) conv_engine: Option<Box<ConvEngine>>,
    /// Polyphase sinc resampler (bypass when sample_rate == 48000).
    /// Held in Box for RT-safe disposal without allocation.
    pub(crate) resampler: Box<NamResampler>,
    /// Current parameters on the audio thread (snapshotted from SPSC at each process()).
    pub(crate) params: RtPluginParams,

    /// Intermediate buffers pre-allocated in activate() — ZERO alloc in process().
    /// 1. Copy of host input (variable sample_rate)
    pub(crate) buf_host_l: AlignedVec<f32>,
    pub(crate) buf_host_r: AlignedVec<f32>,
    /// 2. Post-resampler input / Pre-model (f32 @ 48kHz)
    pub(crate) buf_mid_l: AlignedVec<f32>,
    pub(crate) buf_mid_r: AlignedVec<f32>,
    /// 3. Post-model / Pre-resampler output (f32 @ 48kHz)
    pub(crate) buf_model_l: AlignedVec<f32>,
    pub(crate) buf_model_r: AlignedVec<f32>,
    /// 4. Post-resampler output / Final (variable sample_rate)
    pub(crate) buf_out_l: AlignedVec<f32>,
    pub(crate) buf_out_r: AlignedVec<f32>,

    /// Hysteresis for absolute silence detection.
    pub(crate) silence_hyst: DynamicHysteresis,
    /// Active model for the right channel (None = process as mono or bypass).
    pub(crate) active_model_r: Option<Box<StaticModel>>,
    /// Hysteresis for mono signal detection. Persistent field to avoid
    /// re-initialization on every port_pair iteration.
    pub(crate) mono_hyst: DynamicHysteresis,
    /// Flag indicating whether we are processing in mono (for optimization).
    pub(crate) process_mono: bool,

    /// Status flags for RT telemetry.
    pub(crate) rt_status: Arc<RtStatusFlags>,
    /// Adaptive compute FSM for soft-degrade under CPU pressure.
    pub(crate) adaptive_compute: AdaptiveCompute,
    /// Reference to shared state (to return channels on deactivate).
    pub(crate) shared: &'a NamClapShared,
    /// Smoothers for input and output gains.
    pub(crate) smoother_in: ParamSmoother,
    /// Smoothers for input and output gains.
    pub(crate) smoother_out: ParamSmoother,
    /// Parking lot for model/resampler disposal if the GC channel is full.
    pub(crate) parking_lot: [Option<GcItem>; 16],
    /// SPSC channel: Main Thread -> Audio Thread (Consumer).
    pub(crate) param_rx: Consumer<ClapParamPayload>,
    /// GC channel: Audio Thread -> Main Thread (Producer).
    pub(crate) gc_tx: Producer<GcItem>,
    /// Fallback buffer for GC overflow (overwrite).
    pub(crate) gc_overflow: Arc<GcOverflowBuffer>,
    /// Modulation offsets (CLAP Parameter Modulation).
    pub(crate) mod_input_gain: f32,
    /// Modulation offsets (CLAP Parameter Modulation).
    pub(crate) mod_output_gain: f32,
    /// Modulation offsets (CLAP Parameter Modulation).
    pub(crate) mod_gate_thresh: f32,
    /// Pre-computed thresholds (linear²) — invalidated only when
    /// gate_threshold_db or mod_gate_thresh changes (see S6.T04).
    /// SHARED ALGORITHM: Any change to the cache/invalidation logic
    /// here must be mirrored in src/standalone/pw_host.rs (threshold_open_sq
    /// and threshold_close_sq), and vice-versa. Both pre-calculate thresholds in
    /// linear² via LUT to avoid lookups on the RT hotpath.
    pub(crate) cached_threshold_open_sq: f32,
    pub(crate) cached_threshold_close_sq: f32,
    pub(crate) cached_gate_params: GateParams,
    pub(crate) gate_dirty: bool,
    /// Telemetry decimation: 1-in-16. Cycle counter since last measurement.
    /// SHARED ALGORITHM: Same decimation strategy as `src/standalone/pw_host.rs` (frame_count & 0xF).
    /// Any change to the decimation logic here must be mirrored in pw_host.rs, and vice-versa.
    pub(crate) cycles_since_telemetry: u32,
    /// Host handle for calls on the audio thread.
    pub(crate) host: HostAudioProcessorHandle<'a>,
    /// Per-instance flag for one-time RT priority query on the first block.
    pub(crate) prio_checked: bool,
    /// Monotonic generation counter for GUI param synchronization.
    /// Guard: only load atomics from UiToRt when generation differs.
    pub(crate) last_seen_generation: u32,
    /// Host audio buffer size, used for model buffer realocation on load.
    pub(crate) max_frames_count: usize,
    /// Last seen render mode for transition detection (0 = Realtime, 1 = Offline).
    pub(crate) last_render_mode: u32,
    /// Pre-resolved gain LUT reference, hoisted from process_events hot-path.
    pub(crate) gain_lut: &'static GainLUT,
}
