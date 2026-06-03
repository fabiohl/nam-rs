// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLAP audio processor.
//!
//! Submodules:
//! - `events`: SPSC event drainage (Main Thread → Audio Thread) and host events.
//! - `dsp`: DSP block proper (gate, inference, resampling, output).

mod dsp;
mod events;

use crate::clap::param_smoother::ParamSmoother;
use crate::clap::plugin::{ClapParamPayload, NamClapMainThread, NamClapShared};
use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::gate::DynamicHysteresis;
use crate::dsp::resampler::NamResampler;
use crate::models::DynamicModel;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// RT-safe audio processor. Runs on the host's audio thread.
///
/// Holds pre-allocated buffers and mutable inference state.
/// Created in `activate()` and destroyed in `deactivate()`.
pub struct NamClapProcessor<'a> {
    /// Active model for the left channel (None = bypass).
    model_l: Option<Box<DynamicModel>>,
    /// Polyphase sinc resampler (bypass when sample_rate == 48000).
    /// Held in Box for RT-safe disposal without allocation.
    resampler: Box<NamResampler>,
    /// Current parameters on the audio thread (snapshotted from SPSC at each process()).
    pub(crate) params: NamPluginParams,

    /// Intermediate buffers pre-allocated in activate() — ZERO alloc in process().
    /// 1. Copy of host input (variable sample_rate)
    buf_host_l: Box<[f32]>,
    buf_host_r: Box<[f32]>,
    /// 2. Post-resampler input / Pre-model (f32 @ 48kHz)
    pub(crate) buf_mid_l: Box<[f32]>,
    pub(crate) buf_mid_r: Box<[f32]>,
    /// 3. Post-model / Pre-resampler output (f32 @ 48kHz)
    pub(crate) buf_model_l: Box<[f32]>,
    pub(crate) buf_model_r: Box<[f32]>,
    /// 4. Post-resampler output / Final (variable sample_rate)
    pub(crate) buf_out_l: Box<[f32]>,
    pub(crate) buf_out_r: Box<[f32]>,

    /// Hysteresis for absolute silence detection.
    silence_hyst: DynamicHysteresis,
    /// Active model for the right channel (None = process as mono or bypass).
    active_model_r: Option<Box<DynamicModel>>,
    /// Hysteresis for mono signal detection. Persistent field to avoid
    /// re-initialization on every port_pair iteration.
    mono_hyst: DynamicHysteresis,
    /// Flag indicating whether we are processing in mono (for optimization).
    process_mono: bool,

    /// Status flags for RT telemetry.
    rt_status: Arc<RtStatusFlags>,
    /// Reference to shared state (to return channels on deactivate).
    pub(crate) shared: &'a NamClapShared,
    /// Smoothers for input and output gains.
    smoother_in: ParamSmoother,
    /// Smoothers for input and output gains.
    smoother_out: ParamSmoother,
    /// Parking lot for model/resampler disposal if the GC channel is full.
    parking_lot: [Option<GcItem>; 16],
    /// SPSC channel: Main Thread -> Audio Thread (Consumer).
    param_rx: Consumer<ClapParamPayload>,
    /// GC channel: Audio Thread -> Main Thread (Producer).
    gc_tx: Producer<GcItem>,
    /// Fallback buffer for GC overflow (overwrite).
    gc_overflow: Arc<GcOverflowBuffer>,
    /// Modulation offsets (CLAP Parameter Modulation).
    mod_input_gain: f32,
    /// Modulation offsets (CLAP Parameter Modulation).
    mod_output_gain: f32,
    /// Modulation offsets (CLAP Parameter Modulation).
    mod_gate_thresh: f32,
    /// Pre-computed thresholds (linear²) — invalidated only when
    /// gate_threshold_db or mod_gate_thresh changes (see S6.T04).
    /// SHARED ALGORITHM: Any change to the cache/invalidation logic
    /// here must be mirrored in src/standalone/pw_host.rs (threshold_open_sq
    /// and threshold_close_sq), and vice-versa. Both pre-calculate thresholds in
    /// linear² via LUT to avoid lookups on the RT hotpath.
    cached_threshold_open_sq: f32,
    cached_threshold_close_sq: f32,
    gate_dirty: bool,
    /// Telemetry decimation: 1-in-16. Cycle counter since last measurement.
    /// SHARED ALGORITHM: Same decimation strategy as `src/standalone/pw_host.rs` (frame_count & 0xF).
    /// Any change to the decimation logic here must be mirrored in pw_host.rs, and vice-versa.
    cycles_since_telemetry: u32,
    /// Host handle for calls on the audio thread.
    host: HostAudioProcessorHandle<'a>,
    /// Per-instance flag for one-time RT priority query on the first block.
    prio_checked: bool,
    pub(crate) param_in_changed: bool,
    pub(crate) param_out_changed: bool,
}

impl<'a> NamClapProcessor<'a> {
    /// Attempts to send an item for safe disposal (GC).
    /// If the main channel is full, falls back to the parking lot and then the overflow buffer.
    fn push_to_gc(&mut self, item: GcItem) {
        let mut item = Some(item);

        // 1. Try the main channel (SPSC)
        if let Some(i) = item.take() {
            if let Err(rtrb::PushError::Full(returned)) = self.gc_tx.push(i) {
                item = Some(returned);
            } else {
                return; // Success!
            }
        }

        // 2. If that failed, try the Parking Lot (Array stack-based)
        if let Some(i) = item.take() {
            let mut i_opt = Some(i);
            for slot in self.parking_lot.iter_mut() {
                if slot.is_none() {
                    *slot = i_opt.take();
                    return; // Successfully parked!
                }
            }
            item = i_opt;
        }

        // 3. If even the Parking Lot failed, use the Overflow Buffer (overwrite/controlled leak)
        if let Some(i) = item.take() {
            self.rt_status
                .set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
            self.gc_overflow.push(i);
        }
    }
}

impl<'a> PluginAudioProcessor<'a, NamClapShared, NamClapMainThread<'a>> for NamClapProcessor<'a> {
    fn activate(
        host: HostAudioProcessorHandle<'a>,
        _main_thread: &mut NamClapMainThread<'a>,
        shared: &'a NamClapShared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        #[cfg(feature = "heap-audit")]
        {
            if std::env::var("NAM_HEAP_AUDIT").is_ok() {
                crate::clap::heap_audit::AUDIT_ENABLED.store(true, Ordering::Relaxed);
            }
        }
        // 1. SPSC channel extraction from Shared (ownership transfer)
        let param_rx = shared
            .param_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| PluginError::Message("param_rx consumer has already been extracted"))?;

        let gc_tx = shared
            .gc_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| PluginError::Message("gc_tx producer has already been extracted"))?;

        // 2. Intermediate buffer pre-allocation (Disjoint Stages)
        let buf_capacity = (audio_config.max_frames_count as usize)
            .max(crate::dsp::pipeline::MAX_RESAMP_BUF)
            .max(1024)
            * 2;
        let buf_host_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_host_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_mid_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_mid_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_model_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_model_r = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_out_l = vec![0.0f32; buf_capacity].into_boxed_slice();
        let buf_out_r = vec![0.0f32; buf_capacity].into_boxed_slice();

        // 3. DSP component initialization
        let model_rate = shared.model_sample_rate.load(Ordering::Relaxed);
        let model_rate = if model_rate == 0 { 48000 } else { model_rate };
        let resampler = Box::new(
            NamResampler::new(audio_config.sample_rate as u32, model_rate, buf_capacity).map_err(
                |e| {
                    PluginError::Message(Box::leak(
                        format!("Failed to create NamResampler: {:?}", e).into_boxed_str(),
                    ))
                },
            )?,
        );

        let silence_hyst = DynamicHysteresis::new();
        let mono_hyst = DynamicHysteresis::new();

        // 4. Smoother initialization (Sample-Accurate)
        // Start at 1.0 (unity gain) to avoid silence on the first block.
        let smoother_in = ParamSmoother::new(1.0, audio_config.sample_rate as f32, 20.0);
        let smoother_out = ParamSmoother::new(1.0, audio_config.sample_rate as f32, 20.0);

        // 5. Report initial latency to shared state
        shared.current_latency.store(
            resampler.latency_samples(audio_config.sample_rate as u32),
            Ordering::Relaxed,
        );
        shared
            .sample_rate
            .store(audio_config.sample_rate as u32, Ordering::Relaxed);

        Ok(Self {
            model_l: None,
            resampler,
            params: NamPluginParams::default(),
            buf_host_l,
            buf_host_r,
            buf_mid_l,
            buf_mid_r,
            buf_model_l,
            buf_model_r,
            buf_out_l,
            buf_out_r,
            silence_hyst,
            active_model_r: None,
            mono_hyst,
            process_mono: true,
            rt_status: Arc::clone(&shared.rt_status),
            shared,
            smoother_in,
            smoother_out,
            param_rx,
            gc_tx,
            gc_overflow: Arc::clone(&shared.gc_overflow),
            parking_lot: Default::default(),
            mod_input_gain: 0.0,
            mod_output_gain: 0.0,
            mod_gate_thresh: 0.0,
            cached_threshold_open_sq: 0.0,
            cached_threshold_close_sq: 0.0,
            gate_dirty: true,
            cycles_since_telemetry: 0,
            host,
            prio_checked: false,
            param_in_changed: false,
            param_out_changed: false,
        })
    }

    fn deactivate(self, _main_thread: &mut NamClapMainThread<'a>) {
        let mut param_rx_guard = self
            .shared
            .param_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *param_rx_guard = Some(self.param_rx);

        let mut gc_tx_guard = self.shared.gc_tx.lock().unwrap_or_else(|e| e.into_inner());
        *gc_tx_guard = Some(self.gc_tx);
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        #[cfg(feature = "heap-audit")]
        let _guard = if crate::clap::heap_audit::AUDIT_ENABLED.load(Ordering::Relaxed) {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            let audit_thread = crate::clap::heap_audit::AUDIT_THREAD.load(Ordering::Relaxed);
            if audit_thread == 0 || audit_thread == tid {
                Some(crate::clap::heap_audit::TrackingGuard::new())
            } else {
                None
            }
        } else {
            None
        };

        let start_time = minstant::Instant::now();

        // One-time thread priority query on the first processed block
        if !self.prio_checked {
            self.prio_checked = true;
            unsafe {
                let thread_id = libc::pthread_self();
                let mut policy = 0i32;
                let mut param: libc::sched_param = std::mem::zeroed();
                if libc::pthread_getschedparam(thread_id, &mut policy, &mut param) == 0 {
                    self.rt_status
                        .rt_priority
                        .store(param.sched_priority, Ordering::Relaxed);
                    if policy == libc::SCHED_FIFO || policy == libc::SCHED_RR {
                        self.rt_status
                            .set_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
                    }
                }
            }
        }

        // Event drainage (SPSC + Host + GUI sync + Latency)
        self.process_events(events);

        // DSP block (gate, inference, resampling, output, telemetry)
        self.process_dsp_audio(&mut audio, start_time)
    }
}

#[cfg(test)]
#[path = "../processor_test.rs"]
mod processor_test;
