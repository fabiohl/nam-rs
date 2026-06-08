// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLAP audio processor.
//!
//! Submodules:
//! - `events`: SPSC event drainage (Main Thread → Audio Thread) and host events.
//! - `dsp`: DSP block proper (gate, inference, resampling, output).
//! - `state`: Processor struct definition.
//! - `gc`: Garbage collection (safe disposal from audio thread).

mod dsp;
mod events;
mod gc;
mod state;

pub(crate) use state::NamClapProcessor;

use crate::clap::plugin::{NamClapMainThread, NamClapShared};
use crate::common::params::RtPluginParams;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::gate::DynamicHysteresis;
use crate::dsp::pipeline::MAX_RESAMP_BUF;
use crate::dsp::resampler::NamResampler;
use crate::dsp::smoother::ParamSmoother;
use crate::math::common::AlignedVec;
use clack_plugin::prelude::*;
use minstant::Instant;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Note: the entire `PluginAudioProcessor` impl must live in a single block
/// (Rust E0119 — trait impls cannot be split across modules).
impl<'a> PluginAudioProcessor<'a, NamClapShared, NamClapMainThread<'a>> for NamClapProcessor<'a> {
    /// `activate` is the ONLY allocation site — kept out of `process`.
    fn activate(
        host: HostAudioProcessorHandle<'a>,
        _main_thread: &mut NamClapMainThread<'a>,
        shared: &'a NamClapShared,
        audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        #[cfg(feature = "heap-audit")]
        {
            if std::env::var("NAM_HEAP_AUDIT").is_ok() {
                crate::common::alloc_audit::AUDIT_ENABLED.store(true, Ordering::Relaxed);
            }
        }
        // 1. SPSC channel extraction from Shared (ownership transfer)
        let param_rx = shared
            .cold
            .param_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| PluginError::Message("param_rx consumer has already been extracted"))?;

        let gc_tx = shared
            .cold
            .gc_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| PluginError::Message("gc_tx producer has already been extracted"))?;

        // 2. Intermediate buffer pre-allocation (Disjoint Stages)
        let buf_capacity = (audio_config.max_frames_count as usize)
            .max(MAX_RESAMP_BUF)
            .max(1024)
            * 2;
        let buf_host_l = AlignedVec::new(buf_capacity, 0.0f32);
        let buf_host_r = AlignedVec::new(buf_capacity, 0.0f32);
        let buf_mid_l = AlignedVec::new(buf_capacity, 0.0f32);
        let buf_mid_r = AlignedVec::new(buf_capacity, 0.0f32);
        let buf_model_l = AlignedVec::new(buf_capacity, 0.0f32);
        let buf_model_r = AlignedVec::new(buf_capacity, 0.0f32);
        let buf_out_l = AlignedVec::new(buf_capacity, 0.0f32);
        let buf_out_r = AlignedVec::new(buf_capacity, 0.0f32);

        // 3. DSP component initialization
        let model_rate = shared.cold.model_sample_rate.load(Ordering::Relaxed);
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
        shared.rt_to_ui.current_latency.store(
            resampler.latency_samples(audio_config.sample_rate as u32),
            Ordering::Relaxed,
        );
        shared
            .cold
            .sample_rate
            .store(audio_config.sample_rate as u32, Ordering::Relaxed);
        shared
            .cold
            .buffer_size
            .store(audio_config.max_frames_count, Ordering::Relaxed);

        Ok(Self {
            model_l: None,
            resampler,
            params: RtPluginParams::default(),
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
            rt_status: Arc::clone(&shared.cold.rt_status),
            adaptive_compute: AdaptiveCompute::new(
                crate::common::params::AdaptiveComputeMode::Conservative,
            ),
            shared,
            smoother_in,
            smoother_out,
            param_rx,
            gc_tx,
            gc_overflow: Arc::clone(&shared.cold.gc_overflow),
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
            last_seen_generation: 0,
            max_frames_count: audio_config.max_frames_count as usize,
            last_render_mode: 0,
        })
    }

    fn deactivate(self, _main_thread: &mut NamClapMainThread<'a>) {
        let mut param_rx_guard = self
            .shared
            .cold
            .param_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *param_rx_guard = Some(self.param_rx);

        let mut gc_tx_guard = self
            .shared
            .cold
            .gc_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *gc_tx_guard = Some(self.gc_tx);
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        #[cfg(feature = "heap-audit")]
        let _guard = if crate::common::alloc_audit::AUDIT_ENABLED.load(Ordering::Relaxed) {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            let audit_thread = crate::common::alloc_audit::AUDIT_THREAD.load(Ordering::Relaxed);
            if audit_thread == 0 || audit_thread == tid {
                Some(crate::common::alloc_audit::TrackingGuard::new())
            } else {
                None
            }
        } else {
            None
        };

        let should_measure = self.cycles_since_telemetry & 0xF == 0;
        self.cycles_since_telemetry = self.cycles_since_telemetry.wrapping_add(1);
        let start_time = if should_measure {
            Some(Instant::now())
        } else {
            None
        };

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
                    self.rt_status
                        .confirmed_priority
                        .store(param.sched_priority, Ordering::Relaxed);
                    self.rt_status.rt_policy.store(policy, Ordering::Relaxed);
                    if policy == libc::SCHED_FIFO || policy == libc::SCHED_RR {
                        self.rt_status
                            .set_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
                    }
                }
                let cpu = libc::sched_getcpu();
                self.rt_status.rt_cpu.store(cpu, Ordering::Relaxed);
                crate::math::common::set_daz_ftz();
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
