// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Main thread housekeeping: GC drain, status-flags sync, pending model load, latency.

use super::NamClapMainThread;
use crate::common::spsc::{self, drain_gc_channels};
use crate::models::slimmable::slice_wavenet_model;
use crate::models::{NamModel, StaticModel};
use clack_extensions::tail::HostTail;
use std::sync::atomic::Ordering;

impl<'a> NamClapMainThread<'a> {
    /// GC drain, status flag mirroring, hugepage sync, pending model load, latency notification.
    pub(crate) fn housekeeping(&mut self) {
        // Flush any model deferred by load_model() (F3 fix).
        // Primary mechanism is activate(), this is a fallback for hosts
        // that call state-load between activate() and the first process().
        // Errors set RT_STATUS_MODEL_LOAD_FAILED internally; housekeeping is void.
        let _ = self.flush_pending_model();

        // Drain obsolete models to free memory outside RT.
        let drained = drain_gc_channels(
            &mut self.gc_rx,
            &self.shared.cold.gc_overflow,
            &self.shared.cold.rt_status,
        );
        self.shared
            .cold
            .rt_status
            .drains
            .fetch_add(drained as u32, Ordering::Relaxed);

        let current_bits = self
            .shared
            .cold
            .rt_status
            .status_bits
            .load(Ordering::Relaxed);
        self.shared
            .cold
            .rt_status
            .flags_seen
            .fetch_or(current_bits, Ordering::Relaxed);

        // Sync huge page status from mirror buffer (one-shot).
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static HUGEPAGE_SYNCED: AtomicBool = AtomicBool::new(false);
            if !HUGEPAGE_SYNCED.load(Ordering::Relaxed) {
                crate::dsp::mirror_buf::sync_huge_page_flag(&self.shared.cold.rt_status);
                HUGEPAGE_SYNCED.store(true, Ordering::Relaxed);
            }
        }

        // Slimmable reset failure: RT thread sets flag, main thread emits log.
        if self
            .shared
            .cold
            .rt_status
            .check_and_clear_flag(spsc::RT_STATUS_SLIMMABLE_RESET_FAILED)
        {
            log::error!(
                "ContainerModel submodel reset failed — model may run in previous state."
            );
        }

        // WaveNet slimmable rebuild: main thread performs all allocation,
        // prewarm, and mmap outside the audio-thread callback.
        if self
            .shared
            .cold
            .rt_status
            .check_flag_acquire(spsc::RT_STATUS_NEEDS_SLIMMABLE_REBUILD)
        {
            let target_ch = self
                .shared
                .cold
                .rt_status
                .requested_slimmable_ch
                .load(Ordering::Relaxed) as usize;
            let buffer_size = self.shared.cold.buffer_size.load(Ordering::Relaxed) as usize;

            if target_ch >= 4 {
                let new_model = {
                    let storage = self
                        .shared
                        .cold
                        .full_wavenet_model
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    storage.as_ref().and_then(|m| {
                        if let StaticModel::WavenetDyn(w) = m.as_ref() {
                            match slice_wavenet_model(w, target_ch) {
                                Ok(mut slimmed) => {
                                    slimmed.prewarm();
                                    if buffer_size > 0 {
                                        let _ = slimmed.set_max_buffer_size(buffer_size);
                                    }
                                    Some(Box::new(StaticModel::WavenetDyn(Box::new(slimmed))))
                                }
                                Err(_) => {
                                    self.shared
                                        .cold
                                        .rt_status
                                        .set_flag(spsc::RT_STATUS_SLIMMABLE_SLICE_FAILED);
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    })
                };

                if let Some(model) = new_model {
                    let _ = self.slimmable_tx.push(Some(model));
                }
            }
            self.shared
                .cold
                .rt_status
                .clear_flag_release(spsc::RT_STATUS_NEEDS_SLIMMABLE_REBUILD);
        }

        // Oversampling engine rebuild: main thread performs all allocation
        // (OversampleEngine::new creates filters and buffers) off the audio-thread.
        if self
            .shared
            .cold
            .rt_status
            .check_flag_acquire(spsc::RT_STATUS_NEEDS_OS_REBUILD)
        {
            use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
            use crate::dsp::pipeline::MAX_RESAMP_BUF;

            let factor_val = self
                .shared
                .cold
                .rt_status
                .requested_os_factor
                .load(Ordering::Relaxed);
            let factor = OversampleFactor::from_f32(factor_val as f32);
            if let (Ok(l), Ok(r)) = (
                OversampleEngine::new(factor, MAX_RESAMP_BUF),
                OversampleEngine::new(factor, MAX_RESAMP_BUF),
            ) {
                let _ = self
                    .param_tx
                    .push(crate::clap::plugin::ClapParamPayload::SetOversample {
                        os_l: Box::new(l),
                        os_r: Box::new(r),
                    });
                self.shared
                    .cold
                    .rt_status
                    .clear_flag_release(spsc::RT_STATUS_NEEDS_OS_REBUILD);
            } else {
                crate::common::diagnostics::NamDiagnostic::new(
                    crate::common::diagnostics::NamErrorCode::OutOfMemory,
                    &self.sys,
                )
                .message("Failed to rebuild oversample engine (OOM).")
                .hint("The current oversampling state will be preserved.")
                .emit_warning();
                self.shared
                    .cold
                    .rt_status
                    .clear_flag_release(spsc::RT_STATUS_NEEDS_OS_REBUILD);
            }
        }

        // Propagate dialog_state → ui_pending_model (R2: Arc-backed, UAF-safe)
        if let Some(dialog_state) = self.shared.cold.dialog_state.as_ref() {
            let mut dialog_guard = dialog_state.pending_model.lock().unwrap_or_else(|e| {
                log::error!("PoisonError in dialog_state.pending_model lock: {e:?}");
                e.into_inner()
            });
            if let Some(path) = dialog_guard.take() {
                let mut ui_guard = self
                    .shared
                    .cold
                    .ui_pending_model
                    .lock()
                    .unwrap_or_else(|e| {
                        log::error!("PoisonError in ui_pending_model lock: {e:?}");
                        e.into_inner()
                    });
                *ui_guard = Some(path);
            }
        }

        // Check if there is a pending model sent by the UI
        let pending_model = self
            .shared
            .cold
            .ui_pending_model
            .lock()
            .unwrap_or_else(|e| {
                log::error!("PoisonError in ui_pending_model lock: {e:?}");
                e.into_inner()
            })
            .take();
        if let Some(path) = pending_model {
            let res = self.load_model(&path);
            self.shared.cold.ui_loading.store(false, Ordering::Relaxed);
            match res {
                Ok(_) => {}
                Err(e) => {
                    let err_msg = e.error_code().message();
                    let mut msg_guard =
                        self.shared
                            .cold
                            .ui_load_error_msg
                            .lock()
                            .unwrap_or_else(|e| {
                                log::error!("PoisonError in ui_load_error_msg lock: {e:?}");
                                e.into_inner()
                            });
                    *msg_guard = err_msg.to_string();
                    self.shared
                        .cold
                        .ui_load_error
                        .store(true, Ordering::Relaxed);

                    log::error!("Failed to load model from GUI: {e:?}");
                }
            }
        }

        // Check if there is a pending IR sent by the UI
        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        {
            use crate::clap::plugin::ClapParamPayload;

            // Propagate ir_dialog_state → ui_pending_ir (R2: Arc-backed, UAF-safe)
            if let Some(ir_dialog_state) = self.shared.cold.ir_dialog_state.as_ref() {
                let mut dialog_guard = ir_dialog_state.pending_ir.lock().unwrap_or_else(|e| {
                    log::error!("PoisonError in ir_dialog_state.pending_ir lock: {e:?}");
                    e.into_inner()
                });
                if let Some(path) = dialog_guard.take() {
                    let mut ui_guard = self.shared.cold.ui_pending_ir.lock().unwrap_or_else(|e| {
                        log::error!("PoisonError in ui_pending_ir lock: {e:?}");
                        e.into_inner()
                    });
                    *ui_guard = Some(path);
                }
            }

            let pending_ir = self
                .shared
                .cold
                .ui_pending_ir
                .lock()
                .unwrap_or_else(|e| {
                    log::error!("PoisonError in ui_pending_ir lock: {e:?}");
                    e.into_inner()
                })
                .take();
            if let Some(path) = pending_ir {
                let res = self.load_cabsim(&path);
                self.shared
                    .cold
                    .ui_ir_loading
                    .store(false, Ordering::Relaxed);
                match res {
                    Ok(_) => {}
                    Err(e) => {
                        let err_msg = e.error_code().message();
                        let mut msg_guard = self
                            .shared
                            .cold
                            .ui_ir_load_error_msg
                            .lock()
                            .unwrap_or_else(|e| {
                                log::error!("PoisonError in ui_ir_load_error_msg lock: {e:?}");
                                e.into_inner()
                            });
                        *msg_guard = err_msg.to_string();
                        self.shared
                            .cold
                            .ui_ir_load_error
                            .store(true, Ordering::Relaxed);

                        log::error!("Failed to load cab-sim IR from GUI: {e:?}");
                    }
                }
            }

            if self.shared.cold.ui_clear_ir.swap(false, Ordering::Relaxed) {
                let _ = self
                    .param_tx
                    .push(ClapParamPayload::LoadCabIr { engine: None });
                {
                    let mut ir_guard = self.shared.cold.ir_path.lock().unwrap_or_else(|e| {
                        log::error!("PoisonError in ir_path lock: {e:?}");
                        e.into_inner()
                    });
                    *ir_guard = None;
                }
                {
                    let mut raw_guard =
                        self.shared.cold.ir_raw_samples.lock().unwrap_or_else(|e| {
                            log::error!("PoisonError in ir_raw_samples lock: {e:?}");
                            e.into_inner()
                        });
                    *raw_guard = None;
                }
            }
        }

        // Latency Monitoring: Notify the host if the value changed
        let current_latency = self.shared.rt_to_ui.current_latency.load(Ordering::Relaxed);
        if current_latency != self.last_reported_latency {
            self.last_reported_latency = current_latency;
            if let Some(latency_ext) = self
                .host
                .get_extension::<clack_extensions::latency::HostLatency>()
            {
                latency_ext.changed(&mut self.host);
            }
        }

        // Tail Monitoring: notify the host when the CabSim tail length changes.
        let cabsim_tail = self
            .shared
            .rt_to_ui
            .cabsim_tail_samples
            .load(Ordering::Relaxed);
        if cabsim_tail != self.last_reported_cabsim_tail {
            self.last_reported_cabsim_tail = cabsim_tail;
            if let Some(tail_ext) = self.host.get_extension::<HostTail>() {
                let raw = std::ptr::NonNull::from(self.host.as_raw());
                // SAFETY: HostMainThreadHandle and HostAudioProcessorHandle are
                // repr(transparent) wrappers around the same NonNull<clap_host>.
                // Calling tail.changed() from the main thread is valid per the
                // clap_plugin_tail specification — it posts an event to the
                // host's main-thread event queue.
                let mut audio_host =
                    unsafe { clack_plugin::host::HostAudioProcessorHandle::from_raw(raw) };
                tail_ext.changed(&mut audio_host);
            }
        }
    }
}
