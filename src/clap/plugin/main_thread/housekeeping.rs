// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Main thread housekeeping: GC drain, status-flags sync, pending model load, latency.

use super::NamClapMainThread;
use crate::common::spsc::{self, drain_gc_channels};
use crate::models::slimmable::slice_wavenet_model;
use crate::models::{NamModel, StaticModel};
use clack_extensions::log::{HostLog, LogSeverity};
use std::ffi::CString;
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

        // WaveNet slimmable rebuild: main thread performs all allocation,
        // prewarm, and mmap outside the audio-thread callback.
        if self
            .shared
            .cold
            .rt_status
            .check_flag(spsc::RT_STATUS_NEEDS_SLIMMABLE_REBUILD)
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
                .clear_flag(spsc::RT_STATUS_NEEDS_SLIMMABLE_REBUILD);
        }

        // Check if there is a pending model sent by the UI
        let pending_model = if let Ok(mut pending_guard) = self.shared.cold.ui_pending_model.lock()
        {
            pending_guard.take()
        } else {
            None
        };
        if let Some(path) = pending_model {
            let res = self.load_model(&path);
            self.shared.cold.ui_loading.store(false, Ordering::Relaxed);
            match res {
                Ok(_) => {}
                Err(e) => {
                    let err_msg = e.error_code().message();
                    if let Ok(mut msg_guard) = self.shared.cold.ui_load_error_msg.lock() {
                        *msg_guard = err_msg.to_string();
                    }
                    self.shared
                        .cold
                        .ui_load_error
                        .store(true, Ordering::Relaxed);

                    if let Some(log) = self.host.get_extension::<HostLog>() {
                        let shared = self.host.shared();
                        let err_str = format!("NAM-rs: Failed to load model from GUI: {:?}", e);
                        let sanitized_err = err_str.replace('\0', " ");
                        let msg = CString::new(sanitized_err).unwrap_or_else(|_| {
                            CString::new(
                                "NAM-rs: Failed to load model from GUI due to invalid characters",
                            )
                            .unwrap_or_default()
                        });
                        log.log(&shared, LogSeverity::Error, &msg);
                    }
                }
            }
        }

        // Check if there is a pending IR sent by the UI
        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        {
            use crate::clap::plugin::ClapParamPayload;
            let pending_ir = if let Ok(mut pending_guard) = self.shared.cold.ui_pending_ir.lock() {
                pending_guard.take()
            } else {
                None
            };
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
                        if let Ok(mut msg_guard) = self.shared.cold.ui_ir_load_error_msg.lock() {
                            *msg_guard = err_msg.to_string();
                        }
                        self.shared
                            .cold
                            .ui_ir_load_error
                            .store(true, Ordering::Relaxed);

                        if let Some(log) = self.host.get_extension::<HostLog>() {
                            let shared = self.host.shared();
                            let err_str =
                                format!("NAM-rs: Failed to load cab-sim IR from GUI: {:?}", e);
                            let sanitized_err = err_str.replace('\0', " ");
                            let msg = CString::new(sanitized_err).unwrap_or_else(|_| {
                                CString::new(
                                    "NAM-rs: Failed to load IR from GUI due to invalid characters",
                                )
                                .unwrap_or_default()
                            });
                            log.log(&shared, LogSeverity::Error, &msg);
                        }
                    }
                }
            }

            if self.shared.cold.ui_clear_ir.swap(false, Ordering::Relaxed) {
                let _ = self
                    .param_tx
                    .push(ClapParamPayload::LoadCabIr { engine: None });
                if let Ok(mut ir_guard) = self.shared.cold.ir_path.lock() {
                    *ir_guard = None;
                }
                if let Ok(mut raw_guard) = self.shared.cold.ir_raw_samples.lock() {
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
    }
}
