// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Main thread exclusive state (model loading, state save/load).

use super::shared::{ClapParamPayload, NamClapShared, NamModelMetadata};
use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, drain_gc_channels};
use crate::dsp::resampler::NamResampler;
use crate::loader::load_and_build_model;
use clack_extensions::log::{HostLog, LogSeverity};
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Main thread exclusive state (model loading, state save/load).
pub struct NamClapMainThread<'a> {
    pub(crate) shared: &'a NamClapShared,
    /// Current parameters known by the main thread (mirror of the audio thread params).
    pub params: NamPluginParams,
    /// Host handle for notifications (latency_changed, state, etc.).
    pub host: HostMainThreadHandle<'a>,
    /// System snapshot for emitting diagnostics.
    pub sys: SystemSnapshot,
    /// Producer to send updates to the audio thread.
    pub param_tx: Producer<ClapParamPayload>,
    /// Consumer to collect garbage (obsolete models) from the audio thread.
    pub gc_rx: Consumer<GcItem>,
    /// Cached last latency reported to the host to avoid redundant notifications.
    pub last_reported_latency: u32,
    /// Baseview window handle for GUI lifecycle control.
    #[cfg(feature = "clap-plugin")]
    pub window_handle: Option<baseview::WindowHandle>,
}

impl<'a> PluginMainThread<'a, NamClapShared> for NamClapMainThread<'a> {
    /// Called periodically or in response to host events.
    /// Here we can drain the GC channel.
    fn on_main_thread(&mut self) {
        // Drain obsolete models to free memory outside RT.
        let drained = drain_gc_channels(&mut self.gc_rx, &self.shared.gc_overflow);
        self.shared
            .rt_status
            .drains
            .fetch_add(drained as u32, Ordering::Relaxed);

        let current_bits = self.shared.rt_status.status_bits.load(Ordering::Relaxed);
        self.shared
            .rt_status
            .flags_seen
            .fetch_or(current_bits, Ordering::Relaxed);

        // Sync huge page status from mirror buffer (one-shot).
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static HUGEPAGE_SYNCED: AtomicBool = AtomicBool::new(false);
            if !HUGEPAGE_SYNCED.load(Ordering::Relaxed) {
                crate::dsp::mirror_buf::sync_huge_page_flag(&self.shared.rt_status);
                HUGEPAGE_SYNCED.store(true, Ordering::Relaxed);
            }
        }

        // Check if there is a pending model sent by the UI
        let pending_model = if let Ok(mut pending_guard) = self.shared.ui_pending_model.lock() {
            pending_guard.take()
        } else {
            None
        };
        if let Some(path) = pending_model {
            let res = self.load_model(&path);
            self.shared.ui_loading.store(false, Ordering::Relaxed);
            match res {
                Ok(_) => {}
                Err(e) => {
                    let err_msg = match e.error_code() {
                        NamErrorCode::FileNotFound => "File not found",
                        NamErrorCode::FileReadError => "File read error",
                        NamErrorCode::UnknownExtension => "Unknown extension",
                        NamErrorCode::NamJsonParseError => "Invalid JSON format",
                        NamErrorCode::NambCrc32Mismatch => "CRC32 checksum mismatch",
                        NamErrorCode::NambCrc32Missing => "CRC32 integrity flag missing (v2+)",
                        NamErrorCode::NambInvalidMagic => "Invalid signature",
                        NamErrorCode::NambUnsupportedVersion => "Unsupported version",
                        NamErrorCode::NambTruncated => "Corrupted/truncated file",
                        NamErrorCode::UnsupportedArchitecture => "Unsupported architecture",
                        NamErrorCode::TopologyDetectionFailed => "Topology detection failed",
                        NamErrorCode::WeightCountMismatch => "Weight count mismatch",
                        NamErrorCode::ModelBuildFailed => "Model build failed",
                        NamErrorCode::ModelTooLarge => "Model file too large",
                        _ => "Internal error",
                    };
                    if let Ok(mut msg_guard) = self.shared.ui_load_error_msg.lock() {
                        *msg_guard = err_msg.to_string();
                    }
                    self.shared.ui_load_error.store(true, Ordering::Relaxed);

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

        // RT-Safe logging via atomic flags (consumes transient events)
        if let Some(log) = self.host.get_extension::<HostLog>() {
            let shared = self.host.shared();

            // WHITELIST: CString::new(…).unwrap_or_default() is safe and cannot panic because:
            // - All strings are static ASCII literals with no internal null bytes.
            // - The compiler guarantees such strings contain no '\0' before the terminator.
            // - These calls are on the main thread, outside any RT hotpath or FFI audio.
            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED)
            {
                let msg = CString::new("NAM-rs: Output clipping detected!").unwrap_or_default();
                log.log(&shared, LogSeverity::Warning, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW)
            {
                let msg = CString::new("NAM-rs: GC channel overflow! Possible memory leak.")
                    .unwrap_or_default();
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED)
            {
                let msg = CString::new("NAM-rs: Critical failure! No active model for processing.")
                    .unwrap_or_default();
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HEAP_ALLOC)
            {
                let msg = CString::new(
                    "NAM-rs: Heap allocation detected in audio thread during process()!",
                )
                .unwrap_or_default();
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HUGEPAGE_OK)
            {
                let msg = CString::new(
                    "NAM-rs: Huge pages (2MB) active — reduced TLB pressure on DSP thread.",
                )
                .unwrap_or_default();
                log.log(&shared, LogSeverity::Info, &msg);
            }
        }

        // 2. Latency Monitoring: Notify the host if the value changed
        let current_latency = self.shared.current_latency.load(Ordering::Relaxed);
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

impl<'a> NamClapMainThread<'a> {
    /// Loads a new neural model from the specified path.
    ///
    /// This method performs I/O and memory allocations, being safe to execute
    /// only on the main thread. The loaded model is sent to the RT thread
    /// via a lock-free channel.
    pub fn load_model(&mut self, path: &Path) -> Result<(), Box<NamDiagnostic>> {
        // 1. Loading and building the model pair (L+R)
        let model_pair = load_and_build_model(path, &self.sys).map_err(|e| {
            Box::new(
                NamDiagnostic::new(NamErrorCode::ModelBuildFailed, &self.sys)
                    .message(format!("Failed to load model: {:?}", path))
                    .param("error", e.to_string()),
            )
        })?;

        // Build new resampler for host and model rates
        let host_rate = self.shared.sample_rate.load(Ordering::Relaxed);
        let host_rate = if host_rate == 0 { 48000 } else { host_rate };
        let new_resampler = Box::new(
            NamResampler::new(host_rate, model_pair.sample_rate, 0).map_err(|e| {
                Box::new(
                    NamDiagnostic::new(NamErrorCode::ModelBuildFailed, &self.sys)
                        .message("Failed to build resampler")
                        .param("error", e.to_string()),
                )
            })?,
        );

        // 3. Update path in local params (mirrored)
        self.params.model_path = Some(path.to_path_buf());
        self.params.model_basename = path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string());
        if let Some(parent) = path.parent() {
            let parent_buf = parent.to_path_buf();
            if !self.params.model_search_paths.contains(&parent_buf) {
                self.params.model_search_paths.push(parent_buf);
            }
        }

        // Extract and update model metadata for the GUI
        let metadata = model_pair.metadata.clone();
        let architecture = model_pair.architecture.clone();
        let topology = model_pair.topology.clone();
        if let Ok(mut meta_guard) = self.shared.ui_model_metadata.lock() {
            *meta_guard = Some(NamModelMetadata {
                architecture,
                topology,
                modeled_by: metadata.as_ref().and_then(|m| m.modeled_by.clone()),
                gear_make: metadata.as_ref().and_then(|m| m.gear_make.clone()),
                gear_model: metadata.as_ref().and_then(|m| m.gear_model.clone()),
                gear_type: metadata.as_ref().and_then(|m| m.gear_type.clone()),
                tone_type: metadata.as_ref().and_then(|m| m.tone_type.clone()),
                date: metadata
                    .as_ref()
                    .and_then(|m| m.date.as_ref())
                    .map(|d| match (d.year, d.month, d.day) {
                        (Some(y), Some(m), Some(d)) => format!("{:04}-{:02}-{:02}", y, m, d),
                        (Some(y), Some(m), None) => format!("{:04}-{:02}", y, m),
                        (Some(y), None, None) => format!("{:04}", y),
                        _ => String::new(),
                    })
                    .filter(|s| !s.is_empty()),
            });
        }

        let model_info = crate::common::diagnostics::ModelInfo {
            arch_label: model_pair.architecture.clone(),
            channels: model_pair
                .model_l
                .as_ref()
                .map(|m| m.channels())
                .unwrap_or(0),
            receptive_field: model_pair
                .model_l
                .as_ref()
                .map(|m| m.receptive_field())
                .unwrap_or(0),
            weights_layout: model_pair.weights_layout.clone(),
            path_basename: path.to_string_lossy().into_owned(),
        };
        if let Ok(mut info_guard) = self.shared.ui_model_info.lock() {
            *info_guard = Some(model_info);
        }

        // Update model sample rate
        self.shared
            .model_sample_rate
            .store(model_pair.sample_rate, Ordering::Relaxed);

        // 2. Send to RT thread via SPSC channel
        self.param_tx
            .push(ClapParamPayload::LoadModel(
                Box::new(model_pair),
                new_resampler,
            ))
            .map_err(|_| {
                Box::new(
                    NamDiagnostic::new(NamErrorCode::ParamChannelFull, &self.sys)
                        .message("The communication channel with the audio thread is full.")
                        .hint("Please try loading the model again in a few moments."),
                )
            })?;

        // Update the loaded model name for UI display (basename)
        let basename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if let Ok(mut name_guard) = self.shared.ui_model_name.lock() {
            *name_guard = basename;
        }
        self.shared
            .model_load_counter
            .fetch_add(1, Ordering::Relaxed);

        // Notify host that the active model parameter value changed
        if let Some(params_ext) = self
            .host
            .get_extension::<clack_extensions::params::HostParams>()
        {
            params_ext.rescan(
                &mut self.host,
                clack_extensions::params::ParamRescanFlags::VALUES,
            );
        }

        // 4. Notify host about successful loading (Cold Path)
        if let Some(log) = self.host.get_extension::<HostLog>() {
            let msg = format!("NAM-rs: model loaded ({:?})", path);
            if let Ok(c_msg) = CString::new(msg) {
                log.log(&self.host.shared(), LogSeverity::Info, &c_msg);
            }
        }

        // 5. Notify host that state changed (dirty)
        if let Some(mut state_ext) = self
            .host
            .get_extension::<clack_extensions::state::HostState>()
        {
            state_ext.mark_dirty(&self.host);
        }

        Ok(())
    }
}
