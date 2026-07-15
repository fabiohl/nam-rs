// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Main thread exclusive state (model loading, state save/load).
//!
//! Split into concern-specific sub-modules:
//! - `housekeeping` — GC drain, status flags, hugepage sync, pending model load, latency
//! - `logging` — RT→main transient event logging via atomic flags
//! - `load` — model loading + error_code mapping

mod housekeeping;
mod load;
mod logging;

use super::shared::{ClapParamPayload, NamClapShared, PendingModel};
use crate::common::diagnostics::SystemSnapshot;
use crate::common::params::NamPluginParams;
use crate::common::spsc::{self, GcItem};
use crate::models::{NamModel, StaticModel};
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    /// Producer to send slimmable-rebuilt models to the audio thread.
    pub slimmable_tx: Producer<Option<Box<StaticModel>>>,
    /// Cached last latency reported to the host to avoid redundant notifications.
    pub last_reported_latency: u32,
    /// Baseview window handle for GUI lifecycle control (embedded mode).
    #[cfg(feature = "clap-plugin")]
    pub window_handle: Option<baseview::WindowHandle>,
    /// Thread handle for the floating window event loop.
    #[cfg(feature = "clap-plugin")]
    pub floating_thread_handle: Option<std::thread::JoinHandle<()>>,
    /// Close signal for the floating window.
    #[cfg(feature = "clap-plugin")]
    pub floating_close_signal: Option<Arc<AtomicBool>>,
    /// Handle da thread de file-dialog de modelo (se ativa). Joinado no teardown.
    #[cfg(feature = "clap-plugin")]
    #[allow(dead_code)]
    pub(crate) dialog_handle: Option<std::thread::JoinHandle<()>>,
    /// Estado compartilhado com a thread de file-dialog de modelo.
    #[cfg(feature = "clap-plugin")]
    #[allow(dead_code)]
    pub(crate) dialog_state:
        Option<Arc<crate::clap::gui::ui::zones::dialog_state::DialogSharedState>>,
    /// Handle da thread de file-dialog de IR (se ativa). Joinado no teardown.
    #[cfg(feature = "clap-plugin")]
    #[allow(dead_code)]
    pub(crate) ir_dialog_handle: Option<std::thread::JoinHandle<()>>,
    /// Estado compartilhado com a thread de file-dialog de IR.
    #[cfg(feature = "clap-plugin")]
    #[allow(dead_code)]
    pub(crate) ir_dialog_state:
        Option<Arc<crate::clap::gui::ui::zones::dialog_state::IrDialogSharedState>>,
}

impl<'a> NamClapMainThread<'a> {
    /// Flushes any model deferred by `load_model()` when `buffer_size == 0`
    /// (state-restore-before-activate scenario). Pre-sizes the model on the
    /// main thread and sends it to the audio thread via SPSC.
    ///
    /// F3 fix: this ensures heap alloc + mmap/munmap/memfd_create + drop
    /// never happen on the audio thread.
    pub fn flush_pending_model(&mut self) -> Result<(), PluginError> {
        let pending = if let Ok(mut guard) = self.shared.cold.pending_model.lock() {
            guard.take()
        } else {
            return Ok(());
        };
        let Some(p) = pending else { return Ok(()) };

        let buffer_size = self.shared.cold.buffer_size.load(Ordering::Relaxed) as usize;
        if buffer_size == 0 {
            // Still no buffer_size — store model back for later retry.
            if let Ok(mut guard) = self.shared.cold.pending_model.lock() {
                *guard = Some(p);
            }
            return Ok(());
        }

        let PendingModel {
            model: mut model_l,
            resampler: new_resampler,
            input_mult_adj,
            output_mult_adj,
        } = p;

        if let Some(ref mut model) = model_l
            && let Err(e) = model.set_max_buffer_size(buffer_size)
        {
            self.shared
                .cold
                .rt_status
                .set_flag(spsc::RT_STATUS_MODEL_LOAD_FAILED);
            return Err(PluginError::Message(Box::leak(
                format!(
                    "Failed to resize deferred model buffers for host buffer size ({}): {}",
                    buffer_size, e
                )
                .into_boxed_str(),
            )));
        }

        match self.param_tx.push(ClapParamPayload::LoadModel {
            model_l,
            new_resampler,
            input_mult_adj,
            output_mult_adj,
        }) {
            Ok(()) => Ok(()),
            Err(_) => {
                self.shared
                    .cold
                    .rt_status
                    .set_flag(spsc::RT_STATUS_MODEL_LOAD_FAILED);
                Err(PluginError::Message(
                    "Failed to send deferred model: SPSC channel full",
                ))
            }
        }
    }
    /// Synchronises atomic values from the audio thread and cold state into `self.params`
    /// before serialisation. Shared by `state.rs` and `state_context.rs`.
    pub(crate) fn snapshot_params(&mut self) {
        self.params.input_gain_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_input_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.output_gain_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_output_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.gate_threshold_db = f32::from_bits(
            self.shared
                .ui_to_rt
                .param_gate_thresh
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.bypass = super::super::extensions::params::bypass_u32_to_bool(
            self.shared
                .ui_to_rt
                .param_bypass
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.adaptive_compute = crate::common::params::AdaptiveComputeMode::from_f32(
            self.shared
                .ui_to_rt
                .param_adaptive_compute
                .load(std::sync::atomic::Ordering::Relaxed) as f32,
        );
        self.params.slim_override = crate::dsp::adaptive::SlimOverride::from_f32(
            self.shared
                .ui_to_rt
                .param_slim_override
                .load(std::sync::atomic::Ordering::Relaxed) as f32,
        );
        self.params.oversample = crate::dsp::oversample::OversampleFactor::from_f32(
            self.shared
                .ui_to_rt
                .param_oversample
                .load(std::sync::atomic::Ordering::Relaxed) as f32,
        );
        self.params.activation_precision = crate::common::params::ActivationPrecision::from_f32(
            self.shared
                .ui_to_rt
                .param_activation
                .load(std::sync::atomic::Ordering::Relaxed) as f32,
        );
        #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
        {
            if let Ok(ir_guard) = self.shared.cold.ir_path.lock() {
                self.params.ir_path = ir_guard.as_ref().map(std::path::PathBuf::from);
            }
        }
    }
}

impl<'a> PluginMainThread<'a, NamClapShared> for NamClapMainThread<'a> {
    /// Called periodically or in response to host events.
    /// Delegates to concern-specific sub-module methods.
    fn on_main_thread(&mut self) {
        self.housekeeping();
        self.emit_pending_logs();
    }
}
