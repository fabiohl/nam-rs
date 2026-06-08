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

use super::shared::{ClapParamPayload, NamClapShared};
use crate::common::diagnostics::SystemSnapshot;
use crate::common::params::NamPluginParams;
use crate::common::spsc::GcItem;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

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
    /// Baseview window handle for GUI lifecycle control (embedded mode).
    #[cfg(feature = "clap-plugin")]
    pub window_handle: Option<baseview::WindowHandle>,
    /// Thread handle for the floating window event loop.
    #[cfg(feature = "clap-plugin")]
    pub floating_thread_handle: Option<std::thread::JoinHandle<()>>,
    /// Close signal for the floating window.
    #[cfg(feature = "clap-plugin")]
    pub floating_close_signal: Option<Arc<AtomicBool>>,
}

impl<'a> PluginMainThread<'a, NamClapShared> for NamClapMainThread<'a> {
    /// Called periodically or in response to host events.
    /// Delegates to concern-specific sub-module methods.
    fn on_main_thread(&mut self) {
        self.housekeeping();
        self.emit_pending_logs();
    }
}
