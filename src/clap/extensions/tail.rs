// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the `clap_plugin_tail` extension for NAM-rs.
//!
//! Reports the plugin's tail length (CabSim IR + oversampling/resampling latencies)
//! so DAWs can properly handle offline bounces and silence-timeout processing.

use crate::clap::processor::NamClapProcessor;
use clack_extensions::tail::{PluginTail, PluginTailImpl, TailLength};
use std::sync::atomic::Ordering;

impl PluginTailImpl for NamClapProcessor<'_> {
    /// Returns the current tail length in samples.
    ///
    /// The value is computed from the same `current_latency` shared atomic
    /// used by the latency extension — it already includes the resampler,
    /// oversampling, and CabSim convolution contributions.
    fn get(&self) -> TailLength {
        TailLength::Finite(self.shared.rt_to_ui.current_latency.load(Ordering::Relaxed))
    }
}

/// Marker type for extension registration.
pub type NamPluginTail = PluginTail;
