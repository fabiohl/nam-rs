// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Placeholder for the WaveNet A2 architecture.
//!
//! This module provides `WavenetA2Placeholder`, which allows the system to load
//! A2 models without failing, returning silence until the complete inference
//! engine implementation is ready.

use crate::common::spsc::RtStatusFlags;
use crate::models::NamModel;
use crate::models::sealed;
use std::sync::Arc;

/// Placeholder for the WaveNet A2 architecture.
///
/// This struct allows the system to load A2 models without failing, returning
/// silence until the complete inference engine implementation is ready.
///
/// Stores the channel count reported by the model metadata so the placeholder
/// can signal whether it detected A2 nano (3) or A2 standard (8) architecture.
#[derive(Default)]
pub struct WavenetA2Placeholder {
    /// Number of channels detected from the model topology.
    pub channels: u8,
    /// Flag to emit the log warning only once per instance.
    warned: bool,
    /// Shared RT status flags to signal the placeholder to the UI.
    rt_status: Option<Arc<RtStatusFlags>>,
}

impl WavenetA2Placeholder {
    /// Creates a new placeholder storing the detected channel count.
    ///
    /// The channel count is informational — the placeholder outputs silence
    /// regardless — but allows callers and tests to inspect the architecture
    /// that was detected during loading.
    pub fn new(channels: u8) -> Self {
        Self {
            channels,
            warned: false,
            rt_status: None,
        }
    }

    /// Injects the reference to `RtStatusFlags` so the placeholder can
    /// signal its state to the UI via atomic flags.
    pub fn inject_rt_status(&mut self, rt_status: Arc<RtStatusFlags>) {
        self.rt_status = Some(rt_status);
    }
}

impl sealed::Sealed for WavenetA2Placeholder {}

impl NamModel for WavenetA2Placeholder {
    fn process(&mut self, _input: &[f32], output: &mut [f32]) {
        #[cfg(all(feature = "heap-audit", feature = "clap-plugin"))]
        if crate::common::alloc_audit::AUDIT_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            let audit_thread =
                crate::common::alloc_audit::AUDIT_THREAD.load(std::sync::atomic::Ordering::Relaxed);
            if audit_thread == 0 || audit_thread == tid {
                crate::common::alloc_audit::ALLOC_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        if !self.warned {
            log::warn!(
                "WaveNet A2 (channels={}) architecture detected: Placeholder (Silent) mode active. The real implementation is under development.",
                self.channels
            );
            self.warned = true;
            if let Some(ref rt) = self.rt_status {
                rt.set_flag(crate::common::spsc::RT_STATUS_A2_PLACEHOLDER);
            }
        }

        // Return absolute silence.
        output.fill(0.0);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        // No-op for the placeholder.
    }
}
