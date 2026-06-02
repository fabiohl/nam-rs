// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Architecture (Staging and Placeholder).
//!
//! This module isolates components of the A2 architecture (v0.6+), including
//! stubs for activations, FiLM, gating, and parameters.

pub mod activations;
pub mod film;
pub mod gating;
pub mod params;

use crate::common::spsc::RtStatusFlags;
use crate::models::NamModel;
use std::sync::Arc;

/// Public re-exports for easy access.
pub use activations::{ActivationFn, ActivationType};
pub use film::{FiLMConfig, FiLMLayer};
pub use gating::GatingMode;
pub use params::{HeadParams, LayerArrayParamsA2, LayerParamsA2};

// =============================================================================
// Placeholder for WaveNet A2 (Staging)
// =============================================================================

/// Placeholder for the WaveNet A2 architecture.
///
/// This struct allows the system to load A2 models without failing, returning
/// silence until the complete inference engine implementation is ready.
#[derive(Default)]
pub struct WavenetA2Placeholder {
    /// Flag to emit the log warning only once per instance.
    warned: bool,
    /// Shared RT status flags to signal the placeholder to the UI.
    rt_status: Option<Arc<RtStatusFlags>>,
}

impl WavenetA2Placeholder {
    /// Injects the reference to `RtStatusFlags` so the placeholder can
    /// signal its state to the UI via atomic flags.
    pub fn inject_rt_status(&mut self, rt_status: Arc<RtStatusFlags>) {
        self.rt_status = Some(rt_status);
    }
}

impl NamModel for WavenetA2Placeholder {
    fn process(&mut self, _input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "heap-audit")]
        if crate::clap::heap_audit::AUDIT_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            let audit_thread =
                crate::clap::heap_audit::AUDIT_THREAD.load(std::sync::atomic::Ordering::Relaxed);
            if audit_thread == 0 || audit_thread == tid {
                crate::clap::heap_audit::ALLOC_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        if !self.warned {
            log::warn!(
                "WaveNet A2 architecture detected: Placeholder (Silent) mode active. The real implementation is under development."
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenet_a2_placeholder_silence() {
        let mut model = WavenetA2Placeholder::default();
        let input = [1.0f32; 10];
        let mut output = [1.0f32; 10];
        model.process(&input, &mut output);
        for val in output.iter() {
            assert_eq!(*val, 0.0);
        }
    }
}
