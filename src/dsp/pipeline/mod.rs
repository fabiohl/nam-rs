// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP processing pipeline (Capture → DSP → Bridge).
//!
//! This module isolates the audio processing logic from PipeWire orchestration.
//! It contains the hot-path executed every audio cycle on the real-time thread.

#[cfg(test)]
use crate::models::NamModel;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
mod bridge;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
mod capture;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
mod context;
#[cfg(feature = "standalone")]
mod output_pw;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
mod stages;

// Re-exports — preserve the same visibility as the original pipeline.rs.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use bridge::{BridgeBuffer, BridgeRef, DspBridge, DspBridgeReader, DspBridgeWriter};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use bridge::{MAX_BRIDGE_BUF, MAX_RESAMP_BUF};

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use context::{DspBuffers, DspPipelineContext};

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
#[allow(unused_imports)]
pub(crate) use stages::DENORMAL_DITHER_OFFSET;
#[cfg(feature = "clap-plugin")]
pub(crate) use stages::apply_input_stage;
#[cfg(feature = "clap-plugin")]
pub(crate) use stages::apply_output_stage;
#[cfg(feature = "clap-plugin")]
pub(crate) use stages::run_inference;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use stages::write_bridge;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use stages::{DISABLE_GATE, handle_silence_bypass};

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use capture::capture_dsp_pipeline;

#[cfg(feature = "standalone")]
pub(crate) use output_pw::AppState;
#[cfg(feature = "standalone")]
pub use output_pw::PipewireHostConfig;
#[cfg(feature = "standalone")]
pub(crate) use output_pw::{build_spa_format_pod, playback_dsp_cycle};

#[cfg(test)]
pub(crate) mod test_util {
    pub mod infra {
        pub use crate::common::alloc_audit::{ALLOC_COUNT, CountingAllocator, TrackingGuard};
    }
}

#[cfg(test)]
#[global_allocator]
static GLOBAL: test_util::infra::CountingAllocator = test_util::infra::CountingAllocator;

#[cfg(test)]
mod pipeline_test;

#[cfg(test)]
mod pipeline_block_test;
