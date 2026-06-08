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
mod playback;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
mod stage_bridge;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
mod stage_inference;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
mod stage_input;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
mod stage_output;

// Re-exports — preserve the same visibility as the original pipeline.rs.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use bridge::{BridgeBuffer, BridgeRef, DspBridge, DspBridgeReader, DspBridgeWriter};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use bridge::{MAX_BRIDGE_BUF, MAX_RESAMP_BUF};

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use context::{DspBuffers, DspPipelineContext};

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use stage_bridge::write_bridge;
#[cfg(feature = "clap-plugin")]
pub(crate) use stage_inference::run_inference;
#[cfg(feature = "clap-plugin")]
pub(crate) use stage_input::apply_input_stage;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use stage_input::{DISABLE_GATE, handle_silence_bypass};
#[cfg(feature = "clap-plugin")]
pub(crate) use stage_output::apply_output_stage;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use capture::capture_dsp_pipeline;

#[cfg(feature = "standalone")]
pub(crate) use playback::AppState;
#[cfg(feature = "standalone")]
pub use playback::PipewireHostConfig;
#[cfg(feature = "standalone")]
pub(crate) use playback::{build_spa_format_pod, playback_dsp_cycle};

#[cfg(test)]
pub(crate) mod test_util;

#[cfg(test)]
#[global_allocator]
static GLOBAL: test_util::infra::CountingAllocator = test_util::infra::CountingAllocator;

#[cfg(test)]
mod pipeline_test;

#[cfg(test)]
mod pipeline_block_test;
