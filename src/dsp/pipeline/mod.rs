// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pipeline de processamento DSP (Capture → DSP → Bridge).
//!
//! Este módulo isola a lógica de processamento de áudio da orquestração do PipeWire.
//! Ele contém o "hot-path" que é executado a cada ciclo de áudio na thread de tempo real.

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
mod stages;

// Re-exports — preservam a mesma visibilidade do pipeline.rs original.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use bridge::{BridgeBuffer, BridgeRef, DspBridge, DspBridgeReader, DspBridgeWriter};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use bridge::{MAX_BRIDGE_BUF, MAX_RESAMP_BUF};

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use context::{DspBuffers, DspPipelineContext};

#[cfg(feature = "clap-plugin")]
pub(crate) use stages::{apply_input_stage, apply_output_stage, run_inference};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub use stages::{handle_silence_bypass, write_bridge};

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
