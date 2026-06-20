// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Typed error codes for precise failure triage.
//!
//! Numeric code convention:
//! - **E1xxx**: Model loading (I/O, parse, build)
//! - **E2xxx**: PipeWire / audio (init, stream, resampler, RT)
//! - **E3xxx**: SPSC / inter-thread communication
//! - **E4xxx**: Runtime / CLI (parsing, commands)
//! - **E5xxx**: System / hardware (CPU features, memory)

use std::fmt;

/// Structured error code for precise failure triage.
///
/// Each variant maps to exactly one failure scenario. The numeric code
/// (used in the support block) follows the convention:
/// - **E1xxx**: Model loading (I/O, parse, build)
/// - **E2xxx**: PipeWire / audio (init, stream, resampler, RT)
/// - **E3xxx**: SPSC / inter-thread communication
/// - **E4xxx**: Runtime / CLI (parsing, commands)
/// - **E5xxx**: System / hardware (CPU features, memory)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamErrorCode {
    // E1xxx — Model Loading
    /// Model file not found on the filesystem.
    FileNotFound,
    /// I/O read error while accessing the model file.
    FileReadError,
    /// Unknown file extension (expected .nam or .namb).
    UnknownExtension,
    /// Failed to parse JSON in .nam format.
    NamJsonParseError,
    /// JSON `weights` array exceeds float limit (MAX_WEIGHTS).
    NamJsonWeightsExceedLimit,
    /// JSON `metadata.training` field exceeds size limit (1 MiB).
    NamJsonTrainingTooLarge,
    /// JSON `metadata.training` field exceeds tree depth limit.
    NamJsonTrainingTooDeep,
    /// JSON `config.submodels` array exceeds the maximum count (8).
    NamJsonSubmodelsExceedLimit,
    /// JSON `config.submodels` exceeds container recursion depth limit.
    NamJsonSubmodelsTooDeep,
    /// .namb CRC32 checksum does not match expected value.
    NambCrc32Mismatch,
    /// CRC32 flag missing in .namb v2+ file (mandatory).
    NambCrc32Missing,
    /// Invalid .namb magic signature.
    NambInvalidMagic,
    /// Unsupported .namb format version.
    NambUnsupportedVersion,
    /// Truncated .namb file (smaller than the minimum header).
    NambTruncated,
    /// Declared neural architecture is not supported (neither WaveNet nor LSTM).
    UnsupportedArchitecture,
    /// WaveNet/LSTM topology not detectable from the configuration.
    TopologyDetectionFailed,
    /// Weight count inconsistent with network geometry.
    WeightCountMismatch,
    /// Generic failure building the model via the dispatcher.
    ModelBuildFailed,
    /// Model file exceeds the maximum allowed size (256 MiB).
    ModelTooLarge,

    // E2xxx — PipeWire / Audio
    /// Failed to initialize PipeWire or create context/core.
    PipewireInitFailed,
    /// Failed to connect the PipeWire stream.
    StreamConnectFailed,
    /// Failed to build the NamResampler (native Polyphase Sinc).
    ResamplerBuildFailed,
    /// SPSC resampler channel full (rebuild discarded).
    ResamplerChannelFull,
    /// Permission denied for SCHED_FIFO (no RT privileges).
    SchedFifoDenied,
    /// Failed to set CPU affinity on the DSP thread.
    CpuAffinityFailed,
    /// PipeWire deadline exceeded (DSP processing exceeded budget).
    DeadlineExceeded,

    // E3xxx — SPSC / Communication
    /// CLI→DSP parameter SPSC channel full.
    ParamChannelFull,

    // E4xxx — Runtime / CLI
    /// Invalid gain value (not a valid f32 number).
    InvalidGainValue,
    /// Unknown CLI command.
    UnknownCommand,
    /// Failed to configure the Ctrl-C handler.
    CtrlCHandlerFailed,
    /// Failed to load a cab-sim impulse response (WAV file).
    IrLoadFailed,
    /// Overflow detected in the Garbage Collection (GC) channel.
    GcOverflow,
    /// Corrupted GC overflow buffer slot (inconsistent type/pointer).
    GcCorrupted,
}

impl NamErrorCode {
    /// Returns the numeric code in "Exxxx" format.
    pub fn code(self) -> &'static str {
        match self {
            Self::FileNotFound => "E1100",
            Self::FileReadError => "E1101",
            Self::UnknownExtension => "E1102",
            Self::NamJsonParseError => "E1200",
            Self::NamJsonWeightsExceedLimit => "E1206",
            Self::NamJsonTrainingTooLarge => "E1207",
            Self::NamJsonTrainingTooDeep => "E1208",
            Self::NamJsonSubmodelsExceedLimit => "E1209",
            Self::NamJsonSubmodelsTooDeep => "E1210",
            Self::NambCrc32Mismatch => "E1201",
            Self::NambCrc32Missing => "E1205",
            Self::NambInvalidMagic => "E1202",
            Self::NambUnsupportedVersion => "E1203",
            Self::NambTruncated => "E1204",
            Self::UnsupportedArchitecture => "E1300",
            Self::TopologyDetectionFailed => "E1301",
            Self::WeightCountMismatch => "E1302",
            Self::ModelBuildFailed => "E1303",
            Self::ModelTooLarge => "E1304",
            Self::PipewireInitFailed => "E2100",
            Self::StreamConnectFailed => "E2101",
            Self::ResamplerBuildFailed => "E2200",
            Self::ResamplerChannelFull => "E2201",
            Self::SchedFifoDenied => "E2300",
            Self::CpuAffinityFailed => "E2301",
            Self::DeadlineExceeded => "E2001",
            Self::ParamChannelFull => "E3100",
            Self::GcOverflow => "E3101",
            Self::GcCorrupted => "E3102",
            Self::InvalidGainValue => "E4100",
            Self::UnknownCommand => "E4101",
            Self::CtrlCHandlerFailed => "E4102",
            Self::IrLoadFailed => "E4103",
        }
    }

    /// Returns a human-readable message for GUI display.
    pub fn message(self) -> &'static str {
        match self {
            Self::FileNotFound => "File not found",
            Self::FileReadError => "File read error",
            Self::UnknownExtension => "Unknown extension",
            Self::NamJsonParseError => "Invalid JSON format",
            Self::NamJsonWeightsExceedLimit => "JSON weights exceed limit",
            Self::NamJsonTrainingTooLarge => "Training metadata too large",
            Self::NamJsonTrainingTooDeep => "Training metadata too deeply nested",
            Self::NamJsonSubmodelsExceedLimit => "Submodels count exceeds limit (max 8)",
            Self::NamJsonSubmodelsTooDeep => "Container nesting too deep",
            Self::NambCrc32Mismatch => "CRC32 checksum mismatch",
            Self::NambCrc32Missing => "CRC32 integrity flag missing (v2+)",
            Self::NambInvalidMagic => "Invalid signature",
            Self::NambUnsupportedVersion => "Unsupported version",
            Self::NambTruncated => "Corrupted/truncated file",
            Self::UnsupportedArchitecture => "Unsupported architecture",
            Self::TopologyDetectionFailed => "Topology detection failed",
            Self::WeightCountMismatch => "Weight count mismatch",
            Self::ModelBuildFailed => "Model build failed",
            Self::ModelTooLarge => "Model file too large",
            Self::PipewireInitFailed => "PipeWire initialization failed",
            Self::StreamConnectFailed => "Stream connection failed",
            Self::ResamplerBuildFailed => "Resampler build failed",
            Self::ResamplerChannelFull => "Resampler channel full",
            Self::SchedFifoDenied => "SCHED_FIFO denied",
            Self::CpuAffinityFailed => "CPU affinity setting failed",
            Self::DeadlineExceeded => "Processing deadline exceeded",
            Self::ParamChannelFull => "Parameter channel full",
            Self::GcOverflow => "Garbage collection overflow",
            Self::GcCorrupted => "Garbage collection corruption",
            Self::InvalidGainValue => "Invalid gain value",
            Self::UnknownCommand => "Unknown command",
            Self::CtrlCHandlerFailed => "Ctrl-C handler setup failed",
            Self::IrLoadFailed => "Cab-sim IR load failed",
        }
    }

    /// Returns the human-readable mnemonic name (SCREAMING_SNAKE_CASE) of the code.
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::FileNotFound => "FILE_NOT_FOUND",
            Self::FileReadError => "FILE_READ_ERROR",
            Self::UnknownExtension => "UNKNOWN_EXTENSION",
            Self::NamJsonParseError => "NAM_JSON_PARSE_ERROR",
            Self::NamJsonWeightsExceedLimit => "NAM_JSON_WEIGHTS_EXCEED_LIMIT",
            Self::NamJsonTrainingTooLarge => "NAM_JSON_TRAINING_TOO_LARGE",
            Self::NamJsonTrainingTooDeep => "NAM_JSON_TRAINING_TOO_DEEP",
            Self::NamJsonSubmodelsExceedLimit => "NAM_JSON_SUBMODELS_EXCEED_LIMIT",
            Self::NamJsonSubmodelsTooDeep => "NAM_JSON_SUBMODELS_TOO_DEEP",
            Self::NambCrc32Mismatch => "NAMB_CRC32_MISMATCH",
            Self::NambCrc32Missing => "NAMB_CRC32_MISSING",
            Self::NambInvalidMagic => "NAMB_INVALID_MAGIC",
            Self::NambUnsupportedVersion => "NAMB_UNSUPPORTED_VERSION",
            Self::NambTruncated => "NAMB_TRUNCATED",
            Self::UnsupportedArchitecture => "UNSUPPORTED_ARCHITECTURE",
            Self::TopologyDetectionFailed => "TOPOLOGY_DETECTION_FAILED",
            Self::WeightCountMismatch => "WEIGHT_COUNT_MISMATCH",
            Self::ModelBuildFailed => "MODEL_BUILD_FAILED",
            Self::ModelTooLarge => "MODEL_TOO_LARGE",
            Self::PipewireInitFailed => "PIPEWIRE_INIT_FAILED",
            Self::StreamConnectFailed => "STREAM_CONNECT_FAILED",
            Self::ResamplerBuildFailed => "RESAMPLER_BUILD_FAILED",
            Self::ResamplerChannelFull => "RESAMPLER_CHANNEL_FULL",
            Self::SchedFifoDenied => "SCHED_FIFO_DENIED",
            Self::CpuAffinityFailed => "CPU_AFFINITY_FAILED",
            Self::DeadlineExceeded => "DEADLINE_EXCEEDED",
            Self::ParamChannelFull => "PARAM_CHANNEL_FULL",
            Self::GcOverflow => "GC_OVERFLOW",
            Self::GcCorrupted => "GC_CORRUPTED",
            Self::InvalidGainValue => "INVALID_GAIN_VALUE",
            Self::UnknownCommand => "UNKNOWN_COMMAND",
            Self::CtrlCHandlerFailed => "CTRL_C_HANDLER_FAILED",
            Self::IrLoadFailed => "IR_LOAD_FAILED",
        }
    }
}

impl fmt::Display for NamErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} | {}", self.code(), self.mnemonic())
    }
}
