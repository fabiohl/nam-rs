// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured diagnostics system for NAM-rs.
//!
//! Provides error messages in two layers:
//! 1. **User-friendly message** — readable text for the end user.
//! 2. **Support block** — typed error code, contextual parameters,
//!    version, architecture, and timestamp for precise triage by devs/AI.
//!
//! ## Usage outside the RT thread
//!
//! All formatting and printing of diagnostics occurs **exclusively** in
//! non-RT threads (CLI, PipeWire main loop). The `process()` callback
//! continues using atomic flags (`RtStatusFlags`) for silent signaling.

pub mod diagnostic;
pub mod error_codes;
pub mod system_info;

#[cfg(test)]
pub(crate) use diagnostic::days_to_date;
pub use diagnostic::{
    ACTIVE_MODEL_INFO, ACTIVE_MODEL_NAME, ACTIVE_SAMPLE_RATE, AudioInfo, DiagnosticBundle,
    ErrorContext, HasRuntimeSnapshot, ModelInfo, NamDiagnostic, RtInfo, RuntimeSnapshot,
    TelemetrySnapshot,
};
pub use error_codes::NamErrorCode;
pub use system_info::SystemSnapshot;

#[cfg(test)]
#[path = "../diagnostics_test.rs"]
mod diagnostics_test;
