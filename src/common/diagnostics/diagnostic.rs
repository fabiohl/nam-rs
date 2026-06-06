// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured diagnostic for NAM-rs errors and warnings.
//!
//! Combines a user-friendly message with a copyable technical
//! support block. Formatted for precise triage.

use super::error_codes::NamErrorCode;
use super::system_info::SystemSnapshot;
use std::fmt;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::SystemTime;

/// Thread-safe global storage for the currently active sample rate (populated in active standalone session).
pub static ACTIVE_SAMPLE_RATE: AtomicU32 = AtomicU32::new(0);

/// Thread-safe global storage for the currently active model path/name (populated in active standalone session).
pub static ACTIVE_MODEL_NAME: RwLock<String> = RwLock::new(String::new());

/// Structured diagnostic for NAM-rs errors and warnings.
///
/// Combines a user-friendly message with a copyable technical
/// support block. Formatted for precise triage.
#[derive(Debug)]
pub struct NamDiagnostic {
    /// Typed error code.
    code: NamErrorCode,
    /// User-friendly message (en-US).
    user_message: String,
    /// Suggested action for the user (en-US).
    user_hint: String,
    /// Contextual parameters for the support block (key=value).
    params: Vec<(&'static str, String)>,
    /// System snapshot (captured at startup).
    system: SystemSnapshot,
}

impl NamDiagnostic {
    /// Creates a new diagnostic with the error code and system snapshot.
    #[cold]
    pub fn new(code: NamErrorCode, system: &SystemSnapshot) -> Self {
        Self {
            code,
            user_message: String::new(),
            user_hint: String::new(),
            params: Vec::new(),
            system: system.clone(),
        }
    }

    /// Sets the user-friendly message.
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.user_message = msg.into();
        self
    }

    /// Sets the suggested action for the user.
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.user_hint = hint.into();
        self
    }

    /// Adds a contextual parameter to the support block.
    pub fn param(mut self, key: &'static str, value: impl fmt::Display) -> Self {
        self.params.push((key, value.to_string()));
        self
    }

    /// Generates the ISO 8601 timestamp for the current moment.
    pub(crate) fn timestamp() -> String {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();

        // Manual ISO 8601 formatting without external dependencies
        let secs = now.as_secs();
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Date calculation from days since epoch (1970-01-01)
        let (year, month, day) = days_to_date(days);

        format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
    }

    /// Generates the formatted technical support block for copying.
    pub(crate) fn support_block(&self) -> String {
        DiagnosticBundle {
            system: self.system.clone(),
            runtime: RuntimeSnapshot,
            error: Some(ErrorContext {
                code: self.code,
                params: self.params.clone(),
            }),
            full: false,
        }
        .render()
    }

    /// Prints the complete diagnostic to stderr (user-friendly message + support block).
    pub fn emit(&self) {
        log::error!(
            "{}\n\n{}\n\n{}",
            self.user_message,
            if self.user_hint.is_empty() {
                ""
            } else {
                &self.user_hint
            },
            self.support_block()
        );
    }

    /// Prints as a warning (non-fatal) with a distinct visual prefix.
    pub fn emit_warning(&self) {
        log::warn!(
            "{}\n\n{}\n\n{}",
            self.user_message,
            if self.user_hint.is_empty() {
                ""
            } else {
                &self.user_hint
            },
            self.support_block()
        );
    }

    /// Returns the error code associated with the diagnostic.
    pub fn error_code(&self) -> NamErrorCode {
        self.code
    }
}

impl fmt::Display for NamDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code.code(), self.user_message)
    }
}

impl std::error::Error for NamDiagnostic {}

/// Converts days since 1970-01-01 to Gregorian (year, month, day).
///
/// Minimal implementation of Howard Hinnant's civil algorithm,
/// without external dependencies.
pub(crate) fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Algoritmo civil: https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

/// Context of the error for the diagnostic bundle.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Typed error code.
    pub code: NamErrorCode,
    /// Contextual parameters (key=value).
    pub params: Vec<(&'static str, String)>,
}

/// Snapshot of the dynamic runtime state, captured on-demand.
///
/// Placeholder for this task (empty struct, implementing Default, Clone, Debug).
#[derive(Debug, Clone, Default)]
pub struct RuntimeSnapshot;

/// Decoupled diagnostic bundle containing system information, runtime state, and optional error.
#[derive(Debug, Clone)]
pub struct DiagnosticBundle {
    /// System snapshot (OS, kernel, etc).
    pub system: SystemSnapshot,
    /// Runtime snapshot (buffers, samplerate, etc).
    pub runtime: RuntimeSnapshot,
    /// Error context if captured due to an error.
    pub error: Option<ErrorContext>,
    /// Whether to print absolute paths unredacted.
    pub full: bool,
}

/// Helper function to format model paths.
/// Default (not full): only prints file basename.
/// Full: prints the raw absolute path.
fn format_model_path(path_str: &str, full: bool) -> String {
    if path_str.is_empty() {
        return String::new();
    }
    if full {
        path_str.to_string()
    } else {
        let path = std::path::Path::new(path_str);
        if let Some(file_name) = path.file_name() {
            file_name.to_string_lossy().into_owned()
        } else {
            path_str.to_string()
        }
    }
}

impl DiagnosticBundle {
    /// Captures a nominal diagnostic bundle without error context.
    pub fn capture() -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot,
            error: None,
            full: false,
        }
    }

    /// Captures a diagnostic bundle with the given error code and parameters.
    pub fn capture_with_error(code: NamErrorCode, params: Vec<(&'static str, String)>) -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot,
            error: Some(ErrorContext { code, params }),
            full: false,
        }
    }

    /// Builder method to specify whether to use full (unredacted) paths.
    pub fn with_full(mut self, full: bool) -> Self {
        self.full = full;
        self
    }

    /// Renders the support block as a formatted string.
    pub fn render(&self) -> String {
        let separator = "────────────────────────────────────────────────";
        let mut block = if let Some(ref err) = self.error {
            format!(
                "──── NAM-rs Diagnostic {separator}\n\
                 nam-rs v{} | {} | {}\n",
                self.system.version,
                err.code.code(),
                err.code.mnemonic(),
            )
        } else {
            format!(
                "──── NAM-rs Diagnostic {separator}\n\
                 nam-rs v{}\n",
                self.system.version,
            )
        };

        // Contextual parameters
        if let Some(err) = self.error.as_ref().filter(|e| !e.params.is_empty()) {
            let param_line: String = err
                .params
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ");
            block.push_str(&param_line);
            block.push('\n');
        }

        // Runtime State
        let model_name = ACTIVE_MODEL_NAME
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let active_rate = ACTIVE_SAMPLE_RATE.load(Ordering::Relaxed);

        if !model_name.is_empty() || active_rate != 0 {
            block.push_str("──── Runtime State ─────────────────────────────\n");
            if !model_name.is_empty() {
                let formatted = format_model_path(&model_name, self.full);
                block.push_str(&format!("model={}\n", formatted));
            }
            if active_rate != 0 {
                block.push_str(&format!("sample_rate={}\n", active_rate));
            }
        }

        // System information
        block.push_str(&format!(
            "arch={}\n\
             os={} kernel={}\n\
             pipewire={}\n\
             features={}\n\
             timestamp={}\n\
             {separator}\n\
             Copy the block above when opening a support ticket.",
            self.system.arch,
            self.system.os,
            self.system.kernel,
            self.system.pipewire_version.as_deref().unwrap_or("N/A"),
            if self.system.features.is_empty() {
                "none (baseline x86-64-v3 only)".to_string()
            } else {
                self.system.features.join(", ")
            },
            NamDiagnostic::timestamp(),
        ));

        block
    }
}
