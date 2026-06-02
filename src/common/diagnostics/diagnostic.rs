// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured diagnostic for NAM-rs errors and warnings.
//!
//! Combines a user-friendly message with a copyable technical
//! support block. Formatted for precise triage.

use super::error_codes::NamErrorCode;
use super::system_info::SystemSnapshot;
use std::fmt;
use std::time::SystemTime;

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
        let separator = "────────────────────────────────────────────────";
        let mut block = format!(
            "──── NAM-rs Diagnostic {separator}\n\
             nam-rs v{} | {} | {}\n",
            self.system.version,
            self.code.code(),
            self.code.mnemonic(),
        );

        // Contextual parameters
        if !self.params.is_empty() {
            let param_line: String = self
                .params
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ");
            block.push_str(&param_line);
            block.push('\n');
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
            Self::timestamp(),
        ));

        block
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
