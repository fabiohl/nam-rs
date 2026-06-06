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

/// Thread-safe global storage for the currently active model snapshot info.
pub static ACTIVE_MODEL_INFO: RwLock<Option<ModelInfo>> = RwLock::new(None);

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
            runtime: RuntimeSnapshot::default(),
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

/// Detailed model information for runtime diagnostic snapshot.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Neural network architecture type (e.g. "WaveNet" or "LSTM").
    pub arch_label: String,
    /// Number of channels inside the model.
    pub channels: usize,
    /// Receptive field size of the model.
    pub receptive_field: usize,
    /// Weights layout format (e.g. "Original", "GateMajorLstm", "Interleaved4WaveNet").
    pub weights_layout: String,
    /// Basename of the model file.
    pub path_basename: String,
}

/// Detailed audio configuration for runtime diagnostic snapshot.
#[derive(Debug, Clone)]
pub struct AudioInfo {
    /// Active sample rate in Hz.
    pub sample_rate: u32,
    /// Buffer size in frames.
    pub buffer_size: usize,
    /// Active channel count.
    pub channel_count: usize,
    /// Audio host name ("CLAP" or "PipeWire").
    pub host_name: String,
}

impl Default for AudioInfo {
    fn default() -> Self {
        Self {
            sample_rate: 0,
            buffer_size: 0,
            channel_count: 0,
            host_name: "N/A".to_string(),
        }
    }
}

/// Real-time thread scheduling and affinity information.
#[derive(Debug, Clone)]
pub struct RtInfo {
    /// Thread priority.
    pub thread_priority: i32,
    /// Scheduler policy ("FIFO", "DEADLINE", "OTHER", or "UNKNOWN").
    pub scheduler: String,
    /// CPU core index the thread is pinned to, if any.
    pub cpu_pinned: Option<usize>,
    /// Whether transparent huge pages / hugetlb pages are active.
    pub huge_pages_active: bool,
}

impl Default for RtInfo {
    fn default() -> Self {
        Self {
            thread_priority: -1,
            scheduler: "UNKNOWN".to_string(),
            cpu_pinned: None,
            huge_pages_active: false,
        }
    }
}

/// Snapshot of the dynamic latency telemetry.
#[derive(Debug, Clone, Default)]
pub struct TelemetrySnapshot {
    /// Median latency in microseconds.
    pub p50_us: u64,
    /// 99th percentile latency in microseconds.
    pub p99_us: u64,
    /// 99.9th percentile latency in microseconds.
    pub p999_us: u64,
    /// Maximum latency peak in microseconds.
    pub max_us: u64,
    /// Total processed blocks.
    pub total_blocks: u64,
    /// Total virtual XRUNs/overloads.
    pub xruns: u32,
    /// Total count of GC items successfully drained.
    pub drains: u32,
}

/// Snapshot of the dynamic runtime state, captured on-demand.
#[derive(Debug, Clone, Default)]
pub struct RuntimeSnapshot {
    /// Active model info, if loaded.
    pub model: Option<ModelInfo>,
    /// Audio configuration.
    pub audio: AudioInfo,
    /// Real-time scheduling info.
    pub rt: RtInfo,
    /// Telemetry metrics.
    pub telemetry: TelemetrySnapshot,
    /// Accumulated OR of all RT_STATUS_* flags ever seen since startup.
    pub flags_seen: u64,
}

impl RuntimeSnapshot {
    /// Captures a runtime snapshot from the given provider.
    pub fn capture(provider: &impl HasRuntimeSnapshot) -> Self {
        Self {
            model: provider.model_info(),
            audio: provider.audio_info(),
            rt: provider.rt_info(),
            telemetry: provider.telemetry_snapshot(),
            flags_seen: provider.flags_seen(),
        }
    }
}

/// Trait for querying dynamic runtime snapshot data.
pub trait HasRuntimeSnapshot {
    /// Returns model snapshot info, if loaded.
    fn model_info(&self) -> Option<ModelInfo>;
    /// Returns current audio snapshot configuration.
    fn audio_info(&self) -> AudioInfo;
    /// Returns real-time and scheduler snapshot info.
    fn rt_info(&self) -> RtInfo;
    /// Returns telemetry snapshot data.
    fn telemetry_snapshot(&self) -> TelemetrySnapshot;
    /// Returns OR-accumulated flags seen since startup.
    fn flags_seen(&self) -> u64;
}

impl HasRuntimeSnapshot for crate::common::spsc::RtStatusFlags {
    fn model_info(&self) -> Option<ModelInfo> {
        if let Ok(info_guard) = ACTIVE_MODEL_INFO.read() {
            info_guard.clone()
        } else {
            None
        }
    }

    fn audio_info(&self) -> AudioInfo {
        let sr = self.active_rate.load(Ordering::Relaxed);
        let sr = if sr == 0 {
            ACTIVE_SAMPLE_RATE.load(Ordering::Relaxed)
        } else {
            sr
        };
        let buffer_size = self.last_n_samples.load(Ordering::Relaxed) as usize;
        AudioInfo {
            sample_rate: sr,
            buffer_size,
            channel_count: 2,
            host_name: "PipeWire".to_string(),
        }
    }

    fn rt_info(&self) -> RtInfo {
        let thread_priority = self.confirmed_priority.load(Ordering::Relaxed);
        let scheduler_val = self.rt_policy.load(Ordering::Relaxed);
        let scheduler = if scheduler_val == -1 {
            "UNKNOWN".to_string()
        } else if scheduler_val == libc::SCHED_FIFO || scheduler_val == libc::SCHED_RR {
            "FIFO".to_string()
        } else if scheduler_val == 6 {
            // SCHED_DEADLINE
            "DEADLINE".to_string()
        } else {
            "OTHER".to_string()
        };

        let cpu = self.rt_cpu.load(Ordering::Relaxed);
        let cpu_pinned = if cpu >= 0 { Some(cpu as usize) } else { None };
        let huge_pages_active = self.check_flag(crate::common::spsc::RT_STATUS_HUGEPAGE_OK)
            || (self.flags_seen.load(Ordering::Relaxed)
                & crate::common::spsc::RT_STATUS_HUGEPAGE_OK)
                != 0;

        RtInfo {
            thread_priority,
            scheduler,
            cpu_pinned,
            huge_pages_active,
        }
    }

    fn telemetry_snapshot(&self) -> TelemetrySnapshot {
        let p50_us = self.latency_hist.get_percentile(0.50) / 1000;
        let p99_us = self.latency_hist.get_percentile(0.99) / 1000;
        let p999_us = self.latency_hist.get_percentile(0.999) / 1000;
        let max_us = self.latency_hist.get_max() / 1000;
        let total_blocks = self.latency_hist.total_count();
        let xruns = self.xruns.load(Ordering::Relaxed);
        let drains = self.drains.load(Ordering::Relaxed);

        TelemetrySnapshot {
            p50_us,
            p99_us,
            p999_us,
            max_us,
            total_blocks,
            xruns,
            drains,
        }
    }

    fn flags_seen(&self) -> u64 {
        let current_bits = self.status_bits.load(Ordering::Relaxed);
        self.flags_seen.fetch_or(current_bits, Ordering::Relaxed);
        self.flags_seen.load(Ordering::Relaxed)
    }
}

/// Decoupled diagnostic bundle containing system information, runtime state, and optional error.
///
/// # Privacy & Redaction Policy
/// To protect user privacy when sharing support blocks publicly:
/// - By default (when `full` is false), absolute path prefixes like `$HOME` are replaced with `~`,
///   and `$XDG_RUNTIME_DIR` are replaced with `$XDG_RUNTIME_DIR`. Basenames of model files are shown instead of full paths.
/// - When `full` is true (e.g. `--diagnose-full` or interactive `:diag --full`), paths are shown unredacted (bruto).
/// - Weights content, audio signals, and user/host names are never captured or included.
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

/// Helper function to redact paths by replacing `$HOME` with `~` and `$XDG_RUNTIME_DIR` with `$XDG_RUNTIME_DIR`.
/// Returns the raw (unredacted) path if `full` is true.
fn redact_path(p: &std::path::Path, full: bool) -> String {
    if full {
        return p.to_string_lossy().into_owned();
    }

    // Check XDG_RUNTIME_DIR prefix
    if let Some(xdg_runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR") {
        let xdg_path = std::path::Path::new(&xdg_runtime_dir);
        if let Ok(stripped) = p.strip_prefix(xdg_path) {
            return format!("$XDG_RUNTIME_DIR/{}", stripped.to_string_lossy());
        }
    }

    // Check HOME prefix
    if let Some(home_dir) = std::env::var_os("HOME") {
        let home_path = std::path::Path::new(&home_dir);
        if let Ok(stripped) = p.strip_prefix(home_path) {
            return format!("~/{}", stripped.to_string_lossy());
        }
    }

    p.to_string_lossy().into_owned()
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
            runtime: RuntimeSnapshot::default(),
            error: None,
            full: false,
        }
    }

    /// Captures a nominal diagnostic bundle, capturing dynamic runtime state via the given provider.
    pub fn capture_with_runtime(provider: &impl HasRuntimeSnapshot) -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot::capture(provider),
            error: None,
            full: false,
        }
    }

    /// Captures a diagnostic bundle with the given error code and parameters.
    pub fn capture_with_error(code: NamErrorCode, params: Vec<(&'static str, String)>) -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot::default(),
            error: Some(ErrorContext { code, params }),
            full: false,
        }
    }

    /// Captures a diagnostic bundle with error and runtime provider.
    pub fn capture_with_error_and_runtime(
        code: NamErrorCode,
        params: Vec<(&'static str, String)>,
        provider: &impl HasRuntimeSnapshot,
    ) -> Self {
        Self {
            system: SystemSnapshot::capture(),
            runtime: RuntimeSnapshot::capture(provider),
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
                .map(|(k, v)| {
                    let v_redacted = {
                        let path = std::path::Path::new(v);
                        if path.is_absolute() {
                            redact_path(path, self.full)
                        } else {
                            v.clone()
                        }
                    };
                    format!("{k}={v_redacted}")
                })
                .collect::<Vec<_>>()
                .join(" ");
            block.push_str(&param_line);
            block.push('\n');
        }

        // Runtime State
        let model_name_raw = ACTIVE_MODEL_NAME
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let model_name = if model_name_raw.is_empty() {
            self.runtime
                .model
                .as_ref()
                .map(|m| m.path_basename.clone())
                .unwrap_or_default()
        } else {
            model_name_raw
        };
        let active_rate = {
            let r = ACTIVE_SAMPLE_RATE.load(Ordering::Relaxed);
            if r == 0 {
                self.runtime.audio.sample_rate
            } else {
                r
            }
        };

        block.push_str("──── Runtime State ─────────────────────────────\n");
        let mut model_printed = false;
        if !model_name.is_empty() {
            let formatted = format_model_path(&model_name, self.full);
            block.push_str(&format!("model={}\n", formatted));
            model_printed = true;
        }
        if active_rate != 0 {
            block.push_str(&format!("sample_rate={}\n", active_rate));
        }

        if let Some(ref model) = self.runtime.model {
            let path_val = if self.full {
                model.path_basename.clone()
            } else {
                let path = std::path::Path::new(&model.path_basename);
                if let Some(file_name) = path.file_name() {
                    file_name.to_string_lossy().into_owned()
                } else {
                    model.path_basename.clone()
                }
            };
            block.push_str(&format!(
                "model.arch={}\n\
                 model.channels={}\n\
                 model.receptive_field={}\n\
                 model.weights_layout={}\n\
                 model.path_basename={}\n",
                model.arch_label,
                model.channels,
                model.receptive_field,
                model.weights_layout,
                path_val
            ));
            model_printed = true;
        }

        if !model_printed {
            block.push_str("model=none\n");
        }

        block.push_str(&format!(
            "audio.sr={}\n\
             audio.buffer_size={}\n\
             audio.channel_count={}\n\
             audio.host_name={}\n\
             rt.prio={}\n\
             rt.scheduler={}\n\
             rt.cpu_pinned={}\n\
             rt.huge_pages_active={}\n\
             telemetry.p50_us={}\n\
             telemetry.p99_us={}\n\
             telemetry.p999_us={}\n\
             telemetry.max_us={}\n\
             telemetry.total_blocks={}\n\
             telemetry.xruns={}\n\
             telemetry.drains={}\n\
             flags_seen={:#x}\n",
            self.runtime.audio.sample_rate,
            self.runtime.audio.buffer_size,
            self.runtime.audio.channel_count,
            self.runtime.audio.host_name,
            self.runtime.rt.thread_priority,
            self.runtime.rt.scheduler,
            self.runtime
                .rt
                .cpu_pinned
                .map(|c| c.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.runtime.rt.huge_pages_active,
            self.runtime.telemetry.p50_us,
            self.runtime.telemetry.p99_us,
            self.runtime.telemetry.p999_us,
            self.runtime.telemetry.max_us,
            self.runtime.telemetry.total_blocks,
            self.runtime.telemetry.xruns,
            self.runtime.telemetry.drains,
            self.runtime.flags_seen
        ));

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

#[cfg(test)]
#[path = "diagnostic_test.rs"]
mod diagnostic_test;
