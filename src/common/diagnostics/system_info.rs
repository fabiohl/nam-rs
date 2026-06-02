// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Snapshot of the execution environment, captured once at startup.
//!
//! Includes static system information useful for triage:
//! NAM-rs version, architecture, OS, CPU features, kernel version.

#[cfg(feature = "standalone")]
use crate::standalone::colors::Colorize;

/// Binary version, injected at compile time.
const NAM_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Snapshot of the execution environment, captured once at startup.
///
/// Includes static system information useful for triage:
/// NAM-rs version, architecture, OS, CPU features, kernel version.
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    /// NAM-rs version (e.g. "1.0.0").
    pub version: &'static str,
    /// CPU architecture (e.g. "x86_64").
    pub arch: &'static str,
    /// Operating system (e.g. "linux").
    pub os: &'static str,
    /// Linux kernel version (read from /proc/version).
    pub kernel: String,
    /// Detected CPU features (above x86-64-v3 baseline).
    pub features: Vec<String>,
    /// PipeWire library version in use (only with `standalone` feature).
    pub pipewire_version: Option<String>,
}

impl SystemSnapshot {
    /// Captures system information at the time of the call.
    ///
    /// Should be called once in `main()` and the result propagated
    /// to functions that emit diagnostics.
    pub fn capture() -> Self {
        let kernel = read_kernel_version();
        let features = detect_advanced_features();
        let pipewire_version = Some(pw_library_version());

        Self {
            version: NAM_VERSION,
            arch: std::env::consts::ARCH,
            os: std::env::consts::OS,
            kernel,
            features,
            pipewire_version,
        }
    }

    /// Emits an informational advisory recommending IRQ isolation from the core pinned
    /// by PipeWire, in order to minimize real-time preemption.
    pub fn emit_irq_advisory(&self, core: usize) {
        if core >= 64 {
            return;
        }
        let mask = 0xFFFFFFFFFFFFFFFFu64 ^ (1u64 << core);

        #[cfg(feature = "standalone")]
        log::info!(
            "{} For minimum latency, isolate core {} from system IRQs (as sudo): echo {:X} > /proc/irq/default_smp_affinity",
            "💡".yellow(),
            core,
            mask
        );

        #[cfg(not(feature = "standalone"))]
        log::info!(
            "💡 For minimum latency, isolate core {} from system IRQs (as sudo): echo {:X} > /proc/irq/default_smp_affinity",
            core,
            mask
        );
    }
}

/// Reads the Linux kernel version from `/proc/version`.
fn read_kernel_version() -> String {
    std::fs::read_to_string("/proc/version")
        .ok()
        .and_then(|v| {
            // Format: "Linux version 6.8.0-generic (...)"
            v.split_whitespace().nth(2).map(String::from)
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Detects advanced CPU features (above x86-64-v3 baseline).
fn detect_advanced_features() -> Vec<String> {
    let mut features = Vec::new();
    // Features de x86-64-v4 (AVX-512)
    if std::is_x86_feature_detected!("avx512f") {
        features.push("avx512f".to_string());
    }
    if std::is_x86_feature_detected!("avx512bw") {
        features.push("avx512bw".to_string());
    }
    if std::is_x86_feature_detected!("avx512dq") {
        features.push("avx512dq".to_string());
    }
    if std::is_x86_feature_detected!("avx512vl") {
        features.push("avx512vl".to_string());
    }
    if std::is_x86_feature_detected!("avx512vnni") {
        features.push("avx512vnni".to_string());
    }
    if std::is_x86_feature_detected!("avx512bf16") {
        features.push("avx512bf16".to_string());
    }
    features
}

/// Returns the version of the PipeWire library linked to the binary.
#[cfg(feature = "standalone")]
fn pw_library_version() -> String {
    unsafe extern "C" {
        fn pw_get_library_version() -> *const std::ffi::c_char;
    }
    unsafe {
        let ptr = pw_get_library_version();
        if ptr.is_null() {
            return "unknown".to_string();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[cfg(not(feature = "standalone"))]
fn pw_library_version() -> String {
    "N/A (feature standalone disabled)".to_string()
}
