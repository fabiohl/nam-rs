// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Snapshot do ambiente de execução, capturado uma vez no startup.
//!
//! Inclui informações estáticas do sistema úteis para triagem:
//! versão do NAM-rs, arquitetura, OS, features de CPU, versão do kernel.

#[cfg(feature = "standalone")]
use crate::standalone::colors::Colorize;

/// Versão do binário, injetada em tempo de compilação.
const NAM_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Snapshot do ambiente de execução, capturado uma vez no startup.
///
/// Inclui informações estáticas do sistema que são úteis para triagem:
/// versão do NAM-rs, arquitetura, OS, features de CPU, versão do kernel.
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    /// Versão do NAM-rs (e.g. "1.0.0").
    pub version: &'static str,
    /// Arquitetura do CPU (e.g. "x86_64").
    pub arch: &'static str,
    /// Sistema operacional (e.g. "linux").
    pub os: &'static str,
    /// Versão do kernel Linux (lida de /proc/version).
    pub kernel: String,
    /// Features de CPU detectadas (acima do baseline x86-64-v3).
    pub features: Vec<String>,
    /// Versão da biblioteca PipeWire em uso (apenas com feature `standalone`).
    pub pipewire_version: Option<String>,
}

impl SystemSnapshot {
    /// Captura as informações do sistema no momento da chamada.
    ///
    /// Deve ser chamada uma vez no `main()` e o resultado propagado
    /// para as funções que emitem diagnósticos.
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

    /// Emite aviso informativo recomendando o isolamento de IRQs do núcleo afixado
    /// pelo PipeWire, de modo a minimizar a preempção em tempo real.
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

/// Lê a versão do kernel Linux a partir de `/proc/version`.
fn read_kernel_version() -> String {
    std::fs::read_to_string("/proc/version")
        .ok()
        .and_then(|v| {
            // Formato: "Linux version 6.8.0-generic (...)"
            v.split_whitespace().nth(2).map(String::from)
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Detecta features de CPU avançadas (acima do baseline x86-64-v3).
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

/// Retorna a versão da biblioteca PipeWire linkada ao binário.
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
