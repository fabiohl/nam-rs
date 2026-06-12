// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![warn(missing_docs)]

//! Neural Amp Modeler (NAM-rs) inference engine library.
//!
//! This file defines the "heart" of the project. It is a shared library that:
//! 1. Serves as the foundation for the CLI executable (`main.rs`).
//! 2. Serves as the foundation for plugins (e.g. CLAP).
//!
//! NAM-rs is organized in a modular structure to support multiple build targets:
//!
//! - **Common**: Host-agnostic infrastructure (diagnostics, SPSC communication, parameters).
//! - **Standalone**: Native implementation for PipeWire and CLI utilities.
//!   Enabled via `standalone` feature. Used by `main.rs`.
//! - **CLAP Plugin**: Integration as a CLAP (CLever Audio Plug-in) format plugin with full GUI.
//!   Enabled via `clap-plugin` feature.
//!
//! The inference engine, processing kernels (SIMD), and model loading logic
//! reside here to ensure mathematical parity between the CLI and the Plugin.

/// Common host-agnostic infrastructure layer.
pub mod common;
pub use common::*;

/// Infrastructure for standalone execution (PipeWire + CLI).
#[cfg(feature = "standalone")]
pub mod standalone;
#[cfg(feature = "standalone")]
pub use standalone::*;

/// CLAP plugin format integration.
#[cfg(feature = "clap-plugin")]
pub mod clap;

/// Digital signal processing (DSP) engine.
pub mod dsp;
/// Model loading and construction (.nam, .namb).
pub mod loader;
/// Mathematical primitives and optimized SIMD kernels.
pub mod math;
/// Tensor definitions and neural network topologies.
pub mod models;
/// Testing utilities (signal generation, perceptual metrics, WAV I/O).
#[cfg(any(test, feature = "testing"))]
pub mod testing;

// Backward compatibility with older GLIBC versions (e.g. for Flatpak/Bitwig).
// Redirects math symbols to the stable GLIBC_2.2.5 version.
// Since external dependencies use these symbols, we declare global wrappers
// that intercept calls and jump (jmp) via PLT to the compatible versions.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
core::arch::global_asm!(
    ".global log10f",
    ".type log10f, @function",
    "log10f:",
    "    jmp log10f_compat@PLT",
    ".symver log10f_compat, log10f@GLIBC_2.2.5",
    ".global atan2f",
    ".type atan2f, @function",
    "atan2f:",
    "    jmp atan2f_compat@PLT",
    ".symver atan2f_compat, atan2f@GLIBC_2.2.5",
    ".global acosf",
    ".type acosf, @function",
    "acosf:",
    "    jmp acosf_compat@PLT",
    ".symver acosf_compat, acosf@GLIBC_2.2.5"
);
