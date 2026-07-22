#![doc = include_str!("../README.md")]
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![warn(missing_docs)]

//! Neural Amp Modeler (NAM-rs) — high-performance inference engine for
//! guitar/bass amplifier and pedal neural models.
//!
//! This library is the shared foundation ("heart") of the project, providing
//! the inference engine, SIMD processing kernels, and model loading logic.
//! It serves as the common substrate for:
//!
//! 1. The CLI executable (`main.rs`) — standalone PipeWire audio processing.
//! 2. Plugins (e.g. CLAP) — DAW integration with full GUI.
//!
//! # Architecture Overview
//!
//! NAM-rs is organized in a modular structure to support multiple build
//! targets:
//!
//! - **Common**: Host-agnostic infrastructure (diagnostics, SPSC
//!   communication, parameters). Re-exported at the crate root.
//! - **Standalone**: Native implementation for PipeWire and CLI utilities.
//!   Enabled via `standalone` feature. Used by `main.rs`.
//! - **CLAP Plugin**: Integration as a CLAP (CLever Audio Plug-in) format
//!   plugin with full GUI. Enabled via `clap-plugin` feature.
//!
//! # Quick Start — Loading a Model
//!
//! ```
//! use std::path::Path;
//! use nam_rs::loader::{load_and_build_model, LoadOptions};
//! use nam_rs::SystemSnapshot;
//!
//! let sys = SystemSnapshot::capture();
//! let model = load_and_build_model(
//!     Path::new(concat!(
//!         env!("CARGO_MANIFEST_DIR"),
//!         "/tests/fixtures/models/linear_test.nam",
//!     )),
//!     &sys,
//!     false, // mono
//!     LoadOptions::default(),
//! )
//! .expect("Failed to load model");
//!
//! assert_eq!(model.architecture, "Linear");
//! assert!(model.model_l.is_some());
//! // Mono load: right channel is None
//! assert!(model.model_r.is_none());
//! assert!(model.sample_rate > 0);
//! ```
//!
//! The [`loader::load_and_build_model`] function reads `.nam` (JSON) or `.namb`
//! (binary) files and returns a [`loader::LoadedModelPair`] containing the
//! constructed [`models::StaticModel`] instances, gain calibration, sample rate,
//! and model metadata.
//!
//! For stereo models, pass `true` as the third argument — both `model_l`
//! and `model_r` will be populated.
//!
//! # Activation Functions
//!
//! The crate exposes high-performance scalar activation functions in
//! [`math::activations`]:
//!
//! ```
//! use nam_rs::math::activations::tanh;
//! use nam_rs::math::activations::sigmoid;
//!
//! // tanh: Padé [5,4] rational approximant, clamped to [-1.0, 1.0]
//! let t = tanh(1.0);
//! assert!((t - 0.761594).abs() < 1e-3);
//! assert!(tanh(10.0) <= 1.0);
//! assert!(tanh(-10.0) >= -1.0);
//!
//! // sigmoid: degree-17 minimax polynomial, clamped to [0.0, 1.0]
//! let s = sigmoid(0.0);
//! assert!((s - 0.5).abs() < 1e-2);
//! assert!(sigmoid(10.0) > 0.999);
//! assert!(sigmoid(-10.0) < 0.001);
//! ```
//!
//! Both functions are always available (no feature gate). For slice-based
//! dispatch (auto-selecting SIMD kernels), use `tanh_slice` and
//! `sigmoid_slice` from the same modules.
//!
//! # Module Map
//!
//! | Module     | Purpose                                            |
//! | ---------- | -------------------------------------------------- |
//! | [`loader`] | Model loading and construction (`.nam`, `.namb`)   |
//! | [`math`]   | Mathematical primitives, SIMD kernels, activations |
//! | [`models`] | Neural network architectures and topologies        |
//! | [`dsp`]    | Digital signal processing engine                   |
//! | [`common`] | Diagnostics, SPSC communication, parameters        |

#[cfg(not(target_arch = "x86_64"))]
compile_error!("NAM-rs requires x86_64 architecture");

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
#[cfg(all(target_os = "linux", target_env = "gnu"))]
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
