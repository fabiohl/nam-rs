// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Architecture (Staging and Placeholder).
//!
//! This module isolates components of the A2 architecture (v0.6+), including
//! stubs for activations, FiLM, gating, and parameters.

pub mod activations;
pub mod film;
pub mod gating;
pub mod params;
pub mod placeholder;

/// Public re-exports for easy access.
pub use activations::{ActivationFn, ActivationType};
pub use film::{FiLMConfig, FiLMLayer};
pub use gating::GatingMode;
pub use params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS,
    A2_VALID_CHANNELS, HeadParams, LayerArrayParamsA2, LayerParamsA2,
};
pub use placeholder::WavenetA2Placeholder;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
