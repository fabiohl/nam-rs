// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet Module — Dilated Causal Neural Architecture for amplifier and pedal emulation.
//!
//! This module provides WaveNet inference engines optimized for real-time DSP execution:
//!
//! - **Static Path** (`model`): uses Const Generics for fixed dimensions (e.g., 16 channels),
//!   eliminating bounds checks and maximizing throughput.
//! - **Dynamic Path** (`model_dyn`): fallback for topologies not covered by Const Generics,
//!   with single allocation in the constructor and zero-allocation on the hot-path.
//!
//! ## Sub-modules
//!
//! | Module        | Description                                                             |
//! | ------------- | ----------------------------------------------------------------------- |
//! | `common`      | Fundamental constants and types (`WaveNetLayerState`, `WavenetProcessContext`) |
//! | `conv1d`      | Static causal 1D convolution (`Conv1d`) + `ConvInput` trait             |
//! | `conv1d_dyn`  | Dynamic causal 1D convolution (`Conv1dDyn`)                             |
//! | `dense`       | Static 1x1 dense layer (`DenseLayer`)                                   |
//! | `model`       | Complete static model (`WaveNetModel`, `WaveNetLayerArray`)             |
//! | `model_dyn`   | Dynamic model with runtime dimensions (`WaveNetDynModel`)               |

pub mod common;
pub mod conv1d;
pub mod conv1d_dual;
pub mod conv1d_dyn;
/// Static 1x1 dense layer (`DenseLayer<IN, OUT>`).
pub mod dense;
/// Static model (`WaveNetModel`, `WaveNetLayerArray`, `WaveNetLayer`).
pub mod model;
pub mod model_dyn;

use super::NamModel;

// =============================================================================
// NamModel for WaveNet (Const Generics)
// =============================================================================

impl<const CH: usize, const K: usize, const HEAD: usize> NamModel
    for model::WaveNetModel<CH, K, HEAD>
{
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Delegates to the inherent WaveNetModel::process method (inherent methods have priority)
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        // WaveNet prewarm is one-shot: fills the receptive field via copy_buffer.
        // C++ runs `model->Prewarm()` without a parameter (unlike LSTM).
        self.prewarm();
    }
}

// =============================================================================
// NamModel for Dynamic WaveNet
// =============================================================================

impl NamModel for model_dyn::WaveNetDynModel {
    /// Delegates processing to the dynamic WaveNet model's internal implementation.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// WaveNet's "prewarm" is simplified since it has no infinite memory
    /// like LSTM, only a delay buffer (receptive field).
    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }
}

// =============================================================================
// Public re-exports
// =============================================================================

pub use common::{
    LAYER_ARRAY_BUFFER_PADDING, WAVENET_MAX_NUM_FRAMES, WaveNetLayerState, WavenetProcessContext,
};
pub use conv1d::Conv1d;
pub use conv1d_dyn::{Conv1dDyn, MAX_KERNEL};
pub use dense::DenseLayer;
pub use model::{WaveNetLayer, WaveNetLayerArray, WaveNetModel};
pub use model_dyn::{DenseLayerDyn, WaveNetDynModel, WaveNetLayerArrayDyn, WaveNetLayerDyn};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
