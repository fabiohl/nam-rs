// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LSTM Module — Recurrent Models for NAM Inference.
//!
//! Contains the static (Const Generics) and dynamic (fallback) implementations
//! of LSTM models, their layers, type aliases by performance profile,
//! and integration with the `NamModel` trait for dispatch by the DSP host.
//!
//! ## Architecture
//!
//! ```text
//! lstm/
//! ├── mod.rs        # Re-exports, type aliases, NamModel impls, LstmLike trait
//! ├── layer.rs      # LstmLayer + per-sample SIMD processing macros
//! ├── model1.rs     # LstmModel1 + define_lstm1_process! macro
//! ├── model2.rs     # LstmModel2 + define_lstm2_process_pipelined! macro
//! ├── model_dyn.rs  # LstmDynLayer + LstmDynModel (dynamic fallback)
//! └── tests.rs      # Unit and SIMD vs scalar parity tests
//! ```

use super::NamModel;
use super::sealed;

pub mod layer;
pub mod model1;
pub mod model2;
pub mod model_dyn;
pub mod prewarm;

// =============================================================================
// Re-exports — Public Structs
// =============================================================================

pub use layer::LstmLayer;
pub use model_dyn::{LstmDynLayer, LstmDynModel};
pub use model1::LstmModel1;
pub use model2::LstmModel2;

// =============================================================================
// Type Aliases — Common NAM LSTM Profiles
// =============================================================================

/// LSTM 1 layer × 8 hidden units (Nano/Feather).
pub type Lstm1x8 = LstmModel1<8, 9, 32>;
/// LSTM 1 layer × 12 hidden units (Lite).
pub type Lstm1x12 = LstmModel1<12, 13, 48>;
/// LSTM 1 layer × 16 hidden units (Standard).
pub type Lstm1x16 = LstmModel1<16, 17, 64>;
/// LSTM 1 layer × 24 hidden units (Heavy Standard).
pub type Lstm1x24 = LstmModel1<24, 25, 96>;
/// LSTM 1 layer × 40 hidden units (Tone Matching).
pub type Lstm1x40 = LstmModel1<40, 41, 160>;

/// LSTM 2 layers × 8 hidden units.
pub type Lstm2x8 = LstmModel2<8, 9, 16, 32>;
/// LSTM 2 layers × 12 hidden units.
pub type Lstm2x12 = LstmModel2<12, 13, 24, 48>;
/// LSTM 2 layers × 16 hidden units.
pub type Lstm2x16 = LstmModel2<16, 17, 32, 64>;
/// LSTM 2 layers × 24 hidden units.
pub type Lstm2x24 = LstmModel2<24, 25, 48, 96>;

// =============================================================================
// sealed::Sealed for LSTM types
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H_H4: usize> sealed::Sealed
    for LstmModel1<H, H1_IH, H_H4>
{
}
impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> sealed::Sealed
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
}
impl sealed::Sealed for LstmDynModel {}

// =============================================================================
// NamModel for LSTM — 1 Layer
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H_H4: usize> NamModel
    for LstmModel1<H, H1_IH, H_H4>
{
    /// Executes LSTM audio processing.
    /// Note: `self.process` calls the struct's inherent method, which already has
    /// the optimized SIMD dispatch logic (AVX2/512).
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Safety: Hardware compatibility check is performed at application startup.
        self.process(input, output);
    }

    /// Prewarm is vital for LSTM. Since it's a recurrent model, the internal state (memory)
    /// needs time processing silence to "stabilize" before real audio starts.
    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        lstm_prewarm_common(self, num_samples);
    }

    /// Light reset for LSTM: only zeros internal states without reprocessing
    /// the full prewarm with silence.
    fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        self.reset_states();
    }
}

// =============================================================================
// NamModel for LSTM — 2 Layers
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> NamModel
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Processing identical to the 1-layer model, but operating over the 2-layer chain.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// Prewarm for the stacked model. Both layers are stabilized sequentially.
    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        lstm_prewarm_common(self, num_samples);
    }

    /// Light reset for 2-layer LSTM: only zeros internal states without
    /// reprocessing the full prewarm with silence.
    fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        self.reset_states();
    }
}

// =============================================================================
// NamModel for Dynamic LSTM
// =============================================================================

impl NamModel for LstmDynModel {
    /// Implementation for models where the hidden state size is defined at runtime.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// Dynamic prewarm already encapsulates the silence loop logic internally.
    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        self.prewarm(num_samples);
    }

    /// Light reset for dynamic LSTM: only zeros internal states without
    /// reprocessing the full prewarm with silence.
    fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        self.reset_states();
    }
}

// =============================================================================
// Prewarm Module Import
// =============================================================================

use self::prewarm::*;

#[cfg(test)]
#[path = "tests.rs"]
mod lstm_tests;
