// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Neural Inference Architectures (Brain Engines) module for NAM-rs.
//!
//! This module contains the acoustic brains of the program: neural networks that have learned how,
//! for example, a real amplifier or pedal distorts and colors a guitar sound.

pub mod a2;
pub mod lstm;
pub mod wavenet;

use crate::common::spsc::RtStatusFlags;
use std::sync::Arc;

// =============================================================================
// Sealed Pattern — Prevents external implementations of NamModel
// =============================================================================

mod sealed {
    pub trait Sealed {}
}

// =============================================================================
// Trait NamModel — Public Contract
// =============================================================================

/// The interface (standard connector) for any neural model (amplifiers, pedals, etc.).
///
/// Sealed via private supertrait — only types within this crate can implement `NamModel`.
pub trait NamModel: Send + Sync + sealed::Sealed {
    /// Invoked by the DSP RT-Thread to process acoustic sample blocks (Float32).
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// "Heats up" the virtual tubes of the neural engine (`prewarm`).
    fn prewarm(&mut self, num_samples: usize);

    /// Resets the model's internal state with a new sample rate and max buffer size.
    ///
    /// The default implementation calls `prewarm(max_buffer_size)`, which is suitable for
    /// architectures like WaveNet that need to fill the receptive field with silence.
    /// Architectures with recurrent state (LSTM) may override this for a lighter
    /// reset (only zero the internal states without reprocessing a full prewarm).
    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) {
        self.prewarm(max_buffer_size);
    }

    /// Reallocates internal buffers to support the given maximum block size.
    ///
    /// Models with fixed (const-generic) buffer sizes can use the default no-op.
    /// Dynamic models (e.g., `WaveNetDynModel`) should reallocate `block_buffer`
    /// and `head_accum` when `max_buf` exceeds the current capacity.
    ///
    /// Default: no-op (suitable for static models and LSTM).
    fn set_max_buffer_size(&mut self, _max_buf: usize) {}

    /// Returns the number of samples needed to fully stabilize the model's internal
    /// state (receptive field / recurrent memory depth).
    ///
    /// Default: `0` (suitable for LSTM, which stabilizes via recurrence).
    /// WaveNet variants override this to return `array1.receptive_field_size`.
    fn prewarm_samples(&self) -> usize {
        0
    }
}

/// Wrapper enum for trained model variants.
/// Enables static dispatch of DSP calls to the concrete variant, avoiding vtable overhead.
pub enum DynamicModel {
    /// WaveNet Standard (16 channels, kernel 3, dilation 8).
    WavenetStandard(Box<wavenet::WaveNetModel<16, 3, 8>>),
    /// WaveNet Lite (12 channels, kernel 3, dilation 6).
    WavenetLite(Box<wavenet::WaveNetModel<12, 3, 6>>),
    /// WaveNet Feather (8 channels, kernel 3, dilation 4).
    WavenetFeather(Box<wavenet::WaveNetModel<8, 3, 4>>),
    /// WaveNet Nano (4 channels, kernel 3, dilation 2).
    WavenetNano(Box<wavenet::WaveNetModel<4, 3, 2>>),
    /// WaveNet Dynamic (used as fallback for non-standard architectures).
    WavenetDyn(Box<wavenet::WaveNetDynModel>),
    /// WaveNet A2 (Placeholder for new architecture).
    WavenetA2(Box<a2::WavenetA2Placeholder>),
    /// LSTM 1 Layer × 8 hidden units.
    Lstm1x8(Box<lstm::Lstm1x8>),
    /// LSTM 1 Layer × 12 hidden units.
    Lstm1x12(Box<lstm::Lstm1x12>),
    /// LSTM 1 Layer × 16 hidden units.
    Lstm1x16(Box<lstm::Lstm1x16>),
    /// LSTM 1 Layer × 24 hidden units.
    Lstm1x24(Box<lstm::Lstm1x24>),
    /// LSTM 2 Layers × 8 hidden units.
    Lstm2x8(Box<lstm::Lstm2x8>),
    /// LSTM 2 Layers × 12 hidden units.
    Lstm2x12(Box<lstm::Lstm2x12>),
    /// LSTM 2 Layers × 16 hidden units.
    Lstm2x16(Box<lstm::Lstm2x16>),
    /// LSTM 1 Layer × 40 hidden units.
    Lstm1x40(Box<lstm::Lstm1x40>),
    /// LSTM 2 Layers × 24 hidden units.
    Lstm2x24(Box<lstm::Lstm2x24>),
    /// LSTM Dynamic (used as fallback).
    LstmDyn(Box<lstm::LstmDynModel>),
}

impl DynamicModel {
    /// Injects `RtStatusFlags` into the `WavenetA2` variant so the placeholder
    /// can signal its state to the UI via atomic flags.
    pub fn inject_rt_status(&mut self, rt_status: Arc<RtStatusFlags>) {
        if let Self::WavenetA2(m) = self {
            m.inject_rt_status(rt_status);
        }
    }

    /// Sets the effective number of layers for soft-degrade.
    /// Only applies to WaveNet variants. LSTM handles reduction at the pipeline level.
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        match self {
            Self::WavenetStandard(m) => m.set_effective_layers(n),
            Self::WavenetLite(m) => m.set_effective_layers(n),
            Self::WavenetFeather(m) => m.set_effective_layers(n),
            Self::WavenetNano(m) => m.set_effective_layers(n),
            Self::WavenetDyn(m) => m.set_effective_layers(n),
            // LSTM and A2: no-op — reduction handled at pipeline level
            Self::WavenetA2(_)
            | Self::Lstm1x8(_)
            | Self::Lstm1x12(_)
            | Self::Lstm1x16(_)
            | Self::Lstm1x24(_)
            | Self::Lstm2x8(_)
            | Self::Lstm2x12(_)
            | Self::Lstm2x16(_)
            | Self::Lstm1x40(_)
            | Self::Lstm2x24(_)
            | Self::LstmDyn(_) => {}
        }
    }

    /// Returns the total number of layers for the model (0 for non-WaveNet).
    /// Used by the adaptive FSM to compute how many to keep.
    #[inline(always)]
    pub fn layer_count(&self) -> usize {
        match self {
            Self::WavenetStandard(m) => m.array1.layers.len(),
            Self::WavenetLite(m) => m.array1.layers.len(),
            Self::WavenetFeather(m) => m.array1.layers.len(),
            Self::WavenetNano(m) => m.array1.layers.len(),
            Self::WavenetDyn(m) => m.array1.layers.len(),
            Self::WavenetA2(_) => 0,
            Self::Lstm2x8(_)
            | Self::Lstm2x12(_)
            | Self::Lstm2x16(_)
            | Self::Lstm2x24(_)
            | Self::LstmDyn(_) => 2,
            Self::Lstm1x8(_)
            | Self::Lstm1x12(_)
            | Self::Lstm1x16(_)
            | Self::Lstm1x24(_)
            | Self::Lstm1x40(_) => 1,
        }
    }

    /// Returns `true` if this is an LSTM model.
    #[inline(always)]
    pub fn is_lstm(&self) -> bool {
        matches!(
            self,
            Self::Lstm1x8(_)
                | Self::Lstm1x12(_)
                | Self::Lstm1x16(_)
                | Self::Lstm1x24(_)
                | Self::Lstm2x8(_)
                | Self::Lstm2x12(_)
                | Self::Lstm2x16(_)
                | Self::Lstm1x40(_)
                | Self::Lstm2x24(_)
                | Self::LstmDyn(_)
        )
    }

    /// Returns `true` if this is a WaveNet model.
    #[inline(always)]
    pub fn is_wavenet(&self) -> bool {
        matches!(
            self,
            Self::WavenetStandard(_)
                | Self::WavenetLite(_)
                | Self::WavenetFeather(_)
                | Self::WavenetNano(_)
                | Self::WavenetDyn(_)
                | Self::WavenetA2(_)
        )
    }

    /// Returns the number of channels inside the model.
    pub fn channels(&self) -> usize {
        match self {
            Self::WavenetStandard(_) => 16,
            Self::WavenetLite(_) => 12,
            Self::WavenetFeather(_) => 8,
            Self::WavenetNano(_) => 4,
            Self::WavenetDyn(m) => m.array1.layers.first().map(|l| l.ch).unwrap_or(0),
            Self::WavenetA2(_) => 0,
            Self::Lstm1x8(_) | Self::Lstm2x8(_) => 8,
            Self::Lstm1x12(_) | Self::Lstm2x12(_) => 12,
            Self::Lstm1x16(_) | Self::Lstm2x16(_) => 16,
            Self::Lstm1x24(_) | Self::Lstm2x24(_) => 24,
            Self::Lstm1x40(_) => 40,
            Self::LstmDyn(m) => m.layers.first().map(|l| l.hidden_size).unwrap_or(0),
        }
    }

    /// Returns the receptive field size of the model (or 0 for LSTM).
    pub fn receptive_field(&self) -> usize {
        if self.is_lstm() {
            0
        } else {
            self.prewarm_samples()
        }
    }
}

impl NamModel for DynamicModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        match self {
            Self::WavenetStandard(m) => m.process(input, output),
            Self::WavenetLite(m) => m.process(input, output),
            Self::WavenetFeather(m) => m.process(input, output),
            Self::WavenetNano(m) => m.process(input, output),
            Self::WavenetDyn(m) => m.process(input, output),
            Self::WavenetA2(m) => m.process(input, output),
            Self::Lstm1x8(m) => m.process(input, output),
            Self::Lstm1x12(m) => m.process(input, output),
            Self::Lstm1x16(m) => m.process(input, output),
            Self::Lstm1x24(m) => m.process(input, output),
            Self::Lstm2x8(m) => m.process(input, output),
            Self::Lstm2x12(m) => m.process(input, output),
            Self::Lstm2x16(m) => m.process(input, output),
            Self::Lstm1x40(m) => m.process(input, output),
            Self::Lstm2x24(m) => m.process(input, output),
            Self::LstmDyn(m) => m.process(input, output),
        }
    }

    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        match self {
            Self::WavenetStandard(m) => m.prewarm(),
            Self::WavenetLite(m) => m.prewarm(),
            Self::WavenetFeather(m) => m.prewarm(),
            Self::WavenetNano(m) => m.prewarm(),
            Self::WavenetDyn(m) => m.prewarm(),
            Self::WavenetA2(m) => m.prewarm(num_samples),
            Self::Lstm1x8(m) => m.prewarm(num_samples),
            Self::Lstm1x12(m) => m.prewarm(num_samples),
            Self::Lstm1x16(m) => m.prewarm(num_samples),
            Self::Lstm1x24(m) => m.prewarm(num_samples),
            Self::Lstm2x8(m) => m.prewarm(num_samples),
            Self::Lstm2x12(m) => m.prewarm(num_samples),
            Self::Lstm2x16(m) => m.prewarm(num_samples),
            Self::Lstm1x40(m) => m.prewarm(num_samples),
            Self::Lstm2x24(m) => m.prewarm(num_samples),
            Self::LstmDyn(m) => m.prewarm(num_samples),
        }
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) {
        match self {
            Self::WavenetStandard(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetLite(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetFeather(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetNano(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetDyn(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x8(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x12(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x16(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x24(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x8(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x12(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x16(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x40(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x24(m) => m.reset(sample_rate, max_buffer_size),
            Self::LstmDyn(m) => m.reset(sample_rate, max_buffer_size),
        }
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) {
        match self {
            Self::WavenetStandard(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetLite(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetFeather(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetNano(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetDyn(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x8(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x12(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x16(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x24(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x8(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x12(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x16(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x40(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x24(m) => m.set_max_buffer_size(max_buf),
            Self::LstmDyn(m) => m.set_max_buffer_size(max_buf),
        }
    }

    fn prewarm_samples(&self) -> usize {
        match self {
            Self::WavenetStandard(m) => m.prewarm_samples(),
            Self::WavenetLite(m) => m.prewarm_samples(),
            Self::WavenetFeather(m) => m.prewarm_samples(),
            Self::WavenetNano(m) => m.prewarm_samples(),
            Self::WavenetDyn(m) => m.prewarm_samples(),
            Self::WavenetA2(m) => m.prewarm_samples(),
            Self::Lstm1x8(m) => m.prewarm_samples(),
            Self::Lstm1x12(m) => m.prewarm_samples(),
            Self::Lstm1x16(m) => m.prewarm_samples(),
            Self::Lstm1x24(m) => m.prewarm_samples(),
            Self::Lstm2x8(m) => m.prewarm_samples(),
            Self::Lstm2x12(m) => m.prewarm_samples(),
            Self::Lstm2x16(m) => m.prewarm_samples(),
            Self::Lstm1x40(m) => m.prewarm_samples(),
            Self::Lstm2x24(m) => m.prewarm_samples(),
            Self::LstmDyn(m) => m.prewarm_samples(),
        }
    }
}

impl sealed::Sealed for DynamicModel {}
