// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::{DynamicModel, NamModel};
use std::sync::Arc;

impl DynamicModel {
    /// Injects `RtStatusFlags` into the model so it can signal its state
    /// to the UI via atomic flags.
    pub fn inject_rt_status(&mut self, _rt_status: Arc<crate::common::spsc::RtStatusFlags>) {}

    /// Sets the effective number of layers for soft-degrade.
    /// Only applies to WaveNet variants. LSTM handles reduction at the pipeline level.
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        match self {
            Self::WavenetStandard(m) => m.set_effective_layers(n),
            Self::WavenetLite(m) => m.set_effective_layers(n),
            Self::WavenetFeather(m) => m.set_effective_layers(n),
            Self::WavenetNano(m) => m.set_effective_layers(n),
            // LSTM and A2: no-op — reduction handled at pipeline level
            Self::WavenetA2Full(_)
            | Self::WavenetA2Lite(_)
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
            Self::WavenetA2Full(_) | Self::WavenetA2Lite(_) => crate::models::a2::A2_NUM_LAYERS,
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
                | Self::WavenetA2Full(_)
                | Self::WavenetA2Lite(_)
        )
    }

    /// Returns the number of channels inside the model.
    pub fn channels(&self) -> usize {
        match self {
            Self::WavenetStandard(_) => 16,
            Self::WavenetLite(_) => 12,
            Self::WavenetFeather(_) => 8,
            Self::WavenetNano(_) => 4,
            Self::WavenetA2Full(_) => 8,
            Self::WavenetA2Lite(_) => 3,
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
            Self::WavenetA2Full(m) => m.process(input, output),
            Self::WavenetA2Lite(m) => m.process(input, output),
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
            Self::WavenetA2Full(m) => m.prewarm(),
            Self::WavenetA2Lite(m) => m.prewarm(),
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
            Self::WavenetA2Full(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Lite(m) => m.reset(sample_rate, max_buffer_size),
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
            Self::WavenetA2Full(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Lite(m) => m.set_max_buffer_size(max_buf),
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
            Self::WavenetA2Full(m) => m.prewarm_samples(),
            Self::WavenetA2Lite(m) => m.prewarm_samples(),
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
