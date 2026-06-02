// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model Dispatcher — converts `NamModelData` (JSON/NAMB parsers) into `Box<DynamicModel>`.
//!
//! All construction and allocation logic occurs exclusively on the CLI thread,
//! ensuring the resulting `Box<DynamicModel>` is ready for injection into the
//! DSP thread via SPSC without any allocation on the RT path.
//!
//! Weights are consumed sequentially by a forward-only `WeightCursor`,
//! with an exhaustion check at the end to detect inconsistent models.

use crate::loader::nam_json::NamModelData;
use crate::models::DynamicModel;
use anyhow::bail;

// =============================================================================
// WeightCursor — Deterministic sequential reading of flattened weights
// =============================================================================

/// Forward-only read cursor over the flattened weight vector.
///
/// Ensures:
/// - No weight is read out of bounds (`read_slice` / `read_f32`)
/// - All weights have been consumed at the end (`verify_exhausted`)
pub(crate) struct WeightCursor<'a> {
    /// Reference to the full model weight slice.
    data: &'a [f32],
    /// Current cursor position (advances with each read).
    pos: usize,
    /// Weight layout reported in the binary header.
    pub layout: crate::loader::nam_json::WeightsLayout,
}

impl<'a> WeightCursor<'a> {
    /// Creates a new cursor over the weight slice with the specified layout.
    #[cold]
    pub fn new(data: &'a [f32], layout: crate::loader::nam_json::WeightsLayout) -> Self {
        Self {
            data,
            pos: 0,
            layout,
        }
    }

    /// Checks whether the weights are in interleaved format (WaveNet v2).
    pub fn is_interleaved4(&self) -> bool {
        self.layout == crate::loader::nam_json::WeightsLayout::Interleaved4WaveNet
    }

    /// Checks whether the weights are in Gate-Major format (LSTM v2).
    pub fn is_gate_major_lstm(&self) -> bool {
        self.layout == crate::loader::nam_json::WeightsLayout::GateMajorLstm
    }

    /// Reads a contiguous slice of `len` weights, advancing the cursor.
    fn read_slice(&mut self, len: usize) -> anyhow::Result<&'a [f32]> {
        if self.pos + len > self.data.len() {
            bail!(
                "Insufficient weights: required {} starting from position {}, available {}",
                len,
                self.pos,
                self.data.len()
            );
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    /// Reads a single `f32` scalar, advancing the cursor.
    fn read_f32(&mut self) -> anyhow::Result<f32> {
        let s = self.read_slice(1)?;
        Ok(s[0])
    }

    /// Verifies that all weights have been consumed. Fails if weights remain.
    fn verify_exhausted(&self) -> anyhow::Result<()> {
        if self.pos != self.data.len() {
            bail!(
                "Model with inconsistent weights: consumed {}, total {}",
                self.pos,
                self.data.len()
            );
        }
        Ok(())
    }
}

// =============================================================================
// Public Entry Point
// =============================================================================

/// Builds a `Box<DynamicModel>` from the raw parsed data.
///
/// Branches by architecture (`"WaveNet"` / `"LSTM"`) and delegates to the
/// specialized builders with const generics.
pub fn build_model(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    match data.architecture.as_str() {
        "WaveNet" => wavenet::build_wavenet(data),
        "LSTM" => lstm::build_lstm(data),
        other => bail!("Unsupported architecture: '{}'", other),
    }
}

/// LSTM model builder module
pub mod lstm;
/// WaveNet model builder module
pub mod wavenet;

pub use lstm::build_lstm_dynamic;
pub use wavenet::build_wavenet_dynamic;
