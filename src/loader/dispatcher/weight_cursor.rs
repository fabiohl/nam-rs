// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WeightCursor — Deterministic sequential reading of flattened weights
//!
//! Weights are consumed sequentially by a forward-only cursor,
//! with an exhaustion check at the end to detect inconsistent models.

use anyhow::bail;

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
    pub(crate) fn read_slice(&mut self, len: usize) -> anyhow::Result<&'a [f32]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Insufficient weights: required {} starting from position {}, available {}",
                    len,
                    self.pos,
                    self.data.len()
                )
            })?;
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Reads a single `f32` scalar, advancing the cursor.
    pub(crate) fn read_f32(&mut self) -> anyhow::Result<f32> {
        let s = self.read_slice(1)?;
        Ok(s[0])
    }

    /// Reads a single `f32` scalar and validates finiteness, advancing the cursor.
    /// Use this for critical scalars such as `head_scale`, `head_bias`, etc.
    pub(crate) fn read_f32_finite(&mut self) -> anyhow::Result<f32> {
        let val = self.read_f32()?;
        if !val.is_finite() {
            bail!(
                "Non-finite f32 scalar at weight position {}: {:e}",
                self.pos.wrapping_sub(1),
                val
            );
        }
        Ok(val)
    }

    /// Verifies that all weights have been consumed. Fails if weights remain.
    pub(crate) fn verify_exhausted(&self) -> anyhow::Result<()> {
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
