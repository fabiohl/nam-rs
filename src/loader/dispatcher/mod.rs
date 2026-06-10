// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model Dispatcher — converts `NamModelData` (JSON/NAMB parsers) into `Box<DynamicModel>`.
//!
//! All construction and allocation logic occurs exclusively on the CLI thread,
//! ensuring the resulting `Box<DynamicModel>` is ready for injection into the
//! DSP thread via SPSC without any allocation on the RT path.

use crate::loader::nam_json::NamModelData;
use crate::models::DynamicModel;
use anyhow::bail;

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
/// WeightCursor — Deterministic sequential reading of flattened weights
pub mod weight_cursor;

pub use lstm::build_lstm_dynamic;
pub(crate) use weight_cursor::WeightCursor;
