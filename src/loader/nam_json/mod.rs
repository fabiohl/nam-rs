// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Parser for the .nam format (JSON)
//!
//! Loads tensors and metadata outside the RT path.

pub mod data;
pub mod error;
pub mod model;
pub mod parse;
pub mod topology;
mod validation;

pub use data::{
    JsonError, NamConfig, NamDate, NamLayerConfig, NamMetadata, NamModelData, WeightsLayout,
};
pub use parse::parse_nam_json;
#[cfg(test)]
pub(crate) use topology::parse_semver;
pub use topology::{NamWavenetTopology, get_lstm_topology, get_wavenet_topology, is_a2_shape};

#[cfg(test)]
#[path = "../nam_json_test.rs"]
mod nam_json_test;
