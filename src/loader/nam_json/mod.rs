// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Parser para o formato .nam (JSON)
//!
//! Realiza o carregamento dos tensores e metadados fora do caminho RT.

pub mod data;
pub mod parse;
pub mod topology;

pub use data::{
    JsonError, NamConfig, NamDate, NamLayerConfig, NamMetadata, NamModelData, WeightsLayout,
};
pub use parse::parse_nam_json;
#[cfg(test)]
pub(crate) use topology::parse_semver;
pub use topology::{NamWavenetTopology, get_lstm_topology, get_wavenet_topology};

#[cfg(test)]
#[path = "../nam_json_test.rs"]
mod nam_json_test;
