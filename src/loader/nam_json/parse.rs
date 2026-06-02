// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! High-level parsing function for the `.nam` format (JSON).

use super::data::{JsonError, NamModelData};

/// Raw universal deserialization of the JSON string via `serde_json`.
/// Retorna `JsonError` tipado para falhas de tamanho ou parse.
pub fn parse_nam_json(json_str: &str) -> Result<NamModelData, JsonError> {
    let data: NamModelData = serde_json::from_str(json_str)?;
    Ok(data)
}
