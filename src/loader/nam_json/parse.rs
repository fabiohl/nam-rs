// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Função de parsing de alto nível para o formato `.nam` (JSON).

use super::data::{JsonError, NamModelData};

/// Desserialização universal bruta da string do JSON via `serde_json`.
/// Retorna `JsonError` tipado para falhas de tamanho ou parse.
pub fn parse_nam_json(json_str: &str) -> Result<NamModelData, JsonError> {
    let data: NamModelData = serde_json::from_str(json_str)?;
    Ok(data)
}
