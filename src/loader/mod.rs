// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de carregamento do ecossistema NAM.
//!
//! Contém os parsers dos formatos .nam (JSON) e .namb (Binário).
//! Todo o processo de carga ocorre **fora** da thread RT para
//! evitar qualquer alocação indesejada durante o processamento de áudio.

pub mod nam_json;
pub mod namb;
