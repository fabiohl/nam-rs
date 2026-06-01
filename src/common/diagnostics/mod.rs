// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Sistema de diagnósticos estruturados do NAM-rs.
//!
//! Fornece mensagens de erro em duas camadas:
//! 1. **Mensagem amigável** — texto legível para o usuário final.
//! 2. **Bloco de suporte** — código de erro tipado, parâmetros contextuais,
//!    versão, arquitetura e timestamp para triagem precisa por devs/IA.
//!
//! ## Uso fora da thread RT
//!
//! Toda formatação e impressão de diagnósticos ocorre **exclusivamente** em
//! threads não-RT (CLI, loop principal do PipeWire). O callback `process()`
//! continua usando flags atômicas (`RtStatusFlags`) para sinalização silenciosa.

pub mod diagnostic;
pub mod error_codes;
pub mod system_info;

pub use diagnostic::NamDiagnostic;
#[cfg(test)]
pub(crate) use diagnostic::days_to_date;
pub use error_codes::NamErrorCode;
pub use system_info::SystemSnapshot;

#[cfg(test)]
#[path = "../diagnostics_test.rs"]
mod diagnostics_test;
