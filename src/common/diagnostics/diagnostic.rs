// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Diagnóstico estruturado para erros e avisos do NAM-rs.
//!
//! Combina uma mensagem amigável para o usuário com um bloco técnico
//! copiável para suporte. Formatado para triagem precisa.

use super::error_codes::NamErrorCode;
use super::system_info::SystemSnapshot;
use std::fmt;
use std::time::SystemTime;

/// Diagnóstico estruturado para erros e avisos do NAM-rs.
///
/// Combina uma mensagem amigável para o usuário com um bloco técnico
/// copiável para suporte. Formatado para triagem precisa.
#[derive(Debug)]
pub struct NamDiagnostic {
    /// Código de erro tipado.
    code: NamErrorCode,
    /// Mensagem amigável para o usuário (pt-BR).
    user_message: String,
    /// Sugestão de ação para o usuário (pt-BR).
    user_hint: String,
    /// Parâmetros contextuais para o bloco de suporte (chave=valor).
    params: Vec<(&'static str, String)>,
    /// Snapshot do sistema (capturado no startup).
    system: SystemSnapshot,
}

impl NamDiagnostic {
    /// Cria um novo diagnóstico com o código de erro e snapshot do sistema.
    #[cold]
    pub fn new(code: NamErrorCode, system: &SystemSnapshot) -> Self {
        Self {
            code,
            user_message: String::new(),
            user_hint: String::new(),
            params: Vec::new(),
            system: system.clone(),
        }
    }

    /// Define a mensagem amigável para o usuário.
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.user_message = msg.into();
        self
    }

    /// Define a sugestão de ação para o usuário.
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.user_hint = hint.into();
        self
    }

    /// Adiciona um parâmetro contextual ao bloco de suporte.
    pub fn param(mut self, key: &'static str, value: impl fmt::Display) -> Self {
        self.params.push((key, value.to_string()));
        self
    }

    /// Gera o timestamp ISO 8601 do momento atual.
    pub(crate) fn timestamp() -> String {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();

        // Formatação manual ISO 8601 sem dependências externas
        let secs = now.as_secs();
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Cálculo da data a partir de dias desde epoch (1970-01-01)
        let (year, month, day) = days_to_date(days);

        format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
    }

    /// Gera o bloco de suporte técnico formatado para cópia.
    pub(crate) fn support_block(&self) -> String {
        let separator = "────────────────────────────────────────────────";
        let mut block = format!(
            "──── NAM-rs Diagnostic {separator}\n\
             nam-rs v{} | {} | {}\n",
            self.system.version,
            self.code.code(),
            self.code.mnemonic(),
        );

        // Parâmetros contextuais
        if !self.params.is_empty() {
            let param_line: String = self
                .params
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ");
            block.push_str(&param_line);
            block.push('\n');
        }

        // Informações do sistema
        block.push_str(&format!(
            "arch={}\n\
             os={} kernel={}\n\
             pipewire={}\n\
             features={}\n\
             timestamp={}\n\
             {separator}\n\
             Copy the block above when opening a support ticket.",
            self.system.arch,
            self.system.os,
            self.system.kernel,
            self.system.pipewire_version.as_deref().unwrap_or("N/A"),
            if self.system.features.is_empty() {
                "none (baseline x86-64-v3 only)".to_string()
            } else {
                self.system.features.join(", ")
            },
            Self::timestamp(),
        ));

        block
    }

    /// Imprime o diagnóstico completo no stderr (mensagem amigável + bloco de suporte).
    pub fn emit(&self) {
        log::error!(
            "{}\n\n{}\n\n{}",
            self.user_message,
            if self.user_hint.is_empty() {
                ""
            } else {
                &self.user_hint
            },
            self.support_block()
        );
    }

    /// Imprime como aviso (não-fatal) com prefixo visual diferenciado.
    pub fn emit_warning(&self) {
        log::warn!(
            "{}\n\n{}\n\n{}",
            self.user_message,
            if self.user_hint.is_empty() {
                ""
            } else {
                &self.user_hint
            },
            self.support_block()
        );
    }

    /// Retorna o código de erro associado ao diagnóstico.
    pub fn error_code(&self) -> NamErrorCode {
        self.code
    }
}

impl fmt::Display for NamDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code.code(), self.user_message)
    }
}

impl std::error::Error for NamDiagnostic {}

/// Converte dias desde 1970-01-01 em (ano, mês, dia) gregoriano.
///
/// Implementação mínima do algoritmo civil de Howard Hinnant,
/// sem dependências externas.
pub(crate) fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Algoritmo civil: https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}
