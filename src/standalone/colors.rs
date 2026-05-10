// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Fornece uma trait simples `Colorize` para formatar strings com
//! sequências de escape ANSI em terminais POSIX.
//!
//! **Decisão de Design:** Optamos por manter esta implementação local minimalista
//! em vez de adicionar crates externas (ex: `colored`, `owo-colors`) para manter
//! o grafo de dependências o mais enxuto possível, já que a necessidade do
//! projeto é básica e restrita a threads não-críticas (CLI/Logging).

/// Trait simples para formatar strings com sequências de escape ANSI em terminais POSIX.
pub trait Colorize {
    /// Aplica formatação em negrito.
    fn bold(&self) -> String;
    /// Colore o texto de vermelho.
    fn red(&self) -> String;
    /// Colore o texto de verde.
    fn green(&self) -> String;
    /// Colore o texto de amarelo.
    fn yellow(&self) -> String;
    /// Colore o texto de azul.
    fn blue(&self) -> String;
    /// Colore o texto de ciano.
    fn cyan(&self) -> String;
    /// Colore o texto de vermelho claro.
    fn bright_red(&self) -> String;
    /// Colore o texto de verde claro.
    fn bright_green(&self) -> String;
    /// Colore o texto de azul claro.
    fn bright_blue(&self) -> String;
    /// Colore o texto de ciano claro.
    fn bright_cyan(&self) -> String;
}

impl<T: AsRef<str>> Colorize for T {
    fn bold(&self) -> String {
        format!("\x1b[1m{}\x1b[0m", self.as_ref())
    }

    fn red(&self) -> String {
        format!("\x1b[31m{}\x1b[0m", self.as_ref())
    }

    fn green(&self) -> String {
        format!("\x1b[32m{}\x1b[0m", self.as_ref())
    }

    fn yellow(&self) -> String {
        format!("\x1b[33m{}\x1b[0m", self.as_ref())
    }

    fn blue(&self) -> String {
        format!("\x1b[34m{}\x1b[0m", self.as_ref())
    }

    fn cyan(&self) -> String {
        format!("\x1b[36m{}\x1b[0m", self.as_ref())
    }

    fn bright_red(&self) -> String {
        format!("\x1b[91m{}\x1b[0m", self.as_ref())
    }

    fn bright_green(&self) -> String {
        format!("\x1b[92m{}\x1b[0m", self.as_ref())
    }

    fn bright_blue(&self) -> String {
        format!("\x1b[94m{}\x1b[0m", self.as_ref())
    }

    fn bright_cyan(&self) -> String {
        format!("\x1b[96m{}\x1b[0m", self.as_ref())
    }
}
