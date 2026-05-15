// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Parâmetros agnósticos ao host do NAM-rs.
//!
//! Este módulo define o estado completo de configuração do processamento,
//! permitindo que diferentes hosts (CLI/PipeWire ou CLAP) gerenciem e
//! sincronizem os parâmetros de forma consistente.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Parâmetros globais de processamento do plugin/aplicativo.
///
/// Esta estrutura encapsula todos os controles disponíveis para o usuário,
/// desde ganhos básicos até o caminho do modelo neural carregado.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamPluginParams {
    /// Ganho de entrada em decibéis (dB). Padrão: 0.0.
    pub input_gain_db: f32,
    /// Ganho de saída em decibéis (dB). Padrão: 0.0.
    pub output_gain_db: f32,
    /// Threshold do Noise Gate em decibéis (dB). Padrão: -70.0.
    /// Este valor mapeia para o `threshold_open_db` do motor de gate.
    pub gate_threshold_db: f32,
    /// Caminho para o modelo `.nam` ou `.namb` carregado.
    pub model_path: Option<PathBuf>,
    /// Estado de Bypass (se `true`, o áudio passa sem processamento neural).
    pub bypass: bool,
}

impl Default for NamPluginParams {
    fn default() -> Self {
        Self {
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            gate_threshold_db: -70.0,
            model_path: None,
            bypass: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_default() {
        let params = NamPluginParams::default();
        assert_eq!(params.input_gain_db, 0.0);
        assert_eq!(params.output_gain_db, 0.0);
        assert_eq!(params.gate_threshold_db, -70.0);
        assert_eq!(params.model_path, None);
        assert!(!params.bypass);
    }
}
