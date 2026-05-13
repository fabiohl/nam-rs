// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de gating e blending para a arquitetura NAM A2.
//!
//! Este módulo define os modos de operação e estruturas de configuração para
//! as camadas do WaveNet A2 que utilizam mecanismos de gating ou blending.
//!
//! IMPORTANTE: O suporte à arquitetura A2 está em estágio de "placeholder"
//! aguardando estabilização da implementação de referência.

use super::activations::ActivationType;

/// Modos de gating para as camadas do WaveNet.
///
/// Determina como a camada processa os canais de bottleneck duplicados.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GatingMode {
    /// Sem gating ou blending - ativação padrão.
    #[default]
    None,
    /// Gating tradicional (multiplicação elemento a elemento).
    Gated,
    /// Blending (média ponderada entre valores ativados e pré-ativados).
    Blended,
}

/// Configuração para ativação do tipo Gating.
///
/// Corresponde à classe `GatingActivation` do C++.
#[derive(Debug, Clone, PartialEq)]
pub struct GatingActivationConfig {
    /// Função de ativação para os canais de entrada.
    pub input_activation: ActivationType,
    /// Função de ativação para os canais de gating.
    pub gating_activation: ActivationType,
}

/// Configuração para ativação do tipo Blending.
///
/// Corresponde à classe `BlendingActivation` do C++.
#[derive(Debug, Clone, PartialEq)]
pub struct BlendingActivationConfig {
    /// Função de ativação para os canais de entrada.
    pub input_activation: ActivationType,
    /// Função de ativação para os canais de blending (determina o alpha).
    pub blending_activation: ActivationType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_mode_default() {
        assert_eq!(GatingMode::default(), GatingMode::None);
    }

    #[test]
    fn test_config_construction() {
        let _gated = GatingActivationConfig {
            input_activation: ActivationType::Tanh,
            gating_activation: ActivationType::Sigmoid,
        };
        let _blended = BlendingActivationConfig {
            input_activation: ActivationType::Tanh,
            blending_activation: ActivationType::Sigmoid,
        };
    }
}
