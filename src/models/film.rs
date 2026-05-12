// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de Feature-wise Linear Modulation (FiLM) para a arquitetura NAM A2.
//!
//! O FiLM permite que o modelo adapte seu comportamento com base em sinais
//! de condicionamento externos, aplicando escala e deslocamento (shift) por canal.
//!
//! IMPORTANTE: O suporte à arquitetura A2 está em estágio de "placeholder"
//! aguardando estabilização da implementação de referência.

/// Configuração para uma camada ou operação FiLM.
///
/// Corresponde à struct `_FiLMParams` do C++.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiLMConfig {
    /// Se o FiLM está ativo nesta localização.
    pub active: bool,
    /// Se deve aplicar tanto escala quanto deslocamento (true) ou apenas escala (false).
    pub shift: bool,
    /// Número de grupos para a convolução agrupada no submódulo de condicionamento.
    pub groups: u32,
}

impl Default for FiLMConfig {
    fn default() -> Self {
        Self {
            active: false,
            shift: true,
            groups: 1,
        }
    }
}

/// Trait para implementação de camadas FiLM.
///
/// Define a interface para o processamento de modulação linear baseada em características.
pub trait FiLMLayer {
    /// Processa a modulação FiLM sobre o buffer de entrada.
    ///
    /// Esta é uma implementação stub (placeholder) para a arquitetura A2.
    fn process(&mut self, input: &mut [f32], condition: &[f32]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_film_config_default() {
        let config = FiLMConfig::default();
        assert!(!config.active);
        assert!(config.shift);
        assert_eq!(config.groups, 1);
    }

    #[test]
    fn test_film_config_custom() {
        let config = FiLMConfig {
            active: true,
            shift: false,
            groups: 4,
        };
        assert!(config.active);
        assert!(!config.shift);
        assert_eq!(config.groups, 4);
    }
}
