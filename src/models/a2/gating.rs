// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Gating and blending module for the NAM A2 architecture.
//!
//! This module defines the operation modes and configuration structures for
//! WaveNet A2 layers that use gating or blending mechanisms.
//!
//! NOTE: o fast-path A2-Full/Lite (a2_fast.cpp) utiliza apenas `GatingMode::None`.
//! As variantes `Gated`, `Blended` e as configurações associadas estão reservadas
//! para o motor A2 geral (futuro) e não fazem parte do escopo atual.

use super::activations::ActivationType;

/// Gating modes for WaveNet layers.
///
/// Determines how the layer processes duplicated bottleneck channels.
///
/// NOTE: apenas `None` é utilizado pelo fast-path A2-Full/Lite.
/// `Gated` e `Blended` estão reservados p/ motor A2 geral (futuro).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GatingMode {
    /// No gating or blending - standard activation (fast-path A2).
    #[default]
    None,
    /// Traditional gating (element-wise multiplication).
    /// NOTE: reservado p/ motor A2 geral (futuro).
    Gated,
    /// Blending (weighted average between activated and pre-activated values).
    /// NOTE: reservado p/ motor A2 geral (futuro).
    Blended,
}

/// Configuration for Gating-type activation.
///
/// Corresponds to the `GatingActivation` class in C++.
///
/// NOTE: reservado p/ motor A2 geral (futuro).
#[derive(Debug, Clone, PartialEq)]
pub struct GatingActivationConfig {
    /// Activation function for input channels.
    pub input_activation: ActivationType,
    /// Activation function for gating channels.
    pub gating_activation: ActivationType,
}

/// Configuration for Blending-type activation.
///
/// Corresponds to the `BlendingActivation` class in C++.
///
/// NOTE: reservado p/ motor A2 geral (futuro).
#[derive(Debug, Clone, PartialEq)]
pub struct BlendingActivationConfig {
    /// Activation function for input channels.
    pub input_activation: ActivationType,
    /// Activation function for blending channels (determines alpha).
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
