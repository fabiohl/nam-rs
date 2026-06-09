// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Feature-wise Linear Modulation (FiLM) module for the NAM A2 architecture.
//!
//! FiLM enables the model to adapt its behavior based on external
//! conditioning signals, applying per-channel scale and shift.
//!
//! NOTE: reservado p/ motor A2 geral (futuro). O fast-path A2-Full/Lite (a2_fast.cpp)
//! não utiliza FiLM — toda modulação condicional está fora de escopo no momento.

/// Configuration for a FiLM layer or operation.
///
/// Corresponds to the `_FiLMParams` struct in C++.
///
/// NOTE: reservado p/ motor A2 geral (futuro). O fast-path A2-Full/Lite
/// não utiliza FiLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiLMConfig {
    /// Whether FiLM is active at this location.
    pub active: bool,
    /// Whether to apply both scale and shift (true) or only scale (false).
    pub shift: bool,
    /// Number of groups for grouped convolution in the conditioning submodule.
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

/// Trait for implementing FiLM layers.
///
/// Defines the interface for feature-based linear modulation processing.
///
/// NOTE: reservado p/ motor A2 geral (futuro).
pub trait FiLMLayer {
    /// Processes FiLM modulation over the input buffer.
    ///
    /// This is a stub (placeholder) implementation for the A2 architecture.
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
