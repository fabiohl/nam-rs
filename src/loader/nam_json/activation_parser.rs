// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Advanced activation and gating parser for A2 layer arrays (F8).
//!
//! Parses the per-layer activation, gating mode, and secondary activation arrays
//! from the raw layer JSON, returning typed Rust vectors for consumption by the
//! dynamic A2 engine (Sprint 3 / WaveNetA2Dyn) and topology classification.
//!
//! Each function expects a single layer array's raw JSON (the `layer_raw` field
//! of `NamLayerConfig`) and validates array length against `num_layers` (23 for
//! standard A2).

use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::GatingMode;

/// Parsed per-layer activation configuration for a single A2 layer array.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerActivationConfig {
    /// Primary activation per layer (23 elements for standard A2).
    pub activations: Vec<ActivationType>,
    /// Gating mode per layer (23 elements).
    pub gating_modes: Vec<GatingMode>,
    /// Secondary activation per layer (23 elements, `None` when absent/null).
    pub secondary_activations: Vec<Option<ActivationType>>,
}

/// Parses the full set of activation-related fields from a single layer array's
/// raw JSON.
///
/// Extracts `activation`, `gating_mode`, and `secondary_activation` arrays,
/// validating that each has exactly `num_layers` entries and deserializing
/// each entry to its typed Rust equivalent.
///
/// Returns `None` when the `activation` array is missing or invalid, as A2
/// models require this field.
pub fn parse_layer_activations(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Option<LayerActivationConfig> {
    let activations = parse_activations_from_json(raw, num_layers)?;
    let gating_modes = parse_gating_modes_from_json(raw, num_layers);
    let secondary_activations = parse_secondary_activations_from_json(raw, num_layers);

    Some(LayerActivationConfig {
        activations,
        gating_modes,
        secondary_activations,
    })
}

/// Parses the `activation` JSON array into `Vec<ActivationType>`.
///
/// Returns `None` if the array is missing, has the wrong length, or
/// contains entries that cannot be deserialized as valid `ActivationType`.
pub fn parse_activations_from_json(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Option<Vec<ActivationType>> {
    let arr = raw.get("activation").and_then(|v| v.as_array())?;
    if arr.len() != num_layers {
        return None;
    }
    let mut out = Vec::with_capacity(num_layers);
    for entry in arr {
        let at: ActivationType = serde_json::from_value(entry.clone()).ok()?;
        out.push(at);
    }
    Some(out)
}

/// Parses the `gating_mode` JSON array into `Vec<GatingMode>`.
///
/// If the field is absent or null, returns a vector of `GatingMode::None`
/// with length `num_layers`.
pub fn parse_gating_modes_from_json(raw: &serde_json::Value, num_layers: usize) -> Vec<GatingMode> {
    let arr = match raw.get("gating_mode") {
        None | Some(serde_json::Value::Null) => {
            return vec![GatingMode::None; num_layers];
        }
        Some(v) => match v.as_array() {
            Some(a) if a.len() == num_layers => a,
            _ => return vec![GatingMode::None; num_layers],
        },
    };
    let mut out = Vec::with_capacity(num_layers);
    for entry in arr {
        let mode = match entry.as_str() {
            Some("gated") => GatingMode::Gated,
            Some("blended") => GatingMode::Blended,
            _ => GatingMode::None,
        };
        out.push(mode);
    }
    out
}

/// Parses the `secondary_activation` JSON array into `Vec<Option<ActivationType>>`.
///
/// If the field is absent or null, returns a vector of `None` with length
/// `num_layers`. Individual null entries are mapped to `None`.
pub fn parse_secondary_activations_from_json(
    raw: &serde_json::Value,
    num_layers: usize,
) -> Vec<Option<ActivationType>> {
    let arr = match raw.get("secondary_activation") {
        None | Some(serde_json::Value::Null) => {
            return vec![None; num_layers];
        }
        Some(v) => match v.as_array() {
            Some(a) if a.len() == num_layers => a,
            _ => return vec![None; num_layers],
        },
    };
    let mut out = Vec::with_capacity(num_layers);
    for entry in arr {
        if entry.is_null() {
            out.push(None);
        } else {
            let at = serde_json::from_value(entry.clone()).ok();
            out.push(at);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::a2::A2_NUM_LAYERS;

    fn make_a2_layer_json() -> serde_json::Value {
        let mut activations = Vec::new();
        for _ in 0..A2_NUM_LAYERS {
            activations.push(serde_json::json!({
                "type": "LeakyReLU",
                "negative_slope": 0.01
            }));
        }
        let mut gating = Vec::new();
        for _ in 0..A2_NUM_LAYERS {
            gating.push(serde_json::json!("none"));
        }
        let secondary: Vec<serde_json::Value> = vec![serde_json::Value::Null; A2_NUM_LAYERS];

        serde_json::json!({
            "activation": activations,
            "gating_mode": gating,
            "secondary_activation": secondary
        })
    }

    #[test]
    fn test_parse_standard_a2_activations() {
        let raw = make_a2_layer_json();
        let config = parse_layer_activations(&raw, A2_NUM_LAYERS).expect("should parse");

        assert_eq!(config.activations.len(), A2_NUM_LAYERS);
        for act in &config.activations {
            assert_eq!(
                act,
                &ActivationType::LeakyReLU {
                    negative_slope: 0.01
                }
            );
        }
        for mode in &config.gating_modes {
            assert_eq!(mode, &GatingMode::None);
        }
        for sec in &config.secondary_activations {
            assert!(
                sec.is_none(),
                "expected None secondary activation, got {sec:?}"
            );
        }
    }

    #[test]
    fn test_parse_heterogeneous_activations() {
        let raw = serde_json::json!({
            "activation": [
                {"type": "LeakyReLU", "negative_slope": 0.01},
                {"type": "Tanh"},
                {"type": "ReLU"},
                {"type": "Sigmoid"},
                {"type": "SiLU"},
                {"type": "HardTanh"},
                {"type": "FastTanh"},
                {"type": "Softsign"},
                {"type": "HardSwish"},
                {"type": "PReLU", "negative_slopes": [0.01]},
                {"type": "LeakyHardTanh", "min_val": -1.0, "max_val": 1.0, "min_slope": 0.01, "max_slope": 0.01},
                {"type": "Tanh"},
                {"type": "ReLU"},
                {"type": "Tanh"},
                {"type": "Sigmoid"},
                {"type": "SiLU"},
                {"type": "Tanh"},
                {"type": "HardTanh"},
                {"type": "LeakyReLU", "negative_slope": 0.02},
                {"type": "FastTanh"},
                {"type": "Softsign"},
                {"type": "LeakyReLU", "negative_slope": 0.01},
                {"type": "Tanh"}
            ]
        });

        let activations =
            parse_activations_from_json(&raw, 23).expect("should parse heterogeneous activations");
        assert_eq!(activations.len(), 23);
        assert_eq!(
            activations[0],
            ActivationType::LeakyReLU {
                negative_slope: 0.01
            }
        );
        assert_eq!(activations[1], ActivationType::Tanh);
        assert_eq!(activations[2], ActivationType::ReLU);
        assert_eq!(activations[3], ActivationType::Sigmoid);
        assert_eq!(activations[4], ActivationType::SiLU);
        assert_eq!(
            activations[18],
            ActivationType::LeakyReLU {
                negative_slope: 0.02
            }
        );
    }

    #[test]
    fn test_parse_gating_modes() {
        let raw = serde_json::json!({
            "gating_mode": [
                "none", "gated", "blended", "none", "none",
                "none", "none", "none", "none", "none",
                "none", "none", "none", "none", "none",
                "none", "none", "none", "none", "none",
                "gated", "blended", "none"
            ]
        });

        let modes = parse_gating_modes_from_json(&raw, 23);
        assert_eq!(modes.len(), 23);
        assert_eq!(modes[0], GatingMode::None);
        assert_eq!(modes[1], GatingMode::Gated);
        assert_eq!(modes[2], GatingMode::Blended);
        assert_eq!(modes[20], GatingMode::Gated);
        assert_eq!(modes[21], GatingMode::Blended);
        assert_eq!(modes[22], GatingMode::None);
    }

    #[test]
    fn test_parse_gating_modes_absent() {
        let raw = serde_json::json!({});
        let modes = parse_gating_modes_from_json(&raw, 23);
        assert_eq!(modes.len(), 23);
        for mode in &modes {
            assert_eq!(mode, &GatingMode::None);
        }
    }

    #[test]
    fn test_parse_gating_modes_null() {
        let raw = serde_json::json!({"gating_mode": null});
        let modes = parse_gating_modes_from_json(&raw, 23);
        assert_eq!(modes.len(), 23);
        for mode in &modes {
            assert_eq!(mode, &GatingMode::None);
        }
    }

    #[test]
    fn test_parse_secondary_activations() {
        let raw = serde_json::json!({
            "secondary_activation": [
                null,
                {"type": "Sigmoid"},
                {"type": "Tanh"},
                null, null, null, null, null, null, null,
                null, null, null, null, null, null, null,
                null, null, null, null, null, null
            ]
        });

        let sec = parse_secondary_activations_from_json(&raw, 23);
        assert_eq!(sec.len(), 23);
        assert!(sec[0].is_none());
        assert_eq!(sec[1], Some(ActivationType::Sigmoid));
        assert_eq!(sec[2], Some(ActivationType::Tanh));
        assert!(sec[3].is_none());
        assert!(sec[22].is_none());
    }

    #[test]
    fn test_parse_secondary_activations_absent() {
        let raw = serde_json::json!({});
        let sec = parse_secondary_activations_from_json(&raw, 23);
        assert_eq!(sec.len(), 23);
        for s in &sec {
            assert!(s.is_none());
        }
    }

    #[test]
    fn test_parse_activations_wrong_length_rejected() {
        let raw = serde_json::json!({
            "activation": [
                {"type": "LeakyReLU", "negative_slope": 0.01}
            ]
        });
        assert!(parse_activations_from_json(&raw, 23).is_none());
    }

    #[test]
    fn test_parse_activations_missing_rejected() {
        let raw = serde_json::json!({});
        assert!(parse_activations_from_json(&raw, 23).is_none());
    }

    #[test]
    fn test_parse_layer_activations_integrated() {
        let raw = make_a2_layer_json();
        let config = parse_layer_activations(&raw, A2_NUM_LAYERS).expect("should parse");
        assert_eq!(config.activations.len(), A2_NUM_LAYERS);
        assert_eq!(config.gating_modes.len(), A2_NUM_LAYERS);
        assert_eq!(config.secondary_activations.len(), A2_NUM_LAYERS);
    }
}
