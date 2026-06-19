// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Gating and blending module for the NAM A2 architecture.
//!
//! Implements the real gating and blending operations with parity to
//! `NAM/gating_activations.h`. Buffers are pre-allocated at initialization
//! (zero-alloc on the hot-path, RT-Safe).

use super::activations::{ActivationFn, ActivationType};
use crate::math::common::AlignedVec;

/// Gating modes for WaveNet layers.
///
/// Determines how the layer processes duplicated bottleneck channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GatingMode {
    /// No gating or blending — standard activation (fast-path A2).
    #[default]
    None,
    /// Traditional gating: element-wise multiply of activated input × activated gate.
    Gated,
    /// Blending: weighted average between activated and pre-activated input, with
    /// alpha computed from gate channels.
    Blended,
}

/// Configuration for Gating-type activation.
///
/// Splits a 2×channels buffer into input (first half) and gating (second half).
/// Applies `input_activation` to the input half and `gating_activation` to the
/// gating half, then multiplies them element-wise.
///
/// Gating is fully in-place — no scratch buffer needed. Reads and writes
/// operate on disjoint halves of the buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct GatingActivationConfig {
    /// Activation function for input channels.
    pub input_activation: ActivationType,
    /// Activation function for gating channels.
    pub gating_activation: ActivationType,
}

impl GatingActivationConfig {
    /// Creates a new gating config.
    pub fn new(input_activation: ActivationType, gating_activation: ActivationType) -> Self {
        Self {
            input_activation,
            gating_activation,
        }
    }

    /// Applies gating in-place on a buffer of `2*ch` elements.
    ///
    /// - `buf[0..ch]` — input channels: activated, then multiplied by gate.
    /// - `buf[ch..2*ch]` — gating channels: activated, then used as multiplier.
    ///
    /// After this call, `buf[0..ch]` contains the gated output (the gate half
    /// is consumed). Matches `GatingActivation::process()` in the C++ reference.
    #[inline]
    pub fn apply_gating(&self, buf: &mut [f32]) {
        let ch = buf.len() / 2;
        debug_assert!(
            ch * 2 == buf.len(),
            "gating: buf length must be even (2×ch)"
        );

        self.input_activation.apply(&mut buf[..ch]);
        self.gating_activation.apply(&mut buf[ch..]);

        for i in 0..ch {
            buf[i] *= buf[ch + i];
        }
    }
}

/// Configuration for Blending-type activation.
///
/// Splits a 2×channels buffer into input (first half) and blending (second half).
/// Applies `input_activation` to the input half and `blending_activation` to
/// produce alpha ∈ [0,1] from the second half. Output is the weighted average:
///
/// ```text
/// output[i] = activated_input[i] * alpha[i] + original_input[i] * (1 - alpha[i])
/// ```
///
/// A scratch buffer is pre-allocated at construction to save original input
/// values, ensuring zero allocations on the hot-path (RT-Safe).
#[derive(Debug, Clone)]
pub struct BlendingActivationConfig {
    /// Activation function for input channels.
    pub input_activation: ActivationType,
    /// Activation function for blending channels (determines alpha ∈ [0,1]).
    pub blending_activation: ActivationType,
    /// Scratch buffer for original input values during blending.
    scratch: AlignedVec<f32>,
}

impl PartialEq for BlendingActivationConfig {
    fn eq(&self, other: &Self) -> bool {
        self.input_activation == other.input_activation
            && self.blending_activation == other.blending_activation
            && self.scratch.len() == other.scratch.len()
    }
}

impl BlendingActivationConfig {
    /// Creates a new blending config with a pre-allocated scratch buffer.
    ///
    /// `ch` is the number of channels per half (total buffer = 2×ch).
    pub fn new(
        input_activation: ActivationType,
        blending_activation: ActivationType,
        ch: usize,
    ) -> Self {
        Self {
            input_activation,
            blending_activation,
            scratch: AlignedVec::new(ch, 0.0f32),
        }
    }

    /// Applies blending in-place on a buffer of `2*ch` elements.
    ///
    /// - `buf[0..ch]` — input channels: activated, then blended with original.
    /// - `buf[ch..2*ch]` — blending channels: activated to produce alpha.
    ///
    /// After this call, `buf[0..ch]` contains the blended output.
    /// Matches `BlendingActivation::process()` in the C++ reference.
    ///
    /// # Panics
    ///
    /// Panics in debug if `buf.len()` exceeds the pre-allocated scratch capacity.
    #[inline]
    pub fn apply_blending(&mut self, buf: &mut [f32]) {
        let ch = buf.len() / 2;
        debug_assert!(
            ch * 2 == buf.len(),
            "blending: buf length must be even (2×ch)"
        );
        debug_assert!(
            self.scratch.len() >= ch,
            "blending: scratch too small ({}) for ch={}",
            self.scratch.len(),
            ch
        );

        self.scratch[..ch].copy_from_slice(&buf[..ch]);

        self.input_activation.apply(&mut buf[..ch]);
        self.blending_activation.apply(&mut buf[ch..]);

        for i in 0..ch {
            let alpha = buf[ch + i];
            buf[i] = (buf[i] - self.scratch[i]).mul_add(alpha, self.scratch[i]);
        }
    }

    /// Returns the pre-allocated channel count.
    #[inline]
    pub fn channels(&self) -> usize {
        self.scratch.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_mode_default() {
        assert_eq!(GatingMode::default(), GatingMode::None);
    }

    #[test]
    fn test_gating_apply_tanh_sigmoid() {
        let config = GatingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid);
        let mut buf = vec![0.5f32, -0.3, 0.8, -0.6];
        let mut golden = buf.clone();

        // Golden: apply activations to each half independently, then multiply.
        ActivationType::Tanh.apply(&mut golden[..2]);
        ActivationType::Sigmoid.apply(&mut golden[2..]);
        golden[0] *= golden[2];
        golden[1] *= golden[3];

        config.apply_gating(&mut buf);

        // Compare only the first half (output channels).
        assert!((buf[0] - golden[0]).abs() < 1e-7);
        assert!((buf[1] - golden[1]).abs() < 1e-7);
    }

    #[test]
    fn test_blending_apply_tanh_sigmoid() {
        let mut config =
            BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 2);
        let mut buf = [0.5f32, -0.3f32, 0.8f32, -0.6f32];
        let orig = buf;
        let mut golden = [0.0f32; 2];

        // Golden: apply activations, then blend.
        let mut activated = [orig[0], orig[1]];
        ActivationType::Tanh.apply(&mut activated);
        let mut alpha = [orig[2], orig[3]];
        ActivationType::Sigmoid.apply(&mut alpha);
        for i in 0..2 {
            golden[i] = activated[i].mul_add(alpha[i], orig[i] * (1.0 - alpha[i]));
        }

        config.apply_blending(&mut buf);

        // Compare only the first half (output channels).
        assert!((buf[0] - golden[0]).abs() < 1e-7);
        assert!((buf[1] - golden[1]).abs() < 1e-7);
    }

    #[test]
    fn test_gating_channel_pairs_zero_input() {
        let config = GatingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid);
        let mut buf = vec![0.0f32; 8]; // ch=4
        config.apply_gating(&mut buf);
        // tanh(0)=0, sigmoid(0)=0.5, product=0
        for v in buf.iter().take(4) {
            assert!((*v).abs() < 1e-7);
        }
    }

    #[test]
    fn test_blending_channel_pairs_zero_input() {
        let mut config =
            BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 4);
        let mut buf = vec![0.0f32; 8]; // ch=4
        config.apply_blending(&mut buf);
        // tanh(0)=0, sigmoid(0)=0.5, blended = 0 * 0.5 + 0 * 0.5 = 0
        for v in buf.iter().take(4) {
            assert!((*v).abs() < 1e-7);
        }
    }

    #[test]
    fn test_gating_unity_gate() {
        let config = GatingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid);
        let mut buf = vec![0.5f32, -0.3f32, 10.0f32, 10.0f32]; // ch=2
        let mut expected = [0.5f32, -0.3f32];
        ActivationType::Tanh.apply(&mut expected);
        // sigmoid(10) ≈ 1, so gate multiplier ≈ 1
        config.apply_gating(&mut buf);
        assert!((buf[0] - expected[0]).abs() < 1e-4);
        assert!((buf[1] - expected[1]).abs() < 1e-4);
    }

    #[test]
    fn test_blending_alpha_one() {
        let mut config =
            BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 2);
        let mut buf = vec![0.5f32, -0.3f32, 10.0f32, 10.0f32]; // ch=2
        let mut expected = [0.5f32, -0.3f32];
        ActivationType::Tanh.apply(&mut expected);
        // sigmoid(10) ≈ 1 → blended ≈ activated_input
        config.apply_blending(&mut buf);
        assert!((buf[0] - expected[0]).abs() < 1e-4);
        assert!((buf[1] - expected[1]).abs() < 1e-4);
    }

    #[test]
    fn test_blending_alpha_zero() {
        let mut config =
            BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 2);
        let orig = [0.5f32, -0.3f32];
        let mut buf = [orig[0], orig[1], -10.0f32, -10.0f32]; // ch=2, alpha≈0
        let mut golden = orig;

        // Golden with actual sigmoid: alpha ≈ 0 → blended ≈ original.
        // For x ≤ -8, sigmoid(x) = 0 in our approximation.
        let mut alpha = [-10.0f32, -10.0f32];
        ActivationType::Sigmoid.apply(&mut alpha);
        assert!(alpha[0].abs() < 1e-7);

        let mut activated = orig;
        ActivationType::Tanh.apply(&mut activated);
        for i in 0..2 {
            golden[i] = activated[i].mul_add(alpha[i], orig[i] * (1.0 - alpha[i]));
        }

        config.apply_blending(&mut buf);
        assert!((buf[0] - golden[0]).abs() < 1e-4);
        assert!((buf[1] - golden[1]).abs() < 1e-4);
    }

    #[test]
    fn test_blending_scratch_preserved() {
        let mut config =
            BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 2);
        assert_eq!(config.channels(), 2);

        // Apply twice with same config — scratch must not leak state.
        let mut buf1 = [0.5f32, 0.3f32, 0.8f32, 0.6f32];
        let mut buf2 = [0.5f32, 0.3f32, 0.8f32, 0.6f32];
        config.apply_blending(&mut buf1);
        config.apply_blending(&mut buf2);
        assert_eq!(&buf1[..2], &buf2[..2]);
    }
}
