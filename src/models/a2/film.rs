// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Feature-wise Linear Modulation (FiLM) module for the NAM A2 architecture.
//!
//! FiLM enables the model to adapt its behavior based on external
//! conditioning signals, applying per-channel scale and shift:
//!
//! `output[c] = input[c] * scale[c] + shift[c]`
//!
//! The `_cond_to_scale_shift` operation (C++ `NAM/film.h:28-85`) maps
//! the condition vector to per-channel scale+shift via a Conv1x1
//! implemented as a dense GEMV (matrix-vector multiply with AVX2 FMA).
//!
//! # RT-Safety
//! - The `scale_shift_buf` is pre-allocated in `load()` — zero allocation
//!   on the hot-path.
//! - The `apply_modulation` inner loop uses `chunks_exact_mut` with no
//!   internal branches.
//! - All SIMD operates on 64-byte aligned `AlignedVec` buffers.
//!
//! # Source of truth
//! - `NAM/film.h:28-85` (`_cond_to_scale_shift` via Conv1x1 / Dense)
//! - `NeuralAmpModelerCore/NAM/film.cpp` (per-channel modulation)

use crate::math::common::AlignedVec;

use core::arch::x86_64::*;

/// Configuration for a FiLM layer or operation.
///
/// Corresponds to the `_FiLMParams` struct in C++.
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

/// Concrete FiLM layer — `_cond_to_scale_shift` + per-channel modulation.
///
/// # Weight layout
/// Row-major `[out_rows][cond_size]` where:
/// - When `groups == 1`: `out_rows = channels * (shift ? 2 : 1)`.
///   First `channels` rows produce `scale`; next `channels` rows produce `shift`.
/// - When `groups > 1`: weights are partitioned into `groups` independent blocks,
///   each of size `out_per_group * (cond_size / groups)`.
///
/// The `scale_shift_buf` always has `channels * 2` elements — when `shift == false`
/// the second half remains zero so the modulation step stays branch-free.
#[derive(Clone)]
#[repr(align(64))]
pub struct FiLMLayer {
    /// FiLM configuration (active, shift, groups).
    pub config: FiLMConfig,
    /// Dimension of the conditioning vector.
    pub cond_size: usize,
    /// Number of feature channels to modulate.
    pub channels: usize,
    /// Row-major weight matrix for `_cond_to_scale_shift`.
    pub weights: AlignedVec<f32>,
    /// Bias vector, indexed by global output channel.
    pub bias: AlignedVec<f32>,
    /// Pre-allocated scratch buffer: `[scale[..channels], shift[..channels]]`.
    scale_shift_buf: AlignedVec<f32>,
}

impl FiLMLayer {
    /// Allocates a FiLM layer with the given topology and copies in the weights.
    ///
    /// # Buffers
    /// `scale_shift_buf` is pre-allocated here — no heap traffic on the DSP
    /// hot-path.
    pub fn load(
        config: FiLMConfig,
        cond_size: usize,
        channels: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        Self {
            config,
            cond_size,
            channels,
            weights: AlignedVec::from_vec(weights),
            bias: AlignedVec::from_vec(bias),
            scale_shift_buf: AlignedVec::new(channels * 2, 0.0f32),
        }
    }

    /// Processes FiLM modulation over the input buffer.
    ///
    /// 1. `_cond_to_scale_shift`: maps `condition` → `scale[..channels]` +
    ///    `shift[..channels]` via a dense GEMV with AVX2 FMA.
    /// 2. Per-channel modulation: `input[c] = input[c] * scale[c] + shift[c]`.
    ///
    /// # Safety
    /// `input.len()` must equal `self.channels`.
    /// `condition.len()` must equal `self.cond_size`.
    #[inline(always)]
    pub unsafe fn process(&mut self, input: &mut [f32], condition: &[f32]) {
        self.cond_to_scale_shift(condition);
        self.apply_modulation(input);
    }

    // ── _cond_to_scale_shift ──────────────────────────────────────────

    #[inline(always)]
    unsafe fn cond_to_scale_shift(&mut self, condition: &[f32]) {
        let g = self.config.groups as usize;
        let ch_per_group = self.channels / g;
        let cond_per_group = self.cond_size / g;
        let out_per_group = if self.config.shift {
            ch_per_group * 2
        } else {
            ch_per_group
        };

        for grp in 0..g {
            let cond_slice =
                condition.get_unchecked(grp * cond_per_group..(grp + 1) * cond_per_group);
            let w_offset = grp * out_per_group * cond_per_group;

            for row in 0..out_per_group {
                let global_out = if row < ch_per_group {
                    grp * ch_per_group + row
                } else {
                    self.channels + grp * ch_per_group + (row - ch_per_group)
                };

                let mut sum = *self.bias.get_unchecked(global_out);
                let w_start = w_offset + row * cond_per_group;
                let w_row = self
                    .weights
                    .get_unchecked(w_start..w_start + cond_per_group);
                sum += dot_product_avx2(w_row, cond_slice);
                *self.scale_shift_buf.get_unchecked_mut(global_out) = sum;
            }
        }

        if !self.config.shift {
            for c in self.channels..self.channels * 2 {
                *self.scale_shift_buf.get_unchecked_mut(c) = 0.0;
            }
        }
    }

    // ── Per-channel modulation ────────────────────────────────────────

    /// `input[c] *= scale[c]; input[c] += shift[c]` — AVX2, 8-wide.
    ///
    /// Uses `chunks_exact_mut` for the SIMD block and a scalar tail
    /// for any remainder. The inner SIMD block contains no branches.
    #[inline(always)]
    unsafe fn apply_modulation(&mut self, input: &mut [f32]) {
        let scale = &self.scale_shift_buf[..self.channels];
        let shift = &self.scale_shift_buf[self.channels..self.channels * 2];

        let mut off = 0;
        for in_chunk in input.chunks_exact_mut(8) {
            let v_in = _mm256_loadu_ps(in_chunk.as_ptr());
            let v_scale = _mm256_loadu_ps(scale.as_ptr().add(off));
            let v_shift = _mm256_loadu_ps(shift.as_ptr().add(off));
            _mm256_storeu_ps(
                in_chunk.as_mut_ptr(),
                _mm256_fmadd_ps(v_in, v_scale, v_shift),
            );
            off += 8;
        }

        for c in off..self.channels {
            *input.get_unchecked_mut(c) =
                input.get_unchecked(c) * scale.get_unchecked(c) + shift.get_unchecked(c);
        }
    }
}

// ── FilmBlock: mutable FiLM references for block-level dispatch ──────────

/// Bundle of mutable references to the 8 FiLM insertion points in an A2 layer.
///
/// Used by `layer_forward_ch{3,8}_block` to conditionally apply FiLM
/// at the correct positions in the signal chain without per-point parameter
/// explosion.
pub struct FilmBlock<'a> {
    /// FiLM before dilated convolution.
    pub conv_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after dilated convolution.
    pub conv_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM before input mixin (same insertion point as conv_post_film).
    pub input_mixin_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after input mixin, before activation.
    pub input_mixin_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM before activation (same insertion point as input_mixin_post_film).
    pub activation_pre_film: Option<&'a mut FiLMLayer>,
    /// FiLM after activation.
    pub activation_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM after layer 1x1 residual.
    pub layer1x1_post_film: Option<&'a mut FiLMLayer>,
    /// FiLM after head 1x1 (reserved for future general A2 engine).
    pub head1x1_post_film: Option<&'a mut FiLMLayer>,
}

impl<'a> FilmBlock<'a> {
    /// Creates an empty `FilmBlock` with all fields set to `None`.
    /// Useful for tests and for the fast-path when no FiLM is active.
    pub fn empty() -> Self {
        Self {
            conv_pre_film: None,
            conv_post_film: None,
            input_mixin_pre_film: None,
            input_mixin_post_film: None,
            activation_pre_film: None,
            activation_post_film: None,
            layer1x1_post_film: None,
            head1x1_post_film: None,
        }
    }
}

impl super::layer::A2Layer {
    /// Returns a [`FilmBlock`] with mutable references to all 8 FiLM insertion points.
    #[inline]
    pub fn film_block(&mut self) -> FilmBlock<'_> {
        FilmBlock {
            conv_pre_film: self.conv_pre_film.as_mut(),
            conv_post_film: self.conv_post_film.as_mut(),
            input_mixin_pre_film: self.input_mixin_pre_film.as_mut(),
            input_mixin_post_film: self.input_mixin_post_film.as_mut(),
            activation_pre_film: self.activation_pre_film.as_mut(),
            activation_post_film: self.activation_post_film.as_mut(),
            layer1x1_post_film: self.layer1x1_post_film.as_mut(),
            head1x1_post_film: self.head1x1_post_film.as_mut(),
        }
    }
}

// ── AVX2 dot product ───────────────────────────────────────────────────

/// Dot product `sum(a[i] * b[i])` — AVX2+FMA, 8-wide.
///
/// Uses dual accumulator and horizontal reduction matching the
/// `convolve_mono_avx2` pattern.
#[inline(always)]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;

    while i + 16 <= len {
        let ha0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let hb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(ha0, hb0, sum0);

        let ha1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let hb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        sum1 = _mm256_fmadd_ps(ha1, hb1, sum1);

        i += 16;
    }

    while i + 8 <= len {
        let ha = _mm256_loadu_ps(a.as_ptr().add(i));
        let hb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(ha, hb, sum0);
        i += 8;
    }

    let sum = _mm256_add_ps(sum0, sum1);
    let hi128 = _mm256_extractf128_ps(sum, 1);
    let lo128 = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let r = _mm_add_ss(sums, shuf2);
    let mut out = _mm_cvtss_f32(r);

    while i < len {
        out += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }

    out
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference for `_cond_to_scale_shift` — computes
    /// `scale_shift[o] = bias[o] + Σ weight[o*cond_sz + i] * cond[i]`.
    fn cond_to_scale_shift_ref(
        weights: &[f32],
        bias: &[f32],
        condition: &[f32],
        channels: usize,
        cond_size: usize,
        shift: bool,
        groups: u32,
    ) -> Vec<f32> {
        let g = groups as usize;
        let ch_per_group = channels / g;
        let cond_per_group = cond_size / g;
        let out_per_group = if shift {
            ch_per_group * 2
        } else {
            ch_per_group
        };
        let mut out = vec![0.0f32; channels * 2];

        for grp in 0..g {
            let cond_slice = &condition[grp * cond_per_group..(grp + 1) * cond_per_group];
            let w_offset = grp * out_per_group * cond_per_group;

            for row in 0..out_per_group {
                let global_out = if row < ch_per_group {
                    grp * ch_per_group + row
                } else {
                    channels + grp * ch_per_group + (row - ch_per_group)
                };
                let mut sum = bias[global_out];
                let w_start = w_offset + row * cond_per_group;
                for i in 0..cond_per_group {
                    sum += weights[w_start + i] * cond_slice[i];
                }
                out[global_out] = sum;
            }
        }
        out
    }

    /// Scalar reference for per-channel modulation.
    fn apply_modulation_ref(input: &[f32], scale_shift: &[f32]) -> Vec<f32> {
        let ch = input.len();
        let scale = &scale_shift[..ch];
        let shift = &scale_shift[ch..ch * 2];
        input
            .iter()
            .enumerate()
            .map(|(c, &v)| v * scale[c] + shift[c])
            .collect()
    }

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

    /// FiLM with `groups=1, shift=true` — identity scale + zero shift.
    #[test]
    fn test_film_process_identity_shift() {
        let cond_size = 4;
        let channels = 8;
        let config = FiLMConfig {
            active: true,
            shift: true,
            groups: 1,
        };

        // weights: 16 rows (8 scale + 8 shift) × 4 cond
        let mut weights = vec![0.0f32; channels * 2 * cond_size];
        // Set scale weights to produce scale[c] = 1.0 via identity mapping
        for c in 0..channels {
            weights[c * cond_size + (c % cond_size)] = 1.0;
        }
        let bias = vec![0.0f32; channels * 2];

        let mut layer = FiLMLayer::load(config, cond_size, channels, weights, bias);

        // All-ones condition so every scale[c] = Σ weight[c, i] * 1.0 = 1.0
        let condition = vec![1.0f32; cond_size];
        let mut input = vec![2.0f32, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0];
        let expected_input = input.clone();

        unsafe { layer.process(&mut input, &condition) };

        // Identity scale (≈1.0) + zero shift → output ≈ input
        for c in 0..channels {
            assert!(
                (input[c] - expected_input[c]).abs() < 1e-5,
                "channel {}: expected {}, got {}",
                c,
                expected_input[c],
                input[c]
            );
        }
    }

    /// FiLM with `groups=1, shift=false` — scale-only.
    #[test]
    fn test_film_process_scale_only() {
        let cond_size = 3;
        let channels = 8;
        let config = FiLMConfig {
            active: true,
            shift: false,
            groups: 1,
        };

        // weights: 8 rows (scale only) × 3 cond
        let mut weights = vec![0.0f32; channels * cond_size];
        // Diagonal: scale[c] = condition[c % 3] * 2.0
        for c in 0..channels {
            weights[c * cond_size + (c % cond_size)] = 2.0;
        }
        let bias = vec![0.1f32; channels]; // small bias

        let mut layer = FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone());

        let condition = vec![0.5f32, 0.25, 0.125];
        let mut input = vec![1.0f32; channels];

        // Reference: scale = W * cond + b
        let ref_scale_shift =
            cond_to_scale_shift_ref(&weights, &bias, &condition, channels, cond_size, false, 1);
        let expected = apply_modulation_ref(&input, &ref_scale_shift);

        unsafe { layer.process(&mut input, &condition) };

        for c in 0..channels {
            assert!(
                (input[c] - expected[c]).abs() < 1e-5,
                "channel {}: expected {}, got {}",
                c,
                expected[c],
                input[c]
            );
        }
    }

    /// FiLM with `groups=2, shift=true`.
    #[test]
    fn test_film_process_groups_shift() {
        let cond_size = 6;
        let channels = 8;
        let groups = 2u32;
        let config = FiLMConfig {
            active: true,
            shift: true,
            groups,
        };

        let g = groups as usize;
        let ch_per_group = channels / g;
        let cond_per_group = cond_size / g;
        let out_per_group = ch_per_group * 2; // shift=true → 2×

        // Build weights: group0 → inputs 0..3, group1 → inputs 3..6
        let total_w = g * out_per_group * cond_per_group;
        let mut weights = vec![0.0f32; total_w];
        let mut bias = vec![0.0f32; channels * 2];

        // Scale each group's output to be 2× its condition slice
        for grp in 0..g {
            let w_offset = grp * out_per_group * cond_per_group;
            for row in 0..out_per_group {
                let w_start = w_offset + row * cond_per_group;
                for ic in 0..cond_per_group {
                    weights[w_start + ic] = 2.0;
                }
                let global_out = if row < ch_per_group {
                    grp * ch_per_group + row
                } else {
                    channels + grp * ch_per_group + (row - ch_per_group)
                };
                bias[global_out] = 0.5;
            }
        }

        let mut layer = FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone());

        let condition = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut input = vec![0.5f32; channels];

        let ref_scale_shift = cond_to_scale_shift_ref(
            &weights, &bias, &condition, channels, cond_size, true, groups,
        );
        let expected = apply_modulation_ref(&input, &ref_scale_shift);

        unsafe { layer.process(&mut input, &condition) };

        for c in 0..channels {
            assert!(
                (input[c] - expected[c]).abs() < 1e-4,
                "channel {}: expected {}, got {}",
                c,
                expected[c],
                input[c]
            );
        }
    }

    /// FiLM with odd channel count (3, A2-nano).
    #[test]
    fn test_film_process_odd_channels() {
        let cond_size = 3;
        let channels = 3;
        let config = FiLMConfig {
            active: true,
            shift: true,
            groups: 1,
        };

        let mut weights = vec![0.0f32; channels * 2 * cond_size];
        // Group 0: scale = cond[0] * 2, shift = cond[0] * 0.1
        for c in 0..channels {
            weights[c * cond_size + (c % cond_size)] = 2.0;
            weights[(channels + c) * cond_size + (c % cond_size)] = 0.1;
        }
        let bias = vec![0.0f32; channels * 2];

        let mut layer = FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone());

        let condition = vec![0.5f32, 0.25, 0.125];
        let mut input = vec![0.5f32, 1.0, 1.5];

        let ref_scale_shift =
            cond_to_scale_shift_ref(&weights, &bias, &condition, channels, cond_size, true, 1);
        let expected = apply_modulation_ref(&input, &ref_scale_shift);

        unsafe { layer.process(&mut input, &condition) };

        for c in 0..channels {
            assert!(
                (input[c] - expected[c]).abs() < 1e-5,
                "channel {}: expected {}, got {}",
                c,
                expected[c],
                input[c]
            );
        }
    }

    /// Verify `scale_shift_buf` is zeroed in the shift region when
    /// `shift == false`, so the modulation step produces correct results.
    #[test]
    fn test_film_shift_buffer_zeroed_when_shift_false() {
        let cond_size = 2;
        let channels = 4;
        let config = FiLMConfig {
            active: true,
            shift: false,
            groups: 1,
        };

        let weights = vec![1.0f32; channels * cond_size];
        let bias = vec![0.0f32; channels];

        let mut layer = FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone());

        let condition = vec![2.0f32, 3.0];
        let mut input = vec![1.0f32; channels];

        // Pre-fill shift region with garbage to ensure it gets zeroed
        for c in channels..channels * 2 {
            // SAFETY: buffer is pre-allocated via load()
            layer.scale_shift_buf[c] = 999.0;
        }

        unsafe { layer.process(&mut input, &condition) };

        // Shift region must be zero
        for c in channels..channels * 2 {
            assert_eq!(layer.scale_shift_buf[c], 0.0);
        }
    }

    /// Ensure `load()` copies weights correctly for groups > 1.
    #[test]
    fn test_film_load_copies_weights() {
        let cond_size = 4;
        let channels = 8;
        let groups = 2;
        let config = FiLMConfig {
            active: true,
            shift: true,
            groups,
        };

        let out_per_group = (channels / groups as usize) * 2;
        let cond_per_group = cond_size / groups as usize;
        let total_w = groups as usize * out_per_group * cond_per_group;
        let weights: Vec<f32> = (0..total_w).map(|i| i as f32 + 0.1).collect();
        let bias: Vec<f32> = (0..channels * 2).map(|i| i as f32 + 0.5).collect();

        let layer = FiLMLayer::load(config, cond_size, channels, weights.clone(), bias.clone());

        assert_eq!(layer.weights.len(), total_w);
        assert_eq!(layer.bias.len(), channels * 2);
        assert_eq!(layer.scale_shift_buf.len(), channels * 2);
        for (i, &w) in weights.iter().enumerate() {
            assert_eq!(layer.weights[i], w);
        }
        for (i, &b) in bias.iter().enumerate() {
            assert_eq!(layer.bias[i], b);
        }
    }
}
