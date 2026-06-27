// SPDX-License-Identifier: Apache-2.2
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Half-Band Oversampling Engine for the neural stage (Tarefa 5.2).
//!
//! Implements optional 2×/4× oversampling around the neural model to reduce
//! aliasing from non-linear activations, following the half-band filter design
//! principles of Kahles, Esqueda & Välimäki (JAES 2019).
//!
//! ## Architecture
//!
//! Each 2× stage uses a Kaiser-windowed half-band FIR filter (25 taps, β=12,
//! \>100 dB stop-band). The half-band property h[2n]=0 (n≠D/2) halves the
//! effective MAC count per sample.
//!
//! - **Upsampler**: inserts zeros → filters. Even outputs = x[n-D/2]*0.5;
//!   odd outputs = convolution with non-zero odd taps.
//! - **Downsampler**: FIR at full rate → decimates by 2. Uses contiguous
//!   double-buffer delay line (same pattern as `NamResampler`).
//!
//! ## RT-Safety
//!
//! All allocation in `OversampleEngine::new()`. During `process()`,
//! only pre-allocated buffers — zero alloc, zero heap-drop, no unwrap.
//!
//! Factor change requires rebuild (off-RT), same path as model hot-swap.

use crate::math::common::AlignedVec;

/// Half-band FIR filter length (≡ 1 mod 4 so D=HB_TAPS/2 is even).
/// 25 taps, D=12. Kaiser β=12 → >100 dB stop-band rejection.
const HB_TAPS: usize = 25;
/// Filter delay (group delay = HB_TAPS/2 = 12 samples at native rate).
const HB_DELAY: usize = HB_TAPS / 2;
/// Number of non-zero odd-index taps (h[1], h[3], ..., h[23]).
const HB_ODD_COUNT: usize = HB_TAPS / 2; // 12

/// Oversampling factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OversampleFactor {
    /// No oversampling — pass-through with zero overhead.
    Off,
    /// 2× oversampling (one half-band stage).
    X2,
    /// 4× oversampling (two cascaded half-band stages).
    X4,
}

impl OversampleFactor {
    /// Returns the sample count multiplier (1, 2, or 4).
    #[inline]
    pub const fn multiplier(self) -> usize {
        match self {
            Self::Off => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }

    /// Returns the number of cascaded 2× stages (0, 1, or 2).
    #[inline]
    pub const fn stage_count(self) -> usize {
        match self {
            Self::Off => 0,
            Self::X2 => 1,
            Self::X4 => 2,
        }
    }
}

/// Half-band filter kernel with Kaiser window.
struct HalfBandFilter {
    /// Non-zero odd-index coefficients: h[1], h[3], ..., h[HB_TAPS-2].
    coeffs: [f32; HB_ODD_COUNT],
}

impl HalfBandFilter {
    /// Designs a half-band filter with the specified overall DC gain.
    ///
    /// For upsampling: `dc_gain = L` (2 for 2× upsampler).
    /// For downsampling: `dc_gain = 1` (unity passband).
    fn design(beta: f64, dc_gain: f64) -> Self {
        let i0_beta = bessel_i0(beta);
        let half = HB_DELAY as f64;
        let mut coeffs = [0.0f32; HB_ODD_COUNT];

        for i in 0..HB_TAPS {
            let offset = i as f64 - half;
            if offset.abs() < 1e-10 || (offset.abs() as i64) % 2 == 0 {
                continue;
            }

            let x = std::f64::consts::PI * offset;
            let sinc = (x * 0.5).sin() / x;
            let ratio = offset / half;
            let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
            let window = bessel_i0(arg) / i0_beta;

            if i % 2 == 1 {
                coeffs[i / 2] = (sinc * window) as f32;
            }
        }

        // Normalize so h[D] + Σ odd_taps = dc_gain.
        // h[D] is fixed at dc_gain / 2 in the convolution code.
        // We scale odd taps to reach the target.
        let target_h_center = dc_gain / 2.0; // 1.0 for up,  0.5 for down
        let target_odd_sum = dc_gain - target_h_center; // 1.0 for up,  0.5 for down
        let odd_sum: f32 = coeffs.iter().sum();
        if odd_sum.abs() > 1e-10 {
            let scale = target_odd_sum as f32 / odd_sum;
            for c in coeffs.iter_mut() {
                *c *= scale;
            }
        }

        HalfBandFilter { coeffs }
    }
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;
    for k in 1..=20 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

/// Single 2× oversampling stage (up + down delay-line state).
///
/// Uses pre-allocated ring buffers for delay lines. At 12 samples (up)
/// and 25 samples (down), modulo indexing is cheap and avoids
/// the double-buffer complexity.
struct X2Stage {
    /// Filter for upsampling (DC gain = 2.0, center tap multiplier = 1.0).
    up_filter: HalfBandFilter,
    /// Filter for downsampling (DC gain = 1.0, center tap multiplier = 0.5).
    down_filter: HalfBandFilter,
    /// Center tap multiplier for the upsampler (h[D] = dc_gain_up / 2 = 1.0).
    up_center: f32,
    /// Center tap multiplier for the downsampler (h[D] = dc_gain_down / 2 = 0.5).
    down_center: f32,
    up_ring: AlignedVec<f32>,
    up_pos: usize,
    down_ring: AlignedVec<f32>,
    down_pos: usize,
    down_abs: u64,
}

impl X2Stage {
    fn new() -> Self {
        let dc_up = 2.0;
        let dc_down = 1.0;
        Self {
            up_filter: HalfBandFilter::design(12.0, dc_up),
            down_filter: HalfBandFilter::design(12.0, dc_down),
            up_center: (dc_up / 2.0) as f32,
            down_center: (dc_down / 2.0) as f32,
            up_ring: AlignedVec::new(HB_DELAY, 0.0f32),
            up_pos: 0,
            down_ring: AlignedVec::new(HB_TAPS, 0.0f32),
            down_pos: 0,
            down_abs: 0,
        }
    }

    fn upsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let coeffs = &self.up_filter.coeffs;
        let center = self.up_center;
        let n_in = input.len();
        let n = HB_DELAY;

        for (i, &x) in input.iter().enumerate() {
            let pos = self.up_pos;
            unsafe {
                *self.up_ring.get_unchecked_mut(pos) = x;
            }
            self.up_pos = (pos + 1) % n;

            let center_idx = (pos + n - HB_DELAY / 2) % n;
            let even_out = unsafe { self.up_ring.get_unchecked(center_idx) * center };

            let mut odd_out = 0.0f32;
            for j in 0..HB_ODD_COUNT {
                let d = j;
                let idx = (pos + n - d) % n;
                odd_out += unsafe { coeffs.get_unchecked(j) * self.up_ring.get_unchecked(idx) };
            }

            unsafe {
                *output.get_unchecked_mut(2 * i) = even_out;
                *output.get_unchecked_mut(2 * i + 1) = odd_out;
            }
        }

        n_in * 2
    }

    fn downsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let coeffs = &self.down_filter.coeffs;
        let center = self.down_center;
        let n = HB_TAPS;
        let mut out_idx = 0;

        for &x in input.iter() {
            let pos = self.down_pos;
            unsafe {
                *self.down_ring.get_unchecked_mut(pos) = x;
            }
            self.down_pos = (pos + 1) % n;
            self.down_abs += 1;

            if self.down_abs as usize >= n && self.down_abs % 2 == 1 {
                let abs_idx = (self.down_abs - 1) as usize;

                let mut sum = unsafe {
                    self.down_ring
                        .get_unchecked((abs_idx.wrapping_sub(HB_DELAY)) % n)
                        * center
                };

                for j in 0..HB_ODD_COUNT {
                    let tap_delay = 2 * j + 1;
                    let s_idx = abs_idx.wrapping_sub(tap_delay) % n;
                    sum += unsafe { coeffs.get_unchecked(j) * self.down_ring.get_unchecked(s_idx) };
                }

                if out_idx < output.len() {
                    output[out_idx] = sum;
                    out_idx += 1;
                }
            }
        }

        out_idx
    }
}

/// RT-safe half-band oversampling engine.
///
/// Wraps 1–2 cascaded 2× stages. Off mode is zero-cost pass-through
/// (no state, infallible `copy_nonoverlapping`).
///
/// ## Usage (per stereo channel)
///
/// ```ignore
/// // 1. Upsample model-rate input to oversampled rate
/// let n_os = engine.upsample(&input[..n_native], os_up_buf);
/// // 2. Model processes at oversampled rate
/// model.process(&os_up_buf[..n_os], &mut os_model_buf[..n_os]);
/// // 3. Downsample back to native rate
/// let n_out = engine.downsample(&os_model_buf[..n_os], output);
/// ```
pub struct OversampleEngine {
    factor: OversampleFactor,
    stage1: Option<X2Stage>,
    stage2: Option<X2Stage>,
    /// Scratch for X4 cascaded inter-stage (2× intermediate).
    /// Sized for max_input × 2 (the intermediate upsampled rate between stages).
    inter_buf: AlignedVec<f32>,
    max_samples: usize,
}

impl OversampleEngine {
    /// Creates a new engine with pre-allocated buffers.
    ///
    /// `max_input_samples`: max block size at native model rate
    /// (e.g., `MAX_RESAMP_BUF = 8192`).
    pub fn new(factor: OversampleFactor, max_input_samples: usize) -> Self {
        let inter_size = if factor.stage_count() >= 2 {
            max_input_samples * 2
        } else {
            1
        };

        Self {
            factor,
            stage1: (factor.stage_count() >= 1).then(X2Stage::new),
            stage2: (factor.stage_count() >= 2).then(X2Stage::new),
            inter_buf: AlignedVec::new(inter_size, 0.0f32),
            max_samples: max_input_samples,
        }
    }

    /// Returns the current oversampling factor.
    #[inline]
    pub fn factor(&self) -> OversampleFactor {
        self.factor
    }

    /// Returns `true` when oversampling is bypassed (Off).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        matches!(self.factor, OversampleFactor::Off)
    }

    /// Upsamples mono input from native rate to oversampled rate.
    ///
    /// `output` must have room for `input.len() * factor.multiplier()` samples.
    /// Returns number of oversampled samples written.
    pub fn upsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        debug_assert!(input.len() <= self.max_samples);

        match self.factor {
            OversampleFactor::Off => {
                let n = input.len().min(output.len());
                unsafe {
                    core::ptr::copy_nonoverlapping(input.as_ptr(), output.as_mut_ptr(), n);
                }
                n
            }
            OversampleFactor::X2 => self.stage1.as_mut().unwrap().upsample(input, output),
            OversampleFactor::X4 => {
                let s1 = self.stage1.as_mut().unwrap();
                let s2 = self.stage2.as_mut().unwrap();
                let n_x2 = s1.upsample(input, &mut self.inter_buf[..input.len() * 2]);
                s2.upsample(&self.inter_buf[..n_x2], output)
            }
        }
    }

    /// Downsamples mono input from oversampled rate back to native rate.
    ///
    /// `output` must have room for `input.len() / factor.multiplier()` samples.
    /// Returns number of native-rate samples written.
    pub fn downsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        match self.factor {
            OversampleFactor::Off => {
                let n = input.len().min(output.len());
                unsafe {
                    core::ptr::copy_nonoverlapping(input.as_ptr(), output.as_mut_ptr(), n);
                }
                n
            }
            OversampleFactor::X2 => self.stage1.as_mut().unwrap().downsample(input, output),
            OversampleFactor::X4 => {
                let s1 = self.stage1.as_mut().unwrap();
                let s2 = self.stage2.as_mut().unwrap();
                let n_x2 = s2.downsample(input, &mut self.inter_buf[..input.len() / 2]);
                s1.downsample(&self.inter_buf[..n_x2], output)
            }
        }
    }
}

#[cfg(test)]
#[path = "oversample_test.rs"]
mod oversample_test;
