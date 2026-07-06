// SPDX-License-Identifier: Apache-2.0
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
//! only pre-allocated buffers — zero alloc, zero heap-drop.
//!
//! Factor change requires rebuild (off-RT), same path as model hot-swap.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;
use crate::math::common::hsum_avx2;
use core::arch::x86_64::*;

/// Half-band FIR filter length (≡ 1 mod 4 so D=HB_TAPS/2 is even).
/// 25 taps, D=12. Kaiser β=12 → >100 dB stop-band rejection.
const HB_TAPS: usize = 25;
/// Filter delay (group delay = HB_TAPS/2 = 12 samples at native rate).
const HB_DELAY: usize = HB_TAPS / 2;
/// Number of non-zero odd-index taps (h[1], h[3], ..., h[23]).
const HB_ODD_COUNT: usize = HB_TAPS / 2; // 12

/// Upsampler ring double-buffer length (HB_DELAY × 2 for contiguous access).
const UP_DELAY_LINE_LEN: usize = HB_DELAY * 2;
/// Downsampler even-sample ring size (⌈HB_TAPS/2⌉ = 13).
const DOWN_EVEN_LEN: usize = (HB_TAPS + 1) / 2;
/// Downsampler odd-sample ring size (⌊HB_TAPS/2⌋ = 12).
const DOWN_ODD_LEN: usize = HB_TAPS / 2;
/// Downsampler even double-buffer length.
const DOWN_EVEN_DELAY_LINE_LEN: usize = DOWN_EVEN_LEN * 2;
/// Downsampler odd double-buffer length.
const DOWN_ODD_DELAY_LINE_LEN: usize = DOWN_ODD_LEN * 2;

/// Oversampling factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum OversampleFactor {
    /// No oversampling — pass-through with zero overhead.
    #[default]
    Off,
    /// 2× oversampling (one half-band stage).
    X2,
    /// 4× oversampling (two cascaded half-band stages).
    X4,
}

impl OversampleFactor {
    /// Creates an `OversampleFactor` from a CLAP parameter value (0.0, 1.0, 2.0).
    pub fn from_f32(val: f32) -> Self {
        match val.round() as i32 {
            1 => Self::X2,
            2 => Self::X4,
            _ => Self::Off,
        }
    }

    /// Converts to its CLAP parameter value (0.0, 1.0, 2.0).
    pub fn to_f32(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::X2 => 1.0,
            Self::X4 => 2.0,
        }
    }

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

        // Reverse so coeffs[0] pairs with the oldest sample in the
        // double-buffer window (oldest-first contiguous layout),
        // enabling direct AVX2 FMADD without permutes.
        coeffs.reverse();

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

/// Typed bundle of oversampling stages, eliminating `Option::unwrap`
/// on the RT hot-path. Variant discriminant is a zero-cost compile-time
/// guarantee that the stages (when present) are always valid.
enum OsStages {
    Off,
    X2 { stage1: X2Stage },
    X4 { stage1: X2Stage, stage2: X2Stage },
}

/// Single 2× oversampling stage (up + down delay-line state).
///
/// Uses pre-allocated double-buffer delay lines for contiguous SIMD access,
/// eliminating per-sample modulo indexing from the hot-path.
struct X2Stage {
    /// Filter for upsampling (DC gain = 2.0, center tap multiplier = 1.0).
    up_filter: HalfBandFilter,
    /// Filter for downsampling (DC gain = 1.0, center tap multiplier = 0.5).
    down_filter: HalfBandFilter,
    /// Center tap multiplier for the upsampler (h[D] = dc_gain_up / 2 = 1.0).
    up_center: f32,
    /// Center tap multiplier for the downsampler (h[D] = dc_gain_down / 2 = 0.5).
    down_center: f32,
    /// Upsampler double-buffer delay line (24 entries).
    up_ring: AlignedVec<f32>,
    /// Upsampler write position (0..HB_DELAY-1).
    up_pos: usize,
    /// Downsampler double-buffer for even-position samples (26 entries).
    down_ring_even: AlignedVec<f32>,
    /// Downsampler double-buffer for odd-position samples (24 entries).
    down_ring_odd: AlignedVec<f32>,
    /// Write position in even ring (0..DOWN_EVEN_LEN-1).
    down_pos_even: usize,
    /// Write position in odd ring (0..DOWN_ODD_LEN-1).
    down_pos_odd: usize,
    /// Total samples processed (replaces modulo ring tracking).
    down_total: u64,
}

impl X2Stage {
    fn new() -> Result<Self, NamErrorCode> {
        let dc_up = 2.0;
        let dc_down = 1.0;
        Ok(Self {
            up_filter: HalfBandFilter::design(12.0, dc_up),
            down_filter: HalfBandFilter::design(12.0, dc_down),
            up_center: (dc_up / 2.0) as f32,
            down_center: (dc_down / 2.0) as f32,
            up_ring: AlignedVec::new(UP_DELAY_LINE_LEN, 0.0f32)?,
            up_pos: 0,
            down_ring_even: AlignedVec::new(DOWN_EVEN_DELAY_LINE_LEN, 0.0f32)?,
            down_ring_odd: AlignedVec::new(DOWN_ODD_DELAY_LINE_LEN, 0.0f32)?,
            down_pos_even: 0,
            down_pos_odd: 0,
            down_total: 0,
        })
    }

    fn upsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let coeffs = &self.up_filter.coeffs;
        let center = self.up_center;
        let n = HB_DELAY;
        let n_in = input.len();

        for (i, &x) in input.iter().enumerate() {
            let p = self.up_pos;
            unsafe {
                *self.up_ring.get_unchecked_mut(p) = x;
                *self.up_ring.get_unchecked_mut(p + n) = x;
            }
            self.up_pos = (p + 1) % n;

            let wptr = unsafe { self.up_ring.as_ptr().add(self.up_pos) };

            let even_out = unsafe { *wptr.add(5) * center };

            let odd_out = unsafe {
                let c8 = _mm256_loadu_ps(coeffs.as_ptr());
                let s8 = _mm256_loadu_ps(wptr);
                let acc8 = _mm256_fmadd_ps(c8, s8, _mm256_setzero_ps());
                let mut sum = hsum_avx2(acc8);
                sum += *coeffs.get_unchecked(8) * *wptr.add(8);
                sum += *coeffs.get_unchecked(9) * *wptr.add(9);
                sum += *coeffs.get_unchecked(10) * *wptr.add(10);
                sum += *coeffs.get_unchecked(11) * *wptr.add(11);
                sum
            };

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
        let mut out_idx = 0;

        for &x in input.iter() {
            let is_even = (self.down_total & 1) == 0;
            if is_even {
                let p = self.down_pos_even;
                unsafe {
                    *self.down_ring_even.get_unchecked_mut(p) = x;
                    *self.down_ring_even.get_unchecked_mut(p + DOWN_EVEN_LEN) = x;
                }
                self.down_pos_even = (p + 1) % DOWN_EVEN_LEN;
            } else {
                let p = self.down_pos_odd;
                unsafe {
                    *self.down_ring_odd.get_unchecked_mut(p) = x;
                    *self.down_ring_odd.get_unchecked_mut(p + DOWN_ODD_LEN) = x;
                }
                self.down_pos_odd = (p + 1) % DOWN_ODD_LEN;
            }
            self.down_total += 1;

            if self.down_total >= HB_TAPS as u64 && (self.down_total & 1) == 1 {
                let ev_ptr =
                    unsafe { self.down_ring_even.as_ptr().add(self.down_pos_even) };
                let center_sample = unsafe { *ev_ptr.add(6) };
                let mut sum = center_sample * center;

                let od_ptr =
                    unsafe { self.down_ring_odd.as_ptr().add(self.down_pos_odd) };
                unsafe {
                    let c8 = _mm256_loadu_ps(coeffs.as_ptr());
                    let s8 = _mm256_loadu_ps(od_ptr);
                    let acc8 = _mm256_fmadd_ps(c8, s8, _mm256_setzero_ps());
                    sum += hsum_avx2(acc8);
                    sum += *coeffs.get_unchecked(8) * *od_ptr.add(8);
                    sum += *coeffs.get_unchecked(9) * *od_ptr.add(9);
                    sum += *coeffs.get_unchecked(10) * *od_ptr.add(10);
                    sum += *coeffs.get_unchecked(11) * *od_ptr.add(11);
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
    stages: OsStages,
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
    pub fn new(factor: OversampleFactor, max_input_samples: usize) -> Result<Self, NamErrorCode> {
        let inter_size = if factor.stage_count() >= 2 {
            max_input_samples * 2
        } else {
            1
        };

        let stages = match factor {
            OversampleFactor::Off => OsStages::Off,
            OversampleFactor::X2 => OsStages::X2 {
                stage1: X2Stage::new()?,
            },
            OversampleFactor::X4 => OsStages::X4 {
                stage1: X2Stage::new()?,
                stage2: X2Stage::new()?,
            },
        };

        Ok(Self {
            factor,
            stages,
            inter_buf: AlignedVec::new(inter_size, 0.0f32)?,
            max_samples: max_input_samples,
        })
    }

    /// Returns the current oversampling factor.
    #[inline]
    pub fn factor(&self) -> OversampleFactor {
        self.factor
    }

    /// Returns `true` when oversampling is bypassed (Off).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        matches!(self.stages, OsStages::Off)
    }

    /// Returns the group delay in samples at the native (model) rate.
    ///
    /// Each 2× half-band stage introduces HB_DELAY (= 12) samples.
    /// Off → 0, X2 → 12, X4 → 24.
    #[inline]
    pub fn latency_samples(&self) -> usize {
        match self.factor {
            OversampleFactor::Off => 0,
            OversampleFactor::X2 => HB_DELAY,
            OversampleFactor::X4 => 2 * HB_DELAY,
        }
    }

    /// Upsamples mono input from native rate to oversampled rate.
    ///
    /// `output` must have room for `input.len() * factor.multiplier()` samples.
    /// Returns number of oversampled samples written.
    pub fn upsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        debug_assert!(input.len() <= self.max_samples);
        debug_assert!(
            output.len() >= input.len() * self.factor.multiplier(),
            "oversample: output buffer too small for upsampling factor"
        );

        match &mut self.stages {
            OsStages::Off => {
                let n = input.len().min(output.len());
                unsafe {
                    core::ptr::copy_nonoverlapping(input.as_ptr(), output.as_mut_ptr(), n);
                }
                n
            }
            OsStages::X2 { stage1 } => stage1.upsample(input, output),
            OsStages::X4 { stage1, stage2 } => {
                let n_x2 = stage1.upsample(input, &mut self.inter_buf[..input.len() * 2]);
                stage2.upsample(&self.inter_buf[..n_x2], output)
            }
        }
    }

    /// Downsamples mono input from oversampled rate back to native rate.
    ///
    /// `output` must have room for `input.len() / factor.multiplier()` samples.
    /// Returns number of native-rate samples written.
    pub fn downsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        debug_assert!(
            output.len() >= input.len() / self.factor.multiplier(),
            "oversample: output buffer too small for downsampling factor"
        );
        match &mut self.stages {
            OsStages::Off => {
                let n = input.len().min(output.len());
                unsafe {
                    core::ptr::copy_nonoverlapping(input.as_ptr(), output.as_mut_ptr(), n);
                }
                n
            }
            OsStages::X2 { stage1 } => stage1.downsample(input, output),
            OsStages::X4 { stage1, stage2 } => {
                let n_x2 = stage2.downsample(input, &mut self.inter_buf[..input.len() / 2]);
                stage1.downsample(&self.inter_buf[..n_x2], output)
            }
        }
    }
}

/// Atomic bundle of stereo oversampling engines delivered via SPSC.
///
/// L and R engines are built together on the main thread and consumed
/// together on the RT thread, ensuring they always share the same factor.
pub struct OsEnginePair {
    /// Left-channel oversampling engine.
    pub l: Box<OversampleEngine>,
    /// Right-channel oversampling engine.
    pub r: Box<OversampleEngine>,
}

#[cfg(test)]
#[path = "oversample_test.rs"]
mod oversample_test;
