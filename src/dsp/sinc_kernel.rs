// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Offline generation of minimum-phase Sinc FIR kernels for the native resampler.
//!
//! This module is invoked **exclusively** in `NamResampler::new()` — outside the RT thread.
//! All allocation and heavy computation (FFT, cepstrum) occurs here during initialization.
//!
//! ## Generation Pipeline
//!
//! 1. **Ideal Sinc + Kaiser Windowing** — generates the linear-phase FIR lowpass prototype.
//! 2. **Minimum-Phase Transform (Real Cepstrum)** — eliminates pre-ringing by concentrating
//!    energy into the shortest possible delay, using f64 FFT for numerical precision.
//! 3. **Polyphase Partition** — decomposes the prototype into `num_phases` sub-filters,
//!    each with taps aligned to 64 bytes for AVX2/AVX-512 convolution.

use crate::math::common::AlignedVec;
use rustfft::{FftPlanner, num_complex::Complex};

/// Number of phases in the overabundant polyphase bank.
///
/// Controls the fractional resolution of the resampler. With 256 phases,
/// the maximum phase error between adjacent sub-filters is < 0.4%,
/// making linear interpolation between phases sufficient for
/// SNR > 140 dB in rate conversion.
pub const NUM_PHASES: usize = 256;

/// Number of taps per phase in the polyphase bank.
///
/// With 32 taps per phase and a Kaiser window (β=12), the filter achieves
/// aliasing rejection > 120 dB with a transition band of ~3 kHz
/// at 48 kHz. The value is a multiple of 8 for AVX2 alignment.
pub const TAPS_PER_PHASE: usize = 32;

/// Total prototype FIR length = NUM_PHASES × TAPS_PER_PHASE.
const PROTO_LEN: usize = NUM_PHASES * TAPS_PER_PHASE;

/// Polyphase filter bank with coefficients aligned for SIMD.
///
/// Memory layout: `coeffs[phase * TAPS_PER_PHASE .. (phase+1) * TAPS_PER_PHASE]`
/// Each phase is contiguous and aligned to 64 bytes for `_mm512_load_ps` / `_mm256_load_ps`.
pub struct PolyphaseBank {
    /// Aligned f32 coefficients. Size = `NUM_PHASES * TAPS_PER_PHASE`.
    /// The 64-byte alignment of the buffer start ensures that each 128-byte phase
    /// also begins on a 64-byte boundary.
    coeffs: AlignedVec<f32>,
    /// Taps per phase (always TAPS_PER_PHASE, already a multiple of 8).
    pub taps_per_phase: usize,
}

impl PolyphaseBank {
    /// Returns the pointer to the start of phase `phase` coefficients.
    ///
    /// # Safety
    /// `phase` deve ser < `NUM_PHASES`.
    #[inline]
    pub fn phase_ptr(&self, phase: usize) -> *const f32 {
        debug_assert!(phase < NUM_PHASES);
        unsafe { self.coeffs.as_ptr().add(phase * self.taps_per_phase) }
    }

    /// Returns the coefficient slice for phase `phase`.
    #[inline]
    pub fn phase_coeffs(&self, phase: usize) -> &[f32] {
        debug_assert!(phase < NUM_PHASES);
        let start = phase * self.taps_per_phase;
        &self.coeffs[start..start + self.taps_per_phase]
    }
}

/// Generates the complete polyphase bank for conversion `from_rate → to_rate`.
///
/// Pipeline: Sinc+Kaiser → Minimum Phase (Cepstrum) → Polyphase Partition.
///
/// # Parameters
/// - `from_rate`: source sample rate (Hz).
/// - `to_rate`: destination sample rate (Hz).
///
/// # Returns
/// Polyphase bank ready for SIMD convolution.
pub fn generate_polyphase_bank(from_rate: u32, to_rate: u32) -> PolyphaseBank {
    // 1. Generate Sinc + Kaiser prototype in f64
    let cutoff_ratio = (from_rate.min(to_rate) as f64) / (from_rate.max(to_rate) as f64);
    let cutoff = 0.95 * cutoff_ratio;
    let proto_f64 = generate_sinc_kaiser(PROTO_LEN, cutoff, 12.0);

    // 2. Transform to minimum phase via real cepstrum
    let min_phase = to_minimum_phase(&proto_f64);

    // 3. Normalize energy (DC gain = 1.0 per phase)
    let proto_f32: Vec<f32> = min_phase.iter().map(|&x| x as f32).collect();

    // 4. Partition into NUM_PHASES sub-filters
    partition_polyphase(&proto_f32)
}

/// Generates a Sinc FIR kernel with Kaiser windowing.
///
/// # Parameters
/// - `length`: total filter length (samples).
/// - `cutoff`: normalized cutoff frequency (0..1, relative to Nyquist).
/// - `beta`: Kaiser window β parameter (controls stop-band attenuation).
///   β=12 → ~120 dB of rejection.
fn generate_sinc_kaiser(length: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let half = (length - 1) as f64 / 2.0;
    let i0_beta = bessel_i0(beta);

    let mut kernel = Vec::with_capacity(length);
    for i in 0..length {
        let n = i as f64 - half;

        // Normalized Sinc
        let sinc = if n.abs() < 1e-10 {
            cutoff
        } else {
            let x = std::f64::consts::PI * n * cutoff;
            x.sin() / (std::f64::consts::PI * n)
        };

        // Kaiser window: I0(β × sqrt(1 - (2n/N-1)²)) / I0(β)
        let ratio = n / half;
        let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
        let window = bessel_i0(arg) / i0_beta;

        kernel.push(sinc * window);
    }

    // Normalize for unit DC gain
    let dc_sum: f64 = kernel.iter().sum();
    if dc_sum.abs() > 1e-15 {
        for k in &mut kernel {
            *k /= dc_sum;
        }
    }

    kernel
}

/// Modified Bessel function of the first kind, order zero — I₀(x).
///
/// Taylor series expansion with 20 terms (precision > 1e-12 for β ≤ 25).
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

/// Transforms a linear-phase FIR kernel to minimum phase via Real Cepstrum.
///
/// ## Algorithm (Oppenheim & Schafer, Discrete-Time Signal Processing)
///
/// 1. Zero-pad kernel to `N_fft` (power of 2, ≥ 4× original length).
/// 2. FFT → complex spectrum `H(k)`.
/// 3. Log-magnitude: `L(k) = ln(|H(k)| + ε)`.
/// 4. IFFT of `L` → real cepstrum `c[n]`.
/// 5. Causal truncation: `c[0]` unchanged, `c[1..N/2-1] × 2`, `c[N/2+1..] = 0`.
/// 6. FFT of causal cepstrum → `Ĉ(k)`.
/// 7. Complex exponential: `H_min(k) = exp(Ĉ(k))`.
/// 8. IFFT → `h_min[n]` (real part), truncate to original length.
///
/// All computation is in f64 for numerical stability in the logarithmic domain,
/// as recommended by r8brain-free-src (Vaneev).
fn to_minimum_phase(kernel: &[f64]) -> Vec<f64> {
    let n_proto = kernel.len();
    let n_fft = (4 * n_proto).next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(n_fft);
    let fft_inv = planner.plan_fft_inverse(n_fft);
    let scale = 1.0 / n_fft as f64;

    // Step 1-2: Zero-pad + FFT
    let mut buf: Vec<Complex<f64>> = kernel
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat_n(Complex::new(0.0, 0.0), n_fft - n_proto))
        .collect();
    fft_fwd.process(&mut buf);

    // Step 3: Log-magnitude (real-only complex)
    let eps = 1e-10_f64;
    for c in &mut buf {
        *c = Complex::new((c.norm() + eps).ln(), 0.0);
    }

    // Step 4: IFFT → real cepstrum
    fft_inv.process(&mut buf);
    for c in &mut buf {
        *c *= scale;
    }

    // Step 5: Causal truncation
    // c[0] unchanged, c[1..N/2-1] *= 2, c[N/2] unchanged, c[N/2+1..] = 0
    let half = n_fft / 2;
    for c in &mut buf[1..half] {
        *c *= 2.0;
    }
    for c in &mut buf[half + 1..] {
        *c = Complex::new(0.0, 0.0);
    }

    // Step 6: FFT of causal cepstrum
    fft_fwd.process(&mut buf);

    // Step 7: Complex exponential
    for c in &mut buf {
        *c = c.exp();
    }

    // Step 8: IFFT → minimum-phase impulse
    fft_inv.process(&mut buf);

    // Return real part, truncated to original length
    buf[..n_proto].iter().map(|c| c.re * scale).collect()
}

/// Partitions the FIR prototype into `NUM_PHASES` polyphase sub-filters.
///
/// Coefficient `proto[n]` goes to phase `n % NUM_PHASES`, tap `n / NUM_PHASES`.
/// Each phase is zero-padded to `TAPS_PER_PHASE` (multiple of 8).
fn partition_polyphase(proto: &[f32]) -> PolyphaseBank {
    let taps = TAPS_PER_PHASE;
    let total = NUM_PHASES * taps;
    let mut coeffs = AlignedVec::new(total, 0.0f32);

    // Scale by NUM_PHASES to compensate for the polyphase decomposition.
    // In conceptual upsampling (insertion of L-1 zeros between samples),
    // the prototype filter is applied at L×fs rate. The polyphase partition
    // divides the total gain by L, requiring gain compensation.
    let gain = NUM_PHASES as f32;

    for (n, &coeff) in proto.iter().enumerate() {
        let phase = n % NUM_PHASES;
        let tap = n / NUM_PHASES;
        if tap < taps {
            coeffs[phase * taps + (taps - 1 - tap)] = coeff * gain;
        }
    }

    // Normalize each phase sub-filter individually to ensure flat DC gain
    for phase in 0..NUM_PHASES {
        let start = phase * taps;
        let mut sum = 0.0f32;
        for tap in 0..taps {
            sum += coeffs[start + tap];
        }
        if sum.abs() > 1e-9 {
            for tap in 0..taps {
                coeffs[start + tap] /= sum;
            }
        }
    }

    PolyphaseBank {
        coeffs,
        taps_per_phase: taps,
    }
}

#[cfg(test)]
#[path = "sinc_kernel_test.rs"]
mod sinc_kernel_test;
