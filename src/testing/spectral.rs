// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Spectral fidelity measurement suite.
//!
//! Implements:
//! - **Farina exponential sine sweep** (AES Convention 108, 2000):
//!   Simultaneous measurement of impulse response (FR magnitude/phase)
//!   and THD per harmonic order via deconvolution.
//! - **THD+N per AES17**: 997 Hz pure tone, notch-filtered fundamental,
//!   THD+N = RMS(rest) / RMS(total).
//! - **IMD SMPTE/DIN**: two-tone 60 Hz + 7 kHz (4:1 amplitude ratio),
//!   sidebands around the 7 kHz carrier.
//!
//! ```text
//! Farina method:
//!   x(t) = sin[φ(t)],  φ(t) = ω₁·T / ln(ω₂/ω₁) · (exp(t·ln(ω₂/ω₁)/T) − 1)
//!
//!   inverse filter:  f(t) = x(T−t) · β(t)   where β(t) compensates the
//!   −3 dB/octave spectral envelope of the exponential sweep.
//!
//!   deconvolution:   h(t) = y(t) ∗ f(t)     (via FFT multiplication)
//!
//!   After deconvolution, the linear IR appears at delay ≈ 0,
//!   and the k-th harmonic distortion kernel appears at:
//!       Δtₖ = −T · ln(k) / ln(f₂/f₁)   (expressed as positive time lag)
//! ```
//!
//! All routines are pure analytics (no heap allocation in hot loops
//! aside from FFT planner construction) — safe for off-RT QA use.

use crate::math::dsp::fft::FftPlanner;
use std::f64::consts::TAU;

// =============================================================================
// Common helpers
// =============================================================================

/// Returns the next power-of-two ≥ `n`.
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    if n.is_power_of_two() {
        n
    } else {
        1 << (usize::BITS - (n - 1).leading_zeros())
    }
}

// =============================================================================
// Farina exponential sine sweep
// =============================================================================

/// Farina sweep + deconvolution result.
#[derive(Debug, Clone)]
pub struct FarinaResult {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Start frequency of the sweep in Hz.
    pub f1: f64,
    /// End frequency of the sweep in Hz.
    pub f2: f64,
    /// Sweep duration in seconds.
    pub duration_s: f64,
    /// Linear impulse response (time-domain), length = n_samples.
    pub ir_linear: Vec<f64>,
    /// Frequency response magnitude (dB), length = ir_linear.len()/2 + 1.
    pub fr_magnitude_db: Vec<f64>,
    /// Frequency response phase (radians), length = ir_linear.len()/2 + 1.
    pub fr_phase_rad: Vec<f64>,
    /// Frequency axis in Hz, length = ir_linear.len()/2 + 1.
    pub freq_axis: Vec<f64>,
    /// THD per harmonic order: `Vec<(order, thd_percent)>`.
    /// `order = 1` is the linear component (THD = 0 by definition).
    /// Higher orders are distortion.
    pub thd_by_order: Vec<(u32, f64)>,
    /// Overall THD (%) summed from orders ≥ 2.
    pub thd_total_percent: f64,
}

/// Generates an exponential sine sweep with the Farina method.
///
/// The sweep covers `[f1, f2]` Hz over `duration_s` seconds at `sample_rate` Hz.
/// Returns the sweep signal with peak amplitude ≤ 1.0.
pub fn generate_farina_sweep(f1: f64, f2: f64, duration_s: f64, sample_rate: u32) -> Vec<f64> {
    assert!(f1 > 0.0 && f2 > f1 && duration_s > 0.0 && sample_rate > 0);
    let n = (sample_rate as f64 * duration_s).ceil() as usize;
    let sr = sample_rate as f64;
    let t_scale = sr * duration_s;
    let omega1 = TAU * f1;
    let ln_ratio = (f2 / f1).ln();

    let mut sweep = Vec::with_capacity(n);
    for i in 0..n {
        let t_norm = i as f64 / t_scale;
        // Instantaneous phase
        let phase = omega1 * duration_s / ln_ratio * ((t_norm * ln_ratio).exp_m1());
        sweep.push(phase.sin());
    }
    sweep
}

/// Generates the inverse filter for a Farina sweep.
///
/// Computed directly in the frequency domain:
///   F\[k\] = conj(S\[k\]) / (|S\[k\]|² + ε)
///   f\[n\] = IFFT(F\[k\])  (first n samples)
///
/// This is the mathematically exact matched filter, equivalent to the
/// time-reversed sweep with amplitude compensation.
pub fn generate_farina_inverse_filter(
    sweep: &[f64],
    _f1: f64,
    _f2: f64,
    _duration_s: f64,
    _sample_rate: u32,
) -> Vec<f64> {
    let n = sweep.len();
    let n_fft = next_power_of_two(n);

    let fft = FftPlanner::<f64>::new(n_fft);

    let mut s_re = vec![0.0f64; n_fft];
    let mut s_im = vec![0.0f64; n_fft];
    s_re[..n].copy_from_slice(sweep);

    fft.process(&mut s_re, &mut s_im);

    // H[k] = conj(S[k]) / (|S[k]|² + ε)
    let eps = 1e-10;
    for i in 0..n_fft {
        let mag_sq = s_re[i] * s_re[i] + s_im[i] * s_im[i] + eps;
        s_re[i] /= mag_sq;
        s_im[i] = -s_im[i] / mag_sq;
    }

    fft.process_inverse(&mut s_re, &mut s_im);

    // Take first n samples as the inverse filter
    let mut inv: Vec<f64> = s_re[..n].to_vec();

    // Normalise
    let max_abs = inv.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    if max_abs > 1e-10 {
        let scale = 0.95 / max_abs;
        for v in &mut inv {
            *v *= scale;
        }
    }

    inv
}

/// Deconvolves the system output `y[n]` with the inverse filter `f[n]` via FFT.
///
/// Uses circular convolution (same length as input), as required by the
/// Farina method. Both signals must have the same length.
///
/// Returns the deconvolved impulse response `h[n] = y[n] ⊛ f[n]`.
fn deconvolve_farina(y: &[f64], inv_filter: &[f64]) -> Vec<f64> {
    assert_eq!(
        y.len(),
        inv_filter.len(),
        "y and inv_filter must have same length for Farina deconvolution"
    );
    let n = next_power_of_two(y.len());

    let fft = FftPlanner::<f64>::new(n);

    let mut y_re = vec![0.0f64; n];
    let mut y_im = vec![0.0f64; n];
    let mut f_re = vec![0.0f64; n];
    let mut f_im = vec![0.0f64; n];

    y_re[..y.len()].copy_from_slice(y);
    f_re[..inv_filter.len()].copy_from_slice(inv_filter);

    fft.process(&mut y_re, &mut y_im);
    fft.process(&mut f_re, &mut f_im);

    // Pointwise complex multiply
    for i in 0..n {
        let yr = y_re[i];
        let yi = y_im[i];
        let fr = f_re[i];
        let fi = f_im[i];
        y_re[i] = yr * fr - yi * fi;
        y_im[i] = yr * fi + yi * fr;
    }

    fft.process_inverse(&mut y_re, &mut y_im);

    // Truncate to original length (circular convolution)
    y_re.truncate(y.len());
    y_re
}

/// Runs the full Farina measurement pipeline.
///
/// 1. Generates exponential sweep from `f1` to `f2` over `duration_s`.
/// 2. Processes the sweep through `process_fn` to obtain system output.
/// 3. Deconvolves with inverse filter to obtain separated impulse responses.
/// 4. Extracts linear IR and THD per harmonic order.
///
/// `process_fn` receives the sweep (f64) and must return the system output (f32 or f64).
/// The `max_harmonics` parameter controls how many harmonic orders to extract.
pub fn farina_measure<F>(
    f1: f64,
    f2: f64,
    duration_s: f64,
    sample_rate: u32,
    max_harmonics: u32,
    process_fn: F,
) -> FarinaResult
where
    F: FnOnce(&[f64]) -> Vec<f32>,
{
    let sweep = generate_farina_sweep(f1, f2, duration_s, sample_rate);
    let n = sweep.len();
    let inv_filter = generate_farina_inverse_filter(&sweep, f1, f2, duration_s, sample_rate);

    let output_f32 = process_fn(&sweep);
    assert_eq!(
        output_f32.len(),
        n,
        "process_fn output length {} != sweep length {n}",
        output_f32.len()
    );
    let output: Vec<f64> = output_f32.iter().map(|&x| x as f64).collect();

    let deconv = deconvolve_farina(&output, &inv_filter);

    // The deconvolution result has the linear IR at the start (delay ≈ 0)
    // and harmonic distortion kernels spaced apart.
    // The time lag for harmonic order k relative to the linear IR is:
    //   Δt_k = T · ln(k) / ln(f2/f1)
    let sr = sample_rate as f64;
    let ln_ratio = (f2 / f1).ln();
    let _delay_linear = inv_filter.len() as f64 / sr; // approximate

    // Extract linear IR (from start to first harmonic boundary)
    // The 2nd harmonic appears at: half_range = T·ln(2)/ln(f2/f1) seconds
    let half_point_s = duration_s * (1.5f64).ln() / ln_ratio;
    let half_point_samples = (half_point_s * sr).round() as usize;
    let ir_len = if half_point_samples < n {
        half_point_samples
    } else {
        n / 2
    };

    let ir_linear: Vec<f64> = deconv[..ir_len.min(deconv.len())].to_vec();

    // Compute frequency response from linear IR via FFT
    let fr_n = next_power_of_two(ir_linear.len());
    let mut fr_re = vec![0.0f64; fr_n];
    let mut fr_im = vec![0.0f64; fr_n];
    fr_re[..ir_linear.len()].copy_from_slice(&ir_linear);

    let fft = FftPlanner::<f64>::new(fr_n);
    fft.process(&mut fr_re, &mut fr_im);

    let num_bins = fr_n / 2 + 1;
    let mut fr_magnitude_db = vec![0.0f64; num_bins];
    let mut fr_phase_rad = vec![0.0f64; num_bins];
    let mut freq_axis = vec![0.0f64; num_bins];

    for i in 0..num_bins {
        freq_axis[i] = i as f64 * sr / fr_n as f64;
        let mag = (fr_re[i] * fr_re[i] + fr_im[i] * fr_im[i]).sqrt();
        let mag_db = if mag > 1e-15 {
            20.0 * mag.log10()
        } else {
            -300.0
        };
        fr_magnitude_db[i] = mag_db;
        fr_phase_rad[i] = fr_im[i].atan2(fr_re[i]);
    }

    // Extract THD per harmonic order
    // For each order k, extract the IR segment at time lag Δt_k
    // Time lag for harmonic order k: Δt_k = T · ln(k) / ln(f2/f1)
    let window_len = ir_len;
    let mut harmonic_energies: Vec<(u32, f64)> = Vec::new();

    for k in 1..=max_harmonics {
        let lag_s = duration_s * (k as f64).ln() / ln_ratio;
        let lag_samples = (lag_s * sr).round() as isize;

        let start = lag_samples.max(0) as usize;
        let end = (start + window_len).min(deconv.len());

        if end <= start || start >= deconv.len() {
            break;
        }

        let segment = &deconv[start..end];
        let energy: f64 = segment.iter().map(|&x| x * x).sum();
        harmonic_energies.push((k, energy));
    }

    let fund_energy = harmonic_energies.first().map(|(_, e)| *e).unwrap_or(0.0);

    let mut thd_by_order: Vec<(u32, f64)> = Vec::new();
    for (k, energy) in &harmonic_energies {
        let thd = if *k == 1 {
            0.0
        } else if fund_energy > 1e-20 {
            100.0 * (energy / fund_energy).sqrt()
        } else {
            0.0
        };
        thd_by_order.push((*k, thd));
    }

    // Compute total THD (sum of orders ≥ 2).
    //
    // The Farina method yields harmonic distortion as per-order percentages
    // relative to the fundamental: THD_h% = 100 × √(E_h / E_1). Computing
    // total THD by summing these percentages in quadrature
    // (THD_total% = 100 × √(Σ (THD_h%/100)²)) is mathematically equivalent
    // to computing it from raw energies
    // (THD_total% = 100 × √(Σ E_h / E_1)), but using the per-order
    // percentages directly allows consistent per-harmonic reporting and
    // independent verification of each THD_h% value against Farina's
    // analytical expression.
    let thd_total_percent = if thd_by_order.len() > 1 {
        let sum_sq: f64 = thd_by_order
            .iter()
            .skip(1)
            .map(|(_, thd)| (*thd / 100.0).powi(2))
            .sum();
        100.0 * sum_sq.sqrt()
    } else {
        0.0
    };

    // Recalculate THD for order 1 (should be 0 or very small as it's the linear part)
    if let Some((1, _)) = thd_by_order.first() {}

    FarinaResult {
        sample_rate,
        f1,
        f2,
        duration_s,
        ir_linear,
        fr_magnitude_db,
        fr_phase_rad,
        freq_axis,
        thd_by_order,
        thd_total_percent,
    }
}

// =============================================================================
// THD+N per AES17
// =============================================================================

/// Result of an AES17 THD+N measurement.
#[derive(Debug, Clone)]
pub struct ThdnResult {
    /// Fundamental frequency in Hz.
    pub f0: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// THD+N in percent: 100 · RMS(notched) / RMS(total).
    pub thdn_percent: f64,
    /// THD+N in dB relative to fundamental.
    pub thdn_db: f64,
    /// RMS of the notched signal (everything except fundamental).
    pub rms_notched: f64,
    /// RMS of the total signal.
    pub rms_total: f64,
}

/// Second-order biquad notch filter coefficients.
#[derive(Debug, Clone, Copy)]
struct NotchBiquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl NotchBiquad {
    /// Designs a biquad notch filter.
    ///
    /// `f0` = notch frequency (Hz), `q` = quality factor (typ. 1–5 per AES17),
    /// `fs` = sample rate (Hz).
    fn design(f0: f64, q: f64, fs: f64) -> Self {
        let w0 = TAU * f0 / fs;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);

        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Applies the notch filter in-place to `signal`.
    fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; signal.len()];
        let mut x1 = 0.0f64;
        let mut x2 = 0.0f64;
        let mut y1 = 0.0f64;
        let mut y2 = 0.0f64;

        for (i, &x0) in signal.iter().enumerate() {
            let y0 = self.b0 * x0 + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2;
            out[i] = y0;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
        }
        out
    }
}

/// Generates a pure sine at `f0` Hz with amplitude 1.0.
pub fn generate_sine_f64(f0: f64, sample_rate: u32, num_samples: usize, gain: f64) -> Vec<f64> {
    let sr = sample_rate as f64;
    let omega = TAU * f0 / sr;
    (0..num_samples)
        .map(|i| (i as f64 * omega).sin() * gain)
        .collect()
}

/// Measures THD+N per AES17.
///
/// 1. Generates a `f0` Hz pure tone (typ. 997 Hz per AES17).
/// 2. Processes through `process_fn`.
/// 3. Notch-filters the fundamental (Q ≈ 5).
/// 4. THD+N = 100% · RMS(notched) / RMS(total).
///
/// `stability_blocks` = number of extra blocks to discard at start to avoid
/// transient artefacts from model warm-up (0 for static functions).
pub fn measure_thdn<F>(
    f0: f64,
    sample_rate: u32,
    duration_s: f64,
    notch_q: f64,
    stability_blocks: usize,
    process_fn: F,
) -> ThdnResult
where
    F: FnOnce(&[f64]) -> Vec<f32>,
{
    let n_total = (sample_rate as f64 * duration_s).ceil() as usize + stability_blocks;
    let input = generate_sine_f64(f0, sample_rate, n_total, 1.0);

    let output_f32 = process_fn(&input);
    assert_eq!(output_f32.len(), input.len());

    // Discard stability blocks and convert to f64
    let start = stability_blocks;
    let output: Vec<f64> = output_f32[start..].iter().map(|&x| x as f64).collect();
    let n = output.len();

    // RMS of total signal
    let rms_total = {
        let sum_sq: f64 = output.iter().map(|&x| x * x).sum();
        (sum_sq / n as f64).sqrt()
    };

    // Notch filter the fundamental
    let notch = NotchBiquad::design(f0, notch_q, sample_rate as f64);
    let notched = notch.apply(&output);

    // Discard first 2000 samples to skip biquad transient
    let settle = 2000.min(n.saturating_sub(1));
    let notched_stable = &notched[settle..];

    // RMS of notched signal (stable portion)
    let rms_notched = {
        let sum_sq: f64 = notched_stable.iter().map(|&x| x * x).sum();
        (sum_sq / notched_stable.len() as f64).sqrt()
    };

    let thdn_percent = if rms_total > 1e-15 {
        100.0 * rms_notched / rms_total
    } else {
        0.0
    };

    let thdn_db = if thdn_percent > 1e-15 {
        20.0 * (thdn_percent / 100.0).log10()
    } else {
        f64::NEG_INFINITY
    };

    ThdnResult {
        f0,
        sample_rate,
        thdn_percent,
        thdn_db,
        rms_notched,
        rms_total,
    }
}

// =============================================================================
// IMD SMPTE/DIN
// =============================================================================

/// Result of an IMD SMPTE/DIN measurement.
#[derive(Debug, Clone)]
pub struct SmpteImdResult {
    /// Low frequency in Hz (60 Hz per SMPTE).
    pub f_low: f64,
    /// High frequency in Hz (7 kHz per SMPTE).
    pub f_high: f64,
    /// Amplitude ratio (4:1 per SMPTE).
    pub ratio: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// IMD in percent: 100 · RMS(sidebands) / RMS(carrier).
    pub imd_percent: f64,
    /// IMD in dB: 20 · log10(imd_percent / 100).
    pub imd_db: f64,
    /// Individual sideband amplitudes (normalised to carrier).
    pub sideband_percents: Vec<(i32, f64)>,
}

/// Generates an SMPTE/DIN two-tone test signal.
///
/// `f_low` and `f_high` are the two tone frequencies.
/// `ratio` is the amplitude ratio (f_high : f_low). Standard SMPTE is 4:1.
/// `gain` scales the combined signal such that peak ≤ gain.
pub fn generate_smpte_tones(
    f_low: f64,
    f_high: f64,
    ratio: f64,
    sample_rate: u32,
    num_samples: usize,
    gain: f64,
) -> Vec<f64> {
    let sr = sample_rate as f64;
    let omega_low = TAU * f_low / sr;
    let omega_high = TAU * f_high / sr;

    let amp_high = gain * ratio / (ratio + 1.0);
    let amp_low = gain / (ratio + 1.0);

    (0..num_samples)
        .map(|i| {
            let t = i as f64;
            amp_low * (t * omega_low).sin() + amp_high * (t * omega_high).sin()
        })
        .collect()
}

/// Measures SMPTE IMD.
///
/// 1. Generates two-tone signal (60 Hz + 7 kHz, 4:1).
/// 2. Processes through `process_fn`.
/// 3. Computes FFT of output, identifies carrier and sidebands.
/// 4. IMD(%) = 100 · √(Σ|sidebands|²) / |carrier|
///
/// `stability_blocks` = number of extra blocks discarded from start.
pub fn measure_smpte_imd<F>(
    f_low: f64,
    f_high: f64,
    ratio: f64,
    sample_rate: u32,
    duration_s: f64,
    stability_blocks: usize,
    process_fn: F,
) -> SmpteImdResult
where
    F: FnOnce(&[f64]) -> Vec<f32>,
{
    let n_signal = (sample_rate as f64 * duration_s).ceil() as usize + stability_blocks;
    let input = generate_smpte_tones(f_low, f_high, ratio, sample_rate, n_signal, 0.9);

    let output_f32 = process_fn(&input);
    assert_eq!(output_f32.len(), input.len());

    let start = stability_blocks;
    let output: Vec<f64> = output_f32[start..].iter().map(|&x| x as f64).collect();
    let n = output.len();
    let n_fft = next_power_of_two(n);
    let sr = sample_rate as f64;
    let bin_width = sr / n_fft as f64;

    // Apply 4-term Blackman-Harris window
    let window: Vec<f64> = crate::testing::aliasing::blackman_harris_4term(n);
    let mut re = vec![0.0f64; n_fft];
    let mut im = vec![0.0f64; n_fft];
    for (i, &x) in output.iter().enumerate() {
        re[i] = x * window[i];
    }

    let fft = FftPlanner::<f64>::new(n_fft);
    fft.process(&mut re, &mut im);

    let mag: Vec<f64> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| (r * r + i * i).sqrt())
        .collect();

    // Find the carrier bin (closest to f_high)
    let carrier_bin = (f_high / bin_width).round() as usize;
    let carrier_mag = if carrier_bin < mag.len() {
        mag[carrier_bin]
    } else {
        0.0
    };

    // Search for sidebands at f_high ± n·f_low
    let max_sideband_order = 6i32;
    let bin_search = (2.0 / bin_width).ceil() as usize;
    let mut sideband_percents: Vec<(i32, f64)> = Vec::new();

    for n_order in 1..=max_sideband_order {
        for &sign in &[-1i32, 1i32] {
            let expected_freq = f_high + sign as f64 * n_order as f64 * f_low;
            if expected_freq <= 0.0 || expected_freq >= (sr / 2.0 - bin_width) {
                continue;
            }

            let expected_bin = (expected_freq / bin_width).round() as usize;
            // Find the actual peak in a small range around expected_bin
            let start_bin = expected_bin.saturating_sub(bin_search);
            let end_bin = (expected_bin + bin_search).min(mag.len() - 1);

            if start_bin >= end_bin {
                continue;
            }

            let mut _peak_bin = start_bin;
            let mut peak_val = 0.0f64;
            for b in start_bin + 1..end_bin {
                if mag[b] > mag[b - 1] && mag[b] > mag[b + 1] && mag[b] > peak_val {
                    _peak_bin = b;
                    peak_val = mag[b];
                }
            }

            if peak_val > 0.0 && carrier_mag > 1e-15 {
                let pct = 100.0 * peak_val / carrier_mag;
                sideband_percents.push((n_order * sign, pct));
            }
        }
    }

    let imd_percent = if carrier_mag > 1e-15 {
        let sum_sq: f64 = sideband_percents
            .iter()
            .map(|(_, p)| (p / 100.0).powi(2))
            .sum();
        100.0 * sum_sq.sqrt()
    } else {
        0.0
    };

    let imd_db = if imd_percent > 1e-15 {
        20.0 * (imd_percent / 100.0).log10()
    } else {
        f64::NEG_INFINITY
    };

    SmpteImdResult {
        f_low,
        f_high,
        ratio,
        sample_rate,
        imd_percent,
        imd_db,
        sideband_percents,
    }
}

#[cfg(test)]
#[path = "spectral_test.rs"]
mod tests;
