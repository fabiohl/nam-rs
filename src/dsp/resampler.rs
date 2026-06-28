// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Native Minimum-Phase Polyphase FIR Sinc Resampler with AVX2+FMA SIMD convolution.
//!
//! Implements `NamResampler`, an RT-safe sample rate conversion engine
//! that replaces the `rubato` crate with a custom Polyphase Sinc FIR filter.
//!
//! ## Advantages over rubato (linear phase)
//!
//! - **Pre-ringing elimination**: the minimum-phase transform via Real Cepstrum
//!   concentrates all filter energy into the shortest possible delay, removing 100%
//!   of the pre-echo artifacts on guitar transients.
//! - **Algorithmic latency reduction**: ~50% less latency than equivalent linear-phase.
//! - **Vectorized convolution**: AVX2+FMA inner product with coefficients aligned
//!   to 64 bytes, saturating the processor's FMA port throughput.
//!
//! ## Architecture: Polyphase Oversampled with Interpolation
//!
//! Instead of using discrete L/M phases (impractical for L=160 at ratio 44.1→48),
//! the resampler uses an overabundant bank of 256 phases with linear interpolation
//! between adjacent phases. This yields arbitrary conversion ratios with
//! measured passband ripple < 0.05 dB and stopband ≥ 100 dB (Task 5.4).
//!
//! ## Quality Mode
//!
//! `NamResampler::new()` produces the production default: minimum-phase polyphase
//! with TAPS_PER_PHASE (currently 64). `NamResampler::new_linear()` produces a
//! linear-phase variant for offline/mixdown use. Both are RT-safe and zero-alloc
//! in the hot path after construction.
//!
//! ## Real-Time Guarantees
//!
//! All allocation happens in `NamResampler::new()` / `new_linear()`, outside the DSP thread.
//! In the RT callback, only `process_input()` / `process_output()` are invoked —
//! zero-alloc operations that manipulate pre-allocated ring buffers.

use anyhow::{Result, bail};

use super::sinc_kernel::{
    NUM_PHASES, PolyphaseBank, TAPS_PER_PHASE, generate_polyphase_bank,
    generate_polyphase_bank_linear,
};
use crate::math::common::{
    AlignedVec, Avx2Math, Avx512Math, Avx512VnniBf16Math, InstructionSet, SimdMath,
    effective_instruction_set,
};

/// Delay line size (double-buffer) to ensure contiguous access.
/// Maintains 2 copies of history to avoid wrap logic in the hot-path SIMD.
const DELAY_LINE_LEN: usize = TAPS_PER_PHASE * 2;

/// FIR filter state for one channel (mono).
///
/// Uses the "double-buffer" technique: the sample history is kept in two
/// contiguous copies. When inserting a new sample, it is written to both
/// `[write_pos]` and `[write_pos + TAPS_PER_PHASE]`. This ensures that
/// any window of `TAPS_PER_PHASE` consecutive samples from `write_pos`
/// is always contiguous — eliminating the need for circular wrap logic
/// in the SIMD inner loop.
struct DelayLine {
    /// Sample buffer (size = DELAY_LINE_LEN = 2 × TAPS_PER_PHASE).
    buf: AlignedVec<f32>,
    /// Write position (0..TAPS_PER_PHASE-1, wrapping).
    pos: usize,
}

impl DelayLine {
    fn new() -> Self {
        Self {
            buf: AlignedVec::new(DELAY_LINE_LEN, 0.0f32),
            pos: 0,
        }
    }

    /// Inserts a sample into the delay line (double-write for contiguity).
    #[inline(always)]
    fn push(&mut self, sample: f32) {
        let pos = self.pos;
        debug_assert!(pos < TAPS_PER_PHASE);
        unsafe {
            *self.buf.get_unchecked_mut(pos) = sample;
            *self.buf.get_unchecked_mut(pos + TAPS_PER_PHASE) = sample;
        }
        self.pos += 1;
        if self.pos >= TAPS_PER_PHASE {
            self.pos = 0;
        }
    }

    /// Returns a pointer to TAPS_PER_PHASE contiguous samples (most recent first).
    #[inline(always)]
    fn window_ptr(&self) -> *const f32 {
        debug_assert!(self.pos < TAPS_PER_PHASE);
        debug_assert!(self.pos + TAPS_PER_PHASE <= DELAY_LINE_LEN);
        unsafe { self.buf.as_ptr().add(self.pos) }
    }
}

/// Resampling engine for one direction (input or output).
///
/// Contains the polyphase bank and the fractional state for tracking
/// the position between input and output samples.
struct ResamplerCore {
    /// Polyphase filter bank (32B-aligned coefficients).
    bank: PolyphaseBank,
    /// Left channel delay line.
    state_l: DelayLine,
    /// Right channel delay line.
    state_r: DelayLine,
    /// Fractional position in phase space (0.0 .. NUM_PHASES) in 24.40 fixed point.
    /// Advances by `phase_step` on each output sample.
    phase_accum: u64,
    /// Phase increment per output sample = `from_rate / to_rate * NUM_PHASES` in 24.40 fixed point.
    phase_step: u64,
    /// Cached ISA dispatch for the hot-path stereo method.
    /// Set in `new()` to the monomorphized `process_internal::<Isa>`.
    /// Eliminates the global `dispatch_simd!` match per call.
    /// Micro-opt [T18.5c].
    process_stereo: ProcessFn,
    /// Cached ISA dispatch for the hot-path mono method.
    /// Micro-opt [T18.5c].
    process_mono: ProcessMonoFn,
}

/// Cached function pointer type for stereo resampling (hot path).
type ProcessFn = fn(&mut ResamplerCore, &[f32], &[f32], &mut [f32], &mut [f32]) -> usize;
/// Cached function pointer type for mono resampling (hot path).
type ProcessMonoFn = fn(&mut ResamplerCore, &[f32], &mut [f32], &mut [f32]) -> usize;

/// Trampoline: monomorphized `process_internal` for AVX2.
fn process_internal_avx2(
    core: &mut ResamplerCore,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> usize {
    core.process_internal::<Avx2Math>(in_l, in_r, out_l, out_r)
}

/// Trampoline: monomorphized `process_internal` for AVX-512.
fn process_internal_avx512(
    core: &mut ResamplerCore,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> usize {
    core.process_internal::<Avx512Math>(in_l, in_r, out_l, out_r)
}

/// Trampoline: monomorphized `process_internal` for AVX-512 VNNI BF16.
fn process_internal_avx512bf16(
    core: &mut ResamplerCore,
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> usize {
    core.process_internal::<Avx512VnniBf16Math>(in_l, in_r, out_l, out_r)
}

/// Trampoline: monomorphized `process_internal_mono` for AVX2.
fn process_internal_mono_avx2(
    core: &mut ResamplerCore,
    in_l: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> usize {
    core.process_internal_mono::<Avx2Math>(in_l, out_l, out_r)
}

/// Trampoline: monomorphized `process_internal_mono` for AVX-512.
fn process_internal_mono_avx512(
    core: &mut ResamplerCore,
    in_l: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> usize {
    core.process_internal_mono::<Avx512Math>(in_l, out_l, out_r)
}

/// Trampoline: monomorphized `process_internal_mono` for AVX-512 VNNI BF16.
fn process_internal_mono_avx512bf16(
    core: &mut ResamplerCore,
    in_l: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> usize {
    core.process_internal_mono::<Avx512VnniBf16Math>(in_l, out_l, out_r)
}

impl ResamplerCore {
    fn new(from_rate: u32, to_rate: u32, bank: PolyphaseBank) -> Self {
        // Starts the accumulator at NUM_PHASES so that the first iteration
        // of the processing loop immediately consumes the first input
        // sample, filling the delay line before attempting to produce output.
        let phase_step_f = (from_rate as f64 / to_rate as f64) * NUM_PHASES as f64;
        let phase_step = (phase_step_f * ((1u64 << 40) as f64)).round() as u64;

        // Cache the ISA-dispatched function pointers once, instead of
        // matching on the global `SIMD_MATH` on every hot-path call.
        // Micro-opt [T18.5c].
        let (process_stereo, process_mono) = match effective_instruction_set() {
            InstructionSet::Avx2 => (
                process_internal_avx2 as ProcessFn,
                process_internal_mono_avx2 as ProcessMonoFn,
            ),
            InstructionSet::Avx512 => (
                process_internal_avx512 as ProcessFn,
                process_internal_mono_avx512 as ProcessMonoFn,
            ),
            InstructionSet::Avx512VnniBf16 => (
                process_internal_avx512bf16 as ProcessFn,
                process_internal_mono_avx512bf16 as ProcessMonoFn,
            ),
        };

        Self {
            bank,
            state_l: DelayLine::new(),
            state_r: DelayLine::new(),
            phase_accum: (NUM_PHASES as u64) << 40,
            phase_step,
            process_stereo,
            process_mono,
        }
    }

    /// Processes a stereo block. RT-safe: zero allocations.
    ///
    /// Returns the number of samples written to `out_l` / `out_r`.
    fn process_internal<M: SimdMath>(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let n_in = in_l.len().min(in_r.len());
        let n_out_max = out_l.len().min(out_r.len());
        let mut in_idx = 0usize;
        let mut out_idx = 0usize;

        let num_phases_fp = (NUM_PHASES as u64) << 40;

        while out_idx < n_out_max {
            // Consume input samples as the phase accumulator advances
            while self.phase_accum >= num_phases_fp {
                if in_idx >= n_in {
                    return out_idx;
                }
                unsafe {
                    self.state_l.push(*in_l.get_unchecked(in_idx));
                    self.state_r.push(*in_r.get_unchecked(in_idx));
                }
                self.phase_accum -= num_phases_fp;
                in_idx += 1;
            }

            // Determines the phase and the fractional interpolation factor
            let phase_idx = (self.phase_accum >> 40) as usize;
            const FRAC_MASK: u64 = (1u64 << 40) - 1;
            let frac_bits = self.phase_accum & FRAC_MASK;
            let frac = (frac_bits as i64 as f64 * (1.0 / (1u64 << 40) as f64)) as f32;

            // Next phase (with wrap)
            let phase_next = if phase_idx + 1 >= NUM_PHASES {
                0
            } else {
                phase_idx + 1
            };

            // SIMD convolution + linear interpolation between adjacent phases
            let (y_l, y_r) = unsafe {
                let c0 = self.bank.phase_ptr(phase_idx);
                let c1 = self.bank.phase_ptr(phase_next);
                let x_l = self.state_l.window_ptr();
                let x_r = self.state_r.window_ptr();
                let taps = self.bank.taps_per_phase;

                let ((y0_l, y0_r), (y1_l, y1_r)) = M::convolve_stereo_dual(c0, c1, x_l, x_r, taps);
                (y0_l + frac * (y1_l - y0_l), y0_r + frac * (y1_r - y0_r))
            };

            unsafe {
                *out_l.get_unchecked_mut(out_idx) = y_l;
                *out_r.get_unchecked_mut(out_idx) = y_r;
            }
            out_idx += 1;

            self.phase_accum += self.phase_step;
        }

        // Consume remaining input samples (keep state updated)
        while self.phase_accum >= num_phases_fp && in_idx < n_in {
            unsafe {
                self.state_l.push(*in_l.get_unchecked(in_idx));
                self.state_r.push(*in_r.get_unchecked(in_idx));
            }
            self.phase_accum -= num_phases_fp;
            in_idx += 1;
        }

        out_idx
    }

    /// Processes a mono block. RT-safe: zero allocations.
    ///
    /// Returns the number of samples written to `out_l` / `out_r`.
    fn process_internal_mono<M: SimdMath>(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let n_in = in_l.len();
        let n_out_max = out_l.len().min(out_r.len());
        let mut in_idx = 0usize;
        let mut out_idx = 0usize;

        let num_phases_fp = (NUM_PHASES as u64) << 40;

        while out_idx < n_out_max {
            // Consume input samples as the phase accumulator advances
            while self.phase_accum >= num_phases_fp {
                if in_idx >= n_in {
                    return out_idx;
                }
                unsafe {
                    self.state_l.push(*in_l.get_unchecked(in_idx));
                }
                self.phase_accum -= num_phases_fp;
                in_idx += 1;
            }

            // Determines the phase and the fractional interpolation factor
            let phase_idx = (self.phase_accum >> 40) as usize;
            const FRAC_MASK: u64 = (1u64 << 40) - 1;
            let frac_bits = self.phase_accum & FRAC_MASK;
            let frac = (frac_bits as i64 as f64 * (1.0 / (1u64 << 40) as f64)) as f32;

            // Next phase (with wrap)
            let phase_next = if phase_idx + 1 >= NUM_PHASES {
                0
            } else {
                phase_idx + 1
            };

            // SIMD convolution + linear interpolation between adjacent phases
            let y_l = unsafe {
                let c0 = self.bank.phase_ptr(phase_idx);
                let c1 = self.bank.phase_ptr(phase_next);
                let x_l = self.state_l.window_ptr();
                let taps = self.bank.taps_per_phase;

                let (y0_l, y1_l) = M::convolve_mono_dual(c0, c1, x_l, taps);
                y0_l + frac * (y1_l - y0_l)
            };

            unsafe {
                *out_l.get_unchecked_mut(out_idx) = y_l;
                *out_r.get_unchecked_mut(out_idx) = y_l;
            }
            out_idx += 1;

            self.phase_accum += self.phase_step;
        }

        // Consume remaining input samples (keep state updated)
        while self.phase_accum >= num_phases_fp && in_idx < n_in {
            unsafe {
                self.state_l.push(*in_l.get_unchecked(in_idx));
            }
            self.phase_accum -= num_phases_fp;
            in_idx += 1;
        }

        out_idx
    }
}

/// RT-safe wrapper for bidirectional Minimum-Phase Polyphase Sinc FIR resampling.
///
/// Encapsulates two independent pre-allocated engines (input + output).
/// In the DSP thread only `process_input()` / `process_output()` are called —
/// zero-alloc operations that work on pre-allocated delay lines.
///
/// When `pw_rate == nam_rate`, both engines are bypassed (`None`)
/// and the hot path passes through with zero overhead.
pub struct NamResampler {
    /// Input engine: `pw_rate → nam_rate`. `None` = bypass.
    inner: Option<ResamplerCore>,
    /// Output engine: `nam_rate → pw_rate`. `None` = bypass.
    outer: Option<ResamplerCore>,
    /// PipeWire rate.
    pw_rate: u32,
    /// Target NAM model rate.
    nam_rate: u32,
}

impl NamResampler {
    /// Creates the pair of resamplers (input+output), pre-allocating all buffers.
    ///
    /// Produces a **minimum-phase** polyphase resampler — the production default.
    /// Minimum-phase eliminates pre-ringing at the cost of non-linear phase response,
    /// ideal for live monitoring where transient fidelity dominates.
    ///
    /// If `pw_rate == nam_rate`, full bypass with no overhead.
    ///
    /// # Parameters
    /// - `pw_rate`: PipeWire rate (e.g., 44100, 48000, 96000).
    /// - `nam_rate`: NAM model rate (e.g., 48000).
    /// - `_chunk_size`: kept for API compatibility (not used internally).
    #[cold]
    pub fn new(pw_rate: u32, nam_rate: u32, _chunk_size: usize) -> Result<Self> {
        if pw_rate == 0 || nam_rate == 0 {
            bail!("NamResampler: sample rates must not be null");
        }

        if pw_rate == nam_rate {
            return Ok(Self {
                inner: None,
                outer: None,
                pw_rate,
                nam_rate,
            });
        }

        let inner = ResamplerCore::new(
            pw_rate,
            nam_rate,
            generate_polyphase_bank(pw_rate, nam_rate),
        );
        let outer = ResamplerCore::new(
            nam_rate,
            pw_rate,
            generate_polyphase_bank(nam_rate, pw_rate),
        );

        Ok(Self {
            inner: Some(inner),
            outer: Some(outer),
            pw_rate,
            nam_rate,
        })
    }

    /// Creates the pair of resamplers using **linear-phase** polyphase banks.
    ///
    /// Linear-phase preserves perfect phase linearity at the cost of pre-ringing —
    /// suitable for offline rendering and mixdown where latency is irrelevant and
    /// phase accuracy is paramount.
    ///
    /// If `pw_rate == nam_rate`, full bypass with no overhead.
    #[cold]
    pub fn new_linear(pw_rate: u32, nam_rate: u32, _chunk_size: usize) -> Result<Self> {
        if pw_rate == 0 || nam_rate == 0 {
            bail!("NamResampler: sample rates must not be null");
        }

        if pw_rate == nam_rate {
            return Ok(Self {
                inner: None,
                outer: None,
                pw_rate,
                nam_rate,
            });
        }

        let inner = ResamplerCore::new(
            pw_rate,
            nam_rate,
            generate_polyphase_bank_linear(pw_rate, nam_rate),
        );
        let outer = ResamplerCore::new(
            nam_rate,
            pw_rate,
            generate_polyphase_bank_linear(nam_rate, pw_rate),
        );

        Ok(Self {
            inner: Some(inner),
            outer: Some(outer),
            pw_rate,
            nam_rate,
        })
    }

    /// Returns `true` when `pw_rate == nam_rate` (bypass).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        self.inner.is_none()
    }

    /// Returns the PipeWire rate.
    #[inline]
    pub fn pw_rate(&self) -> u32 {
        self.pw_rate
    }

    /// Returns the NAM model rate.
    #[inline]
    pub fn nam_rate(&self) -> u32 {
        self.nam_rate
    }

    /// Computes the total latency (input + output) in host-rate samples.
    ///
    /// Latency is deterministic and uses a conservative estimate of
    /// TAPS_PER_PHASE/2 per filter (64 taps → 32 sample group delay per stage).
    ///
    /// # Parameters
    /// - `host_rate`: host sample rate (e.g., 44100, 48000, 96000).
    ///
    /// # Returns
    /// Total latency in samples at `host_rate`.
    pub fn latency_samples(&self, host_rate: u32) -> u32 {
        if self.is_bypass() || host_rate == 0 {
            return 0;
        }

        // TAPS_PER_PHASE (64) is the order of each sub-filter.
        // The group average latency is ~ (TAPS_PER_PHASE / 2).
        let taps_half = TAPS_PER_PHASE as f64 / 2.0;

        // Input filter latency (host -> nam):
        // measured in host-rate samples.
        let latency_in = taps_half * (self.pw_rate as f64 / self.nam_rate as f64);

        // Output filter latency (nam -> host):
        // measured in host-rate samples.
        let latency_out = taps_half;

        (latency_in + latency_out).round() as u32
    }

    /// **Input resampling** (input path): `pw_rate → nam_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// # Returns
    /// Number of samples written to `out_l` / `out_r`.
    pub fn process_input(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.inner else {
            let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
            unsafe {
                core::ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                core::ptr::copy_nonoverlapping(in_r.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return n;
        };
        (core.process_stereo)(core, in_l, in_r, out_l, out_r)
    }

    /// **Output resampling** (output path): `nam_rate → pw_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// # Returns
    /// Number of samples written to `out_l` / `out_r`.
    pub fn process_output(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.outer else {
            let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
            unsafe {
                core::ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                core::ptr::copy_nonoverlapping(in_r.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return n;
        };
        (core.process_stereo)(core, in_l, in_r, out_l, out_r)
    }

    /// **Mono input resampling** (input path): `pw_rate → nam_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// # Returns
    /// Number of samples written to `out_l` / `out_r`.
    pub fn process_input_mono(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.inner else {
            let n = in_l.len().min(out_l.len()).min(out_r.len());
            unsafe {
                core::ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                core::ptr::copy_nonoverlapping(in_l.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return n;
        };
        (core.process_mono)(core, in_l, out_l, out_r)
    }

    /// **Mono output resampling** (output path): `nam_rate → pw_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// # Returns
    /// Number of samples written to `out_l` / `out_r`.
    pub fn process_output_mono(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.outer else {
            let n = in_l.len().min(out_l.len()).min(out_r.len());
            unsafe {
                core::ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                core::ptr::copy_nonoverlapping(in_l.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return n;
        };
        (core.process_mono)(core, in_l, out_l, out_r)
    }
}

#[cfg(test)]
#[path = "resampler_test.rs"]
mod resampler_test;
