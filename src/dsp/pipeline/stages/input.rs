// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 1: Gate, Input Gains, Mono Detection, and Silence Bypass.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::common::spsc::RtStatusFlags;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::GateState;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
#[cfg(feature = "stereo")]
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::math::dsp::stereo::{compute_energy_stereo, compute_max_diff};

use super::super::bridge::DspBridgeWriter;
use super::super::context::DspPipelineContext;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Global control to disable the noise gate/silence bypass during profiling/benchmarks.
pub static DISABLE_GATE: AtomicBool = AtomicBool::new(false);

/// Ultra-low DC offset injected at the input stage to prevent subnormal floats
/// from propagating through neural network activations during fade-out/silence.
/// At -220 dBFS, this bias is 76 dB below the 24-bit DAC noise floor and inaudible.
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
pub(crate) const DENORMAL_DITHER_OFFSET: f32 = 1.0e-11;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Silence Bypass: signals silence and zeros the bridge so that playback emits silence.
/// Delegates gate-flag reporting to the canonical `report_gate_flags`.
#[cold]
#[inline(never)]
pub fn handle_silence_bypass(bridge: Option<DspBridgeWriter>, rt_status: &RtStatusFlags) {
    crate::dsp::gate_flags::report_gate_flags(rt_status, crate::dsp::gate::GateState::Closed);

    if let Some(writer) = bridge {
        writer.write_silence();
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Stage 1: Gate, Input Gains, and Mono Detection.
#[inline(always)]
pub(crate) fn apply_input_stage(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
) -> GateState {
    // Uses the maximum energy of both channels: any channel with active signal
    // must keep the gate open. Using the fused kernel to reduce cache traffic.
    #[cfg(feature = "stereo")]
    let energy_ms =
        unsafe { compute_energy_stereo(&samples_l[..n_samples], &samples_r[..n_samples]) };
    #[cfg(not(feature = "stereo"))]
    let energy_ms = crate::math::common::dispatch_simd!(compute_energy(&samples_l[..n_samples]));

    #[cfg(not(feature = "stereo"))]
    let _ = samples_r;

    // 1. UPDATE THE NOISE GATE
    // Decides whether the sound is strong enough to pass or should be silenced to save processing.
    ctx.silence_hysteresis.update(
        energy_ms,
        ctx.threshold_open_sq,
        ctx.threshold_close_sq,
        ctx.gate_params,
        n_samples,
    );

    // If the gate is fully closed (absolute silence), we stop here to save battery/CPU.
    if ctx.silence_hysteresis.state() == GateState::Closed {
        if DISABLE_GATE.load(Ordering::Relaxed) {
            // Keep running the model, return FadingIn or Open so the pipeline runs neural inference
            return GateState::Open;
        } else {
            return GateState::Closed;
        }
    }

    #[cfg(feature = "stereo")]
    {
        // 2. MONO SOUND DETECTION (SAME ON BOTH SIDES)
        // Computes the difference between left and right to check if the sound is the same.
        let max_diff =
            unsafe { compute_max_diff(&samples_l[..n_samples], &samples_r[..n_samples]) };

        ctx.mono_hysteresis.update(
            max_diff,
            ctx.gate_params.mono_epsilon,
            ctx.gate_params.mono_epsilon * 0.9,
            ctx.gate_params,
            n_samples,
        );

        // If the sound is identical on both sides (mono), notify the system to process only one side.
        // This cuts the workload in half without losing quality!
        *ctx.process_mono = ctx.mono_hysteresis.state() == GateState::Closed
            || ctx.mono_hysteresis.state() == GateState::FadingOut;
    }
    #[cfg(not(feature = "stereo"))]
    {
        *ctx.process_mono = true;
    }

    // 3. INPUT VOLUME ADJUSTMENT (GAIN)
    // Applies the initial user-defined gain (volume).
    crate::math::dsp::gain::apply_gain_simd(&mut samples_l[..n_samples], ctx.input_gain_mult);

    // Only adjust the right side if the sound is NOT mono (to save processing).
    #[cfg(feature = "stereo")]
    if !*ctx.process_mono {
        crate::math::dsp::gain::apply_gain_simd(&mut samples_r[..n_samples], ctx.input_gain_mult);
    }

    // 4. DENORMAL SUPPRESSION: Inject ultra-low DC offset
    // Prevents subnormal floats from reaching neural network activations
    // during fade-out/silence, ensuring smooth decay without digital artifacts.
    crate::math::dsp::gain::apply_dither_add_simd(
        &mut samples_l[..n_samples],
        DENORMAL_DITHER_OFFSET,
    );

    #[cfg(feature = "stereo")]
    if !*ctx.process_mono {
        crate::math::dsp::gain::apply_dither_add_simd(
            &mut samples_r[..n_samples],
            DENORMAL_DITHER_OFFSET,
        );
    }

    ctx.silence_hysteresis.state()
}
