// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Individual DSP pipeline stages: input gate/gain,
//! neural inference, output gain/clipping, and bridge writing.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::common::spsc::RtStatusFlags;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::adaptive::{AdaptiveCompute, AdaptiveState};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::{DynamicHysteresis, GateState};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::math::common::dispatch_simd;
#[cfg(feature = "stereo")]
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::math::dsp::stereo::{compute_energy_stereo, compute_max_diff};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::models::{DynamicModel, NamModel};

use super::bridge::{DspBridgeWriter, MAX_RESAMP_BUF};
use super::context::DspPipelineContext;

/// Ultra-low DC offset injected at the input stage to prevent subnormal floats
/// from propagating through neural network activations during fade-out/silence.
/// At -220 dBFS, this bias is 76 dB below the 24-bit DAC noise floor and inaudible.
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
const DENORMAL_DITHER_OFFSET: f32 = 1.0e-11;

/// Silence Bypass: signals silence and zeros the bridge so that playback emits silence.
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
#[cold]
#[inline(never)]
pub fn handle_silence_bypass(bridge: Option<DspBridgeWriter>, rt_status: &RtStatusFlags) {
    rt_status.set_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
    rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);

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
        return GateState::Closed;
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
    unsafe {
        for i in 0..n_samples {
            *samples_l.get_unchecked_mut(i) += DENORMAL_DITHER_OFFSET;
        }
        #[cfg(feature = "stereo")]
        if !*ctx.process_mono {
            for i in 0..n_samples {
                *samples_r.get_unchecked_mut(i) += DENORMAL_DITHER_OFFSET;
            }
        }
    }

    ctx.silence_hysteresis.state()
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Updates model effective layers for soft-degrade based on the adaptive FSM state.
/// Returns `true` if the LSTM model should be fully bypassed (Minimal passthrough).
#[inline(always)]
fn configure_adaptive_model(
    model_l: &mut Option<Box<crate::models::DynamicModel>>,
    model_r: &mut Option<Box<crate::models::DynamicModel>>,
    adaptive: &AdaptiveCompute,
) -> bool {
    if adaptive.mode() == crate::dsp::adaptive::AdaptiveComputeMode::Off {
        return false;
    }

    // Hold effective layers at the previous level while crossfading.
    // The model structural change is deferred until the crossfade completes,
    // preventing audible discontinuities from abrupt layer count changes.
    if adaptive.is_crossfading() {
        return false;
    }

    match adaptive.state() {
        AdaptiveState::Full => {
            if let Some(m) = model_l {
                let layers = m.layer_count();
                m.set_effective_layers(layers);
            }
            if let Some(m) = model_r {
                let layers = m.layer_count();
                m.set_effective_layers(layers);
            }
            false
        }
        AdaptiveState::Reduced => {
            if let Some(m) = model_l.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                m.set_effective_layers(effective);
            }
            if let Some(m) = model_r.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                m.set_effective_layers(effective);
            }
            false
        }
        AdaptiveState::Minimal => {
            let lstm_skip = model_l.as_ref().is_some_and(|m| m.is_lstm());
            if lstm_skip {
                return true;
            }
            if let Some(m) = model_l.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                m.set_effective_layers(effective);
            }
            if let Some(m) = model_r.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                m.set_effective_layers(effective);
            }
            false
        }
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Unified helper for mono/stereo processing of neural models.
///
/// Processes the L channel model (_always_) and decides whether the R channel is a mono copy
/// or independent processing via the active R model.
#[inline(always)]
fn run_stereo_or_mono(
    active_model_l: &mut Option<Box<DynamicModel>>,
    active_model_r: &mut Option<Box<DynamicModel>>,
    model_in_l: &[f32],
    model_in_r: &[f32],
    m_out_l: &mut [f32],
    m_out_r: &mut [f32],
    process_mono: bool,
) {
    if let Some(model_l) = active_model_l {
        model_l.process(model_in_l, m_out_l);
    } else {
        m_out_l.copy_from_slice(model_in_l);
    }

    if process_mono {
        m_out_r.copy_from_slice(m_out_l);
    } else if let Some(model_r) = active_model_r {
        model_r.process(model_in_r, m_out_r);
    } else {
        m_out_r.copy_from_slice(model_in_r);
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Stage 2: Neural Inference and Resampling.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_inference(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
    resamp_mid_l: &mut [f32],
    resamp_mid_r: &mut [f32],
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    model_out_l: &mut [f32],
    model_out_r: &mut [f32],
) -> usize {
    let is_resamp_bypass = ctx.resampler.is_bypass();
    let n = n_samples.min(MAX_RESAMP_BUF);

    // Soft-degrade: configure model layers based on CPU pressure
    let lstm_passthrough =
        configure_adaptive_model(ctx.active_model_l, ctx.active_model_r, ctx.adaptive);

    // PATH A: Quality adjustment off (Resampler in Bypass).
    if is_resamp_bypass {
        let model_in_l = &samples_l[..n];
        let model_in_r = if *ctx.process_mono {
            &samples_l[..n]
        } else {
            &samples_r[..n]
        };
        let m_out_l = &mut resamp_out_l[..n];
        let m_out_r = &mut resamp_out_r[..n];

        if lstm_passthrough {
            // LSTM Minimal: passthrough input with gain compensation
            m_out_l.copy_from_slice(model_in_l);
            m_out_r.copy_from_slice(model_in_r);
        } else {
            run_stereo_or_mono(
                ctx.active_model_l,
                ctx.active_model_r,
                model_in_l,
                model_in_r,
                m_out_l,
                m_out_r,
                *ctx.process_mono,
            );
        }

        n
    } else {
        // PATH B: Quality adjustment on (Active Resampler).

        // 1. Translates the sound to the frequency the neural "Brain" understands (usually 48kHz).
        let n_48k = if *ctx.process_mono {
            ctx.resampler.process_input_mono(
                &samples_l[..n],
                &mut resamp_mid_l[..MAX_RESAMP_BUF],
                &mut resamp_mid_r[..MAX_RESAMP_BUF],
            )
        } else {
            ctx.resampler.process_input(
                &samples_l[..n],
                &samples_r[..n],
                &mut resamp_mid_l[..MAX_RESAMP_BUF],
                &mut resamp_mid_r[..MAX_RESAMP_BUF],
            )
        };

        let model_in_l = &resamp_mid_l[..n_48k];
        let model_in_r = &resamp_mid_r[..n_48k];
        let m_out_l = &mut model_out_l[..n_48k];
        let m_out_r = &mut model_out_r[..n_48k];

        // 2. Applies the amplifier simulation (Neural Model).
        if lstm_passthrough {
            m_out_l.copy_from_slice(model_in_l);
            m_out_r.copy_from_slice(model_in_r);
        } else {
            run_stereo_or_mono(
                ctx.active_model_l,
                ctx.active_model_r,
                model_in_l,
                model_in_r,
                m_out_l,
                m_out_r,
                *ctx.process_mono,
            );
        }

        // 3. Translates the sound back to the original frequency of your sound card.
        if *ctx.process_mono {
            ctx.resampler.process_output_mono(
                m_out_l,
                &mut resamp_out_l[..MAX_RESAMP_BUF],
                &mut resamp_out_r[..MAX_RESAMP_BUF],
            )
        } else {
            ctx.resampler.process_output(
                m_out_l,
                m_out_r,
                &mut resamp_out_l[..MAX_RESAMP_BUF],
                &mut resamp_out_r[..MAX_RESAMP_BUF],
            )
        }
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Stage 3: Output Gain, Fading, Clipping Detection, and Degrade Crossfade.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_output_stage(
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    n_pw: usize,
    output_gain_mult: f32,
    silence_hysteresis: &mut DynamicHysteresis,
    rt_status: &RtStatusFlags,
    process_mono: bool,
    adaptive: &mut AdaptiveCompute,
    sample_rate: u32,
) {
    // 0. DENORMAL SUPPRESSION COMPENSATION: Subtract the injected DC offset
    // Compensates the ultra-low bias added at the input stage. Any residual is
    // far below the DAC noise floor and inaudible.
    unsafe {
        for i in 0..n_pw {
            *resamp_out_l.get_unchecked_mut(i) -= DENORMAL_DITHER_OFFSET;
        }
        if !process_mono {
            for i in 0..n_pw {
                *resamp_out_r.get_unchecked_mut(i) -= DENORMAL_DITHER_OFFSET;
            }
        }
    }

    // 1. FINAL VOLUME ADJUSTMENT AND CLIPPING PROTECTION
    // Applies output volume and checks if the sound has "blown past" the digital limit.
    let has_clipped = crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_stereo(
        &mut resamp_out_l[..n_pw],
        &mut resamp_out_r[..n_pw],
        output_gain_mult
    ));

    // 2. NOISE GATE SMOOTHING (FADING)
    // Applies smooth opening/closing of the sound (fade) to avoid pops or clicks.
    dispatch_simd!(
        silence_hysteresis,
        apply_gain_rt_stereo,
        &mut resamp_out_l[..n_pw],
        &mut resamp_out_r[..n_pw],
        n_pw
    );

    // If the sound "blew past" the limit at any moment, we raise a warning flag in the system.
    if has_clipped {
        rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
    }

    if process_mono {
        resamp_out_r[..n_pw].copy_from_slice(&resamp_out_l[..n_pw]);
    }

    // ── Degrade Crossfade ──
    // Advances the crossfade timer. During active crossfade,
    // effective model layers are held at their previous configuration
    // (see configure_adaptive_model), avoiding discontinuities
    // from abrupt structural changes. The crossfade timer acts as a
    // deferral mechanism — when it completes, the model switches to
    // the new degradation level in a single block transition.
    adaptive.crossfade_multiplier(sample_rate, n_pw);
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Stage 4: Write to DspBridge.
#[inline(always)]
pub fn write_bridge(
    resamp_out_l: &[f32],
    resamp_out_r: &[f32],
    n_pw: usize,
    bridge: Option<DspBridgeWriter>,
) {
    if let Some(writer) = bridge {
        writer.write_block(resamp_out_l, resamp_out_r, n_pw);
    }
}
