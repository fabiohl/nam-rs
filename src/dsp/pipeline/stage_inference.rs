// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 2: Neural Inference and Resampling.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::adaptive::{AdaptiveCompute, AdaptiveState};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::models::{DynamicModel, NamModel};

use super::bridge::MAX_RESAMP_BUF;
use super::context::DspPipelineContext;

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
