// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 3: Output Gain, Fading, Clipping Detection, and Degrade Crossfade.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::common::spsc::RtStatusFlags;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::adaptive::AdaptiveCompute;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::DynamicHysteresis;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::math::common::dispatch_simd;

use super::input::DENORMAL_DITHER_OFFSET;

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
    crate::math::dsp::gain::apply_dither_add_simd(
        &mut resamp_out_l[..n_pw],
        -DENORMAL_DITHER_OFFSET,
    );
    if !process_mono {
        crate::math::dsp::gain::apply_dither_add_simd(
            &mut resamp_out_r[..n_pw],
            -DENORMAL_DITHER_OFFSET,
        );
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
