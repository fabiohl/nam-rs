// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Context and working buffers for the DSP pipeline.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::common::spsc::RtStatusFlags;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::adaptive::AdaptiveCompute;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::cabsim::conv::ConvEngine;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::{DynamicHysteresis, GateParams};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::resampler::NamResampler;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::models::StaticModel;

use super::bridge::DspBridgeWriter;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::oversample::OversampleEngine;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Data context for the DSP pipeline hot-path.
pub struct DspPipelineContext<'a> {
    /// Active resampler for sample rate conversion.
    pub resampler: &'a mut NamResampler,
    /// Optional half-band oversampling engine for the left channel.
    pub os_l: &'a mut OversampleEngine,
    /// Optional half-band oversampling engine for the right channel.
    pub os_r: &'a mut OversampleEngine,
    /// Active model for the left channel.
    pub active_model_l: &'a mut Option<Box<StaticModel>>,
    /// Active model for the right channel.
    pub active_model_r: &'a mut Option<Box<StaticModel>>,
    /// Input gain multiplier applied in `apply_input_stage` (pipeline-level).
    /// In the CLAP plugin this is always `1.0` because input gain is applied
    /// separately via `NamClapProcessor::apply_input_gain` (which uses the
    /// smoothed user-configured gain). In standalone mode this reflects the
    /// combined user + model gain multiplier.
    pub input_gain_mult: f32,
    /// Output gain multiplier applied in `apply_output_stage` (pipeline-level).
    /// In the CLAP plugin this is always `1.0` because output gain is applied
    /// separately via `NamClapProcessor::apply_output_gain` (which uses the
    /// smoothed user-configured gain). In standalone mode this reflects the
    /// combined user + model gain multiplier.
    pub output_gain_mult: f32,
    /// Noise Gate parameters.
    pub gate_params: &'a GateParams,
    /// Hysteresis for silence detection.
    pub silence_hysteresis: &'a mut DynamicHysteresis,
    /// Hysteresis for mono signal detection.
    pub mono_hysteresis: &'a mut DynamicHysteresis,
    /// Opening threshold (squared).
    pub threshold_open_sq: f32,
    /// Closing threshold (squared).
    pub threshold_close_sq: f32,
    /// Flag indicating mono processing.
    pub process_mono: &'a mut bool,
    /// RT status flags.
    pub rt_status: &'a RtStatusFlags,
    /// Adaptive compute state for soft-degrade.
    pub adaptive: &'a mut AdaptiveCompute,
    /// Reference to the audio monitoring bridge (optional).
    pub bridge_writer: Option<DspBridgeWriter>,
    /// Active cab-sim convolution engine (None = bypass, zero cost).
    pub conv: Option<&'a mut ConvEngine>,
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Set of working buffers for the DSP pipeline.
/// Intermediate working buffers for the DSP pipeline.
pub struct DspBuffers<'a> {
    /// Intermediate post-resampler L buffer.
    pub resamp_mid_l: &'a mut [f32],
    /// Intermediate post-resampler R buffer.
    pub resamp_mid_r: &'a mut [f32],
    /// Resampler output buffer L.
    pub resamp_out_l: &'a mut [f32],
    /// Resampler output buffer R.
    pub resamp_out_r: &'a mut [f32],
    /// Model output buffer L.
    pub model_out_l: &'a mut [f32],
    /// Model output buffer R.
    pub model_out_r: &'a mut [f32],
    /// Oversampled input buffer L (pre-model, at 2×/4× rate).
    pub os_in_l: &'a mut [f32],
    /// Oversampled input buffer R.
    pub os_in_r: &'a mut [f32],
    /// Oversampled model output buffer L (post-model, at 2×/4× rate).
    pub os_model_l: &'a mut [f32],
    /// Oversampled model output buffer R.
    pub os_model_r: &'a mut [f32],
}
