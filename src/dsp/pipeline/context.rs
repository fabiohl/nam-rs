// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Context and working buffers for the DSP pipeline.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::common::spsc::RtStatusFlags;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::adaptive::AdaptiveCompute;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::{DynamicHysteresis, GateParams};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::resampler::NamResampler;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::models::DynamicModel;

use super::bridge::DspBridgeWriter;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Data context for the DSP pipeline hot-path.
pub struct DspPipelineContext<'a> {
    /// Active resampler for sample rate conversion.
    pub resampler: &'a mut NamResampler,
    /// Active model for the left channel.
    pub active_model_l: &'a mut Option<Box<DynamicModel>>,
    /// Active model for the right channel.
    pub active_model_r: &'a mut Option<Box<DynamicModel>>,
    /// Input gain multiplier.
    pub input_gain_mult: f32,
    /// Output gain multiplier.
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
}
