// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP state aggregated from the preamble of `setup_capture_stream`.
//!
//! Groups all stack-allocated locals initialized before the RT `process()` closure.
//! The struct is captured by `move` into the closure — all fields remain on the
//! stack, with no heap indirection, preserving RT safety.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode};
use crate::common::params::AdaptiveComputeMode;
use crate::common::spsc::GcItem;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::cabsim::conv::ConvEngine;
use crate::dsp::gate::{DynamicHysteresis, GateParams};
use crate::dsp::pipeline::MAX_RESAMP_BUF;
use crate::dsp::resampler::NamResampler;
use crate::math::dsp::gain_lut;
use crate::models::StaticModel;
use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

pub struct CaptureState {
    pub active_model_l: Option<Box<crate::models::StaticModel>>,
    pub active_model_r: Option<Box<crate::models::StaticModel>>,
    pub resampler: Box<NamResampler>,
    pub active_cabsim: Option<Box<ConvEngine>>,
    pub current_nam_rate: u32,
    pub resamp_mid_l: [f32; MAX_RESAMP_BUF],
    pub resamp_out_l: [f32; MAX_RESAMP_BUF],
    pub resamp_mid_r: [f32; MAX_RESAMP_BUF],
    pub resamp_out_r: [f32; MAX_RESAMP_BUF],
    pub model_out_l: [f32; MAX_RESAMP_BUF],
    pub model_out_r: [f32; MAX_RESAMP_BUF],
    pub user_input_gain_mult: f32,
    pub user_output_gain_mult: f32,
    pub model_input_mult_adj: f32,
    pub model_output_mult_adj: f32,
    pub input_gain_mult: f32,
    pub output_gain_mult: f32,
    pub gate_params: GateParams,
    pub silence_hysteresis: DynamicHysteresis,
    pub mono_hysteresis: DynamicHysteresis,
    pub process_mono: bool,
    pub adaptive_compute: AdaptiveCompute,
    pub threshold_open_sq: f32,
    pub threshold_close_sq: f32,
    pub shared_target_rate: Arc<AtomicU32>,
    pub parking_lot: [Option<GcItem>; 16],
    pub frame_count: u32,
    pub thread_configured: bool,
    pub ir_raw_samples: Option<Vec<f32>>,
    pub slimmable_rx: Option<Consumer<Option<Box<StaticModel>>>>,
}

impl CaptureState {
    pub fn init(sys: &crate::common::diagnostics::SystemSnapshot) -> Self {
        let resampler = NamResampler::new(48_000, 48_000, 2048).unwrap_or_else(|e| {
            NamDiagnostic::new(NamErrorCode::ResamplerBuildFailed, sys)
                .message("Failed to create initial NamResampler (using 48k bypass).")
                .hint("The engine remains in bypass mode. The resampler will be recreated upon receiving the actual rate from PipeWire.")
                .param("initial_rate", 48_000_u32)
                .param("detail", &e)
                .emit_warning();
            NamResampler::new(48_000, 48_000, 2048).expect("bypass cannot fail")
        });

        let gate_params = GateParams::default();
        let lut = gain_lut::get_gain_lut();
        let open_lin = lut.db_to_linear(gate_params.threshold_open_db);
        let close_lin = lut.db_to_linear(gate_params.threshold_close_db);

        Self {
            active_model_l: None,
            active_model_r: None,
            resampler: Box::new(resampler),
            active_cabsim: None,
            current_nam_rate: 48_000,
            resamp_mid_l: [0.0f32; MAX_RESAMP_BUF],
            resamp_out_l: [0.0f32; MAX_RESAMP_BUF],
            resamp_mid_r: [0.0f32; MAX_RESAMP_BUF],
            resamp_out_r: [0.0f32; MAX_RESAMP_BUF],
            model_out_l: [0.0f32; MAX_RESAMP_BUF],
            model_out_r: [0.0f32; MAX_RESAMP_BUF],
            user_input_gain_mult: 1.0,
            user_output_gain_mult: 1.0,
            model_input_mult_adj: 1.0,
            model_output_mult_adj: 1.0,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params,
            silence_hysteresis: DynamicHysteresis::new(),
            mono_hysteresis: DynamicHysteresis::new(),
            process_mono: false,
            adaptive_compute: AdaptiveCompute::new(AdaptiveComputeMode::Off),
            threshold_open_sq: open_lin * open_lin,
            threshold_close_sq: close_lin * close_lin,
            shared_target_rate: Arc::new(AtomicU32::new(0)),
            parking_lot: Default::default(),
            frame_count: 0,
            thread_configured: false,
            ir_raw_samples: None,
            slimmable_rx: None,
        }
    }
}
