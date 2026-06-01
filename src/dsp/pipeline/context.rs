// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Contexto e buffers de trabalho para o pipeline DSP.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::common::spsc::RtStatusFlags;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::{DynamicHysteresis, GateParams};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::resampler::NamResampler;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::models::DynamicModel;

use super::bridge::DspBridgeWriter;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Contexto de dados para a pipeline DSP hot-path.
pub struct DspPipelineContext<'a> {
    /// Resampler ativo para conversão de sample rate.
    pub resampler: &'a mut NamResampler,
    /// Modelo ativo para o canal esquerdo.
    pub active_model_l: &'a mut Option<Box<DynamicModel>>,
    /// Modelo ativo para o canal direito.
    pub active_model_r: &'a mut Option<Box<DynamicModel>>,
    /// Multiplicador de ganho de entrada.
    pub input_gain_mult: f32,
    /// Multiplicador de ganho de saída.
    pub output_gain_mult: f32,
    /// Parâmetros do Noise Gate.
    pub gate_params: &'a GateParams,
    /// Histerese para detecção de silêncio.
    pub silence_hysteresis: &'a mut DynamicHysteresis,
    /// Histerese para detecção de sinal mono.
    pub mono_hysteresis: &'a mut DynamicHysteresis,
    /// Limiar de abertura (ao quadrado).
    pub threshold_open_sq: f32,
    /// Limiar de fechamento (ao quadrado).
    pub threshold_close_sq: f32,
    /// Flag indicando processamento em mono.
    pub process_mono: &'a mut bool,
    /// Flags de status RT.
    pub rt_status: &'a RtStatusFlags,
    /// Referência para a ponte de monitoração de áudio (opcional).
    pub bridge_writer: Option<DspBridgeWriter>,
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Conjunto de buffers de trabalho para o pipeline DSP.
/// Buffers de trabalho intermediários para o pipeline DSP.
pub struct DspBuffers<'a> {
    /// Buffer intermediário pós-resampler L.
    pub resamp_mid_l: &'a mut [f32],
    /// Buffer intermediário pós-resampler R.
    pub resamp_mid_r: &'a mut [f32],
    /// Buffer de saída do resampler L.
    pub resamp_out_l: &'a mut [f32],
    /// Buffer de saída do resampler R.
    pub resamp_out_r: &'a mut [f32],
    /// Buffer de saída do modelo L.
    pub model_out_l: &'a mut [f32],
    /// Buffer de saída do modelo R.
    pub model_out_r: &'a mut [f32],
}
