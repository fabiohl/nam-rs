// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Architecture.
//!
//! Este módulo isola componentes da arquitetura A2 (v0.6+), incluindo
//! ativações, FiLM, gating, parâmetros e os modelos A2.
//!
//! ## Status
//!
//! O scaffolding existente inclui a definição completa de parâmetros e
//! constantes arquiteturais (espelhando `a2_fast.h`). O fast-path
//! A2-Full/Lite (`Épico 1`) utilizará um subconjunto destes structs;
//! os campos de FiLM, gating (`Gated`/`Blended`), `head1x1`, `bottleneck ≠ channels`
//! e ativações heterogêneas estão reservados para o motor A2 geral (futuro).

pub mod activations;
pub mod conv1d;
pub mod conv1d_ch3;
pub mod conv1d_ch8;
pub mod conv1d_fallback;
pub mod film;
pub mod gating;
pub mod head;
pub mod layer;
pub mod model;
pub mod params;
/// Public re-exports for easy access.
pub use activations::{ActivationFn, ActivationType};
pub use conv1d::A2Conv1d;
pub use film::{FiLMConfig, FiLMLayer};
pub use gating::GatingMode;
pub use head::{A2HeadConv, a2_head_block_scalar_ref, a2_head_single_frame_scalar_ref};
pub use layer::{A2Layer, a2_layer_single_frame_scalar_ref};
pub use model::WaveNetA2;
pub use params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS,
    A2_VALID_CHANNELS, HeadParams, LayerArrayParamsA2, LayerParamsA2,
};

use crate::models::NamModel;
use crate::models::sealed;

impl<const CH: usize> sealed::Sealed for model::WaveNetA2<CH> {}

impl<const CH: usize> NamModel for model::WaveNetA2<CH> {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }

    fn prewarm_samples(&self) -> usize {
        self.receptive_field_size
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) {
        self.set_max_buffer_size(max_buf);
    }

    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) {
        self.reset(_sample_rate, max_buffer_size);
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
