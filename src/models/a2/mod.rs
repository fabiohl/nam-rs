// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 Architecture.
//!
//! Este módulo isola componentes da arquitetura A2 (v0.6+), incluindo
//! ativações, FiLM, gating, parâmetros e o placeholder atual.
//!
//! ## Status
//!
//! O scaffolding existente inclui a definição completa de parâmetros e
//! constantes arquiteturais (espelhando `a2_fast.h`). O fast-path
//! A2-Full/Lite (`Épico 1`) utilizará um subconjunto destes structs;
//! os campos de FiLM, gating (`Gated`/`Blended`), `head1x1`, `bottleneck ≠ channels`
//! e ativações heterogêneas estão reservados para o motor A2 geral (futuro).
//!
//! `WavenetA2Placeholder` será aposentado ao final do Épico 1 (T1.9).

pub mod activations;
pub mod conv1d;
pub mod conv1d_fallback;
pub mod film;
pub mod gating;
pub mod head;
pub mod params;
pub mod placeholder;

/// Public re-exports for easy access.
pub use activations::{ActivationFn, ActivationType};
pub use conv1d::A2Conv1d;
pub use film::{FiLMConfig, FiLMLayer};
pub use gating::GatingMode;
pub use head::{A2HeadConv, a2_head_block_scalar_ref, a2_head_single_frame_scalar_ref};
pub use params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS,
    A2_VALID_CHANNELS, HeadParams, LayerArrayParamsA2, LayerParamsA2,
};
pub use placeholder::WavenetA2Placeholder;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
