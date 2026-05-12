// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo central de ativações com despacho (dispatch) automático de SIMD.

pub mod fused;
pub mod prelu;
pub mod relu;
pub mod sigmoid;
pub mod silu;
pub mod softsign;
pub mod tanh;

#[cfg(test)]
mod tests;

pub use fused::*;
pub use prelu::*;
pub use relu::*;
pub use sigmoid::*;
pub use silu::*;
pub use softsign::*;
pub use tanh::*;

use crate::math::simd::{InstructionSet, SIMD_MATH};

/// Aplica a ativação Tanh a um slice de f32 com despacho automático para a melhor implementação SIMD.
#[inline(always)]
pub fn tanh_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { tanh::tanh_slice_avx512(data) };
        }
        _ => {
            unsafe { tanh::tanh_slice_avx2(data) };
        }
    }
}

/// Aplica a ativação Sigmoid a um slice de f32 com despacho automático.
#[inline(always)]
pub fn sigmoid_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { sigmoid::sigmoid_slice_avx512(data) };
        }
        _ => {
            unsafe { sigmoid::sigmoid_slice_avx2(data) };
        }
    }
}

/// Aplica a ativação ReLU a um slice de f32 com despacho automático.
#[inline(always)]
pub fn relu_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { relu::relu_slice_avx512(data) };
        }
        _ => {
            unsafe { relu::relu_slice_avx2(data) };
        }
    }
}

/// Aplica a ativação PReLU a um slice de f32 com despacho automático.
#[inline(always)]
pub fn prelu_slice(data: &mut [f32], slopes: &[f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { prelu::prelu_slice_avx512(data, slopes) };
        }
        _ => {
            unsafe { prelu::prelu_slice_avx2(data, slopes) };
        }
    }
}

/// Aplica a ativação Softsign a um slice de f32 com despacho automático.
#[inline(always)]
pub fn softsign_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { softsign::softsign_slice_avx512(data) };
        }
        _ => {
            unsafe { softsign::softsign_slice_avx2(data) };
        }
    }
}

/// Aplica a ativação SiLU a um slice de f32 com despacho automático.
#[inline(always)]
pub fn silu_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx512 | InstructionSet::Avx512Vnni | InstructionSet::Avx512VnniBf16 => {
            unsafe { silu::silu_slice_avx512(data) };
        }
        _ => {
            unsafe { silu::silu_slice_avx2(data) };
        }
    }
}
