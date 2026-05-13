// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Operações DSP de ganho, detecção de clipping e rampa estéreo.
//!
//! Despacha dinamicamente para o backend SIMD configurado.

/// Aplica ganho constante em um buffer mono via despacho SIMD.
///
/// # Safety
/// O buffer deve ser válido.
pub unsafe fn apply_gain(data: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain(data, gain))
}

/// Aplica ganho constante em estéreo via despacho SIMD.
///
/// # Safety
/// Os buffers devem ser válidos e ter o mesmo tamanho.
pub unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain_stereo(left, right, gain))
}

/// Aplica ganho e detecta clipping em estéreo em uma única passagem.
/// Retorna `true` se qualquer amostra resultante possuir `|x| > 1.0`.
///
/// # Safety
/// Os buffers devem ser válidos e ter o mesmo tamanho.
pub unsafe fn apply_gain_and_detect_clipping_stereo(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_stereo(left, right, gain))
}

/// Aplica rampa linear de ganho em estéreo via despacho SIMD.
///
/// # Safety
/// Os buffers devem ser válidos e ter o mesmo tamanho.
pub unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    crate::math::common::dispatch_simd!(apply_ramp_stereo(left, right, start, step))
}
