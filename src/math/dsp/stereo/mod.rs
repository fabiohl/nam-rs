// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Operações DSP para processamento estéreo e medição de sinal.

mod convolution_avx2;
mod convolution_avx512;
mod energy;
mod max_diff;

pub use convolution_avx2::{convolve_mono_avx2, convolve_stereo_avx2, convolve_stereo_dual_avx2};
pub use convolution_avx512::{convolve_mono_avx512, convolve_stereo_avx512, convolve_stereo_dual_avx512};
pub use energy::{
    compute_energy_avx2,
    compute_energy_avx512,
    compute_energy_stereo_avx2,
    compute_energy_stereo_avx512,
};
pub use max_diff::{compute_max_diff_avx2, compute_max_diff_avx512};

/// Calcula o máximo da energia entre dois canais de áudio via despacho SIMD.
///
/// # Safety
/// Utiliza despacho dinâmico via v-table global.
pub unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_energy_stereo(l, r))
}

/// Calcula a diferença absoluta máxima entre dois blocos via despacho SIMD.
///
/// # Safety
/// Utiliza despacho dinâmico via v-table global.
pub unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_max_diff(a, b))
}

/// Convolução estéreo (usada no resampler) via despacho SIMD.
/// Realiza o produto escalar entre um banco de coeficientes e dois buffers de entrada (L/R).
///
/// # Safety
/// `coeffs`, `input_l` e `input_r` devem ser ponteiros válidos para pelo menos `taps` elementos.
/// `coeffs` deve estar alinhado conforme o registrador SIMD.
pub unsafe fn convolve_stereo(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    crate::math::common::dispatch_simd!(convolve_stereo(coeffs, input_l, input_r, taps))
}

/// Convolução mono (usada no resampler) via despacho SIMD.
/// Realiza o produto escalar entre um banco de coeficientes e um buffer de entrada.
///
/// # Safety
/// `coeffs` e `input` devem ser ponteiros válidos para pelo menos `taps` elementos.
/// `coeffs` deve estar alinhado conforme o registrador SIMD.
pub unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    crate::math::common::dispatch_simd!(convolve_mono(coeffs, input, taps))
}
