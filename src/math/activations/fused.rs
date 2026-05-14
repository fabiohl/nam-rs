// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de ativação fundidos para performance (ex: Sigmoid + ReLU, LSTM Gates).

use super::relu::{simd_relu_avx2, simd_relu_avx512, simd_relu_dual_avx2};
use super::sigmoid::{simd_sigmoid_avx2, simd_sigmoid_avx512, simd_sigmoid_dual_avx2};
use super::tanh::{simd_tanh_avx2, simd_tanh_avx512};
use core::arch::x86_64::*;

/// Aplica Sigmoid seguido de ReLU (fundido).
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_fused_sigmoid_relu_avx2(x: __m256) -> __m256 {
    let s = unsafe { simd_sigmoid_avx2(x) };
    unsafe { simd_relu_avx2(s) }
}

/// Aplica Sigmoid seguido de ReLU (Dual, 16 floats).
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_fused_sigmoid_relu_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let (s1, s2) = unsafe { simd_sigmoid_dual_avx2(x1, x2) };
    unsafe { simd_relu_dual_avx2(s1, s2) }
}

/// Aplica Tanh em x1 e Sigmoid em x2 (Dual).
/// Utilizado em blocos Gated Activation (ex: Wavenet).
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_sigmoid_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let t1 = unsafe { simd_tanh_avx2(x1) };
    let s2 = unsafe { simd_sigmoid_avx2(x2) };
    (t1, s2)
}

/// Aplica Tanh em x1 e Sigmoid em x2 (Dual, AVX-512).
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_sigmoid_dual_avx512(x1: __m512, x2: __m512) -> (__m512, __m512) {
    let t1 = unsafe { simd_tanh_avx512(x1) };
    let s2 = unsafe { simd_sigmoid_avx512(x2) };
    (t1, s2)
}

/// Aplica Sigmoid seguido de ReLU (AVX-512).
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_fused_sigmoid_relu_avx512(x: __m512) -> __m512 {
    let s = unsafe { simd_sigmoid_avx512(x) };
    unsafe { simd_relu_avx512(s) }
}

/// Aplica a ativação fundida Sigmoid+ReLU a um slice usando AVX2.
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_sigmoid_relu_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let (y1, y2) = simd_fused_sigmoid_relu_dual_avx2(x1, x2);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
        }
        i += 16;
    }

    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            let y = simd_fused_sigmoid_relu_avx2(x);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 8;
    }

    for item in slice.iter_mut().skip(i) {
        let s = 1.0 / (1.0 + (-*item).exp());
        *item = if s < 0.0 { 0.0 } else { s };
    }
}

/// Aplica a ativação fundida Sigmoid+ReLU a um slice usando AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_sigmoid_relu_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_fused_sigmoid_relu_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 16;
    }

    for item in slice.iter_mut().skip(i) {
        let s = 1.0 / (1.0 + (-*item).exp());
        *item = if s < 0.0 { 0.0 } else { s };
    }
}
