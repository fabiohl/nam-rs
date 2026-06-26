// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-benchmarks for GEMV kernels — Épico G Sprint 6, Tarefa 1.
//!
//! Isolates measurement of `fused_add_gemv_avx2` (generic) vs fully-unrolled
//! specialized prototypes for the dimensions listed in the sprint plan:
//! 1×4, 4×4, 4×6, 8×4, 8×6, 8×8 (Out × In).
//!
//! All kernels operate on f16c-quantized weights and f32 inputs/outputs.
//! The fused variant (`fused_add_gemv`) is used because it is the dominant
//! hot-path call in WaveNet and LSTM inference.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench gemv_bench
//! ```

use core::arch::x86_64::*;
use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::math::common::half::f16_bits_to_f32_f16c;
use nam_rs::math::common::scalar_ref::fused_add_gemv_fallback;
use nam_rs::math::gemm::gemv::fused_add_gemv_avx2;

// ── Synthetic test data ────────────────────────────────────────────────────────

struct GemvTestData {
    in_frame: Vec<f32>,
    weights: Vec<u16>,
    bias: Vec<f32>,
    out_frame: Vec<f32>,
}

/// Creates deterministic test data for a given (in_len, out_len) pair.
/// Weights are derived from a sinusoidal pattern to avoid degenerate values.
fn make_test_data(in_len: usize, out_len: usize) -> GemvTestData {
    let in_frame: Vec<f32> = (0..in_len).map(|i| (i as f32 * 0.17).sin()).collect();
    let bias: Vec<f32> = (0..out_len)
        .map(|i| (i as f32 * 0.31).cos() * 0.1)
        .collect();
    let weights: Vec<u16> = (0..in_len * out_len)
        .map(|i| {
            let v = (i as f32 * 0.13).sin() * 0.5;
            // Software f32→f16 conversion (no F16C hardware requirement for test setup).
            let u = v.to_bits();
            let sign = (u >> 16) & 0x8000;
            let exp = (u >> 23) & 0xFF;
            let frac = (u & 0x7F_FFFF) >> 13;
            if exp < 112 {
                0 // underflow to zero
            } else if exp > 142 {
                (sign | 0x7BFF) as u16 // saturate to max
            } else {
                (sign | ((exp - 112) << 10) | (frac & 0x3FF)) as u16
            }
        })
        .collect();
    let out_frame = vec![0.0; out_len];
    GemvTestData {
        in_frame,
        weights,
        bias,
        out_frame,
    }
}

// ── Specialized kernels (AVX2, fully-unrolled by dimension) ────────────────────

/// Specialized fused GEMV for: 1 input × 4 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv_specialized_1x4(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let v_in = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w_ptr = weights.as_ptr();
        let vw = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
        let mut acc = _mm256_mul_ps(v_in, vw);
        if do_bias {
            let vb = _mm256_loadu_ps(bias.as_ptr());
            acc = _mm256_add_ps(acc, vb);
        }
        let out_ptr = out_frame.as_ptr();
        let out_val = _mm256_loadu_ps(out_ptr);
        acc = _mm256_add_ps(acc, out_val);
        _mm256_storeu_ps(out_frame.as_mut_ptr(), acc);
    }
}

/// Specialized fused GEMV for: 4 inputs × 4 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv_specialized_4x4(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let w_ptr = weights.as_ptr();
        let mut acc = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
        acc = _mm256_fmadd_ps(v0, w0, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(4) as *const __m128i));
        acc = _mm256_fmadd_ps(v1, w1, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(8) as *const __m128i));
        acc = _mm256_fmadd_ps(v2, w2, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(12) as *const __m128i));
        acc = _mm256_fmadd_ps(v3, w3, acc);

        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc = _mm256_add_ps(acc, out_val);
        _mm256_storeu_ps(out_frame.as_mut_ptr(), acc);
    }
}

/// Specialized fused GEMV for: 4 inputs × 6 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv_specialized_4x6(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let w_ptr = weights.as_ptr();
        // Process outputs 0..3 in one YMM, 4..5 via scalar (or load YMM anyway and scalar tail).
        let mut acc = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = _mm_loadu_si128(w_ptr as *const __m128i);
        // Pad the 6 f16 entries to 8 for cvtph_ps (safe: we own 6*4=24 bytes, load 16 bytes for lanes 0-3).
        // Lanes 4-5 are fallback; we use scalar for those.
        let w0_ps = _mm256_cvtph_ps(w0);
        acc = _mm256_fmadd_ps(v0, w0_ps, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = _mm_loadu_si128(w_ptr.add(6) as *const __m128i);
        let w1_ps = _mm256_cvtph_ps(w1);
        acc = _mm256_fmadd_ps(v1, w1_ps, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = _mm_loadu_si128(w_ptr.add(12) as *const __m128i);
        let w2_ps = _mm256_cvtph_ps(w2);
        acc = _mm256_fmadd_ps(v2, w2_ps, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = _mm_loadu_si128(w_ptr.add(18) as *const __m128i);
        let w3_ps = _mm256_cvtph_ps(w3);
        acc = _mm256_fmadd_ps(v3, w3_ps, acc);

        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc = _mm256_add_ps(acc, out_val);

        // Store to temporary and handle lanes 4-5 via scalar.
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);

        // Lanes 4-5: compute scalars for the remaining 2 outputs.
        for oc in 4..6 {
            let w = f16_bits_to_f32_f16c(*weights.get_unchecked(oc));
            let mut sum = *in_frame.get_unchecked(0) * w;
            let w = f16_bits_to_f32_f16c(*weights.get_unchecked(6 + oc));
            sum += *in_frame.get_unchecked(1) * w;
            let w = f16_bits_to_f32_f16c(*weights.get_unchecked(12 + oc));
            sum += *in_frame.get_unchecked(2) * w;
            let w = f16_bits_to_f32_f16c(*weights.get_unchecked(18 + oc));
            sum += *in_frame.get_unchecked(3) * w;
            if do_bias {
                sum += bias[oc];
            }
            tmp[oc] = *out_frame.get_unchecked(oc) + sum;
        }

        // Copy back lanes 0..5.
        for (i, &val) in tmp.iter().enumerate().take(6) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized fused GEMV for: 8 inputs × 4 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv_specialized_8x4(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let w_ptr = weights.as_ptr();
        let mut acc = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
        acc = _mm256_fmadd_ps(v0, w0, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(4) as *const __m128i));
        acc = _mm256_fmadd_ps(v1, w1, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(8) as *const __m128i));
        acc = _mm256_fmadd_ps(v2, w2, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(12) as *const __m128i));
        acc = _mm256_fmadd_ps(v3, w3, acc);

        let v4 = _mm256_set1_ps(*in_frame.get_unchecked(4));
        let w4 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(16) as *const __m128i));
        acc = _mm256_fmadd_ps(v4, w4, acc);

        let v5 = _mm256_set1_ps(*in_frame.get_unchecked(5));
        let w5 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(20) as *const __m128i));
        acc = _mm256_fmadd_ps(v5, w5, acc);

        let v6 = _mm256_set1_ps(*in_frame.get_unchecked(6));
        let w6 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(24) as *const __m128i));
        acc = _mm256_fmadd_ps(v6, w6, acc);

        let v7 = _mm256_set1_ps(*in_frame.get_unchecked(7));
        let w7 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(28) as *const __m128i));
        acc = _mm256_fmadd_ps(v7, w7, acc);

        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc = _mm256_add_ps(acc, out_val);
        _mm256_storeu_ps(out_frame.as_mut_ptr(), acc);
    }
}

/// Specialized fused GEMV for: 8 inputs × 6 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv_specialized_8x6(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let out_len = 6usize;
        let w_ptr = weights.as_ptr();

        // Handle lanes 0..3 SIMD, lanes 4..5 scalar, for each of 8 inputs.
        let mut tmp = [0.0f32; 8];

        // Load bias / zero for lanes 0..3.
        let mut acc_simd = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };
        // Load initial out values.
        let out_simd = _mm256_loadu_ps(out_frame.as_ptr());
        acc_simd = _mm256_add_ps(acc_simd, out_simd);

        // Lanes 4..5: scalar accumulators.
        let mut sum4 = if do_bias { bias[4] } else { 0.0 };
        let mut sum5 = if do_bias { bias[5] } else { 0.0 };

        for in_c in 0..8 {
            let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
            let w_ptr_row = w_ptr.add(in_c * out_len);
            let w_simd = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr_row as *const __m128i));
            acc_simd = _mm256_fmadd_ps(vs, w_simd, acc_simd);
            sum4 += *in_frame.get_unchecked(in_c)
                * f16_bits_to_f32_f16c(*weights.get_unchecked(in_c * out_len + 4));
            sum5 += *in_frame.get_unchecked(in_c)
                * f16_bits_to_f32_f16c(*weights.get_unchecked(in_c * out_len + 5));
        }

        _mm256_storeu_ps(tmp.as_mut_ptr(), acc_simd);
        tmp[4] = *out_frame.get_unchecked(4) + sum4;
        tmp[5] = *out_frame.get_unchecked(5) + sum5;

        for (i, &val) in tmp.iter().enumerate().take(6) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized fused GEMV for: 8 inputs × 8 outputs.
///
/// This matches the inner block size of the generic kernel (8 SIMD accumulators × step 8),
/// but without the outer loop overhead.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv_specialized_8x8(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let w_ptr = weights.as_ptr();

        let mut acc0 = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut acc4 = _mm256_setzero_ps();
        let mut acc5 = _mm256_setzero_ps();
        let mut acc6 = _mm256_setzero_ps();
        let mut acc7 = _mm256_setzero_ps();

        // Fully unrolled 8-input loop.
        let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let vs2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let vs3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let vs4 = _mm256_set1_ps(*in_frame.get_unchecked(4));
        let vs5 = _mm256_set1_ps(*in_frame.get_unchecked(5));
        let vs6 = _mm256_set1_ps(*in_frame.get_unchecked(6));
        let vs7 = _mm256_set1_ps(*in_frame.get_unchecked(7));

        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
        acc0 = _mm256_fmadd_ps(vs0, w0, acc0);
        let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(8) as *const __m128i));
        acc1 = _mm256_fmadd_ps(vs1, w1, acc1);
        let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(16) as *const __m128i));
        acc2 = _mm256_fmadd_ps(vs2, w2, acc2);
        let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(24) as *const __m128i));
        acc3 = _mm256_fmadd_ps(vs3, w3, acc3);
        let w4 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(32) as *const __m128i));
        acc4 = _mm256_fmadd_ps(vs4, w4, acc4);
        let w5 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(40) as *const __m128i));
        acc5 = _mm256_fmadd_ps(vs5, w5, acc5);
        let w6 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(48) as *const __m128i));
        acc6 = _mm256_fmadd_ps(vs6, w6, acc6);
        let w7 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(56) as *const __m128i));
        acc7 = _mm256_fmadd_ps(vs7, w7, acc7);

        // Reduction tree.
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc4 = _mm256_add_ps(acc4, acc5);
        acc6 = _mm256_add_ps(acc6, acc7);
        acc0 = _mm256_add_ps(acc0, acc2);
        acc4 = _mm256_add_ps(acc4, acc6);
        acc0 = _mm256_add_ps(acc0, acc4);

        // Fused add: accumulate with existing out_frame.
        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc0 = _mm256_add_ps(acc0, out_val);
        _mm256_storeu_ps(out_frame.as_mut_ptr(), acc0);
    }
}

// ── Benchmarks ─────────────────────────────────────────────────────────────────

macro_rules! bench_dim {
    ($c:expr, $name:literal, $in_len:expr, $out_len:expr, $specialized:path) => {{
        let mut group = $c.benchmark_group($name);
        let data = make_test_data($in_len, $out_len);

        group.bench_function("generic_avx2", |b| {
            b.iter(|| {
                let mut out = data.out_frame.clone();
                unsafe {
                    fused_add_gemv_avx2(&data.in_frame, &data.weights, &data.bias, &mut out, true);
                }
                out
            })
        });

        group.bench_function("specialized_avx2", |b| {
            b.iter(|| {
                let mut out = data.out_frame.clone();
                unsafe {
                    $specialized(&data.in_frame, &data.weights, &data.bias, &mut out, true);
                }
                out
            })
        });

        group.bench_function("scalar_fallback", |b| {
            b.iter(|| {
                let mut out = data.out_frame.clone();
                unsafe {
                    fused_add_gemv_fallback(
                        &data.in_frame,
                        &data.weights,
                        &data.bias,
                        &mut out,
                        true,
                    );
                }
                out
            })
        });

        group.finish();
    }};
}

fn bench_gemv_1x4(c: &mut Criterion) {
    bench_dim!(c, "gemv_1x4", 1, 4, gemv_specialized_1x4);
}

fn bench_gemv_4x4(c: &mut Criterion) {
    bench_dim!(c, "gemv_4x4", 4, 4, gemv_specialized_4x4);
}

fn bench_gemv_4x6(c: &mut Criterion) {
    bench_dim!(c, "gemv_4x6", 4, 6, gemv_specialized_4x6);
}

fn bench_gemv_8x4(c: &mut Criterion) {
    bench_dim!(c, "gemv_8x4", 8, 4, gemv_specialized_8x4);
}

fn bench_gemv_8x6(c: &mut Criterion) {
    bench_dim!(c, "gemv_8x6", 8, 6, gemv_specialized_8x6);
}

fn bench_gemv_8x8(c: &mut Criterion) {
    bench_dim!(c, "gemv_8x8", 8, 8, gemv_specialized_8x8);
}

criterion_group!(
    gemv_benches,
    bench_gemv_1x4,
    bench_gemv_4x4,
    bench_gemv_4x6,
    bench_gemv_8x4,
    bench_gemv_8x6,
    bench_gemv_8x8,
);
criterion_main!(gemv_benches);
