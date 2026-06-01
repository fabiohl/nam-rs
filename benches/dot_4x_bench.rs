// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Benchmarks para kernels dot_product_4x (AVX2 e AVX-512).
//!
//! Mede throughput de dot product interleaved 4x para tamanhos de estado
//! representativos de topologias LSTM e WaveNet.

use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::math::common::scalar_ref;
use nam_rs::math::gemm::dot_4x;

fn generate_test_data(len: usize) -> (Vec<[u16; 4]>, Vec<f32>, Vec<f32>) {
    let weights: Vec<[u16; 4]> = (0..len)
        .map(|i| {
            let v = (i as f32 * 0.1).sin() * 0.5 + 0.5;
            let bits = half::f16::from_f32(v).to_bits();
            [bits; 4]
        })
        .collect();
    let state_f0: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
    let state_f1: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07 + 1.0).sin()).collect();
    (weights, state_f0, state_f1)
}

fn bench_dot_4x_interleaved_avx512(c: &mut Criterion) {
    let sizes = [16, 64, 256, 1024, 4096];
    let mut group = c.benchmark_group("dot_4x_interleaved_avx512");

    for &size in &sizes {
        let (weights, state, _) = generate_test_data(size);

        group.bench_function(format!("fallback_{}", size), |b| {
            b.iter(|| unsafe {
                scalar_ref::dot_product_4x_interleaved_fallback(&weights, &state)
            })
        });

        group.bench_function(format!("avx2_{}", size), |b| {
            b.iter(|| unsafe { dot_4x::dot_product_4x_interleaved_avx2(&weights, &state) })
        });

        if std::is_x86_feature_detected!("avx512f") {
            group.bench_function(format!("avx512_{}", size), |b| {
                b.iter(|| unsafe {
                    dot_4x::dot_product_4x_interleaved_avx512(&weights, &state)
                })
            });
        }
    }
    group.finish();
}

fn bench_dot_4x_interleaved_dual_frame_avx512(c: &mut Criterion) {
    let sizes = [16, 64, 256, 1024, 4096];
    let mut group = c.benchmark_group("dot_4x_dual_frame_avx512");

    for &size in &sizes {
        let (weights, state_f0, state_f1) = generate_test_data(size);

        group.bench_function(format!("fallback_{}", size), |b| {
            b.iter(|| unsafe {
                scalar_ref::dot_product_4x_interleaved_dual_frame_fallback(
                    &weights, &state_f0, &state_f1,
                )
            })
        });

        group.bench_function(format!("avx2_{}", size), |b| {
            b.iter(|| unsafe {
                dot_4x::dot_product_4x_interleaved_dual_frame_avx2(&weights, &state_f0, &state_f1)
            })
        });

        if std::is_x86_feature_detected!("avx512f") {
            group.bench_function(format!("avx512_{}", size), |b| {
                b.iter(|| unsafe {
                    dot_4x::dot_product_4x_interleaved_dual_frame_avx512(
                        &weights, &state_f0, &state_f1,
                    )
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_4x_interleaved_avx512,
    bench_dot_4x_interleaved_dual_frame_avx512,
);
criterion_main!(benches);
