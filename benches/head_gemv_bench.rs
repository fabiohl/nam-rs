// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-benchmark for the Lite CH12 head GEMV masked path (in=12, out=6).
//!
//! Sprint S5, T2.S5.1 — Isolates `gemv_no_bias_f32_avx2(12, 6, 1)` to measure
//! the cost of the masked tail path (`out_len=6 < 8`) vs full 8-wide path.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench head_gemv_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::math::gemm::gemv::gemv_no_bias_f32_avx2;

fn make_test_data(in_len: usize, out_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_frames: Vec<f32> = (0..in_len).map(|i| (i as f32 * 0.17).sin()).collect();
    let weights: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.13).sin() * 0.5)
        .collect();
    let out_frames = vec![0.0f32; out_len];
    (in_frames, weights, out_frames)
}

fn bench_head_gemv_12x6(c: &mut Criterion) {
    let (in_frames, weights, mut out_frames) = make_test_data(12, 6);
    c.bench_function("head_gemv_12x6_masked", |b| {
        b.iter(|| unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_frames, 1);
        });
    });
}

fn bench_head_gemv_12x8(c: &mut Criterion) {
    let (in_frames, weights, mut out_frames) = make_test_data(12, 8);
    c.bench_function("head_gemv_12x8_clean", |b| {
        b.iter(|| unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_frames, 1);
        });
    });
}

criterion_group!(
    name = head_gemv;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1))
        .noise_threshold(0.02);
    targets = bench_head_gemv_12x6, bench_head_gemv_12x8
);

criterion_main!(head_gemv);
