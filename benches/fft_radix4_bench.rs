// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Benchmarks: Radix-2 DIT vs Radix-4 DIT FFT — Research Artifact (Task A9, Spr.4).
//!
//! **Status: CONCLUÍDO. Preservado para reprodutibilidade da decisão de engenharia.**
//!
//! Mede o throughput do FFT forward/inverse de `FftPlanner` (Radix-2) vs
//! `FftPlannerRadix4` (Radix-4) nos tamanhos relevantes ao CabSim
//! (N=256, N=1024, f32 escalar). Resultados demonstram que Radix-4 é
//! 7–19% mais lento, levando à decisão de manter Radix-2 DIT com SIMD
//! como algoritmo canônico do projeto.
//!
//! Resultados e análise completa: [`TODO-sprints.md`](file:///home/fabio/nam-rs/TODO-sprints.md), Tarefa A9.
//!
//! # Running
//! ```sh
//! cargo bench --bench fft_radix4_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use nam_rs::math::dsp::fft::FftPlanner;
use nam_rs::math::dsp::fft_radix4::FftPlannerRadix4;
use std::hint::black_box;

fn make_complex_input_impl(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    use std::num::Wrapping;
    let mut s = Wrapping(seed);
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for _ in 0..n {
        s = s * Wrapping(6364136223846793005u64) + Wrapping(1442695040888963407u64);
        let v1 = ((s.0 & 0xFFFFFF) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0;
        s = s * Wrapping(6364136223846793005u64) + Wrapping(1442695040888963407u64);
        let v2 = ((s.0 & 0xFFFFFF) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0;
        re.push(v1);
        im.push(v2);
    }
    (re, im)
}

fn bench_fft_radix2_f32(c: &mut Criterion, n: usize) {
    let planner = FftPlanner::<f32>::new(n);
    let (re_orig, im_orig) = make_complex_input_impl(n, 42);
    let mut re = re_orig.clone();
    let mut im = im_orig.clone();
    c.bench_function(&format!("Radix2 FFT forward N={n} (f32)"), |b| {
        b.iter(|| {
            re.copy_from_slice(&re_orig);
            im.copy_from_slice(&im_orig);
            planner.process(black_box(&mut re), black_box(&mut im));
        })
    });
}

fn bench_fft_radix4_f32(c: &mut Criterion, n: usize) {
    let planner = FftPlannerRadix4::<f32>::new(n);
    let (re_orig, im_orig) = make_complex_input_impl(n, 42);
    let mut re = re_orig.clone();
    let mut im = im_orig.clone();
    c.bench_function(&format!("Radix4 FFT forward N={n} (f32)"), |b| {
        b.iter(|| {
            re.copy_from_slice(&re_orig);
            im.copy_from_slice(&im_orig);
            planner.process(black_box(&mut re), black_box(&mut im));
        })
    });
}

fn bench_fft_radix4_inverse_f32(c: &mut Criterion, n: usize) {
    let planner = FftPlannerRadix4::<f32>::new(n);
    let (mut re, mut im) = make_complex_input_impl(n, 42);
    planner.process(&mut re, &mut im);
    let fft_re = re.clone();
    let fft_im = im.clone();
    let mut re_copy = fft_re.clone();
    let mut im_copy = fft_im.clone();
    c.bench_function(&format!("Radix4 FFT inverse N={n} (f32)"), |b| {
        b.iter(|| {
            re_copy.copy_from_slice(&fft_re);
            im_copy.copy_from_slice(&fft_im);
            planner.process_inverse(black_box(&mut re_copy), black_box(&mut im_copy));
        })
    });
}

fn bench_fft_radix2_inverse_f32(c: &mut Criterion, n: usize) {
    let planner = FftPlanner::<f32>::new(n);
    let (mut re, mut im) = make_complex_input_impl(n, 42);
    planner.process(&mut re, &mut im);
    let fft_re = re.clone();
    let fft_im = im.clone();
    let mut re_copy = fft_re.clone();
    let mut im_copy = fft_im.clone();
    c.bench_function(&format!("Radix2 FFT inverse N={n} (f32)"), |b| {
        b.iter(|| {
            re_copy.copy_from_slice(&fft_re);
            im_copy.copy_from_slice(&fft_im);
            planner.process_inverse(black_box(&mut re_copy), black_box(&mut im_copy));
        })
    });
}

fn bench_fft_radix2v4(c: &mut Criterion) {
    for n in [256, 1024] {
        bench_fft_radix2_f32(c, n);
        bench_fft_radix4_f32(c, n);
        bench_fft_radix2_inverse_f32(c, n);
        bench_fft_radix4_inverse_f32(c, n);
    }
}

criterion_group!(benches, bench_fft_radix2v4);
criterion_main!(benches);
