// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Benchmarks of the CabSim convolution engine (`ConvEngine`).
//!
//! Covers uniform-partitioned fast convolution for short, medium, and long
//! impulse responses at two block sizes (64 and 256 samples), plus the
//! offline construction/allocation cost of the engine.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench cabsim_bench
//! ```

use criterion::{criterion_group, criterion_main};

fn synth_ir(len: usize, freq: f32, decay: f32) -> Vec<f32> {
    let sample_rate = 48000u32;
    (0..len)
        .map(|n| {
            let t = n as f32 / sample_rate as f32;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}

fn bench_cabsim_process_block(
    c: &mut criterion::Criterion,
    ir_len: usize,
    partition_size: usize,
    label: &str,
) {
    use nam_rs::dsp::cabsim::conv::ConvEngine;

    let ir = synth_ir(ir_len, 440.0, 10.0);
    let mut engine = ConvEngine::new(&ir, partition_size).expect("bench allocation failed");

    let mut input = vec![0.0f32; partition_size];
    let mut output = vec![0.0f32; partition_size];

    for (j, v) in input.iter_mut().enumerate() {
        *v = (j as f32 * 0.01).sin();
    }

    for _ in 0..engine.num_partitions().max(1) {
        engine.process(&input, &mut output);
    }

    c.bench_function(label, |b| {
        b.iter(|| {
            for (j, v) in input.iter_mut().enumerate() {
                *v = (j as f32 * 0.01).sin();
            }
            engine.process(
                std::hint::black_box(&input),
                std::hint::black_box(&mut output),
            );
        });
    });
}

fn bench_cabsim_short_ir_64samp(c: &mut criterion::Criterion) {
    bench_cabsim_process_block(c, 64, 64, "Cabsim_ShortIR_64samp");
}

fn bench_cabsim_medium_ir_64samp(c: &mut criterion::Criterion) {
    bench_cabsim_process_block(c, 2048, 64, "Cabsim_MediumIR_2048_64samp");
}

fn bench_cabsim_long_ir_64samp(c: &mut criterion::Criterion) {
    bench_cabsim_process_block(c, 16384, 64, "Cabsim_LongIR_16384_64samp");
}

fn bench_cabsim_256samp_block(c: &mut criterion::Criterion) {
    bench_cabsim_process_block(c, 2048, 256, "Cabsim_MediumIR_2048_256samp");
}

fn bench_cabsim_engine_construction(c: &mut criterion::Criterion) {
    use nam_rs::dsp::cabsim::conv::ConvEngine;

    let ir = synth_ir(2048, 440.0, 10.0);

    c.bench_function("Cabsim_Engine_Construction_2048_64", |b| {
        b.iter(|| {
            let engine = ConvEngine::new(&ir, 64).expect("bench allocation failed");
            std::hint::black_box(engine);
        });
    });
}

fn bench_cabsim_engine_construction_long(c: &mut criterion::Criterion) {
    use nam_rs::dsp::cabsim::conv::ConvEngine;

    let ir = synth_ir(16384, 440.0, 10.0);

    c.bench_function("Cabsim_Engine_Construction_16384_64", |b| {
        b.iter(|| {
            let engine = ConvEngine::new(&ir, 64).expect("bench allocation failed");
            std::hint::black_box(engine);
        });
    });
}

criterion_group! {
    name = cabsim_benches;
    config = criterion::Criterion::default().sample_size(50);
    targets = bench_cabsim_short_ir_64samp,
    bench_cabsim_medium_ir_64samp,
    bench_cabsim_long_ir_64samp,
    bench_cabsim_256samp_block,
    bench_cabsim_engine_construction,
    bench_cabsim_engine_construction_long
}

criterion_main!(cabsim_benches);
