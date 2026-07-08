// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  RT jitter & xrun stress test — measures tail latency under concurrent CPU load.
//
//  Runs the DSP processing loop while background stress threads consume CPU,
//  simulating a loaded RT system. Reports p50/p99/p99.9/exact_max latency
//  and counts deadline violations.
//
//  ## Running
//
//  ```sh
//  cargo test --release --test rt_jitter -- --ignored --nocapture
//  ```
//
//  ## Stress methodology
//
//  Spawns N background threads that spin on a `num_cpus::get()` loop doing
//  floating-point work (no I/O, no allocations), creating CPU contention.
//  The DSP thread runs concurrently and records latency via `LatencyHistogram`.
//
//  Marked `#[ignore]` — runs during `tests-long.sh` Phase 6.

use super::common;
use common::*;

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use nam_rs::dsp::telemetry::LatencyHistogram;
use nam_rs::loader::dispatcher::build_model;
use nam_rs::loader::nam_json::parse_nam_json;
use nam_rs::models::NamModel;

const RT_DEADLINE_US: u64 = 1330;
const BLOCK_SIZE: usize = 64;
const MEASURE_BLOCKS: usize = 2048;

/// Background stress worker — burns CPU with floating-point arithmetic.
fn stress_worker(running: Arc<AtomicBool>) {
    let mut x: f64 = 0.0;
    while running.load(Ordering::Relaxed) {
        // Approximate sin/cos via Taylor to keep FPU busy without libm calls
        for _ in 0..256 {
            x = (x + 0.123456789).fract();
            let x2 = x * x;
            let sin_approx = x * (1.0 - x2 * (1.0 / 6.0 - x2 / 120.0));
            let cos_approx = 1.0 - x2 * (0.5 - x2 * (1.0 / 24.0 - x2 / 720.0));
            // Prevent compiler from optimizing away the work
            std::hint::black_box(sin_approx + cos_approx);
        }
    }
}

fn load_and_prewarm(filename: &str) -> Option<nam_rs::models::StaticModel> {
    let path = model_path(filename);
    if !path.exists() {
        eprintln!("SKIP: {} not found.", filename);
        return None;
    }
    let json_data = fs::read_to_string(&path).ok()?;
    let model_data = parse_nam_json(&json_data).ok()?;
    let mut model = build_model(&model_data).ok()?;
    model.prewarm(2048);
    Some(*model)
}

fn run_jitter_test(label: &str, model: &mut nam_rs::models::StaticModel, stress_threads: usize) {
    if stress_threads == 0 {
        println!(
            "[{label}] No stress — baseline measurement, \
             {MEASURE_BLOCKS} blocks"
        );
    } else {
        println!(
            "[{label}] Stress: {stress_threads} background CPU-burn threads, \
             {MEASURE_BLOCKS} blocks"
        );
    }

    let running = Arc::new(AtomicBool::new(true));
    let mut workers = Vec::with_capacity(stress_threads);

    for _ in 0..stress_threads {
        let r = running.clone();
        workers.push(thread::spawn(move || stress_worker(r)));
    }

    // Give stress threads time to spin up and saturate cores
    thread::sleep(std::time::Duration::from_millis(100));

    let input = generate_sine_440hz(BLOCK_SIZE);
    let mut output = vec![0.0f32; BLOCK_SIZE];
    let hist = LatencyHistogram::new();
    let mut violations: u64 = 0;

    for _ in 0..MEASURE_BLOCKS {
        let start = std::time::Instant::now();
        model.process(&input, &mut output);
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        hist.record(elapsed_ns);

        if elapsed_ns > RT_DEADLINE_US * 1000 {
            violations += 1;
        }

        assert!(
            output.iter().all(|s| s.is_finite()),
            "[{label}] Non-finite output under stress"
        );
    }

    // Stop stress threads
    running.store(false, Ordering::Relaxed);
    for w in workers {
        let _ = w.join();
    }

    // Report statistics
    let p50 = hist.get_percentile(0.50) / 1000;
    let p99 = hist.get_percentile(0.99) / 1000;
    let p999 = hist.get_percentile(0.999) / 1000;
    let exact_max = hist.get_exact_max() / 1000;
    let violation_pct = (violations as f64 / MEASURE_BLOCKS as f64) * 100.0;

    println!(
        "[{label}] P50={p50}μs  P99={p99}μs  \
         P99.9={p999}μs  exact_max={exact_max}μs  \
         violations={}/{} ({:.2}%)",
        violations, MEASURE_BLOCKS, violation_pct
    );
}

/// Baseline — no stress.
#[test]
#[ignore]
fn test_jitter_baseline_wavenet_standard() {
    if let Some(mut model) = load_and_prewarm("BossWN-standard.nam") {
        run_jitter_test("WaveNet-Std-baseline", &mut model, 0);
    }
}

/// Light stress — 1 background CPU-burn thread.
#[test]
#[ignore]
fn test_jitter_stress_1_wavenet_standard() {
    if let Some(mut model) = load_and_prewarm("BossWN-standard.nam") {
        run_jitter_test("WaveNet-Std-stress-1", &mut model, 1);
    }
}

/// Moderate stress — 2 background CPU-burn threads.
#[test]
#[ignore]
fn test_jitter_stress_2_wavenet_standard() {
    if let Some(mut model) = load_and_prewarm("BossWN-standard.nam") {
        run_jitter_test("WaveNet-Std-stress-2", &mut model, 2);
    }
}

/// Heavy stress — saturate all but 1 core (for the DSP thread).
#[test]
#[ignore]
fn test_jitter_stress_saturate_wavenet_standard() {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let stress_threads = num_cpus.saturating_sub(1).max(1);
    if let Some(mut model) = load_and_prewarm("BossWN-standard.nam") {
        run_jitter_test(
            &format!("WaveNet-Std-saturate-{stress_threads}"),
            &mut model,
            stress_threads,
        );
    }
}
