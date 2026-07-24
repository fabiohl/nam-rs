---
trigger: glob
description: Rust guidelines for RT Audio, Neural Inference, and Plugins.
globs: **/*.rs, **/*.toml
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Technical and Performance Guidelines (NAM-rs)

## 1. Absolute Real-Time Safety (RT-Safety)

* **Zero Heap Drop:** Heap objects (`Box`, `Vec`, `Arc`, strings) must never go out of scope on the RT thread. Transfer via SPSC GC (`gc_producer.push(GcItem::...)`).
* **Zero Blocking I/O:** No `println!`, `eprintln!`, `format!`, file I/O, or locks on the RT thread. Use `RtStatusFlags` (atomic bitmask) to signal states; the main thread does the logging.
* **Panic Prevention:** Stack unwinding allocates and breaks RT determinism. No `unwrap()`/`expect()` — use `.get()` with fallbacks. Structure loops for static elimination of bounds checks.

---

## 2. DSP & RT-Safe Math

* **Denormals:** Configure FTZ+DAZ at the start of the processing loop. Alternatively, zero states with `if val.abs() < 1e-15 { val = 0.0; }`.
* **FastMath:** Native `f32::tanh()`/`exp()` are prohibitive on the hot-path. Never call them directly; use the existing `simd_tanh`/`simd_sigmoid` kernels (`src/math/activations/`). Do not invent new approximations without measuring the error budget — see [docs/fastmath-approximations.md](../../docs/fastmath-approximations.md) and [docs/audio_fidelity_map.md](../../docs/audio_fidelity_map.md) for the current numbers and precision modes.
* **Casting:** Avoid frequent `as` between `f32` and integers. Use `.round()`, `.floor()`, or vectorized operations.
* **Unsafety:** Do your best to keep "unsafe" block the most restrict possible.

---

## 3. SIMD & Auto-Vectorization (target: x86-64-v3)

* **Mandatory Baseline (x86-64-v3):** The project enforces `x86-64-v3` (AVX2, FMA, BMI2) as the strict minimum baseline in `.cargo/config.toml`. **Never** write scalar fallbacks or use `is_x86_feature_detected!("avx2")`, `if is_x86_feature_detected!("avx2") && ...)`, `if !avx2 { fallback }` or similar. AVX2 and FMA instructions must be used natively and unconditionally throughout the entire codebase, including outside the hot path. Dynamic dispatch should only exist for higher extensions like AVX-512.
* **Loops:** Use `.chunks_exact(N)` / `.chunks_exact_mut(N)` + `zip`. Bodies free of complex branches — prefer branchless (SIMD masks).
* **Alignment:** Always `AlignedVec<T>` (64 bytes) for buffers, coefficients, and tensors. Prevents unaligned load/store penalties.

---

## 4. Lock-Free Concurrency (SPSC & Cache)

* **False Sharing:** Shared structures RT↔Main annotated with `#[repr(align(128))]` to isolate cache lines.
* **Ordering:** No `SeqCst` on the hot-path. `Relaxed` for telemetry; `Acquire`/`Release` for SPSC pointers.

---

## 5. Quality Modes: Live vs. HQ / Offline

NAM-rs operates in two distinct quality modes — **Live** (default: oversampling `Off`, adaptive compute active, zero added latency) and **HQ/Offline** (oversampling `4×`, adaptive compute disabled for deterministic output). Activation precision defaults to `Standard` (exact-grade) in both modes; `Fast` (Padé approximations) is an explicit opt-in for CPU-constrained scenarios. Never write code that silently mixes the two modes (e.g. a code path that reads adaptive-compute state while `RenderMode::Offline` is active). Full mode matrix and rationale: [README.md](../../README.md#quality-and-operational-modes) and [docs/audio_fidelity_map.md](../../docs/audio_fidelity_map.md).

Off-RT resource swaps (oversampling factor, model, cab IR) always go through the SPSC → GC-cascade protocol — never mutate or allocate these on the audio thread. See [docs/clap_integration.md](../../docs/clap_integration.md) §6.3.

---

## 6. Measurement & Off-RT QA Framework

Measurement and spectral analysis functions are **strictly off-RT** — they allocate on the heap and are never called from the audio thread. Placement, gate calibration, and f64-oracle-wins rules are defined once in [.agents/rules/testing.md](testing.md) — do not duplicate them here.

* **True-peak prohibition (RT-specific):** BS.1770-4 Annex 2 true-peak (4× polyphase FIR, 48 taps) is too expensive for the RT thread. Use sample-peak detection on the audio-thread hot-path; true-peak only in integration tests (`src/testing/perceptual.rs`).

---

## 7. Logging & Diagnostics Standards (Off-RT Logging)

* **Unified `log::*` Facade:** All off-RT modules (constructors, builders, parsers, configuration functions, plugin lifecycle, preset loading) MUST use the unified `log::*` facade (`info!`, `warn!`, `error!`, `debug!`). Never create disconnected custom logging routines or depend solely on manual host logger calls.
* **Strict Off-RT Enforcement:** `log::*` macros are **strictly prohibited** inside the hot-path audio thread (`process()`, audio callback, inner DSP sample loops). Signaling RT state transitions or anomalies MUST be done exclusively via atomic bitmasks (`RtStatusFlags`) or lock-free SPSC channels, which are then consumed off-RT by main-thread loops (`poll_rt_status()` / `emit_pending_logs()`).
* **Comprehensive Domain Coverage:**
  * **Model & IR Loaders (`src/loader/`, `src/dsp/cabsim/loader.rs`):** Log file path/basename, file size in bytes, format detection (`.nam` vs `.namb`), parsing duration, model topology (WaveNet, LSTM, ConvNet, Linear), receptive field, weight counts, sample rate, and CabSim IR metadata (sample rate, channels, frames, resample status).
  * **DSP Infrastructure (`src/dsp/` constructors/config):** Log resampler initialization (ratios, filter mode), oversampling mode transitions (`Off`, `2×`, `4×`) with added latency in samples and ms, noise gate thresholds/toggles, and adaptive compute state changes.
  * **CLAP Plugin Lifecycle (`src/clap/`):** Log DAW host name/version, CLAP API version, render mode changes (`Realtime` vs `Offline` HQ), and preset path/name.
  * **Standalone & PipeWire Host (`src/standalone/`):** Log PipeWire quantum/buffer renegotiations, CPU affinity/SCHED_FIFO status, and HugeTLB allocation attempts/fallbacks.
* **Log Buffer & Diagnostics Integration:** All `log::*` calls populate the central `NamLogger` ring buffer (`LogBuffer`), ensuring that a recent execution trace (`Recent Log Trace`) is automatically included in support bundles (`DiagnosticBundle::render()`) and crash reports (`~/.cache/nam-rs/crash-*.txt`).
