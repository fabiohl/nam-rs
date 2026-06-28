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
* **FastMath:** Native `f32::tanh()`/`exp()` are prohibitive on the hot-path. Use Minimax/Padé approximations with error < −80 dB (~1e-4).
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

NAM-rs operates in two distinct quality modes. Code must honor these distinctions:

* **Live Mode (default):** Oversampling `Off`, activation precision `Standard`, adaptive compute active. Zero added latency. All hot-path code must complete within the RT deadline budget.
* **HQ / Offline Mode:** Oversampling `4×`, activation precision `HighFidelity`, adaptive compute disabled. Maximum fidelity with deterministic output — no soft-degradation allowed. The CLAP host signals this via `RenderMode::Offline`.

**Off-RT rebuild protocol:** Factor changes (oversampling, model swap, cab IR) are never applied on the audio thread. The main thread constructs new resources (filter allocation, buffer allocation), pushes them via SPSC, and the audio thread atomically swaps. Old resources are disposed via the GC cascade (SPSC → parking-lot → overflow).

**Deterministic offline bounce (CLAP):** When `RenderMode::Offline` is active:

* `AdaptiveCompute` is forced to `Off` (FSM reset to Full, no degradation).
* All `RT_STATUS_DEGRADE_*` flags are cleared.
* Block deadline measurements are ignored.
* User-initiated adaptive-compute mode changes are guarded and rejected.

---

## 6. Measurement & Off-RT QA Framework

Measurement and spectral analysis functions are **strictly off-RT** — they allocate on the heap and are never called from the audio thread.

* **Placement:** All measurement functions belong in `src/testing/`. Never in `src/dsp/` or hot-path code.
* **True-peak prohibition:** BS.1770-4 Annex 2 true-peak (4× polyphase FIR, 48 taps) is too expensive for the RT thread. Use sample-peak detection on the audio-thread hot-path; true-peak only in integration tests.
* **f64 oracle authority:** The f64 reference oracle (`src/testing/reference_oracle.rs`) is the absolute mathematical ground truth. When it disagrees with C++ NAMCore or golden vectors, the f64 oracle wins.
* **Gate calibration:** All metric thresholds must be explicitly documented with measurement comments. `// Measured: SNR=..., ESR=...` format required for every calibrated entry in golden threshold tables.
* **Baseline versioning:** Metric baselines (ASR, THD+N, Farina FR) are versioned in `tests/fixtures/spectral_fidelity_baseline.json`.
