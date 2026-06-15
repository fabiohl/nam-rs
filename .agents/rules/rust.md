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

---

## 3. SIMD & Auto-Vectorization (target: x86-64-v3)

* Nam-rs is x86-64-v3 first: *Alway* try to optimize to use modern ISA instructions.
* **Loops:** Use `.chunks_exact(N)` / `.chunks_exact_mut(N)` + `zip`. Bodies free of complex branches — prefer branchless (SIMD masks).
* **Alignment:** Always `AlignedVec<T>` (64 bytes) for buffers, coefficients, and tensors. Prevents unaligned load/store penalties.

---

## 4. Lock-Free Concurrency (SPSC & Cache)

* **False Sharing:** Shared structures RT↔Main annotated with `#[repr(align(128))]` to isolate cache lines.
* **Ordering:** No `SeqCst` on the hot-path. `Relaxed` for telemetry; `Acquire`/`Release` for SPSC pointers.
