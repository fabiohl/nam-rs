---
trigger: glob
description: Diretrizes Rust para Áudio RT, Inferência Neural e Plugins.
globs: **/*.rs, **/*.toml
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Diretrizes Técnicas e de Performance (NAM-rs)

## 1. Segurança Absoluta em Tempo Real (RT-Safety)

* **Zero Heap Drop:** Objetos heap (`Box`, `Vec`, `Arc`, strings) nunca devem sair de escopo na thread RT. Transfira via SPSC GC (`gc_producer.push(GcItem::...)`).
* **Zero I/O Bloqueante:** Sem `println!`, `eprintln!`, `format!`, I/O de arquivo ou locks na thread RT. Use `RtStatusFlags` (bitmask atômico) para sinalizar estados; a main thread faz o logging.
* **Panic Prevention:** Stack unwinding aloca e quebra determinismo RT. Sem `unwrap()`/`expect()` — use `.get()` com fallbacks. Estruture loops para eliminação estática de bounds checks.

---

## 2. DSP & Matemática RT-Safe

* **Denormals:** Configure FTZ+DAZ no início do loop de processamento. Alternativamente, zere estados com `if val.abs() < 1e-15 { val = 0.0; }`.
* **FastMath:** `f32::tanh()`/`exp()` nativos são proibitivos no hot-path. Use aproximações Minimax/Padé com erro < −80 dB (~1e-4).
* **Casting:** Evite `as` frequente entre `f32` e inteiros. Use `.round()`, `.floor()` ou operações vetorizadas.

---

## 3. SIMD & Auto-Vetorização (meta: x86-64-v3)

* **Loops:** Use `.chunks_exact(N)` / `.chunks_exact_mut(N)` + `zip`. Corpos livres de branches complexos — preferir branchless (máscaras SIMD).
* **Alinhamento:** Sempre `AlignedVec<T>` (64 bytes) para buffers, coeficientes e tensores. Previne penalidades de unaligned loads/stores.

---

## 4. Concorrência Lock-Free (SPSC & Cache)

* **False Sharing:** Estruturas compartilhadas RT↔Main anotadas com `#[repr(align(128))]` para isolar linhas de cache.
* **Ordering:** Sem `SeqCst` no hot-path. `Relaxed` para telemetria; `Acquire`/`Release` para ponteiros SPSC.
