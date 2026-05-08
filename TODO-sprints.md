<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO-sprints.md — Roteiro Técnico de Auditoria NAM-rs

---

## Épico 9 — Testes, Benchmarks & CI

> **Módulos**: `dsp/pipeline_test.rs`, `dsp/vring.rs`, `dsp/telemetry.rs`, `tests/nam_infer_test.rs`

### 🔴 Tarefa 9.1 — `pipeline_test.rs` — Lacunas de Cobertura do Hot-Path

- **Arquivo**: `dsp/pipeline_test.rs`
- **Achado**: Os 4 testes existentes cobrem apenas o **caminho de bypass** (sem modelo). Edge cases do hot-path não estão cobertos:
  1. **Gate Closed**: verificar `n_samples == 0` no bridge e flag `RT_STATUS_IS_SILENT`.
  2. **Gate Fading**: validar flag `RT_STATUS_IS_FADING` em `FadingIn/FadingOut`.
  3. **Clipping Detection**: confirmar que `RT_STATUS_HAS_CLIPPED` é setado quando sinal satura.
  4. **Dropped Frames**: exercitar a condição de drop no `write_bridge`/`handle_silence_bypass`.
- **Proposta**: 4 testes unitários dedicados. Os que tocam o hot-path devem usar `CountingAllocator` para zero-alloc guard.

### 🔴 Tarefa 9.2 — Zero-Alloc Guard para `capture_dsp_pipeline`

- **Arquivo**: `tests/nam_infer_test.rs`
- **Achado**: `test_zero_alloc_process_*` cobrem apenas `.process()` do modelo. O `capture_dsp_pipeline` completo (gate + resampler bypass + bridge write) **não tem guard de alocação**. Uma alocação acidental em `pipeline.rs` não seria detectada.
- **Proposta**: Adicionar `test_zero_alloc_capture_pipeline` em `tests/nam_infer_test.rs` envolvendo `capture_dsp_pipeline` com `CountingAllocator`.
