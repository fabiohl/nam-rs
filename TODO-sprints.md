<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO-sprints.md — Roteiro Técnico de Auditoria NAM-rs

---

## Épico 8 — Loader & Parsing

> **Módulos**: `loader/mod.rs`, `loader/dispatcher/`

### 🔴 Tarefa 8.2 — `loader/mod.rs` — `io_error` Ausente no Diagnóstico `.nam`

- **Arquivo**: `loader/mod.rs:84-89`
- **Achado**: O branch `.nam` **não inclui** `.param("io_error", &e)` no `NamDiagnostic`, ao contrário do branch `.namb` (linha 65). O diagnóstico perde o `io::Error` real (ex: "Permission denied", "No such file").
- **Correção**:

  ```rust
  let json = std::fs::read_to_string(path).map_err(|e| {
      NamDiagnostic::new(NamErrorCode::FileReadError, sys)
          .message(format!("Não conseguimos ler o arquivo \"{}\".", path_str))
          .param("file", &path_str)
          .param("io_error", &e)   // ← adicionar
          .emit();
      anyhow::Error::from(e)
  })?;
  ```

### 🟡 Tarefa 8.3 — `WeightCursor` — Abstração do Campo `layout`

- **Arquivo**: `loader/dispatcher/wavenet.rs` (múltiplos pontos)
- **Achado**: `cursor.layout == WeightsLayout::Interleaved4WaveNet` é comparado diretamente em vários locais sem abstração. A adição de um terceiro layout (ex: NAMB v3) exigiria atualização manual de todos os `if/else`.
- **Proposta**: Extrair método `cursor.is_interleaved4() -> bool` ou converter para `match` exaustivo. Reduz risco de regressão silenciosa ao evoluir o formato.

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

### 🔴 Tarefa 9.3 — Zero-Alloc Guard para `capture_dsp_pipeline`

- **Arquivo**: `tests/nam_infer_test.rs`
- **Achado**: `test_zero_alloc_process_*` cobrem apenas `.process()` do modelo. O `capture_dsp_pipeline` completo (gate + resampler bypass + bridge write) **não tem guard de alocação**. Uma alocação acidental em `pipeline.rs` não seria detectada.
- **Proposta**: Adicionar `test_zero_alloc_capture_pipeline` em `tests/nam_infer_test.rs` envolvendo `capture_dsp_pipeline` com `CountingAllocator`.

### 🟡 Tarefa 9.2 — Golden — WaveNet Dyn Gated

- **Arquivo**: `tests/regression_goldens.rs`
- **Achado**: Não há golden auto-referência específico para WaveNet dinâmico com `gated=true`. A cobertura de `gated_activation_and_accumulate_block` em diferentes `block_size` é inferida indiretamente pelos testes de paridade.
- **Proposta**: Adicionar `test_golden_wavenet_dyn_gated` com fixture dedicada. Baixa urgência — funcionalidade coberta pelos testes de paridade existentes.

### 🟡 Tarefa 9.4 — `VirtualRingBuffer` — Teste Determinístico de Wraparound

- **Arquivo**: `dsp/vring.rs`
- **Achado**: Não há teste unitário determinístico para o caso exato de wraparound (write pointer cruzando o limite físico). `test_vring_long_run` (soak) cobre isso indiretamente mas está marcado `#[ignore]`.
- **Proposta**: Adicionar `test_vring_wraparound_exact` que escreve exatamente `capacity` amostras e valida a leitura com dilatação `capacity/2` após o wraparound.

### 🟡 Tarefa 9.5 — `LatencyHistogram` — Teste de Regressão de Saturação

- **Arquivo**: `dsp/telemetry.rs`
- **Achado**: `fetch_add` saturante está implementado. Não há teste unitário que exercite o limite superior, deixando a correção sem guarda contra regressão futura.
- **Proposta**: Adicionar teste que incrementa o contador até o limite e confirma que ele satura (não faz overflow).
