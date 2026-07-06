<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO Sprints

Este documento organiza a execução das otimizações planejadas em Epics do projeto `nam-rs`.

## Sprint 1: Otimização SIMD do Hot-Path WaveNet/LSTM (x86-64-v3)

**Meta da Sprint:** Reduzir a latência do hot-path WaveNet/LSTM em sistemas x86-64-v3 eliminando stores escalares desnecessários e reduzindo atomic loads e branches redundantes nos loops críticos.

### Tarefas Técnicas

#### [x86-64-v3] T-PERF-1.1: Otimização de Stores de Acumuladores em WaveNet (`conv_input.rs`)

- **Objetivo:** Substituir loops de stores escalares nas funções `store_16_accums`, `store_8_accums` e `store_4_accums` por instruções SIMD `_mm256_storeu_ps` / `_mm_storeu_ps`.
- **Finding Associado:** P-1
- **Arquivos:**
  - [conv_input.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv_input.rs)
- **Risco:** Mínimo. Mudança puramente estrutural em stores de buffers temporários.
- **Validação:**
  - Garantir compilação com target x86-64-v3.
  - Verificar fidelidade via `utils/quality-dashboard.sh` comparando com o baseline salvo em `E-PERF-1_quality-dashboard.txt`.

#### [x86-64-v3] T-PERF-1.2: Otimização de Branch e Atomic Loads em Gates LSTM (`gates.rs`)

- **Objetivo:** Hoistar a verificação `activation_precision() == ActivationPrecision::HighFidelity` para fora do loop principal em `fused_lstm_gates_dyn_avx2` através de bifurcação de loops (loop-fission/bifurcation) e criação de variantes especializadas de processamento de gates para AVX2.
- **Finding Associado:** P-2
- **Arquivos:**
  - [gates.rs](file:///home/fabio/nam-rs/src/math/lstm/gates.rs)
- **Risco:** Baixo.
- **Validação:**
  - Execução dos testes unitários e de integração (`utils/tests-quick.sh`).
  - Comparação de performance contra o benchmark salvo de Criterion.
