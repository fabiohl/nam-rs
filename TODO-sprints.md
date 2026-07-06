<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO Sprints

Este documento organiza a execução das otimizações planejadas em Epics do projeto `nam-rs`.

## Sprint 1: Otimização SIMD do Hot-Path WaveNet/LSTM (x86-64-v3)

**Meta da Sprint:** Reduzir a latência do hot-path WaveNet/LSTM em sistemas x86-64-v3 eliminando stores escalares desnecessários e reduzindo atomic loads e branches redundantes nos loops críticos.

### Tarefas Técnicas

#### [x86-64-v3] T-PERF-1.1: Otimização de Stores de Acumuladores em WaveNet (`conv_input.rs`) [DONE]

- **Objetivo:** Substituir loops de stores escalares nas funções `store_16_accums`, `store_8_accums` e `store_4_accums` por instruções SIMD `_mm256_storeu_ps` / `_mm_storeu_ps`.
- **Finding Associado:** P-1
- **Arquivos:**
  - [conv_input.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv_input.rs)
- **Risco:** Mínimo. Mudança puramente estrutural em stores de buffers temporários.
- **Validação:**
  - Garantir compilação com target x86-64-v3.
  - Verificar fidelidade via `utils/quality-dashboard.sh` comparando com o baseline salvo em `E-PERF-1_quality-dashboard.txt`.

#### [x86-64-v3] T-PERF-1.2: Otimização de Branch e Atomic Loads em Gates LSTM (`gates.rs`) [DONE]

- **Objetivo:** Hoistar a verificação `activation_precision() == ActivationPrecision::HighFidelity` para fora do loop principal em `fused_lstm_gates_dyn_avx2` através de bifurcação de loops (loop-fission/bifurcation) e criação de variantes especializadas de processamento de gates para AVX2.
- **Finding Associado:** P-2
- **Arquivos:**
  - [gates.rs](file:///home/fabio/nam-rs/src/math/lstm/gates.rs)
- **Risco:** Baixo.
- **Validação:**
  - Execução dos testes unitários e de integração (`utils/tests-quick.sh`).
  - Comparação de performance contra o benchmark salvo de Criterion.

---

## Sprint 2: Otimização do Resampler e Oversampler (x86-64-v3)

**Meta da Sprint:** Otimizar o overhead aritmético do resampler polyphase e vetorizar a convolução escalar do oversampler half-band.

### [x86-64-v3] T-PERF-2.1: Otimização do Fator de Interpolação Fracional no Resampler (`resampler.rs`)

- **Objetivo:** Eliminar a conversão `f64` no cálculo de `frac` usando shift à direita por 9 e conversão para `i32` multiplicada por `1.0 / (1u32 << 31) as f32`.
- **Finding Associado:** P-3
- **Arquivos:**
  - [resampler.rs](file:///home/fabio/nam-rs/src/dsp/resampler.rs)
- **Risco:** Médio (precisão de interpolação fracional).
- **Validação:**
  - Rodar `utils/quality-dashboard.sh` aferindo que o ESR dos modelos em taxas ≠ 48 kHz não apresentou regressão em relação ao ideal.

### [x86-64-v3] T-PERF-2.2: Vetorização AVX2 do Filtro Half-Band no Oversampler (`oversample.rs`)

- **Objetivo:** Implementar double-buffering e separar ring-buffers para amostras pares e ímpares, possibilitando loads SIMD contíguos de 8 e 4 elementos e eliminando indexações modulares por amostra.
- **Finding Associado:** P-4
- **Arquivos:**
  - [oversample.rs](file:///home/fabio/nam-rs/src/dsp/oversample.rs)
- **Risco:** Baixo.
- **Validação:**
  - Execução do suite de testes rápidos (`utils/tests-quick.sh`).
