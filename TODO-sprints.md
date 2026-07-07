<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO Sprints

Este documento organiza a execução das otimizações planejadas em Epics do projeto `nam-rs`.

## Sprint 1: Otimização SIMD do Hot-Path WaveNet/LSTM (x86-64-v3) [DONE]

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

## Sprint 2: Otimização do Resampler e Oversampler (x86-64-v3) [DONE]

**Meta da Sprint:** Otimizar o overhead aritmético do resampler polyphase e vetorizar a convolução escalar do oversampler half-band.

### [x86-64-v3] T-PERF-2.1: Otimização do Fator de Interpolação Fracional no Resampler (`resampler.rs`) [DONE]

- **Objetivo:** Eliminar a conversão `f64` no cálculo de `frac` usando shift à direita por 9 e conversão para `i32` multiplicada por `1.0 / (1u32 << 31) as f32`.
- **Finding Associado:** P-3
- **Arquivos:**
  - [resampler.rs](file:///home/fabio/nam-rs/src/dsp/resampler.rs)
- **Risco:** Médio (precisão de interpolação fracional).
- **Validação:**
  - Rodar `utils/quality-dashboard.sh` aferindo que o ESR dos modelos em taxas ≠ 48 kHz não apresentou regressão em relação ao ideal.
- **Conclusão:** Substituída a conversão `frac_bits as i64 as f64 * (1.0 / (1u64 << 40) as f64)) as f32` por `((frac_bits >> 9) as i32 as f32) * (1.0 / (1u32 << 31) as f32)`, eliminando as instruções `vcvtqq2pd` + `vmulsd` + `vcvtsd2ss` (~3-5 cycles) do hot-path stereo e mono. Todos os 24 testes do resampler passaram (SNR ≥ 25 dB contra libsoxr, micro-soak 5000 iterações × 5 pares de taxas, drift fixed-point < 1e-7). Emissão de `vcvtqq2ps` única no lugar da cadeia f64.

### [x86-64-v3] T-PERF-2.2: Vetorização AVX2 do Filtro Half-Band no Oversampler (`oversample.rs`) [DONE]

- **Objetivo:** Implementar double-buffering e separar ring-buffers para amostras pares e ímpares, possibilitando loads SIMD contíguos de 8 e 4 elementos e eliminando indexações modulares por amostra.
- **Finding Associado:** P-4
- **Arquivos:**
  - [oversample.rs](file:///home/fabio/nam-rs/src/dsp/oversample.rs)
- **Risco:** Baixo.
- **Validação:**
  - Execução do suite de testes rápidos (`utils/tests-quick.sh`).
- **Conclusão:** Upsampler: `up_ring` convertido para double-buffer (24 entradas) com double-write, eliminando `(pos + n - d) % n` por acesso contíguo via `wptr + 5` (center tap) + `_mm256_fmadd_ps` 8-wide + 4 scalar (12 odd taps). Coeficientes invertidos em `HalfBandFilter::design()` para casar com layout oldest-first. Downsampler: `down_ring` substituído por `down_ring_even` (26, double-buffer) + `down_ring_odd` (24, double-buffer), tornando os 12 taps ímpares contíguos no buffer ímpar — mesmo padrão AVX2 FMADD 8+4. Eliminados 12×2 MACs escalares (upsample) e 13×2 MACs escalares (downsample) por amostra. Todos os 15 testes de oversample passaram (DC gain, X2/X4 round-trip, aliasing rejection, perceptual 4×). Suite `tests-quick.sh` completa limpa.

---

## Sprint 3: Modernização da Stack do Kernel Linux (Requisito: Linux 6.3+)

**Meta da Sprint:** Adotar de forma mandatória os recursos modernos de Huge Pages e descritores seguros do Kernel Linux (mínimo Kernel 6.3 para cobertura completa das LTS recentes).

### [Linux 6.1+] T-PERF-3.1: ✅ Promoção Síncrona de Huge Pages com `MADV_COLLAPSE` (`huge_alloc.rs`) — CONCLUÍDO 2026-07-06

- **Objetivo:** Chamar `madvise` com o flag `MADV_COLLAPSE` obrigatoriamente e sem fallbacks de verificação.
- **Finding Associado:** P-7
- **Arquivos:**
  - [huge_alloc.rs](file:///home/fabio/nam-rs/src/math/common/huge_alloc.rs)
  - [mirror_buf/alloc.rs](file:///home/fabio/nam-rs/src/dsp/mirror_buf/alloc.rs) (THP path do MirroredBuffer)
- **Risco:** Mínimo. Requer Kernel Linux 6.1+.
- **Validação:**
  - Validação funcional no Linux garantindo compilação e execução corretas em kernel moderno.
- **Conclusão:** `MADV_COLLAPSE` adicionado após `MADV_HUGEPAGE` nos dois call sites THP (`huge_alloc.rs:120` + `mirror_buf/alloc.rs:235`). Sem verificações de versão de kernel (confia no requisito Linux 6.1+). Constante já disponível em libc 0.2.186. `cargo check`, `cargo clippy` e 4 testes `huge_alloc_tests` + 9 testes `diagnostic_bundle` passam limpo.

### [Linux 6.3+] T-PERF-3.2: Hardening de Memfds com `MFD_NOEXEC_SEAL` (`huge_alloc.rs`)

- **Objetivo:** Passar obrigatoriamente o flag `MFD_NOEXEC_SEAL` na chamada de `memfd_create` para buffers hugetlb e comuns, sem ramificações de fallback.
- **Finding Associado:** P-8
- **Arquivos:**
  - [huge_alloc.rs](file:///home/fabio/nam-rs/src/math/common/huge_alloc.rs)
- **Risco:** Mínimo. Requer Kernel Linux 6.3+.
- **Validação:**
  - Validação funcional garantindo que memfds não executáveis funcionem corretamente para buffers de modelo e alocações de huge pages.
