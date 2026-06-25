<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Sprints de Engenharia — Resolução dos Épicos E2 e E3

Este documento apresenta o planejamento ágil para as sprints de implementação e hardening dos Épicos **E2** e **E3** identificados em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

---

## Épico E2 — Endurecimento de contratos `unsafe` e ordenação de memória 🟡

### Sprint 1: Robustez de Buffers e SIMD (F-03 & F-08)

#### 📋 Tarefa E2.1: Proteção de limites no bypass estéreo do Resampler (F-03)

* **Objetivo:** Impedir leituras/escritas fora dos limites (UB) no resampler quando o canal direito for assíncrono ou assimétrico.
* **Componente:** DSP / Resampler
* **Arquivos afetados:**
  * [src/dsp/resampler.rs](file:///home/fabio/nam-rs/src/dsp/resampler.rs) (linhas ~490 e ~514)
* **Mudança proposta:**
  * No `process_input` e `process_output` (ramos bypass), calcular `n` incluindo ambos os canais (`in_l`, `in_r`, `out_l`, `out_r`):

    ```rust
    let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
    ```

* **Risco:** BAIXO.
* **Critério de Aceitação:**
  * Teste unitário adicionado que valida buffers assimétricos copiando com segurança no ramo bypass sem pânico ou UB.

#### 📋 Tarefa E2.2: Hardening dos invariantes de padding SIMD em release (F-08)

* **Objetivo:** Promover a checagem de alocação de padding SIMD de `debug_assert!` para `assert!` em tempo de execução frio (carga do modelo).
* **Componente:** Model Loader / WaveNet Kernels
* **Arquivos afetados:**
  * [src/loader/dispatcher/wavenet/traits.rs](file:///home/fabio/nam-rs/src/loader/dispatcher/wavenet/traits.rs)
* **Mudança proposta:**
  * No construtor `from_parts` de `Conv1d` e `Conv1dDyn`, recalcular o total padded esperado (`num_blocks * interleave_width * in_ch * k_size`) e aplicar assertiva estrita:

    ```rust
    assert!(weights.len() >= padded_total, "Conv1d weights buffer is too small");
    ```

* **Risco:** BAIXO (não afeta o hot-path real-time).
* **Critério de Aceitação:**
  * Teste unitário em `traits_test.rs` (ou correspondente) que força a construção de um `Conv1d` subdimensionado e assegura que ele provoca pânico seguro em release.

---

### Sprint 2: Ordenação Concorrente RT↔Main (F-04)

#### 📋 Tarefa E2.3: Sincronização SPSC Status Flags com Barreiras Acquire-Release (F-04)

* **Objetivo:** Garantir a coerência de dados transferidos via variáveis atômicas `requested_*` entre a thread RT (produtora) e a thread Main (consumidora) eliminando ordenação `Relaxed` nos handshakes.
* **Componente:** Concorrência SPSC
* **Arquivos afetados:**
  * [src/common/spsc/status.rs](file:///home/fabio/nam-rs/src/common/spsc/status.rs)
  * [src/clap/processor/events.rs](file:///home/fabio/nam-rs/src/clap/processor/events.rs)
  * [src/standalone/pw_host/rt_callback/commands.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/rt_callback/commands.rs)
  * [src/clap/plugin/main_thread/housekeeping.rs](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/housekeeping.rs)
  * [src/standalone/pw_host/run.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/run.rs)
* **Mudança proposta:**
  * Adicionar métodos Acquire/Release específicos na `RtStatusFlags` em [status.rs](file:///home/fabio/nam-rs/src/common/spsc/status.rs):
    * `set_flag_release(&self, flag: u64)` -> usa `fetch_or(flag, Ordering::Release)`
    * `check_flag_acquire(&self, flag: u64)` -> usa `load(Ordering::Acquire)`
    * `clear_flag_release(&self, flag: u64)` -> usa `fetch_and(!flag, Ordering::Release)`
  * Substituir chamadas de handshake de dados (rebuilds de slimmable, cabsim, resampler) para usar estes novos métodos de barreira, mantendo as flags puras de telemetria com a ordenação `Relaxed`.
* **Risco:** MÉDIO. Requer precisão e auditoria rigorosa de cada ponto de controle atômico.
* **Critério de Aceitação:**
  * Compilação limpa, sem regressões nos testes concorrentes de SPSC.
  * Validação teórica e estrutural da sincronização e ausência de cache reordering em arquiteturas de memória fraca.

---

## Épico E3 — Honestidade de telemetria e _test oracles_ 🟡

### Sprint 3: Verificação de Qualidade e Observabilidade Genuínas

#### 📋 Tarefa E3.1: Correção do Fidelity Margin check honesto em validation.rs (F-06)

* **Objetivo:** Fazer com que o marcador `✓` de margem de fidelidade perceptual reflita genuinamente se o delta SNR supera o alvo de 8.0 dB.
* **Componente:** Test Oracles / QA
* **Arquivos afetados:**
  * [tests/common/validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs) (linha ~242)
* **Mudança proposta:**
  * Alterar a definição de `is_satisfactory` no teste de margem de fidelidade para depender unicamente do alvo:

    ```rust
    let is_satisfactory = delta_snr > 8.0;
    ```

* **Risco:** BAIXO.
* **Critério de Aceitação:**
  * Execução dos testes de validação com logs honestos. Modelos com margem de fidelidade abaixo de 8.0 dB devem exibir `?` ao invés de `✓`, enquanto a asserção rígida de `min_snr_db` continua sendo exigida no final da validação.

#### 📋 Tarefa E3.2: Separação de estados de alocação de Huge Page (HugeTLB vs THP) (F-07)

* **Objetivo:** Diferenciar telemetria e diagnóstico entre HugeTLB estrito (páginas físicas garantidas) de Transparent Huge Pages (THP, apenas hint do kernel).
* **Componente:** Telemetria & OS Integration
* **Arquivos afetados:**
  * [src/common/spsc/status.rs](file:///home/fabio/nam-rs/src/common/spsc/status.rs) (adicionar flag `RT_STATUS_THP_ACTIVE`)
  * [src/dsp/mirror_buf.rs](file:///home/fabio/nam-rs/src/dsp/mirror_buf.rs)
  * [src/dsp/mirror_buf/alloc.rs](file:///home/fabio/nam-rs/src/dsp/mirror_buf/alloc.rs)
  * [src/common/diagnostics/snapshot.rs](file:///home/fabio/nam-rs/src/common/diagnostics/snapshot.rs)
  * [src/common/diagnostics/bundle.rs](file:///home/fabio/nam-rs/src/common/diagnostics/bundle.rs)
  * [src/standalone/rt_setup/telemetry.rs](file:///home/fabio/nam-rs/src/standalone/rt_setup/telemetry.rs)
  * [src/clap/plugin/main_thread/logging.rs](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/logging.rs)
* **Mudança proposta:**
  * Definir `RT_STATUS_THP_ACTIVE = 1 << 18` em `status.rs`.
  * Rastrear o tipo de alocação usando um `MIRROR_BUF_HUGEPAGE_STATE: AtomicU8` em `mirror_buf.rs` (0 = Standard, 1 = THP, 2 = HugeTLB).
  * Propagar a flag correspondente na inicialização via `sync_huge_page_flag`.
  * Atualizar o `snapshot.rs` para incluir o campo de diagnóstico `huge_page_mode` e preenchê-lo adequadamente.
  * Formatar o log e bundle para refletir o modo exato em uso.
* **Risco:** BAIXO.
* **Critério de Aceitação:**
  * Verificação via logs de execução standalone e CLAP que identificam transparent hugepages vs hugetlb corretamente.
