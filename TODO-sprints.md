<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO-sprints.md — Planejamento de Sprints Técnicas (E3, E4, E5)

> **Origem:** Skill `planejador-arquiteto`
> **Objetivo:** Decompor os achados dos épicos E3, E4 e E5 (referenciados em `TODO-findings.md`) em sprints lógicas e tarefas técnicas granulares, atômicas e direcionadas aos especialistas (`implementador`, `refatora-rust`, etc).

---

## 🟠 Épico E3 — Pureza RT do Hot-Path (Zero-panic, Zero-IO, Zero-denormal)

**Meta:** Eliminar os últimos desvios da política RT-safety §1, garantindo execução sem falhas no hot-path.

### Sprint E3.1: Remoção de Código Perigoso no RT Thread [DONE]

*Foco: Remover `unwrap` e logs de I/O do hot-path.*

- [x] **Tarefa E3.1.1 [F9]: Substituir `.unwrap()` por fallback seguro no descarregamento a frio.**
  - **Especialista:** `implementador`
  - **Arquivo:** `src/clap/processor/events.rs`
  - **Ação:** Em `cold_load_model`, alterar `self.model_l.take().unwrap()` para `if let Some(failed) = self.model_l.take() { self.push_to_gc(GcItem::Model(failed)); }`.

- [x] **Tarefa E3.1.2 [F10]: Trocar `log::error!` por sinalização atômica em operações de rebuild.**
  - **Especialista:** `implementador`
  - **Arquivos:** `src/clap/processor/events.rs`
  - **Ação:** Em `try_slimmable_rebuild`, substituir o log direto no RT thread por uma atualização numa flag atômica (ex.: `RT_STATUS_SLIMMABLE_SLICE_FAILED`) a ser consumida pelo thread principal para logar de forma segura.

### Sprint E3.2: Consistência Numérica e Anti-Denormal em Ativações [DONE]

*Foco: Impedir subnormalização nos tails escalares e unificar aproximações polinomiais nos LSTMs.*

- [x] **Tarefa E3.2.1 [F11]: Adicionar guardas de denormal nos tails escalares.**
  - **Especialista:** `implementador`
  - **Arquivos:** `src/math/activations/sigmoid.rs`, `src/math/activations/tanh/production.rs`
  - **Ação:** Implementar o flush-to-zero manual (`if val.abs() < f32::MIN_POSITIVE { *val = 0.0; }`) nos loops escalares pós-SIMD.

- [x] **Tarefa E3.2.2 [F19]: Unificar polinômios de aproximação no tail escalar dos gates LSTM.**
  - **Especialista:** `implementador` / `pesquisador-inovador` (math)
  - **Arquivos:** `src/math/lstm/gates.rs`
  - **Ação:** Substituir chamadas diretas a `(-x).exp()`/`x.tanh()` pelas mesmas aproximações (`scalar_minimax_sigmoid`/`scalar_pade_tanh`) usadas no corpo SIMD para manter continuidade. *Atenção:* Exige revalidar os thresholds dos testes de paridade.

---

## 🟡 Épico E4 — Robustez de Concorrência e Ciclo de Vida (CLAP/Standalone)

**Meta:** Endurecer os mecanismos de persistência, gerenciamento de estado e garbage collection lock-free.

### Sprint E4.1: Organização e Limpeza de Scaffolding

*Foco: Limpeza do scaffolding CLAP stereo e traits não utilizados.*

- [ ] **Tarefa E4.1.1 [F12]: Remover scaffolding stereo morto (CLAP mono).**
  - **Especialista:** `refatora-rust`
  - **Arquivos:** `src/clap/processor/events.rs`, `src/clap/processor/dsp/orchestrator.rs`, `src/clap/processor_test.rs`
  - **Ação:** NAM-rs é native mono. Remover `model_r` em `cold_load_model`, `active_model_r` nas structs, e fluxos R no orchestrator/rebuild. Corrigir os comentários nos testes.

- [ ] **Tarefa E4.1.2 [F17]: Remover trait `AudioHost` inerte.**
  - **Especialista:** `refatora-rust`
  - **Arquivo:** `src/common/audio_host.rs`
  - **Ação:** Como não é utilizado no CLAP ou PipeWire, removê-lo do projeto para diminuir código morto.

### Sprint E4.2: Robustez de Estado e Interfaces de Áudio

*Foco: Refinamentos no GC e estado do plugin.*

- [ ] **Tarefa E4.2.1 [F13]: Corrigir ordering do buffer SPSC no `GcOverflowBuffer`.**
  - **Especialista:** `implementador` (concorrência)
  - **Arquivo:** `src/common/spsc/gc.rs`
  - **Ação:** Analisar e ajustar a ordem de operações (`fetch_add(1, Relaxed)` vs `swap`) para garantir semântica estrita de sincronização ou documentar explícita e tecnicamente como "drain deferido".

- [ ] **Tarefa E4.2.2 [F14, F15]: Estruturar `migrate()` de estado e limpar escape ANSI.**
  - **Especialista:** `implementador`
  - **Arquivo:** `src/clap/extensions/state.rs`
  - **Ação:** Estruturar um pipeline básico de migração v0->v1 (`match version`) e remover o escape ANSI (`\r\x1b[K`) em retornos de log, trocando por strings limpas.

- [ ] **Tarefa E4.2.3 [F16]: Adicionar asserções de bounds de callback no PipeWire.**
  - **Especialista:** `implementador`
  - **Arquivo:** `src/standalone/pw_host/rt_callback/process.rs`
  - **Ação:** Adicionar `debug_assert!(offset + n_bytes <= len)` e `debug_assert!(n_samples * 4 == n_bytes)` documentando e garantindo em runtime o contrato do host.

---

## 🟡 Épico E5 — Soundness e Performance SIMD

**Meta:** Prevenir potenciais falhas de memória UB e extrair máxima performance das CPUs AVX-512.

### Sprint E5.1: Soundness (UB Latente) e Intrínsecos

*Foco: Ajustes preventivos de segurança e higiene no math.*

- [ ] **Tarefa E5.1.1 [F18]: Garantir restrição de alinhamento nas convoluções SIMD.**
  - **Especialista:** `implementador`
  - **Arquivos:** `src/math/dsp/stereo/convolution_avx2.rs`, `src/math/dsp/stereo/convolution_avx512.rs`
  - **Ação:** Inserir checks defensivos `debug_assert!(coeffs.as_ptr() as usize % alinhamento == 0)` para fail-fast se ponteiros não-alinhados entrarem nesses kernels.

- [ ] **Tarefa E5.1.2 [F21]: Remover intrínsecos de redução deprecados.**
  - **Especialista:** `implementador`
  - **Arquivos:** `src/math/dsp/stereo/convolution_avx2.rs` etc.
  - **Ação:** Substituir usos passivos de `_mm_cvtss_f32` por `_mm_store_ss(&mut out, r)`.

### Sprint E5.2: Performance AVX-512

*Foco: Implementar pipelines de ativação mais largos.*

- [ ] **Tarefa E5.2.1 [F20]: Implementar variantes AVX-512 para as 4 ativações A2.**
  - **Especialista:** `pesquisador-inovador` / `implementador`
  - **Arquivos:** `src/math/common/dispatch/detect.rs`, ativações.
  - **Ação:** Implementar kernels AVX-512 e dispatch para `hard_tanh`, `hard_swish`, `fast_tanh`, e `leaky_hard_tanh`. Testar contra referências escalares para garantir bit-parity ou tolerância adequada.
