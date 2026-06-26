<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Sprints (NAM-rs)

Este documento contém o planejamento de sprints e tarefas técnicas estruturadas para o desenvolvimento do NAM-rs, garantindo paridade total com o NeuralAmpModelerCore v0.5.4.

---

## Sprint 3: Épico J — "Container Reset Seletivo + Staging" (F9, F10)

**Escopo:** Implementar reset seletivo apenas do sub-modelo ativo no ContainerModel, armazenar `sample_rate`/`max_buffer_size` para uso em `set_slimmable_size()`, e verificar propagação de prewarm no SlimmableWavenet.
**Objetivo de Paridade:** Garantir que a troca de sub-modelos de qualidade (Slimmable) seja livre de clicks e de alocações de memória no thread de tempo real, realizando o reset/prewarm correto do modelo destino no instante de sua ativação.
**Estimativa:** 1 sprint.
**Risco Geral:** 🟡 Médio — Requer garantia estrita de zero alocação no thread de tempo real (RT-Safety) ao chamar `reset` e `prewarm` na transição do `ContainerModel`.

---

### Tarefa 1. [MODEL] Reset Seletivo de Sub-Modelos no `ContainerModel` (F9) [DONE]

- **Status:** `[X]` **Concluída**
- **Arquivos Alvo:**
  - [`src/models/container.rs`](file:///home/fabio/nam-rs/src/models/container.rs)
- **Descrição:**
  - Modificar o método `reset` do bloco `impl NamModel for ContainerModel`.
  - Em vez de resetar todos os sub-modelos em um loop (`for (_, model) in &mut self.submodels`), o método deve:
    1. Atualizar os campos locais `self.sample_rate` e `self.max_buffer_size`.
    2. Redimensionar o `scratch_buffer` local.
    3. Atualizar a `crossfade_duration`.
    4. Propagar a chamada de `set_max_buffer_size(max_buffer_size)` para todos os sub-modelos para garantir que os tamanhos de buffer estejam corretos.
    5. Chamar `reset(sample_rate, max_buffer_size)` apenas no sub-modelo ativo atual (`self.submodels[self.active_index].1.reset(sample_rate, max_buffer_size)`).
- **Risco:** Baixo. Modificação local e alinhada com o comportamento do C++.

---

### Tarefa 2. [MODEL] Otimizar `set_max_buffer_size` em `WaveNetA2` para Evitar Alocações (F9 / RT-Safety) [DONE]

- **Status:** `[X]` **Concluída**
- **Arquivos Alvo:**
  - [`src/models/a2/model/static/mod.rs`](file:///home/fabio/nam-rs/src/models/a2/model/static/mod.rs)
  - [`src/models/a2/model/dynamic/mod.rs`](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/mod.rs)
- **Descrição:**
  - Modificar `set_max_buffer_size` nas implementações estática e dinâmica de `WaveNetA2`.
  - Se o `max_buf` solicitado for igual ao `self.max_buffer_size` atual e a estrutura já estiver inicializada, redefinir as variáveis de estado e preencher os buffers existentes com zero (`fill(0.0)`) em vez de liberar e realocar `MirroredBuffer` e `AlignedVec`.
  - Isso remove qualquer alocação de memória do caminho de chamada de `reset()` no thread RT.
- **Risco:** Médio. Exige validação cuidadosa de que todos os estados/ponteiros (como `head_write_pos` e `layer_buffer_starts`) foram redefinidos perfeitamente sem fugas.
- **Nota de conclusão:** Implementado. `set_max_buffer_size` agora possui três caminhos: (1) `max_buf < self.max_buffer_size` → no-op; (2) `max_buf == self.max_buffer_size` → zero-fill in-place de `head_accum`, `layer_buffers` (MirroredBuffer) e `layer_in`, reset de `head_write_pos`=rf e `layer_buffer_starts`=ring_sizes, sem nenhuma alocação no heap; (3) `max_buf > self.max_buffer_size` → realocação completa como antes. Testes existentes (`test_wavenet_a2_set_max_buffer_size_noop_on_smaller`, `test_wavenet_a2_set_max_buffer_size_grows`, `test_wavenet_a2_reset_reallocates_and_prewarms`, e equivalentes dyn) passam. A condição `has_weights()` sugerida na descrição original não foi necessária — o zero-fill in-place é sempre seguro e evita branches na decisão de RT.

---

### Tarefa 3. [MODEL] Reset e Prewarm Seletivo em `ContainerModel::set_slimmable_size` (F9) [DONE]

- **Status:** `[X]` **Concluída**
- **Arquivos Alvo:**
  - [`src/models/container.rs`](file:///home/fabio/nam-rs/src/models/container.rs)
- **Descrição:**
  - No método `set_slimmable_size(&mut self, val: f32)`, identificar se haverá uma transição de sub-modelo (quando o sub-modelo correspondente ao novo valor `val` for diferente do ativo e do pendente).
  - Antes de definir `self.pending_index = Some(next);`, chamar o método `reset(self.sample_rate, self.max_buffer_size)` no sub-modelo de destino (`self.submodels[next].1`).
  - Isso garante que o sub-modelo destino esteja limpo e pré-aquecido no sample rate e tamanho de buffer vigentes antes do início do crossfade, eliminando artefatos.
- **Risco:** Baixo a Médio. Depende da garantia de que o `reset()` do modelo destino seja RT-safe (livre de alocações).
- **Nota de conclusão:** Implementado. `set_slimmable_size` agora chama `reset()` no sub-modelo destino antes de definir `pending_index`. Para garantir zero-alocação, `ContainerModel::new()` foi modificado para chamar `set_max_buffer_size(default_buf)` em todos os sub-modelos na inicialização, evitando realocação tardia no `reset()` do destino. O `reset()` interno do A2 é RT-safe graças às otimizações das Tarefas 2 (set_max_buffer_size igual → zero-fill) e 4 (prewarm com stack arrays).

---

### Tarefa 4. [MODEL] Otimizar `prewarm` dos Modelos A2 para Evitar Alocações no Thread RT (RT-Safety) [DONE]

- **Status:** `[X]` **Concluída**
- **Arquivos Alvo:**
  - [`src/models/a2/model/static/prewarm.rs`](file:///home/fabio/nam-rs/src/models/a2/model/static/prewarm.rs)
  - [`src/models/a2/model/dynamic/prewarm.rs`](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/prewarm.rs)
- **Descrição:**
  - Substituir o uso de `vec![0.0f32; block]` por um buffer estático/de pilha de tamanho fixo `[0.0f32; WAVENET_MAX_NUM_FRAMES]` nos métodos `prewarm` dos modelos A2.
  - Isso evita alocação e desalocação de vetores no heap durante a fase de pré-aquecimento.
- **Risco:** Baixo. Apenas alteração mecânica de tipo de buffer.
- **Nota de conclusão:** Implementado. `vec![0.0f32; block]` substituído por `[0.0f32; WAVENET_MAX_NUM_FRAMES]` (64 elementos, 256 bytes) em ambos `prewarm.rs` (static e dynamic). Testes de prewarm, wavenet, golden vectors, zero-alloc e container_slimmable passam sem regressões.

---

### Tarefa 5. [DOC] Documentar Divergência Intencional de Staging de `SlimmableWavenet` (F10) [DONE]

- **Status:** `[X]` **Concluída**
- **Arquivos Alvo:**
  - [`src/models/slimmable.rs`](file:///home/fabio/nam-rs/src/models/slimmable.rs)
- **Descrição:**
  - Adicionar documentação detalhada (comentários de módulo ou de struct) explicando que o NAM-rs diverge intencionalmente do design de staging de C++ `SlimmableWavenet` (que usa `std::atomic<shared_ptr>`).
  - Documentar que o NAM-rs usa o canal de comunicação SPSC GC (Single Producer Single Consumer Garbage Collector) para transferir a desalocação do modelo antigo para a thread principal de housekeeping/Pipewire, mantendo o thread de áudio livre de locks e contenção.
- **Risco:** Baixo. Apenas documentação e alinhamento de design.
- **Nota de conclusão:** Documentação de módulo expandida com seção "Architectural divergence from C++ NAM — SlimmableWavenet staging" (~44 linhas). Cobre: (1) problema do `std::atomic<shared_ptr<WaveNet>>` do C++ (destrutor pode rodar na thread RT); (2) pipeline SPSC GC do NAM-rs (`gc_cascade` → `drain_gc_channels`) com RT thread como produtor e main thread como consumer; (3) mecanismo de overflow via `parking_lot` de 16 slots; (4) lifecycle completo de um swap slimmable (slice → replace → gc_cascade). Apenas comentários de documentação, sem alterações de código.

---

### Tarefa 6. [TEST] Testes de Integração de Reset Seletivo e Zero-Alloc para ContainerModel (F9, F10) [TODO]

- **Status:** `[ ]` **Pendente**
- **Arquivos Alvo:**
  - [`tests/container_slimmable.rs`](file:///home/fabio/nam-rs/tests/container_slimmable.rs)
  - [`tests/zero_alloc_infer.rs`](file:///home/fabio/nam-rs/tests/zero_alloc_infer.rs)
- **Descrição:**
  - Criar novos testes ou expandir os testes existentes para assegurar que:
    1. O `ContainerModel::reset` redefina apenas o sub-modelo ativo.
    2. A mudança de tamanho através de `set_slimmable_size` resete o modelo destino com sucesso.
    3. Nenhuma alocação ocorra no heap ao mudar de modelo através do `set_slimmable_size` (garantido pelo teste com `TrackingGuard`).
- **Risco:** Baixo.
