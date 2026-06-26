<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Sprints (NAM-rs)

Este documento contém o planejamento de sprints e tarefas técnicas estruturadas para o desenvolvimento do NAM-rs, garantindo paridade total com o NeuralAmpModelerCore v0.5.4 para os Épicos F, H e K.

---

## Sprint 4: Épicos F e H — "API Slimmable Breakpoints e Fixture A2" (F5, F6)

**Escopo:** Expor os breakpoints de transição do `SlimmableModel` para uso em hosts e plugins CLAP, e introduzir o modelo oficial `A2.nam` como fixture para testes de integração de paridade C++ e consistência.
**Objetivo de Paridade:** Permitir que o plugin identifique onde as transições de qualidade ocorrem, permitindo mapeamento e snapping preciso de parâmetros discretos de tamanho do modelo, além de garantir que a nova topologia slimmable seja testada contra o comportamento de referência do C++.
**Estimativa:** 1 sprint.
**Risco Geral:** 🟢 Baixo — Alterações locais focadas na API de breakpoints e adição de fixtures de teste, sem interferência no hot-path de processamento de áudio existente.

---

### Tarefa 1. [MODEL] Definir a API de Breakpoints no Trait `SlimmableModel` (F5) [DONE]

- **Status:** `[x]` **Concluída** — `fn slimmable_breakpoints(&self) -> Vec<f64>` adicionado ao trait `SlimmableModel` com default `vec![]`.
- **Arquivos Alvo:**
  - [`src/models/slimmable.rs`](file:///home/fabio/nam-rs/src/models/slimmable.rs)
- **Descrição:**
  - Adicionar a assinatura `fn slimmable_breakpoints(&self) -> Vec<f64>` ao trait `SlimmableModel`.
  - Definir o retorno padrão como `vec![]` para manter a compatibilidade com modelos que não utilizam submodelos ou breakpoints discretos.
- **Risco:** Baixo. Modificação simples de interface.

---

### Tarefa 2. [MODEL] Implementar Breakpoints em `ContainerModel` (F5) [DONE]

- **Status:** `[x]` **Concluída** — `slimmable_breakpoints()` implementado em `impl SlimmableModel for ContainerModel`, retornando `max_value` de todos submodelos (exceto o último) convertidos para `f64`. Paridade com `ContainerModel::GetSlimmableSizeBreakpoints()` do C++.
- **Arquivos Alvo:**
  - [`src/models/container.rs`](file:///home/fabio/nam-rs/src/models/container.rs)
- **Descrição:**
  - Implementar o método `slimmable_breakpoints` para `ContainerModel` no bloco `impl SlimmableModel for ContainerModel`.
  - Retornar os limites (`max_value`) dos submodelos ordenados, exceto o último, convertidos para `f64`, em total paridade com `ContainerModel::GetSlimmableSizeBreakpoints()` do C++.
- **Risco:** Baixo. A lista `submodels` já é validada e ordenada na construção do `ContainerModel`.

---

### Tarefa 3. [MODEL] Delegar Breakpoints no `StaticModel` e `NamModel` (F5) [DONE]

- **Status:** `[x]` **Concluída** — `fn slimmable_breakpoints(&self) -> Vec<f64>` adicionado ao trait `NamModel` com default `vec![]`. Método inerente e trait-implementado em `StaticModel` delegando para `ContainerModel` via `SlimmableModel::slimmable_breakpoints()`, retornando `vec![]` para demais variantes.
- **Arquivos Alvo:**
  - [`src/models/static_model.rs`](file:///home/fabio/nam-rs/src/models/static_model.rs)
  - [`src/models/mod.rs`](file:///home/fabio/nam-rs/src/models/mod.rs)
- **Descrição:**
  - Adicionar `fn slimmable_breakpoints(&self) -> Vec<f64>` ao trait `NamModel` (com padrão `vec![]`).
  - Adicionar o método público/delegado correspondente em `StaticModel`. Se o modelo for `StaticModel::Container(c)`, delegar para o container; caso contrário, retornar `vec![]`.
- **Risco:** Baixo.

---

### Tarefa 4. [TEST] Cobertura de Testes para Slimmable Breakpoints (F5) [DONE]

- **Status:** `[x]` **Concluída** — 10 testes adicionados em `tests/container_slimmable.rs`: unitários com LSTM dummy (1/2/3 submodelos, não-Container, roundtrip via NamModel/SlimmableModel/inerente, edge cases) + integração com A2 fixtures. Todos os caminhos cobertos: `SlimmableModel::slimmable_breakpoints()` direto, `NamModel::slimmable_breakpoints()` via `StaticModel::Container`, e método inerente de `StaticModel`.
- **Arquivos Alvo:**
  - [`tests/container_slimmable.rs`](file:///home/fabio/nam-rs/tests/container_slimmable.rs)
- **Descrição:**
  - Criar testes de unidade ou integração para instanciar um `ContainerModel` (ou usar fixtures existentes) e certificar-se de que `slimmable_breakpoints()` retorna os valores corretos.
- **Risco:** Baixo.

---

### Tarefa 5. [TEST] Copiar e Integrar `A2.nam` como Fixture Oficial (F6) [DONE]

- **Status:** `[x]` **Concluída** — `A2.nam` copiado para `tests/fixtures/models/a2_example.nam`. Teste de determinismo `test_auto_consistency_a2_example_slimmable` adicionado em `tests/self_consistency.rs`.
- **Arquivos Alvo:**
  - `tests/fixtures/models/a2_example.nam` (Novo arquivo - cópia)
  - [`tests/self_consistency.rs`](file:///home/fabio/nam-rs/tests/self_consistency.rs)
- **Descrição:**
  - Copiar o modelo de exemplo original `A2.nam` de `tests/fixtures/NeuralAmpModelerCore/example_models/A2.nam` para `tests/fixtures/models/a2_example.nam`.
  - Registrar e mapear esse modelo em `tests/self_consistency.rs` para validar que o loader do NAM-rs analisa perfeitamente a estrutura de `SlimmableContainer` oficial distribuída no core do C++.
- **Risco:** Baixo.

---

## Sprint 5: Épico K — "Metadados e Parser Linear" (F11, F12)

**Escopo:** Expor a API de metadados de loudness e níveis (input/output level) e implementar o parser do campo `implementation` no modelo Linear de forma case-insensível.
**Objetivo de Paridade:** Garantir que o host DAW ou plugins possam inspecionar os níveis RMS de calibração originais do modelo, e corrigir o parser de Linear para aceitar `"auto"`, `"direct"`, `"fft"`, `"legacy"` em letras minúsculas (como gerado oficialmente pelo exportador do C++).
**Estimativa:** 1 sprint.
**Risco Geral:** 🟢 Baixo — Focado em melhorias de parsing e adição de métodos de leitura à API pública.

---

### Tarefa 1. [MODEL/LOADER] Corrigir Parser de `LinearImplementation` para Case-Insensitive (F12)

- **Status:** `[ ]` **Não Iniciada**
- **Arquivos Alvo:**
  - [`src/loader/nam_json/model.rs`](file:///home/fabio/nam-rs/src/loader/nam_json/model.rs)
- **Descrição:**
  - Modificar a implementação de `std::str::FromStr` para `LinearImplementation`.
  - Converter a string de entrada para lowercase antes do match, permitindo que `"auto"`, `"direct"` e `"fft"` (em minúsculas, conforme o JSON real de exportação) sejam interpretados corretamente.
- **Risco:** Baixo. Corrige o bug de fallback silencioso para `Auto` quando as strings do JSON vêm em minúsculas.

---

### Tarefa 2. [LOADER] Expor Metadados de Níveis em `LoadedModelPair` (F11)

- **Status:** `[ ]` **Não Iniciada**
- **Arquivos Alvo:**
  - [`src/loader/loaded_model_pair.rs`](file:///home/fabio/nam-rs/src/loader/loaded_model_pair.rs)
- **Descrição:**
  - Implementar métodos auxiliares de leitura em `LoadedModelPair`:
    - `pub fn loudness(&self) -> Option<f32>`
    - `pub fn input_level_dbu(&self) -> Option<f32>`
    - `pub fn output_level_dbu(&self) -> Option<f32>`
    - `pub fn has_loudness(&self) -> bool`
    - `pub fn has_input_level_dbu(&self) -> bool`
    - `pub fn has_output_level_dbu(&self) -> bool`
  - Estes métodos extraem diretamente os valores de `self.metadata` quando disponíveis.
- **Risco:** Baixo. Sem qualquer impacto no pipeline de processamento RT.

---

### Tarefa 3. [TEST] Testar Metadados e Parser Case-Insensitive (F11, F12)

- **Status:** `[ ]` **Não Iniciada**
- **Arquivos Alvo:**
  - [`src/loader/nam_json_test.rs`](file:///home/fabio/nam-rs/src/loader/nam_json_test.rs)
- **Descrição:**
  - Escrever testes unitários verificando se metadados de calibração são expostos corretamente através das novas funções públicas.
  - Testar o parser de Linear com `"implementation": "auto"` (minúsculo) e confirmar que o enum `LinearImplementation::Auto` é mapeado com sucesso em vez de falhar.
- **Risco:** Baixo.
