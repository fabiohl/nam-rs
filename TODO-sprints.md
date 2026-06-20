<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento de Sprints e Tarefas Técnicas

> **Skill:** `planejador-arquiteto`
> **Foco:** ÉPICO A — Sincronização Documentação ↔ Implementação (paridade real) (conforme `TODO-audit.md`)

---

## ÉPICO A — Sincronização Documentação ↔ Implementação (paridade real)

**Objetivo Central:** Garantir que a documentação (em especial `docs/cpp_parity_map.md`, `docs/architecture.md` e `docs/testing.md`) reflita com total precisão o estado real da engine, que agora suporta engines dinâmicos e arquitetura ConvNet por padrão.
**Criticidade:** 🟢 Baixo Risco Técnico, mas **CRÍTICO** para o alinhamento arquitetural e para balizar os próximos Épicos (B, C, D e E). Requer leitura atenta de código (cross-check).

---

### Sprint A.1: Atualização do Mapeamento de Paridade (C++ Parity Map)

**Foco:** Refatorar o `docs/cpp_parity_map.md` para remover falsas afirmações de rejeição/remoção e mapear corretamente os novos módulos (WaveNet Dyn, LSTM Dyn, WaveNetA2 Dyn, ConvNet) e a natureza dos goldens A2.
**Findings Relacionados:** F1, F2

* [x] **Tarefa A.1.1: Atualizar seções de WaveNet e LSTM Dinâmicos (`docs/cpp_parity_map.md`)**
  * **Especialista:** `documentador`
  * **Ação:** Em `docs/cpp_parity_map.md`, seção `3.3 Legacy Dynamic WaveNet (removed)` e `9. A1 Topology Table`:
    * Remover qualquer linguagem que afirme que `WaveNet Dyn` ou `LSTM Dyn` foram removidos.
    * Alterar o título da Seção 3.3 para refletir o suporte real aos engines dinâmicos.
    * Adicionar entradas para `WaveNetModelDyn`, `LstmModelDyn` e `WaveNetA2Dyn` nas tabelas correspondentes, indicando que operam como fallback genérico para geometrias livres, `condition_size ≠ 1`, post-stack heads e outros cenários dinâmicos.
    * Adicionar à matriz o mapeamento de `ConvNet` para o respectivo fonte C++ (`convnet.cpp`).
* [x] **Tarefa A.1.2: Corrigir documentação sobre `condition_size` e `head` (`docs/cpp_parity_map.md`)**
  * **Especialista:** `documentador`
  * **Ação:** Em `docs/cpp_parity_map.md`, seção `10.1 Architecture`:
    * Modificar a divergência que afirma que `condition_size ≠ 1` ou `head` não-nulo são rejeitados no carregamento.
    * Documentar que essas propriedades são agora capturadas por `loader/nam_json/topology.rs` (`get_wavenet_topology` → `Free`) e direcionadas para processamento seguro via `WaveNetModelDyn`.
* [ ] **Tarefa A.1.3: Reconciliar a natureza dos goldens A2 (`docs/cpp_parity_map.md`)**
  * **Especialista:** `documentador`
  * **Ação:** Na seção `5. A2 Architecture (Fixed fast-path port)` de `docs/cpp_parity_map.md`:
    * Remover ou reformular o aviso "C++ Live Cross-Validation Blocked (Upstream Bug) … self-golden pattern", uma vez que os testes atuais para A2 já utilizam goldens reais compilados (`tests/fixtures/golden_gen_build.sh`).
    * Alinhar o texto com o conteúdo de `tests/fixtures/README.md` que indica SNR/ESR reais em cross-reference. Se ainda houver limitações específicas (ex: falhas upstream sob sample rates != 48kHz), detalhá-las com precisão no lugar de um alerta global de bloqueio.

---

### Sprint A.2: Atualização da Arquitetura e Estratégia de Testes

**Foco:** Garantir que o `docs/architecture.md` cite os módulos dinâmicos e que a flag `dynamic-engine` seja bem esclarecida. Atualizar a matriz de testes no `docs/testing.md` para incluir a cobertura pretendida.
**Findings Relacionados:** F1

* [ ] **Tarefa A.2.1: Incluir Dinâmicos e ConvNet na Arquitetura (`docs/architecture.md`)**
  * **Especialista:** `documentador`
  * **Ação:** Em `docs/architecture.md`, nas seções de dispatch e descrição da Microarquitetura SIMD:
    * Mencionar o suporte ativo à arquitetura `ConvNet` e aos modelos dinâmicos (`WaveNetModelDyn`, `LstmModelDyn`).
    * Explicar como o `StaticModel` realiza o routing dinâmico por trás da enum, sem penalidade de vtable mas acomodando blocos imprevisíveis de modelos não classificados ("free-shape").
* [ ] **Tarefa A.2.2: Documentar o escopo da feature flag `dynamic-engine` (`docs/architecture.md`)**
  * **Especialista:** `documentador`
  * **Ação:** No arquivo `docs/architecture.md` (na seção de flags/condicionais):
    * Deixar explícito que a feature `dynamic-engine` do Cargo **NÃO** controla se os paths dinâmicos (WaveNetModelDyn, etc.) são compilados.
    * Aclarar que o real escopo desta flag é controlar um ramo escalar interno do path rápido do `WaveNetA2<CH>`. Os engines dinâmicos principais permanecem sempre compilados.
* [ ] **Tarefa A.2.3: Atualizar Matriz de Cobertura (`docs/testing.md`)**
  * **Especialista:** `documentador`
  * **Ação:** Em `docs/testing.md`:
    * Adicionar ou atualizar as descrições na tabela de matriz de rastreabilidade (ex: `golden_vectors`, `soak_test`, `cpp_parity`) para citar explicitamente a validação de `ConvNet` e motores dinâmicos, refletindo o objetivo de testes que será implementado nos próximos épicos.
