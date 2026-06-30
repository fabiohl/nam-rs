<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Roadmap de Sprints — Épicos A, C & D

Este documento organiza o planejamento ágil e tarefas técnicas para o **Épico A (PM-01, PM-02, PM-08 — Sincronização Documental de Paridade)**, o **Épico C (PM-05 — Cobertura de Modelos Reais A2-FiLM)** e o **Épico D (PM-06 — SlimmableWavenet)** no `nam-rs`, com base nas descobertas consolidadas em `TODO-findings.md`.

---

## SPRINT S9 — Sincronização Documental e Alinhamento de Paridade (A2-FiLM & SlimmableWavenet)

### Objetivos da Sprint

1. Sincronizar toda a documentação de paridade ao estado real do motor, eliminando avisos obsoletos sobre a WaveNet Lite e corrigindo referências cruzadas mortas.
2. Formalizar a cobertura do motor A2 FiLM sob fixtures sintéticas devido à indisponibilidade de capturas reais compatíveis.
3. Definir a fronteira de escopo e os critérios de aceitação para o fatiamento de canais dinâmico (`SlimmableWavenet`), mantendo-o como item diferido.

---

### Tarefas Técnicas

#### [ ] Task S9.1 — Auditoria de Fixtures e Conformismo A2-FiLM (PM-05)

* **Responsável:** Engenheiro de DSP / QA
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  O motor `WaveNetA2Dyn` suporta FiLM, mas os modelos reais de FiLM disponíveis (como `wavenet_a2_max.nam` com `condition_size=8`) são rejeitados por incompatibilidade com a assinatura esperada pelo loader dinâmico de A2 (que exige geometrias específicas). Como não existem fixtures reais compatíveis no diretório `tests/fixtures/models-nondist/` nem em `tests/fixtures/models/`, a suíte de testes deve se conformar com as fixtures sintéticas `wavenet_a2_film_full.nam` e `wavenet_a2_film_lite.nam` para garantir a correção do motor matemático.
* **Critérios de Aceitação:**
  1. Confirmar que os testes de vetores dourados (`tests/golden_vectors.rs`) exercitam corretamente as fixtures sintéticas de FiLM.
  2. Verificar que o modelo real incompatível `wavenet_a2_max.nam` é rejeitado graciosamente e coberto pelo teste `test_loader_gap_wavenet_a2_max` sem quebras silenciosas.

#### [ ] Task S9.2 — Documentação de Conformismo FiLM A2 em `cpp_parity_map.md` (PM-05)

* **Responsável:** Documentador Técnico
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  Registrar formalmente no mapa de paridade (`docs/cpp_parity_map.md`, seção 13) o conformismo às fixtures sintéticas para validação do motor FiLM devido à ausência de capturas reais suportadas, garantindo rastreabilidade futura.
* **Critérios de Aceitação:**
  1. Atualizar a entrada de tabela **"A2 official real-amp FiLM captures"** no §13 para refletir o status de conformismo temporário com modelos sintéticos.
  2. Documentar o motivo técnico (incompatibilidade estrutural dos modelos reais existentes) no §13.1.

#### [ ] Task S9.3 — Fronteira de Escopo e Critérios para `SlimmableWavenet` Diferido (PM-06)

* **Responsável:** Arquiteto de Software
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  Garantir que o `SlimmableWavenet` (fatiamento de canais em runtime num único arquivo) permaneça diferido sem causar confusão com o `SlimmableContainer` (que já resolve a qualidade adaptativa via múltiplos modelos e crossfade).
* **Critérios de Aceitação:**
  1. Atualizar a tabela e notas do §13 em `docs/cpp_parity_map.md` delimitando o escopo do `SlimmableContainer` (pronto e testado) vs. `SlimmableWavenet` (diferido).
  2. Definir explicitamente os critérios de aceitação para eventual implementação futura do `SlimmableWavenet`:
     * Parser para ler múltiplas larguras de canal de um único arquivo `.nam`.
     * Fatiamento dinâmico de pesos em runtime de forma segura para tempo real (RT-safe).
     * Paridade matemática bit-a-bit com a implementação equivalente do NAMCore C++.

#### [ ] Task S9.4 — Resolução de Avisos Obsoletos da WaveNet Lite em `fastmath-approximations.md` (PM-02)

* **Responsável:** Documentador Técnico / Auditor
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  O arquivo `docs/fastmath-approximations.md` §9.4 contém um bloco `> [!CAUTION]` obsoleto e um título indicando que o Lite possui divergência estrutural de 0.9 dB. O bug real de alinhamento no buffer circular de delay lines (`MirroredBuffer`) foi sanado no código (`MirroredBuffer::new_aligned`) e a paridade do modelo real `EVH-5150-Lite.nam` está estabelecida em 122.3 dB. É necessário atualizar este trecho para refletir o estado de paridade estabelecida e documentar a causa-raiz resolvida.
* **Critérios de Aceitação:**
  1. Renomear a seção 9.4 para indicar a resolução (ex: "9.4 Lite Architectures — Resolved").
  2. Descrever detalhadamente a causa-raiz (arredondamento do buffer circular sem levar em conta o alinhamento de stride de canais para não-potências de dois) e sua correção (`MirroredBuffer::new_aligned`).
  3. Remover/substituir o bloco de cautela por uma nota de contexto histórico de resolução.

#### [ ] Task S9.5 — Correção de Referências Quebradas (PM-08)

* **Responsável:** Documentador Técnico
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  Eliminar referências mortas ao arquivo inexistente `TODO-problemas.md` na documentação do projeto, substituindo-as por referências corretas ao mapa de paridade e aos findings correspondentes de `TODO-findings.md`.
* **Critérios de Aceitação:**
  1. Mapear as referências de `TODO-problemas.md` a problemas reais e redefinir seus links:
     * `TODO-problemas.md:155` (silêncio/denormais) apontará para `fastmath-approximations.md` §6.
     * `TODO-problemas.md#P1` e `TODO-problemas.md:47` (Lite) apontarão para `docs/cpp_parity_map.md` §9.1 e `PM-02`.
     * `TODO-problemas.md:92` (asimetria) e `TODO-problemas.md:353` (lo-fi) apontarão para `docs/fastmath-approximations.md` §9.5.
