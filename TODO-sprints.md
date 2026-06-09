<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints — Plano de Sprints

---

## Épico: Documentação e Rastreabilidade (gerado por `/refatora-doc`)

Objetivo: garantir que toda a documentação do projeto (docs/*.md, comentários de código, testes) reflita com precisão o estado real do código, eliminando divergências que causam confusão para novos colaboradores e auditores.

---

### Sprint D1 — Sincronização docs ↔ código (prioridade imediata)

#### Tarefa D1.T01 — Verificar e atualizar referências de `file://` em `docs/architecture.md`

- **Onde:** [`docs/architecture.md`](docs/architecture.md) — seções §8.2.1 e §8.2.2.
- **Problema:** Os links de código em `architecture.md` (e.g. `[PluginAudioProcessor::process](file:///home/fabio/nam-rs/src/clap/processor/mod.rs#L178-L242)`) usam caminhos absolutos à máquina do desenvolvedor. Em outros ambientes, os números de linha podem estar desatualizados.
- **Solução técnica:**
  1. Auditar todos os links `file:///...` no documento e confirmar que os números de linha batem com o código atual.
  2. Usar referências a funções (`fn process_dsp_audio`) em vez de números de linha fixos onde possível, para robustez futura.
  3. Links de markdown interno (e.g. `docs/clap_integration.md`) devem ser convertidos para o esquema `file://` absoluto conforme o padrão do projeto.
- **Critérios de aceitação:**
  - Todos os links `file://` em `docs/architecture.md` resolvem corretamente e apontam para as linhas exatas da implementação atual.
- **Especialista:** `documentador`.
- **Esforço:** 0,5 dia.

---

#### Tarefa D1.T02 — Auditar e corrigir `docs/clap_integration.md` — §7 Target DAWs

- **Onde:** [`docs/clap_integration.md`](docs/clap_integration.md) — §7 "Target DAWs for Validation".
- **Problema:** O documento menciona "Fender Studio Pro" como alvo futuro requerendo "Wayland native mode". Verificar se esse DAW ainda é alvo ativo ou se a nomenclatura/estratégia mudou.
- **Solução técnica:**
  1. Confirmar o nome correto do DAW e seu status de suporte a CLAP em Wayland.
  2. Atualizar o texto conforme a realidade atual (alvo ativo, descontinuado ou especulativo).
- **Critérios de aceitação:**
  - A seção §7 descreve apenas DAWs ativamente testados ou planejados, com status explícito.
- **Especialista:** `documentador`.
- **Esforço:** 0,25 dia.

---

#### Tarefa D1.T03 — Documentar o comportamento adaptativo do VU meter (mono/stereo) em `docs/functional-tests.md` e `docs/clap_integration.md`

- **Onde:** [`docs/functional-tests.md`](docs/functional-tests.md) §2C; [`docs/clap_integration.md`](docs/clap_integration.md) §5 nota sobre mono.
- **Problema:** O meter da Zone 3 é adaptativo: exibe 1 barra centrada (mono) ou 2 barras L/R (stereo) com base em `active_channel_count` do host. A nota em `clap_integration.md` §5 ainda afirma que o plugin "opera estritamente em mono" sem mencionar que o meter reflete a contagem de canais do host.
- **Solução técnica:**
  1. Adicionar nota em `clap_integration.md` §5 esclarecendo que o DSP é mono, mas o VU meter exibe L/R quando o host declara ≥2 canais de saída.
  2. Confirmar como `active_channel_count` é populado pelo host e documentar o contrato.
- **Critérios de aceitação:**
  - A distinção entre "DSP mono" e "meter adaptativo" está claramente descrita em ambos os documentos.
- **Especialista:** `documentador`.
- **Esforço:** 0,25 dia.

---

#### Tarefa D1.T04 — Criar `docs/gui-architecture.md`: guia detalhado da arquitetura GUI

- **Onde:** novo arquivo `docs/gui-architecture.md`.
- **Problema:** O módulo `src/clap/gui/` cresceu significativamente (11 módulos, 3 submódulos, GPU shaders, status bar modular) e hoje está documentado apenas de forma superficial em `architecture.md` §8.3.1. Falta uma referência arquitetural dedicada que oriente contribuidores na estrutura da GUI.
- **Solução técnica:**
  1. Criar `docs/gui-architecture.md` com:
     - Diagrama de módulos (`ui/`, `window/`, `zones/`, `status_bar/`, `meter/`).
     - Ciclo de vida do frame: `on_frame()` → `draw_ui()` → zona por zona.
     - Protocolo de sincronização UI↔RT (atomics, geração, SPSC).
     - Estratégia de renderização condicional (idle skip, peak hold, throttle 22ms).
     - GPU VU meter: shaders GLSL, VAO, fallback CPU.
     - Fluxo do toast e do diagnóstico.
     - Regras de acessibilidade (Tab/focus cycle via `focus.rs`).
  2. Referenciar o novo doc em `architecture.md` §8.3 e `clap_integration.md` §8.
- **Critérios de aceitação:**
  - Um novo desenvolvedor consegue entender a estrutura GUI apenas lendo o documento, sem precisar navegar pelo código.
  - Todos os módulos de `src/clap/gui/` têm ao menos uma linha de descrição no guia.
- **Especialista:** `documentador` + `implementador`.
- **Esforço:** 1 dia.

---

### Sprint D2 — Melhoria de cobertura dos testes funcionais

#### Tarefa D2.T01 — Adicionar seção de teste para o VU meter stereo em `docs/functional-tests.md`

- **Onde:** [`docs/functional-tests.md`](docs/functional-tests.md) §2C.
- **Problema (levantado no /refatora-doc):** O §2C agora cobre o comportamento adaptativo (mono/stereo), mas não existe um roteiro de teste específico para inserir o plugin em uma faixa stereo e verificar os dois medidores L e R independentemente (sinal no L, silêncio no R; e vice-versa).
- **Solução técnica:**
  1. Adicionar checklist para teste com sinal apenas no canal L, verificar que barra R fica em mínimo.
  2. Adicionar checklist para sinal em fase e verificar que ambas as barras se movem simetricamente.
  3. Documentar como reprovar (e.g.: barras sempre iguais mesmo com sinal assimétrico = bug).
- **Critérios de aceitação:**
  - §2C contém ≥4 novos itens de checklist cobrindo o comportamento stereo.
- **Especialista:** `documentador` + UX specialist.
- **Esforço:** 0,5 dia.

---

#### Tarefa D2.T02 — Incluir checklist de teste para o "Open Folder" button no toast (§2L)

- **Onde:** [`docs/functional-tests.md`](docs/functional-tests.md) §2L.
- **Problema:** O §2L já menciona o botão "Open Folder", mas o roteiro não cobre o caso de falha (e.g.: `xdg-open` indisponível ou `HOME` não definido).
- **Solução técnica:**
  1. Adicionar checklist: `xdg-open` não instalado → botão desaparece ou produz mensagem de erro amigável (sem crash).
  2. Documentar o comportamento esperado em ambientes sem desktop (e.g.: servidor headless via SSH).
- **Critérios de aceitação:**
  - §2L tem ≥2 itens adicionais cobrindo falhas do `xdg-open`.
- **Especialista:** `documentador`.
- **Esforço:** 0,25 dia.

---

#### Tarefa D2.T03 — Adicionar número de seção explícito a todos os itens de checklist de `functional-tests.md`

- **Onde:** [`docs/functional-tests.md`](docs/functional-tests.md) — todos os blocos.
- **Problema:** O template de bug report (§ ao final) pede `<ID, e.g.: 2C.4>`, mas os checklists não têm numeração explícita nos itens. Isso torna difícil reportar um bug específico.
- **Solução técnica:**
  1. Adicionar numeração inline a cada item: `- [ ] **2C.1** Insert NAM-rs...`.
  2. Garantir que itens novos (dos testes D2.T01 e D2.T02) também recebam IDs.
- **Critérios de aceitação:**
  - Cada item de checklist tem um ID único no formato `<seção>.<número>`.
  - O template de bug report é atualizado com um exemplo de ID real.
- **Especialista:** `documentador`.
- **Esforço:** 0,5 dia.

---

### Sprint D3 — Rastreabilidade de decisões arquiteturais

#### Tarefa D3.T01 — Documentar a decisão arquitetural do meter adaptativo em `docs/architecture.md`

- **Onde:** [`docs/architecture.md`](docs/architecture.md) §8.3.1.
- **Problema:** A decisão de exibir L/R no VU meter com base em `active_channel_count` (em vez de fixar em mono, alinhado ao DSP mono) é uma decisão de UX/arquitetura não documentada. Sem contexto, futuros desenvolvedores podem "corrigir" isso e regredir a feature.
- **Solução técnica:**
  1. Adicionar um bloco de decisão técnica (similar ao §6.1 ou §5.1) explicando:
     - Motivação: informar o usuário do nível do sinal no canal processado e detectar desequilíbrios de roteamento no host.
     - Implementação: `shared.rt_to_ui.active_channel_count` populado pelo CLAP audio ports; `zones/meters.rs` usa esse valor para selecionar layout.
     - Trade-off: VU stereo com DSP mono não introduz custo RT (apenas o canal L é processado; R é peak do mesmo buffer ou zero).
- **Critérios de aceitação:**
  - `architecture.md` contém um bloco de decisão técnica explicando o VU adaptativo.
- **Especialista:** `documentador` + `revisor-auditor`.
- **Esforço:** 0,5 dia.

---

#### Tarefa D3.T02 — Adicionar ADR (Architecture Decision Record) para a estratégia de renderização condicional da GUI (G3.T01 idle reduce)

- **Onde:** [`docs/architecture.md`](docs/architecture.md) §8.3.1 ou nova seção §8.3.3.
- **Problema:** A estratégia de renderização condicional (skip frame quando idle, `should_skip` em `window/handler.rs`) é complexa e afeta: peak-hold decay, animações de toast, pulso de automação. Não está documentada como decisão arquitetural.
- **Solução técnica:**
  1. Adicionar bloco ADR com:
     - Contexto: GUI abre em loop 30ms mas DAW pode ter dezenas de plugins; CPU sem idle = queixa de usuário.
     - Decisão: frame skip quando `!dirty && !has_short_repaint && !hold_changed && time_since_paint < 22ms`.
     - Consequências: animações de toast e loading não podem depender apenas de `request_repaint_after`; devem chamar `request_repaint()` ativo enquanto ativas (já implementado).
- **Critérios de aceitação:**
  - O ADR é compreensível por um desenvolvedor Rust sem contexto de egui/baseview.
- **Especialista:** `documentador`.
- **Esforço:** 0,5 dia.
