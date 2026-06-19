<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Planejamento de Sprints e Tarefas Técnicas (TODO-sprints.md)

> **Mega-Tópico:** MT2 — 🟠 Motor A2 Geral (F3 + F8 + F9)
> **Data do Planejamento:** 19/jun/2026
> **Skills Envolvidas:** revisor-auditor, planejador-arquiteto

## Visão Geral e Riscos

O Mega-Tópico 2 visa implementar a generalização completa da arquitetura A2 do Neural Amp Modeler Core, permitindo configurações não-padronizadas que atualmente são rejeitadas ou que quebram a paridade com o C++ v0.5.3.

**Riscos Identificados:**

1. **Regressão de Performance:** A principal ameaça é degradar o fast-path atual (`WaveNetA2<3>` e `WaveNetA2<8>`). A solução híbrida (A2 Dinâmico vs A2 Const-Generic) é obrigatória.
2. **RT-Safety:** As novas ativações de gating/blending exigem buffers temporários intermediários. Isso deve ser pré-alocado no load (`new()`) para garantir zero alocações (`zero-alloc`) no hot-path.
3. **Complexidade de Dispatch:** O JSON precisa ser profundamente interpretado para determinar corretamente quando instanciar a versão dinâmica ou fazer downcast para o fast-path.

---

## Sprint 1: Parsing e Relaxamento de Topologia (A2 Geral)

**Objetivo:** Permitir que o parser e as checagens de arquitetura (JSON) deixem de rejeitar modelos A2 não-padronizados, mantendo a detecção estrita apenas para o fast-path const-generic.

- [x] **Tarefa 1.1: Refatoração de `topology.rs` para o Motor Dinâmico**
  - **Especialista:** `implementador` / `refatora-rust`
  - **Ação:** Em `src/loader/nam_json/topology.rs`, a função `is_a2_shape` aplica regras severas (`check_groups_are_1`, `check_activations_are_leaky_relu`, etc.). Modificar esta lógica para retornar variantes (ex: `A2Topology::FastPath(u8)` vs `A2Topology::Dynamic`).
  - **Requisito:** Modelos que usam `head1x1`, `layer1x1` com `groups>1`, ativações heterogêneas ou gating devem ser aceitos como `Dynamic`.

- [x] **Tarefa 1.2: Parsing da Biblioteca de Ativações (F8) e Gating**
  - **Especialista:** `implementador`
  - **Ação:** Implementar a desserialização avançada de `ActivationConfig` no loader. Portar a lógica para ler arrays de ativações JSON, mapeando para o enum `ActivationType` em `src/models/a2/activations.rs`.
  - **Requisito:** Obter a lista completa de ativações por camada (23 elementos) para instanciar as funções corretamente.
  - **Nota:** `NamLayerConfig::parse_activation_config()` é o ponto de entrada para o motor dinâmico (T3.1). Para usar, chamar com `num_layers=23` (ou usar `A2_NUM_LAYERS`). Retorna `LayerActivationConfig { activations, gating_modes, secondary_activations }`.

## Sprint 2: Fundações Numéricas e Gating/Blending (F8 + F9)

**Objetivo:** Preparar os blocos de processamento essenciais (convoluções agrupadas e modos de gating) que o motor dinâmico montará.

- [x] **Tarefa 2.1: Integração de Convoluções Agrupadas (F9)**
  - **Especialista:** `implementador` / `debugger`
  - **Ação:** O arquivo `src/models/a2/grouped_conv1d.rs` já possui a base vetorial AVX2 para `A2GroupedConv1d`. Conectá-la de forma fluida nas abstrações de camada (ou generalizar `src/models/a2/conv1d.rs`) para que a arquitetura instancie `Conv1dDyn` quando `groups == 1` ou `A2GroupedConv1d` quando `groups > 1`.
  - **Critério de Aceite:** Kernel depthwise (`groups == channels`) otimizado e ativado sob demanda.

- [ ] **Tarefa 2.2: Implementação Real do Gating e Blending**
  - **Especialista:** `implementador`
  - **Ação:** Dar vida ao arquivo `src/models/a2/gating.rs`. Implementar os métodos de `apply_gating` ou `apply_blending` que operam em canais adjacentes.
  - **Requisito RT-Safety:** Pré-alocar os buffers auxiliares na inicialização. Escrever laços branchless auto-vetorizáveis ou usar intrinsics equivalentes aos que usamos em ativações inline. Paridade total com `NAM/gating_activations.h`.

## Sprint 3: O Motor A2 Dinâmico (F3) e Golden Tests

**Objetivo:** Aglutinar as fundações em um motor escalável e híbrido, assegurando cross-validation.

- [ ] **Tarefa 3.1: Criação do `WaveNetA2Dyn`**
  - **Especialista:** `implementador` / `planejador-arquiteto`
  - **Ação:** Criar `src/models/a2/model/dynamic.rs`. Esta será a versão de `mod.rs` do A2, porém, alocada dinamicamente com base nos tamanhos detectados:
    - Suporte a `bottleneck != channels`.
    - Suporte aos pontos ativos para `head1x1` e `layer1x1`.
    - Chamadas dinâmicas para as ativações heterogêneas e gating/blending.
  - **Requisito:** Manter o SPSC / MirroredBuffer design para eficiência no anel de memória.

- [ ] **Tarefa 3.2: Dispatch e Integração no Core**
  - **Especialista:** `implementador`
  - **Ação:** No dispatcher central, se a topologia A2 for classificada como `Dynamic`, instanciar o novo `WaveNetA2Dyn`. Garantir que o fast-path continue intocado para os modelos Standard e Lite clássicos.

- [ ] **Tarefa 3.3: Golden Vectors e C++ Parity**
  - **Especialista:** `revisor-auditor` / `pesquisador-inovador`
  - **Ação:** Gerar golden vectors contra o C++ v0.5.3 (ex: usando `wavenet_a2_max.nam` ou modelos A2 sintéticos com gating e bottleneck) e atingir ESR/SNR dentro da margem de tolerância. Executar `utils/tests-long.sh`.
