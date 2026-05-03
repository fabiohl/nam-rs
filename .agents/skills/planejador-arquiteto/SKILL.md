---
name: planejador-arquiteto
description: Transformar planos em Sprints e Tarefas técnicas. Painel multi-disciplinar (no caso, as disciplinas envolvidas na demanda) de cientistas, arquitetos e engenheiros sêniors, além de especialistas de UX e de negócios.
---

# Skill: Planejador Arquiteto

## When to use this skill

Use esta skill focando em **Planejamento técnico sob metodologias ágeis**. Quebre entregas maiores em tarefas menores atômicas direcionadas aos especialistas capazes de cumpri-las com perfeição. Assegure uma entrega coesa e perfeitamente atendente ao que foi solicitado.

## Instructions

Vide `.agents/rules/rust.md` para diretrizes técnicas mandatórias (RT-Safety, SIMD, SPSC).

Organizar as atividades em Sprints e Tarefas Técnicas coerentes, auto-contidas e granulares em `TODO-sprints.md`. Cada tarefa deve ser detalhada o suficiente para que qualquer agente possa implementá-la sem ambiguidades.

### 1. Fundamentos Analíticos

Ao planejar, carregue o contexto denso dos seguintes artefatos:

1. `docs/architecture.md` — Fonte primária de verdade arquitetural.
2. `.agents/rules/rust.md` — Condições inegociáveis de código.
3. `TODO-sprints.md` — Estado atual do backlog.

### 2. Sincronização

- Ao concluir o planejamento, valide a consistência com a rule `.agents/rules/linting.md`.
- Acione a skill `documentador` para garantir que as documentações acompanhem a evolução prevista.
