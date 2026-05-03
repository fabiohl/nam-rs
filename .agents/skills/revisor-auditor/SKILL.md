---
name: revisor-auditor
description: Painel de auditores, caçadores de bugs, cientistas, engenheiros sêniors e especialistas em diversas disciplinas associadas ao projeto.
---

# Skill: Revisor Auditor

## When to use this skill

Use para inspecionar, revisar e diagnosticar estrita aderência arquitetural e correta compatibilidade com a implementação de referência [Neural Amp Modeler Core](https://github.com/sdatkinson/NeuralAmpModelerCore).
Use também para revisar e auditar o projeto como todo em prol de correções de bugs, melhorias de segurança e performance.

## Instructions

Vide `.agents/rules/rust.md` para diretrizes técnicas mandatórias (RT-Safety, SIMD, SPSC).

### 1. Ingestão de Referenciais

Revise seu contexto mental com base nos seguintes documentos:

- **Regras de código**: `.agents/rules/rust.md` (condições inegociáveis de RT-safety).
- **Arquitetura atual**: `docs/architecture.md` (fonte primária de verdade).

### 2. Vetores de Auditoria

Inspecione linha-de-código detectando categoricamente:

- **Violações RT (Heap, I/O, Locks)**: Verifique se `.process()` ou funções auxiliares invocadas pelo callback incorrem em alocações, syscalls bloqueantes ou travas.
- **Microarquitetura**: Certifique-se do uso de SoA, `const generics` e alinhamento `#[repr(align(128))]` em estruturas compartilhadas.
- **Resampling e Ganhos**: Valide se o `NamResampler` opera em bypass quando as taxas coincidem e se as compensações de ganho refletem os metadados do modelo.
- **Fidelidade Numérica**: Verifique se os testes de inferência possuem assertivas numéricas com tolerâncias adequadas.

### 3. Retificações Bit-Perfect

Submeta patches cirurgicamente, sem alterar padrões consolidados. A operação final exige validação incondicional pelas skills de linting e testes. Tudo deve operar nas janelas perfeccionistas de baixa latência.

### 4. Acionar o Planejamento e Execução

- Acione a skill `planejador-arquiteto` para transformar as ideias levantadas em sprints e tarefas técnicas granulares em `TODO-sprints.md`.
