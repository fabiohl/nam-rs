---
name: planejador-arquiteto
description: Use esta habilidade atuando como um painel multi-disciplinar (no caso, as disciplinas envolvidas na demanda) de cientistas, arquitetos e engenheiros sêniores, além de especialistas de UX e de negócios.
---

# Skill: Planejador Arquiteto

## When to use this skill

Use esta skill focando em **Planejamento Técnico e Arquitetura Inferencia Inegociável**. A matriz da arquitetura aborda problemas espaciais temporais (DSP) destrinchando as necessidades base do `NAM-rs` em tarefas ágeis via `implementation_plan.md` listando micro passos exatos de refatoração para a linguagem de máquina (SIMD/FMA).

## Instructions

### 1. Fundamentos e Diretrizes Analíticas

- Carregue o contexto denso proveniente de artefatos essenciais:
  - `docs/NAM-rs-referência.md` e manifesto `docs/NAM-rs-sprints.md`.
  - `.agents/rules/rust.md` sobre as condições inegociáveis de código.

### 2. Subdivisões do Motor Matemático e Concorrência

- Modele requisitos complexos das Redes Neurais convertendo os _models_ Tone3000 nativos suportando implementações `Const Generics`.
- Ao introduzir um novo fluxo do Host pipewire, proteja o acesso determinístico isolando estritamente em barreiras Single-Producer Single-Consumer (Ring Buffers SPSC).
- Abstrações não focadas à simulação matemática de amplificadores percussivos (tais quais rotinas herdadas como gravação em HD com _io_uring_ de arquivos) devem ser barradas ou planejados expurgos literais se encontrados.

### 3. Organização do Roteamento Computacional

- Especifique para a equipe técnica como e em qual parte exata a thread `Audio/DSP` `low latency` deve assimilar bibliotecas externas `fastmath` no lugar do STD base. Descreva os mecanismos dinâmicos contendo multiversioning SIMD compilados pelo `cargo build` e delineados.

### 4. Atividades finais

- Encerre solicitando à infraestrutura de dev (`lints.sh`) validação do plano macro na base estrita de sintaxe do rust antes de fechar sua sessão com a tag da skill `documentador`.
