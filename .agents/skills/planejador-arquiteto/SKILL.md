---
name: planejador-arquiteto
description: Use esta habilidade atuando como um painel multi-disciplinar (no caso, as disciplinas envolvidas na demanda) de cientistas, arquitetos e engenheiros sêniores, além de especialistas de UX e de negócios.
---

# Skill: Planejador Arquiteto

## When to use this skill

Use esta skill antes de sair programando, focado em **Definição, Requisitos e Estratégia (Upstream)**. Esta skill gera planos de ação robustos (como `task.md` e `implementation_plan.md`) e destrincha demandas macro em micro-tarefas detalhadas.
Novas demandas acionadas pelo Agent Manager do Google Antigravity (ou prompt inicial equivalente em outras IDEs) deve acionar esta skill.

## Instructions

### 1. Foco em Definição e Requisitos

- Seja visionário e pragmático. Entenda profundamente as necessidades do produto lendo os artefatos:
  - `docs/architecture.md` — Visão arquitetural e decisões fundamentais do áudio Bit Perfect.
  - `.agent/rules/rust.md` — Regras inegociáveis de operação e domínio.
  - `TODO.txt` — Backlog e intenção de sprint atual.
- Traduza as dores do usuário em "soluções técnicas viáveis e documentadas". A arquitetura de áudio real-time requer planejamento rigoroso de dependências (sem bloqueios mútuos) entre as threads.

### 2. Geração de Artefatos (Planos)

- Quebre demandas complexas em passos menores e atômicos criando uma lista de tarefas bem definida.
- Produza planos claros que possam ser lidos e consumidos pelas personas de negócio ou pela skill `implementador`.

### 3. Consciência Arquitetural do Projeto

Toda decisão de nova arquitetura ou plugin desenhada em seu plano deve prever isolamento do Ring Buffer restrito a Produtor Único e Consumidor Único (SPSC), e prever como isso será integrado ao `io_uring`, garantindo resiliência ao Graceful Shutdown.

### 4. Atividades finais

Ao final, para concluir, sempre acione a skill `documentador` para garantir que a documentação esteja atualizada e rode os lints via utils/lints.sh e só encerre quando não houver mais nenhum erro.
