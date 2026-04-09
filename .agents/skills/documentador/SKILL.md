---
name: documentador
description: Especialista em documentação técnica e arquitetural. Garante que o conhecimento do projeto (arquitetura e requisitos) esteja sempre sincronizado com a implementação.
---

# Skill: Documentador

## When to use this skill

Use esta skill ao final de cada ciclo de desenvolvimento ou quando houver necessidade de manter a "fonte da verdade" do projeto atualizada. Deve ser ativada ao receber solicitações expressas para documentar o sistema, mormente em mudanças de DSP, SIMD Vetorial (WaveNet, LSTM) ou algoritmos FastMath, mitigando perda de sabedoria endêmica da arquitetura.

## Instructions

### 1. Análise de Impacto

Antes de consolidar a documentação final ou propor mudanças, identifique quais documentos em `docs/` e diretrizes de IAs (`.agents/`) são afetados. Assegure total rastreabilidade da documentação nativa do repositório.

### 2. Hierarquia de Documentos (ordem de prioridade)

1. **`docs/architecture.md`** — é a **bíblia atual do projeto** e a única fonte de verdade sobre arquitetura. Toda mudança arquitetural deve ser refletida aqui **primeiro**.
2. **`README.md`** — visão geral para novos colaboradores (instalação, uso básico, dependências de sistema).
3. **`docs/NAM-rs-referência.md`** e **`docs/NAM-rs-sprints.md`** — documentação histórica e roadmap de sprints. Editar apenas de forma cirúrgica (notas e observações). Não contradizer `architecture.md`.
4. **`.agents/`** — definições de IA. Atualizar quando a arquitetura evoluir de forma que as skills descrevam padrões obsoletos.

### 3. Boas Práticas

- Nunca apague arquivos contendo a matriz mental do sistema de tempo real sem realocação sistemática.
- Edição do arquivo `docs/NAM-rs-referência.md` só é permitida se for cirúrgicas, ou na forma de notas e observações.
- Aproveite a documentação para justificar decisões de design (por que SCHED_FIFO? por que `#[repr(align(128))]`? por que `rubato 0.16` e não 2.0?). Isso evita regressões arquiteturais futuras.
