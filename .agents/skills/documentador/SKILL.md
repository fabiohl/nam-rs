---
name: documentador
description: Especialista em documentação técnica e arquitetural. Garante que o conhecimento do projeto (arquitetura e requisitos) esteja sempre sincronizado com a implementação.
---

# Skill: Documentador

## When to use this skill

Use esta skill ao final de cada ciclo de desenvolvimento ou quando houver necessidade de manter a "fonte da verdade" do projeto atualizada. Deve ser ativada ao receber solicitações expressas para documentar o sistema, mormente em mudanças de DSP, SIMD Vetorial (WaveNet, LSTM) ou algoritmos FastMath, mitigando perda de sabedoria endêmica da arquitetura.

## Instructions

Vide `.agents/rules/rust.md` para diretrizes técnicas mandatórias (RT-Safety, SIMD, SPSC).

### 1. Hierarquia de Documentos

1. **`docs/architecture.md`** — Bíblia de arquitetura e fonte primária de verdade.
2. **`README.md`** — Visão geral, instalação e uso.
3. **`.agents/`** — Definições de IA. Devem ser atualizadas se houver mudança nos padrões de implementação.

### 2. Princípios de Documentação

- **Guia, não substituto**: A documentação justifica o *porquê* das decisões e orienta o uso de patterns (diagnósticos, SPSC, RT-safety). Detalhes de implementação pertencem ao código.
- **Rastreabilidade**: Sempre aponte para o arquivo ou função no código-fonte (ex: "Veja `src/diagnostics.rs`").
- **DRY (Don't Repeat Your Code)**: Nunca duplique código verbatim na documentação. Explique o conceito e referencie o arquivo.
- **Sincronia**: Mantenha o catálogo de erros `Exxxx` sincronizado entre `docs/architecture.md` e o enum `NamErrorCode`.

### 3. Boas Práticas

- Justifique decisões críticas (ex: por que `SCHED_FIFO`? por que `#[repr(align(128))]`?) para evitar regressões por desconhecimento do histórico.
- Use a rule `.agents/rules/linting.md` para validar a higiene dos documentos gerados.
