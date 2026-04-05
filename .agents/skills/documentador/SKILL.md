---
name: documentador
description: Especialista em documentação técnica e arquitetural. Garante que o conhecimento do projeto (arquitetura e requisitos) esteja sempre sincronizado com a implementação.
---

# Skill: Documentador

## When to use this skill

Use esta skill ao final de cada ciclo de desenvolvimento ou quando houver necessidade de manter a "fonte da verdade" do projeto atualizada. Deve ser ativada ao receber solicitações expressas para documentar o sistema, mormente em mudanças de DSP, SIMD Vetorial (WaveNet, LSTM) ou algoritmos FastMath, mitigando perda de sabedoria endêmica da arquitetura.

## Instructions

### 1. Análise de Impacto

Antes de consolidar a documentação final ou propor mudanças, identifique quais documentos em `docs/` e diretrizes de IAs (.agents) são afetados. Assegure total rastreabilidade da documentação nativa do repositório ou de links associados.

### 2. Sincronização de Conhecimento Rust x DSP

- Qualquer adoção paramétrica inovadora ou refinamento envolvendo superamostragem `Sinc Interporlation` FIR temporal de fase linear, ganhos parametrizados Tone3000 `.namb` ou metadados e rotinas numéricas com AVX-512 devem estar presentes integralmente nas documentações (`docs/architecture.md`).
- A aplicação suprimiu antigas vias orientadas à persistência e roteamento em discos remotos (ex: AudioRip io_uring). Descarte referências inúteis na arquitetura.

### 3. Boas Práticas

- Foque na documentação visando a justificação de mitigações de memory cache misses e escalonamentos low latency no Host Linux restrito. Nunca apague arquivos contendo a matriz mental do sistema de tempo real sem realocação sistemática.
- Nunca edite os arquivos docs/NAM-rs-referência.md e docs/NAM-rs-sprints.md. Eles são documentos mestres mantidos pelo desenvolvedor humano do projeto.
