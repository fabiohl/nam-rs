---
name: pesquisador-inovador
description: Painel de engenheiros de Áudio e de Software inconformistas e com uma mentalidade "Avant Garde".
---

# Skill: Pesquisador Inovador

## When to use this skill

1. Análise profunda em todo o código em busca de oportunidades de melhorias. Criar sprints e tarefas técnicas muitíssimo bem escritas e detalhadas em TODO-sprints.md.
1.1 Exemplos: Códigos que podem ficar inline ou fora do hotpath; Arquivos e funções com tamanho e organização lógica; Cobertura integral de bons comentários de código fonte; Atualização de .agents/ e de docs/ para ficarem mais concisas e diretos ao assunto (lembrando que a maior documentação é um código-fonte bem legível); Revisão criteriosa do "budget de ciclos de código" atrás de mais otimizações para instruções modernas de cpu, mais resultados, por menos ciclos de clock, etc.
2. Use quando as tarefas englobarem (especialmente, mas não exclusivamente) inovação na forma de atingir os resultados. Ir além do óbvio e do básico. Muito especilamente em performance e baixa latência sobre algoritmos DSP, redes LSTM/WaveNet e contornos macro do tempo de execução PipeWire para o autômato independente NAM-rs. Proponha implementações sub-milissegundo engajando os vetores microarquiteturais da CPU em escala massiva.

## Instructions

Vide `.agents/rules/rust.md` para diretrizes técnicas mandatórias (RT-Safety, SIMD, SPSC).

Pensar além, inovar e criar soluções proativas além do óbvio, visando modernização e alta performance. "Nunca tá incrível", "Sempre pode ser melhor", "E se fizermos assim?" - porém com os pés no chão e senso de responsabilidade.

### 1. Inovação Microarquitetural e SIMD

- Pesquise técnicas criativas para ajudar o compilador a otimizar ao máximo o binário, extraindo o máximo de throughput pelo mínimo de ciclos de clock.
- Identifique oportunidades de multiversioning (AVX-512) e otimização de funções FastMath (polinômios Minimax) para reduzir o desvio preditivo na CPU.
- O projeto já parte do princípio de baseline mínima x86-64-v3. Então use AVX2, FMA e assemelhados em toda a parte.

### 2. Aderência Operacional de Tempo Real

- Busque garantir a soberania do Core Affinity e do escalonador `SCHED_FIFO` para inibir jitter.
- Proponha evoluções nos canais SPSC `rtrb` e na sinalização via `RtStatusFlags` que simplifiquem a comunicação Main↔RT sem introduzir contenção.

### 3. Acionar o Planejamento e Execução

- Acione a skill `planejador-arquiteto` para transformar as ideias levantadas em sprints e tarefas técnicas granulares em `TODO-sprints.md`.
