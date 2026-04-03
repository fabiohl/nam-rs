---
name: pesquisador-inovador
description: Use esta habilidade para pensar além, inovar e criar soluções proativas além do óbvio, visando modernização e alta performance.
---

# Skill: Pesquisador Inovador

## When to use this skill

Use quando as tarefas exigirem pesquisa de ponta para ecossistema Linux Audio (PipeWire), Kernel (io_uring), SIMD (DSP algorithms), Lock-Free concurrency ou CLI otimizadas. Proponha implementações de baixíssima latência e altíssima performance (nesta ordem).

## Instructions

### 1. Performance First

- Toda sugestão deve considerar o impacto realístico em processamento matemático, branch prediction, L1/L2 cache locality e latência sub-milisegundo.
- Analise o uso de `std::simd` (Portable SIMD) para processamento DSP intenso ou detecção numérica rápida de silêncio se for o caso.
- Considere a otimização extrema nos caminhos de recepção de bytes e nos Ring Buffers (ex: contornando bounds checks sem abrir brechas de segurança, com asserts estáticos).

### 2. Linux Kernel e Audio Architecture

- Sugira inovações baseadas ou combináveis com isolamento de Core (Core Affinity), Isolcpus e flags assíncronas de gravação direta via disco (`io_uring`, pre-alocação otimizada como `fallocate`).
- Mapeie arquiteturas resilientes que suportem perfeitamente o dynamic roteamento no PipeWire (re-roteamento de streams nativo ou mudança abrupta de taxa de bits sem pânicos em lock).
- Esteja atentos às novidades em toda a stack envolvida (Kernel Linux, Pipewire, etc).

### 3. CLI Minimalista e Feedback de Estado

- Adote abordagens "Zero Overhead" e que não oneram IO para imprimir na CLI. A CLI e o backend de tempo real devem estar rigorosamente isolados para não competirem pelos mesmos schedulers críticos da CPU.
