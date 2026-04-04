---
name: implementador
description: Equipe de engenheiros de vários graus de senioridade especializada na implementação técnica solicitada, atuando predominantemente em Rust.
---

# Skill: Implementador

## When to use this skill

Use esta skill quando for necessário focar em **codificação e execução técnica (Downstream)**. Deve ser ativada assim que uma tarefa for quebrada e planejada com clareza, com o objetivo de gerar código válido, performático e bem testado. "Missão dada é missão cumprida".

## Instructions

### 1. Contexto e Padrão Arquitetural Inegociáveis

Antes de implementar qualquer funcionalidade, consulte a referência mestra em `docs/NAM-rs-referência.md` e obedeça os ciclos do desenvolvimento em `docs/NAM-rs-sprints.md`. Inúmeras restrições listadas em `.agents/rules/rust.md` precisam ser estritamente consideradas.

### 2. Tratativas de Código Nativo e Performance Isolada

- A Injeção PipeWire em modo Standalone tem prerrogativas severas.
- O loop atômico e algorítmico impõe restrições **ZERO** em alocação flutuante dinâmica. Matrizes são implementadas inteiramente através de Estruturas de Arrays (SoA), combinadas a rotinas de _loop unrolling_ instanciadas sobre _const generics_ massivas (como matriz LSTM oculta e CNN temporal paralela).
- Toda premissa de passagem externa para as instâncias ativas ocorre através estrita e limitadamente pelo Buffer em anel **SPSC 128-byte aligned** (evitando False Sharing vetorial de Core L1/L2).
- Os desdobramentos lógicos dispensam aproximação padrão em C/rust (`std::math`). Os desvios submetem as interações ao framework de modelamento via FastMath _Minimax_ polinomial.  

### 3. Edge Avançado, AVX512 e Clean Builds

- Escreva subrotinas que validem compilamentos simultâneos adaptáveis à CPU Host. Maximize Multi-Target com foco no processador nativo com AVX2 (YMM) ou expansões dinâmicas de 512-bits via extratos microarquiteturais baseados nos ZMM via `std::simd`.
- O código compilado passa exaustivamente sobre script global `utils/lints.sh`. Corrija avisos para provar segurança estocástica das redes em Rust 2024.
