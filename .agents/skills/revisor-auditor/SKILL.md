---
name: revisor-auditor
description: Painel de auditores, arquitetos, caçadores de bugs, cientistas, engenheiros sêniors e especialistas em diversas disciplinas associadas ao projeto nam-rs (Rust, Linux Low Latency, Pipewire, CLAP, DSP e redes neurais, etc).
---

# Skill: Revisor Auditor

## When to use this skill

Use esta skill para revisar geral o projeto em busca de melhorias.

## Instructions

* Analise profundamente todo o código em busca de oportunidades de melhorias. Exemplos (não exaustivos):
  * Bugs de aderência arquitetural, funcionalidade, segurança, performance, baixa latência, etc.
  * Inspecionar e diagnosticar estrita aderência arquitetural e correta compatibilidade com a implementação de referência [Neural Amp Modeler Core](https://github.com/sdatkinson/NeuralAmpModelerCore).
  * Códigos que podem ficar inline ou fora do hotpath;
  * Arquivos e funções com tamanho e organização lógica;
  * Revisão criteriosa do "budget de ciclos de código" atrás de mais otimizações para instruções modernas de cpu, mais resultados, por menos ciclos de clock, etc.
  * Cobertura integral de bons comentários de código fonte;
  * Documentação (skill `documentador`) exemplar.
* Acione a skill `planejador-arquiteto` para transformar as ideias levantadas em sprints e tarefas técnicas granulares, muitíssimo bem escritas e detalhadas, em `TODO-sprints.md`.
