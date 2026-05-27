---
trigger: glob
description: Diretrizes mandatórias de garantia de qualidade (Linting) para o encerramento das submissões da IA.
globs: **/*.rs, **/*.toml
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Qualidade e Linting ao Fim das Atividades

1. **Compilação incremental**: Execute `cargo check` e `cargo build` em momentos oportunos durante o trabalho.
2. **Documentação**: Se houve alteração arquitetural relevante, acione a skill `documentador`.
3. **Testes (se `.rs` alterado)**: `cargo test` — nenhuma quebra de funcionalidade.
4. **Benchmarks (se `.rs` alterado com objetivo de performance)**: `cargo bench` — verificar ganho ou ao menos não-regressão.
5. **Correção exaustiva**: Analise cada fase e só prossiga quando passar sem erros, warnings ou mensagens suspeitas. Ciclo: identificar fonte → corrigir → reexecutar.
6. **Higiene do repo**: Remova arquivos temporários, logs ou artefatos de debug não listados no `.gitignore`.
