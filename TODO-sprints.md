<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento de Sprints e Tarefas Técnicas

Este documento organiza a execução das melhorias mapeadas em `TODO-findings.md` em sprints e tarefas técnicas estruturadas de forma ágil.

---

## Sprint 1: Polimento da Interface Standalone (CLI) e Logs

Foco em limpar redundâncias de parâmetros na CLI, melhorar a clareza das opções e enriquecer os logs de inicialização.

### Épico 1: Ajustes e Polimento de Interface CLI e Logs (F01, F03, F04, F06)

* **Risco/Criticidade:** Baixo. Alterações simples de parsing de argumentos e exibição na CLI.
* **Tarefas Técnicas:**

#### [T1.1] Simplificação dos Parâmetros de Oversampling (F01) [DONE]

* **Objetivo:** Remover o atalho `--os`, mantendo apenas `--oversample`.
* **Ações:**
  * Remover `--os` da ajuda em `print_help()` em `src/standalone/cli.rs`.
  * Remover correspondência de `Long("os")` em `parse_args_from()` em `src/standalone/cli.rs`.
  * Garantir que testes unitários continuem válidos e usem apenas `--oversample`.

#### [T1.2] Simplificação dos Parâmetros de Activation Precision (F03) [DONE]

* **Objetivo:** Remover o atalho `--act`, mantendo apenas `--activation`.
* **Ações:**
  * Remover `--act` da ajuda em `print_help()` em `src/standalone/cli.rs`.
  * Remover correspondência de `Long("act")` em `parse_args_from()` em `src/standalone/cli.rs`.
  * Atualizar o teste `test_parse_args_activation_hf` para usar `--activation`.

#### [T1.3] Exibição de Activation Precision no Log de Inicialização (F04) [DONE]

* **Objetivo:** Adicionar log informativo mostrando a precisão de ativação selecionada ao iniciar.
* **Ações:**
  * Em `src/main.rs`, após a chamada de `set_activation_precision(args.activation)`, logar:
    `log::info!("{} Activation precision set to {:?}", "⚡".yellow(), args.activation);`

#### [T1.4] Melhoria da Descrição do Parâmetro `--slim` (F06) [DONE]

* **Objetivo:** Alterar a mensagem de ajuda para descrever `--slim` com mais clareza.
* **Ações:**
  * Alterar em `src/standalone/cli.rs` o texto de ajuda do `--slim` para:
    `Adaptive Compute (downgrades the active model) Auto = Under CPU Pressure [default: auto]`

---

## Sprint 2: Correção do Pipeline de Build de Release

Foco em acelerar o passo de perfilamento e resolver falhas silenciosas de pipe em validações de símbolos do CLAP.

### Épico 2: Correção do Pipeline de Build de Release (F07, F08)

* **Risco/Criticidade:** Médio. Alterações em scripts Bash críticos de build e integração. É essencial que os scripts continuem executando corretamente e instalando os artefatos nos caminhos esperados.
* **Tarefas Técnicas:**

#### [T2.1] Otimização do `perf annotate` no Build de Release (F07) [DONE]

* **Objetivo:** Restringir a análise do `perf annotate` ao binário do `nam-rs`, reduzindo o tempo de execução de ~14 minutos para apenas alguns segundos.
* **Ações:**
  * No arquivo [build-release.sh](file:///home/fabio/nam-rs/utils/build-release.sh), mover a definição da variável `NAM_RS_BIN` para antes da chamada de `perf annotate`.
  * Passar `"$NAM_RS_BIN"` como argumento para o comando `perf annotate`.

#### [T2.2] Correção do SIGPIPE na Validação de Símbolos do CLAP (F08)

* **Objetivo:** Substituir o uso de `grep -q` em pipelines para evitar que sinais `SIGPIPE` retornem código de erro e façam o script falhar sob `set -o pipefail`.
* **Ações:**
  * No arquivo [build-release.sh](file:///home/fabio/nam-rs/utils/build-release.sh), substituir `grep -q` por `grep ... >/dev/null` nas verificações de `clap_entry` e `SONAME`.
  * No arquivo [tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh), aplicar a mesma modificação nas checagens de `SONAME` e `clap_entry`.
