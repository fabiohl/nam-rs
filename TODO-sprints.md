<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Plano de Sprints: Cirurgia de Separação em 3 Repositórios Git Independente

**Referência de Achado:** [TODO-refactor_split.md](TODO-refactor_split.md)
**Objetivo:** Dividir o monolito `nam-rs` em 3 repositórios Git independentes em `target/transicao/` com zero regressão, 100% de cobertura de testes passando e zero duplicidade de código DSP.

> **Regra de Transição Importante:**
> A pasta raiz `./` **NÃO** será tocada ou alterada durante a migração. O staging funcional `target/transicao/nam-rs` receberá a cópia inicial e sofrerá os comandos `mv` para popular os novos repositórios (`NeuralAmpModeler-rs`, `NAM-Audio-Pipe`, `NAM-Plug`). É estritamente proibido commitar alterações na raiz `./` ou em `target/transicao/nam-rs`.
>
> **Nota do PO:** Em todos os novos repositórios, estaremos trabalhando na branch "dev" (sem push para o github). Quando tudo estiver bem, dar merge na branch "main". Dai dar "push origin".

---

## Épico 1: Isolamento do Kernel DSP (`NeuralAmpModeler-rs`)

### Sprint 1.1 — Inicialização de Diretórios e Crate Base

- [OK] **Tarefa 1.1.1:** Criar estrutura de diretórios em `target/transicao/[audiorip, nam-rs, NeuralAmpModeler-rs, NAM-Audio-Pipe, NAM-Plug]`.
- [OK] **Tarefa 1.1.2:** Copiar o repositório original integralmente para `target/transicao/nam-rs` como staging ativo de movimentação.
- [OK] **Tarefa 1.1.3:** Configurar o repositório Git <https://github.com/fabiohl/NeuralAmpModeler-rs> em `target/transicao/NeuralAmpModeler-rs` (branch `dev`).

### Sprint 1.2 — População e Ajuste de Dependências de `NeuralAmpModeler-rs`

> **Detalhamento completo:** Ver [sprint_1_2_detalhado.md](/home/fabio/.gemini/antigravity-ide/brain/f0ecbeaf-b6eb-42b4-a768-7e8a614be911/sprint_1_2_detalhado.md) (peer review integrado, comandos precisos, critérios de conclusão verificáveis).
>
> **Regra de ouro:** Todos os `mv`/`cp` operam entre `target/transicao/nam-rs` (fonte) e
> `target/transicao/NeuralAmpModeler-rs` (destino). A raiz `./` e o staging `nam-rs` **jamais**
> recebem `git commit`.

- **Tarefa 1.2.1 — Migração dos Módulos-Fonte DSP:**
  `mv` de `src/math`, `src/models`, `src/loader`, `src/dsp`, `src/common`, `src/testing`
  de `target/transicao/nam-rs/` para `NeuralAmpModeler-rs/src/`.
  **⚠️ NÃO mover** `src/standalone/`, `src/clap/`, `src/main.rs`, `src/bin/`.
  **Reescrever** `src/lib.rs` (não copiar) — expondo apenas os módulos DSP, sem `mod standalone` nem `mod clap`.
  Copiar `.cargo/config.toml` (vital: força `target-cpu=x86-64-v3`), `build.rs`, `.gitignore`, `variables.env`, `LICENSE.txt`, `NOTICE.txt`.

- **Tarefa 1.2.2 — Migração de Testes e Benchmarks DSP:**
  `mv` dos testes de integração DSP: `tests/models.rs`, `tests/models/`, `tests/parity.rs`, `tests/parity/`,
  `tests/rt_constraints.rs`, `tests/rt_constraints/`, `tests/loom_tests.rs`, `tests/perf_soak.rs`, `tests/perf_soak/`,
  `tests/fixtures/`, `tests/common/`, `tests/proptest-regressions/`.
  **⚠️ NÃO mover** `tests/clap.rs`, `tests/clap/`, `tests/clap_e0_containment_test.rs`, `tests/clap_e2_proptest.rs`.
  `mv` de 14 suítes de benchmarks DSP: todos em `benches/` **exceto** `benches/clap_bench.rs`.
  Incluir subdirs `benches/gemv/` e `benches/inference/` e o helper `benches/common.rs`.

- **Tarefa 1.2.3 — Documentação e Scripts `utils/` Adaptados:**
  `cp` de 12 docs DSP (excluir `docs/clap_integration.md`). Incluir `docs/quality-contract.txt`.
  `cp` de `utils/{_lib.sh, lints.sh, tests-quick.sh, quality-dashboard.sh, tests-long.sh, check-model.py, tests-performance-regression.sh}`.
  **Adaptar** cada script no destino: remover blocos de feature `clap-plugin`/`standalone`,
  referências ao binário `nam-rs`, crate name `nam_rs`. Validar sintaxe com `bash -n <script>`.
  **⚠️ NÃO copiar** `utils/run-standalone.sh`, `utils/build-release.sh`, `utils/mod-update.sh`.

- **Tarefa 1.2.4 — Criar `Cargo.toml` (Especificação Completa):**
  `name = "NeuralAmpModeler-rs"` / `lib.name = "neural_amp_modeler_rs"` / `crate-type = ["rlib"]` (sem `cdylib`).
  Dependências: `thiserror`, `libc`, `rtrb`, `serde`+`serde_json`, `sha2`, `log`, `anyhow`.
  **NÃO incluir** `env_logger` (responsabilidade do consumer), `lexopt` (CLI only), `pipewire`, `clack-*`, `egui`, `baseview`, `rfd`.
  Dev-deps: `criterion`, `proptest`, `loom`, `serial_test`.
  Features: `default = []`, `stereo`, `testing`, `heap-audit`, `long_bench`, `pgo`.
  13 entradas `[[bench]]` (um por arquivo bench DSP — ver detalhamento).
  Perfis `release`/`dist`/`dev` idênticos ao original. `[lints.*]` idênticos.

- **Tarefa 1.2.5 — Compilação Limpa, Zero Lints, 100% Testes Operacionais:**
  **A.** Grep e corrigir imports inválidos: `crate::standalone`, `crate::clap`, `nam_rs::`.
  **B.** `cargo check` em todas as combos de features (6 combinações — ver detalhamento).
  **C.** `bash utils/lints.sh` — zero erros/warnings, SPDX em todos os arquivos.
  **D.** `bash utils/tests-quick.sh` — 100% de testes não-ignored passando.
  **E.** `cargo bench --no-run --all-features` — todos os 14 benchmarks compilam.
  **F.** `git add -A && git commit` em `NeuralAmpModeler-rs` (mensagem de encerramento de sprint).
  **⚠️ Não dar push** — push somente no Épico 4.

---

## Épico 2: Extração e Construção do CLI Standalone (`NAM-Audio-Pipe`)

### Sprint 2.1 — Inicialização e Dependência Local

- [OK] **Tarefa 2.1.1:** Configurar o repositório Git <https://github.com/fabiohl/NAM-Audio-Pipe> em `target/transicao/NAM-Audio-Pipe` (branch `dev`).
- **Tarefa 2.1.2:** Mover `src/standalone` e `src/main.rs` de `target/transicao/nam-rs` para `NAM-Audio-Pipe`.
- **Tarefa 2.1.3:** Configurar `Cargo.toml` declarando a dependência por caminho: `NeuralAmpModeler-rs = { path = "../NeuralAmpModeler-rs" }` e dependências nativas (`pipewire`, `lexopt`).

### Sprint 2.2 — Adaptação de Importações e Módulo de Gravação em Disco (WAV)

> **Nota do PO:** Aproveita muito do que já foi feito no projeto <https://github.com/fabiohl/audiorip>, aqui espelhado em `target/transicao/audiorip`.

- **Tarefa 2.2.1:** Atualizar importações em `src/standalone/` para utilizar o crate `NeuralAmpModeler_rs`.
- **Tarefa 2.2.2:** Criar estrutura do novo módulo `src/recording/mod.rs` para gravação assíncrona/lock-free do stream de áudio exclusivamente em formato **WAV** no disco.
- **Tarefa 2.2.3:** Mover os testes e benchmarks que exercitam o módulo PipeWire (`src/standalone/cli_test.rs`, benchmarks de quantum PipeWire) e herdar documentação e scripts `utils/` adaptados ao contexto Standalone (`run-standalone.sh`, `lints.sh`, `tests-quick.sh`, `build-release.sh`).
- **Tarefa 2.2.4:** Gestão e fixes internos em `NAM-Audio-Pipe` até que todos os testes passem de primeira.

---

## Épico 3: Extração e Construção do Plugin CLAP (`NAM-Plug`)

### Sprint 3.1 — Inicialização e Dependências de GUI/CLAP

- [OK] **Tarefa 3.1.1:** Configurar o repositório Git <https://github.com/fabiohl/NAM-Plug> em `target/transicao/NAM-Plug` (branch `dev`).
- **Tarefa 3.1.2:** Mover `src/clap/`, `src/lib.rs` (cdylib) e `src/bin/pgo_profiling_workload.rs` para `NAM-Plug`.
- **Tarefa 3.1.3:** Mover testes do plugin (`tests/clap.rs`, `tests/clap/`, `tests/clap_e0_containment_test.rs`, `tests/clap_e2_proptest.rs`) e benchmarks (`benches/clap_bench.rs`) para `NAM-Plug`.
- **Tarefa 3.1.4:** Configurar `Cargo.toml` com a dependência por caminho `NeuralAmpModeler-rs = { path = "../NeuralAmpModeler-rs" }` e dependências `clack-*`, `egui`, `glow`, `baseview`, `rfd`.

### Sprint 3.2 — Adaptação de Imports, Documentação e Scripts Adaptados

- **Tarefa 3.2.1:** Atualizar importações no código do CLAP de `crate::dsp::...` para `NeuralAmpModeler_rs::...`.
- **Tarefa 3.2.2:** Copiar documentação dedicada (`docs/clap_integration.md`) e utilitários enxutos/adaptados (`utils/build-release.sh`, `utils/lints.sh`, `utils/tests-quick.sh`).
- **Tarefa 3.2.3:** Gestão e fixes internos em `NAM-Plug` garantindo compilação limpa, zero lints e 100% de passagem de testes de plugin.

---

## Épico 4: Fechamento Final

- Dar merge das branches "dev" de todos os repos nas respectivas branches "main". Só ai dar "git push origin" (apenas de "main").
- Montar um Workspace local funcional dos 3 novos repositórios no VScode/Antigravity. Assegurar que todos os testes passam.
- Publicar o novo `NeuralAmpModeler-rs` no crates.io/docs.rs e assegurar que os demais projetos consigam utilizá-lo como dependência tanto local como remota.
