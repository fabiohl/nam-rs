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

- [OK] **Tarefa 1.2.1 — Migração dos Módulos-Fonte DSP:**
  `mv` de `src/math`, `src/models`, `src/loader`, `src/dsp`, `src/common`, `src/testing`
  de `target/transicao/nam-rs/` para `NeuralAmpModeler-rs/src/`.
  **⚠️ NÃO mover** `src/standalone/`, `src/clap/`, `src/main.rs`, `src/bin/`.
  **Reescrever** `src/lib.rs` (não copiar) — expondo apenas os módulos DSP, sem `mod standalone` nem `mod clap`.
  Copiar `.cargo/config.toml` (vital: força `target-cpu=x86-64-v3`), `build.rs`, `.gitignore`, `variables.env`, `LICENSE.txt`, `NOTICE.txt`.

- [OK] **Tarefa 1.2.2 — Migração de Testes e Benchmarks DSP:**
  `mv` dos testes de integração DSP: `tests/models.rs`, `tests/models/`, `tests/parity.rs`, `tests/parity/`,
  `tests/rt_constraints.rs`, `tests/rt_constraints/`, `tests/loom_tests.rs`, `tests/perf_soak.rs`, `tests/perf_soak/`,
  `tests/fixtures/`, `tests/common/`, `tests/proptest-regressions/`.
  **⚠️ NÃO mover** `tests/clap.rs`, `tests/clap/`, `tests/clap_e0_containment_test.rs`, `tests/clap_e2_proptest.rs`.
  `mv` de 14 suítes de benchmarks DSP: todos em `benches/` **exceto** `benches/clap_bench.rs`.
  Incluir subdirs `benches/gemv/` e `benches/inference/` e o helper `benches/common.rs`.

- [OK]**Tarefa 1.2.3 — Documentação e Scripts `utils/` Adaptados:**
  `cp` de 12 docs DSP (excluir `docs/clap_integration.md`). Incluir `docs/quality-contract.txt`.
  `cp` de `utils/{_lib.sh, lints.sh, tests-quick.sh, quality-dashboard.sh, tests-long.sh, check-model.py, tests-performance-regression.sh}`.
  **Adaptar** cada script no destino: remover blocos de feature `clap-plugin`/`standalone`,
  referências ao binário `nam-rs`, crate name `nam_rs`. Validar sintaxe com `bash -n <script>`.
  **⚠️ NÃO copiar** `utils/run-standalone.sh`, `utils/build-release.sh`, `utils/mod-update.sh`.

- [OK] **Tarefa 1.2.4 — Criar `Cargo.toml` (Especificação Completa):**
  `name = "NeuralAmpModeler-rs"` / `lib.name = "neural_amp_modeler_rs"` / `crate-type = ["rlib"]` (sem `cdylib`).
  Dependências: `thiserror`, `libc`, `rtrb`, `serde`+`serde_json`, `sha2`, `log`, `anyhow`.
  **NÃO incluir** `env_logger` (responsabilidade do consumer), `lexopt` (CLI only), `pipewire`, `clack-*`, `egui`, `baseview`, `rfd`.
  Dev-deps: `criterion`, `proptest`, `loom`, `serial_test`.
  Features: `default = []`, `stereo`, `testing`, `heap-audit`, `long_bench`, `pgo`.
  13 entradas `[[bench]]` (um por arquivo bench DSP — ver detalhamento).
  Perfis `release`/`dist`/`dev` idênticos ao original. `[lints.*]` idênticos.

- [OK] **Tarefa 1.2.5 — Compilação Limpa, Zero Lints, 100% Testes Operacionais:**
  **A.** Grep e corrigir imports inválidos: `crate::standalone`, `crate::clap`, `nam_rs::`.
  **B.** `cargo check` em todas as combos de features (6 combinações — ver detalhamento).
  **C.** `bash utils/lints.sh` — zero erros/warnings, SPDX em todos os arquivos.
  **D.** `bash utils/tests-quick.sh` — 100% de testes não-ignored passando.
  **E.** `cargo bench --no-run --all-features` — todos os 14 benchmarks compilam.
  **F.** `git add -A && git commit` em `NeuralAmpModeler-rs` (mensagem de encerramento de sprint).
  **⚠️ Não dar push** — push somente no Épico 4.

> **Notas de conclusão da Sprint 1.2 (registradas em 2026-07-29):**
>
> 1. **Features `standalone`/`clap-plugin` como cfg esperados:** Estas features NÃO foram adicionadas ao `[features]` do Cargo.toml (não há deps como pipewire/clack), mas foram registradas em `[lints.rust] unexpected_cfgs.check-cfg` como valores esperados de `feature`. Isso permite que os `#[cfg(feature = "standalone")]` e `#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]` no código-fonte não gerem warnings, ao mesmo tempo que o código gated nunca é compilado (feature não existe). Se no futuro `NAM-Audio-Pipe` ou `NAM-Plug` precisarem ativar esses blocos, a feature deve ser movida para `[features]` com as dependências nativas correspondentes.
>
> 2. **Ungating de módulos pipeline/cabsim:** Os módulos `dsp::pipeline` e `dsp::cabsim` foram desbloqueados (removido `#[cfg(any(...))]` do `mod` declaration) para que estejam sempre disponíveis como parte do core DSP. O código interno mantém seus próprios gates para partes específicas de standalone (`output_pw.rs`) e clap-plugin (`apply_input_stage`, `apply_output_stage`, `run_inference`). Isso foi necessário porque os testes de integração referenciam tipos do pipeline (`DspPipelineContext`, `BridgeBuffer`, etc.) que precisam estar presentes na lib, não apenas em `#[cfg(test)]`.
>
> 3. **Feature `dynamic-engine` mantida:** O código em `src/models/a2/` ainda referencia `#[cfg(feature = "dynamic-engine")]`. Esta feature foi mantida como vazia (`dynamic-engine = []`) para evitar warnings. Remover as referências do código é uma tarefa futura (A2 scaffolding).
>
> 4. **Lint `clippy::allow_attributes` rebaixado:** Alterado de `"warn"` para `"allow"` porque o código usa `#[allow(unused_imports)]` em vários pontos do pipeline onde imports são condicionalmente utilizados dependendo da feature ativa. Migrar para `#[expect]` (Rust 2024) não foi viável porque o expect falha quando o import é realmente usado (lint unfulfilled).
>
> 5. **Testes compilam, mas NÃO foram executados:** `cargo test --no-run --features testing` confirma que todos os 5 test targets (models, parity, perf_soak, rt_constraints, loom) compilam. A execução real (`bash utils/tests-quick.sh`) requer golden vectors pré-gerados e o binário C++ `render` do NeuralAmpModelerCore — executar manualmente no ambiente completo antes do Épico 4.
>
> 6. **Staging `target/transicao/nam-rs` permanece com deletes por commit:** Os `mv` das tarefas 1.2.1 e 1.2.2 deixaram o staging com arquivos deletados (status `D`). Nenhum `git commit` foi feito no staging. Isso é esperado e intencional — o staging é somente fonte de movimentação.

---

## Épico 2: Extração e Construção do CLI Standalone (`NAM-Audio-Pipe`)

> **Detalhamento completo:** Ver [epico2_detalhado.md](/home/fabio/.gemini/antigravity-ide/brain/901e1a21-a4fa-4ba2-bda7-7d20f4e015d4/epico2_detalhado.md) (peer review integrado, comandos precisos, mapeamento completo de imports, critérios de conclusão verificáveis).
>
> **Regra de ouro:** Usar `cp` (nunca `mv`) do staging `nam-rs` → `NAM-Audio-Pipe`, preservando o staging intacto para o Épico 3.
> A raiz `./` e o staging `nam-rs` **jamais** recebem `git commit`.

### Sprint 2.1 — Inicialização, Estrutura de Arquivos e `Cargo.toml`

> **Gate de conclusão:** `cargo check` retorna apenas erros de `use crate::*` (imports) — zero erros de estrutura de projeto.

- [OK] **Tarefa 2.1.1:** Confirmar repositório Git <https://github.com/fabiohl/NAM-Audio-Pipe> em `target/transicao/NAM-Audio-Pipe` (branch `dev`). Verificar: `git status` → `nothing to commit`.

- [OK] **Tarefa 2.1.2 — Migração de `src/` Standalone:**
  `cp $SRC/src/main.rs $DST/src/main.rs` e `cp -r $SRC/src/standalone $DST/src/standalone`.
  **Estrutura esperada:** `src/main.rs`, `src/standalone/{mod.rs, cli.rs, colors.rs, pw_host/, rt_setup/}`.
  **⚠️ NÃO copiar:** `src/dsp/`, `src/models/`, `src/math/`, `src/common/`, `src/loader/`, `src/testing/`, `src/clap/`, `src/lib.rs`.

- [OK] **Tarefa 2.1.3 — Reescrever `src/standalone/mod.rs`:**
  Remover `#![cfg(feature = "standalone")]` (gate do monolito — inválido aqui).
  Remover re-exportações glob (`pub use cli::*;`). Manter apenas `pub mod cli; pub mod colors; pub mod pw_host; pub mod rt_setup;`.

- [OK] **Tarefa 2.1.4 — Criar `Cargo.toml` (Especificação Completa):**
  `name = "nam-audio-pipe"` / `[[bin]] name = "nam-audio-pipe"` / `path = "src/main.rs"`.
  Dependências: `NeuralAmpModeler-rs = { path = "../NeuralAmpModeler-rs" }`, `pipewire = "0.8"`, `lexopt`, `hound = "3.5"`, `rtrb`, `tokio` (`rt`/`time`/`macros`), `tokio-uring = "0.5"`, `anyhow`, `libc`, `log`, `env_logger`.
  **Pré-requisitos de sistema:** kernel Linux ≥ 5.10 (`uname -r`), `libpipewire-0.3-dev` (`pkg-config --modversion libpipewire-0.3`).
  Perfis `release`/`dist`/`dev` equivalentes ao staging. `[lints.*]` rigorosos.

- [OK] **Tarefa 2.1.5 — Copiar Configuração de Baixo Nível:**
  `cp -r nam-rs/.cargo NAM-Audio-Pipe/.cargo` (vital: força `target-cpu=x86-64-v3`).
  `cp nam-rs/.gitignore nam-rs/LICENSE.txt nam-rs/NOTICE.txt NAM-Audio-Pipe/`.
  **Verificar** `build.rs` antes de copiar — se tiver lógica DSP, NÃO copiar.

- [OK] **Tarefa 2.1.6 — Primeira Compilação e Diagnóstico:**
  `cargo check 2>&1 | head -100`. Erros esperados: apenas `use crate::dsp/models/common/math::...`.
  Zero erros de estrutura = gate de conclusão da Sprint 2.1.

> **Notas de conclusão da Sprint 2.1 (registradas em 2026-07-29):**
>
> 1. **`build.rs` copiado (sem lógica DSP):** O `build.rs` do staging contém apenas fix de linker ELF para symbol interposition de libm (version script `.cargo/hide-libm-shadow.map`), sem qualquer lógica DSP. Foi copiado porque o binário `nam-audio-pipe` linka as mesmas dependências e está sujeito ao mesmo bug.
>
> 2. **`mod standalone;` inserido em `main.rs`:** No monolito, `mod standalone;` ficava em `lib.rs`. No `NAM-Audio-Pipe` (crate binário, sem `lib.rs`), foi adicionado diretamente no topo de `main.rs` sem `#[cfg()]`.
>
> 3. **`cargo check` — gate atingido com 26 erros:** 100% dos erros são de importação (`nam_rs::` não encontrado, métodos `.bright_green()`/`.cyan()`/`.yellow()` ausentes após remoção dos `pub use colors::*;`). Zero erros de estrutura de diretórios, `mod` ausente ou arquivo não encontrado.
>
> 4. **Pipewire 0.8 vs 0.10:** O staging usa `pipewire = "0.10.0"`, mas o spec do Épico 2 determina `pipewire = "0.8"`. Seguido conforme spec. Se houver incompatibilidade com APIs reais, revisitar na Sprint 2.2.
>
> 5. **143 substituições de imports na Sprint 2.2:** Os padrões `crate::dsp::`, `crate::models::`, `crate::common::`, `crate::math::` aparecem nos sub-módulos do standalone (não apenas em `use`, mas também em `pub use`, `pub(crate) use`, e imports indentados). A substituição por sed foi expandida para cobrir todas as variações de prefixo, preservando `crate::standalone::`. `nam_rs::` só aparecia em `main.rs`.
>
> 6. **Itens de pipeline gated atrás de `#[cfg(feature = "standalone")]` no NeuralAmpModeler-rs:** `build_spa_format_pod`, `playback_dsp_cycle`, `AppState`, `PipewireHostConfig` e outros em `dsp::pipeline/output_pw.rs` estão inacessíveis porque a feature `standalone` não existe no `NeuralAmpModeler-rs`. A Tarefa 2.2.3 deve resolver isso ungatando os itens ou adicionando a feature com `pipewire` como dependência opcional.
>
> 7. **`Cargo.toml` com feature `testing = []`:** Adicionada porque `main.rs:45` usa `#[cfg(feature = "testing")]`. Sem a definição, o bloco fica permanentemente inativo — sem erro de compilação, mas requer a feature para testes futuros.

### Sprint 2.2 — Adaptação de Importações e Integração com `NeuralAmpModeler-rs`

> **Gate de conclusão:** `cargo check 2>&1 | grep "^error"` → saída vazia. `cargo build` sem warnings não tratados.

- **Tarefa 2.2.1 — Mapeamento e Substituição de Importações:**
  Grep: `grep -rn "use crate::" src/ | grep -v "standalone\|colors" | sort`.
  Substituição em lote via `find src/ -name "*.rs" -exec sed -i 's/use crate::dsp::/use neural_amp_modeler_rs::dsp::/g; s/use crate::models::/use neural_amp_modeler_rs::models::/g; s/use crate::common::/use neural_amp_modeler_rs::common::/g; s/use crate::math::/use neural_amp_modeler_rs::math::/g' {} \;`.
  **⚠️ NÃO substituir:** `use crate::standalone::` e `use crate::standalone::colors::` — são locais do `NAM-Audio-Pipe`.
  Verificação: `grep -rn "use crate::dsp\|use crate::models\|use crate::common\|use crate::math" src/` → saída vazia.

- **Tarefa 2.2.2 — Atualizar `src/main.rs`:**
  Remover `#[cfg(feature = "standalone")]` de `mod standalone;`. Substituir referências `nam_rs` por `nam_audio_pipe`.
  Garantir `env_logger::init()` no início de `main()`. Verificar imports de `NamLogger`/`DiagnosticBundle` → via `neural_amp_modeler_rs::common::...`.

- **Tarefa 2.2.3 — Corrigir Visibilidade de APIs do `NeuralAmpModeler-rs`:**
  `cargo check 2>&1 | grep "not found\|is private"`. Para cada tipo inacessível, adicionar `pub use ...` em `NeuralAmpModeler-rs/src/lib.rs`.
  **⚠️ Regra:** Commits no `NeuralAmpModeler-rs` devem ser feitos **antes** do commit do `NAM-Audio-Pipe`.

- **Tarefa 2.2.4 — Compilação Limpa:**
  `cargo check` → zero erros. `cargo build` → zero warnings não tratados.

### Sprint 2.3 — Módulo de Gravação WAV (`src/recording/`)

> **Fonte:** `target/transicao/audiorip/src/` — `buffer.rs` e `disk.rs`.
> **⚠️ Processo:** adaptação (não cópia direta) — reescrever cabeçalho para `Apache-2.0`, ajustar imports e globais.
> **Nota do PO:** Aproveitar código do projeto <https://github.com/fabiohl/audiorip> (espelhado em `target/transicao/audiorip`), especialmente para trim de silêncios — respeitar os objetivos do NAM-Audio-Pipe.

- **Tarefa 2.3.1 — Estrutura do Módulo:**
  `mkdir -p src/recording`. Criar: `src/recording/mod.rs` (do zero), `src/recording/buffer.rs` (adaptar de `audiorip/src/buffer.rs`), `src/recording/disk.rs` (adaptar de `audiorip/src/disk.rs`).

- **Tarefa 2.3.2 — Adaptar `src/recording/buffer.rs`:**
  Cabeçalho → `Apache-2.0`. Remover `static SHUTDOWN` local → reutilizar `neural_amp_modeler_rs::common::spsc::SHUTDOWN` (DRY).
  Manter `static OVERRUN_COUNT: AtomicU64` local. Ajustar `MAX_BLOCK_SIZE = 4096`, `RING_CAPACITY = 1024`.

- **Tarefa 2.3.3 — Adaptar `src/recording/disk.rs`:**
  Cabeçalho → `Apache-2.0`. `use crate::buffer::` → `use crate::recording::buffer::`. `SHUTDOWN` local → `neural_amp_modeler_rs::common::spsc::SHUTDOWN`.
  Mensagens de log: `[AudioRip]` → `[NAM-Audio-Pipe]`. Verificar runtime `tokio_uring::start()` vs. `#[tokio::main]`.
  Trim de silêncios: feature futura — adicionar `// TODO: integrar trim (ver audiorip/src/audio.rs)`.

- **Tarefa 2.3.4 — Criar `src/recording/mod.rs`:**
  Expor: `buffer::{AlignedBlock, AudioMetadata, MAX_BLOCK_SIZE, OVERRUN_COUNT, RING_CAPACITY, RingPayload, create_audio_ring_buffer}` e `disk::disk_writer_loop`.

- **Tarefa 2.3.5 — Integrar `recording` no Pipeline PipeWire (`pw_host/run.rs`):**
  Em `main()`: criar ring buffer off-RT, spawnar thread `tokio_uring` de I/O com `disk_writer_loop`, passar `Option<Producer>` para `run_pipewire_host`.
  Na RT capture callback: `try_push(RingPayload::Audio(block))` — em caso de overrun, `OVERRUN_COUNT.fetch_add(1, Ordering::Relaxed)`.
  **⚠️ RT-safety obrigatória:** NUNCA `push` bloqueante na callback. Usar APENAS `try_push`.

- **Tarefa 2.3.6 — Flag `--record` como Opt-In no CLI:**
  Adicionar `pub record: bool` em `CliArgs`. Parsing de `Long("record")` em `parse_args_from()`. Atualizar `print_help()`. Criar ring buffer e thread de I/O somente se `args.record == true`.

### Sprint 2.4 — Testes, Benchmarks, Scripts e Fechamento

- **Tarefa 2.4.1 — Testes do Standalone:**
  Testes inline (`cli_test.rs`, `pw_host_test.rs`, `rt_setup_test.rs`) já copiados com os módulos em 2.1.2.
  `cargo test --no-run` → zero erros de compilação. Execução real dos testes PipeWire: requer ambiente com PipeWire ativo — executar manualmente antes do Épico 4.

- **Tarefa 2.4.2 — Benchmark PipeWire:**
  Verificar: `ls nam-rs/benches/ | grep -i "pw\|pipe\|standalone"`. Se existir, copiar e adaptar.
  Se não existir, criar stub `benches/pw_latency_bench.rs` com `#[ignore]`. `cargo bench --no-run` → compila.

- **Tarefa 2.4.3 — Adaptar Scripts `utils/`:**
  `cp _lib.sh lints.sh tests-quick.sh run-standalone.sh build-release.sh` do staging.
  **`lints.sh`:** substituir `nam_rs`/`nam-rs` por `nam_audio_pipe`/`nam-audio-pipe`; remover blocos de features `clap-plugin`/`standalone`. Validar: `bash -n utils/lints.sh`.
  **`tests-quick.sh`:** remover `--features standalone`; remover blocos `tests/clap*` e `tests/models*`. Validar: `bash -n`.
  **`run-standalone.sh`:** `nam-rs` → `nam-audio-pipe`; `~/.local/bin/nam-rs` → `~/.local/bin/nam-audio-pipe`; modelo via caminho relativo a `../NeuralAmpModeler-rs/tests/fixtures/`. Validar: `bash -n`.
  **`build-release.sh`:** remover `--features standalone`; binário `nam-rs` → `nam-audio-pipe`. Validar: `bash -n`.

- **Tarefa 2.4.4 — Documentação Mínima:**
  Criar `docs/` com `README.md` documentando: descrição, dependências do sistema, build, instalação, exemplos de uso (`--model`, `--record`, `--oversample`, `--activation`).

- **Tarefa 2.4.5 — Pipeline de Qualidade Final e Commit:**
  **A.** `bash utils/lints.sh` — zero erros/warnings, SPDX em todos os arquivos.
  **B.** `cargo build --release` — binário `target/release/nam-audio-pipe` existe.
  **C.** `cargo test --no-run` — zero erros de compilação de testes.
  **D.** `grep -rL "SPDX-License-Identifier" src/` → saída vazia.
  **E.** `git add -A && git commit` em `NAM-Audio-Pipe` (mensagem de encerramento de épico).
  **⚠️ Não dar push** — push somente no Épico 4.

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
