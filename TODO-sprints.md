<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Sprints de Melhoria da Crate Pública (NeuralAmpModeler-rs)

Este documento descreve as Sprints e Tarefas Técnicas para refatorar e preparar a crate pública
`NeuralAmpModeler-rs` (disponível no [crates.io](https://crates.io/crates/NeuralAmpModeler-rs) e
[docs.rs](https://docs.rs/NeuralAmpModeler-rs/latest/nam_rs/)), garantindo simplicidade, segurança
e facilidade de consumo por terceiros sem comprometer o funcionamento original (`standalone`, `clap-plugin`,
benches e testes).

As tarefas são baseadas nas análises de `TODO-crate-publica.md`.

---

## 🎯 Visão Geral dos Épicos e Sprints

| Sprint / Épico | Nome                                                        | Foco                                                                                           | Risco                                                                        |
| -------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Sprint 1**   | **E1: Correções Estruturais e Isolamento de Linker/Target** | F1, F2, F14, F15, `crate-type`, `global_asm!`, version script e flags de CPU                   | 🔴 Alto (Requer testes rigorosos para evitar regressão em `standalone`/CLAP) |
| **Sprint 2**   | **E2: Saneamento da API Pública & Documentação**            | F3, F4, F5, F6, F7, F8, F9, F10, re-exports seletivos, visibilidade de módulos e `Debug` impls | 🟡 Médio (Re-exports podem requerer ajuste de imports internos)              |
| **Sprint 3**   | **E3: Projeto Exemplo & Developer Experience (DX)**         | F13, criação de `examples/basic_inference.rs`, empacotamento (`include`) e smoke-test          | 🟢 Baixo (Apenas adição de novo código de exemplo e validação de build)      |

---

## 🚀 Sprint 1 — E1: Correções Estruturais e Isolamento de Linker/Target

**Objetivo:** Garantir que consumidores da biblioteca externa (`rlib` com `default-features = false`) construam limpo e sem poluição de símbolos globais ou quebras de linker, preservando as necessidades específicas dos alvos `standalone` e `clap-plugin`.

---

### Task 1.1: Restringir `crate-type` do alvo de biblioteca para apenas `rlib`

- **Referência:** `TODO-crate-publica.md` — Finding F1
- **Arquivos impactados:** `Cargo.toml`
- **Descrição detalhada:**
  1. Alterar `crate-type = ["rlib", "cdylib"]` para `crate-type = ["rlib"]` no `Cargo.toml`.
  2. Verificar o comportamento do build do plugin CLAP (`cargo build --features clap-plugin`). Se a compilação do plugin CLAP necessitar de formato dinâmico (`.so`), configurar a compilação via flag/target apropriado ou garantir que o build CI continue gerando a biblioteca dinâmica sem forçar downstream a gerar `cdylib` por padrão.
- **Critério de Aceite:**
  - `cargo check --no-default-features` gera apenas o arquivo `.rlib`.
  - `cargo build --features clap-plugin` continua gerando o plugin CLAP funcional.

---

### Task 1.2: Isolar o bloco `global_asm!` de compatibilidade GLIBC sob alvos binários/plugin

- **Referência:** `TODO-crate-publica.md` — Finding F2
- **Arquivos impactados:** `src/lib.rs` (linhas 209-226)
- **Descrição detalhada:**
  1. O bloco `global_asm!` em `src/lib.rs` injeta símbolos globais (`log10f`, `atan2f`, `acosf`) redirecionados via PLT para a GLIBC 2.2.5.
  2. Proteger o bloco com o atributo `#[cfg(any(feature = "standalone", feature = "clap-plugin"))]`.
  3. Validar se consumidores de `rlib` (com `default-features = false`) não compilam esses símbolos globais, prevenindo conflitos e symbol interposition em aplicações terceiras.
  4. Executar os testes unitários e de integração locais para confirmar que a alteração não ressuscita o hang de interposition documentado em `docs/postmortem-libm-symbol-interposition.md`.
- **Critério de Aceite:**
  - Em builds de biblioteca pura (`--no-default-features`), a tabela de símbolos de `nam_rs` não exporta `log10f`, `atan2f`, ou `acosf`.
  - O suite de testes da aplicação nativa continua rodando e passando integralmente.

---

### Task 1.3: Ajustar emissão incondicional de Linker Version Script no `build.rs` e empacotamento do arquivo `.map`

- **Referência:** `TODO-crate-publica.md` — Findings F14 e F15
- **Arquivos impactados:** `build.rs`, `Cargo.toml`
- **Descrição detalhada:**
  1. Garantir que `.cargo/hide-libm-shadow.map` esteja explicitamente incluído na tabela `include` do `Cargo.toml` para que seja publicado junto ao crate no crates.io.
  2. No `build.rs`, verificar se o arquivo `.map` existe no manifesto antes de emitir a instrução `cargo:rustc-link-arg`.
  3. Adicionar uma checagem no `build.rs` para verificar se as instruções SIMD mínimas (`avx2`, `fma`) estão habilitadas para o target atual. Caso não estejam (por exemplo, quando o consumidor não definiu `-Ctarget-cpu=x86-64-v3`), emitir um `cargo:warning` claro orientando o desenvolvedor sobre como passar as flags de otimização de CPU para obter o máximo desempenho do NAM-rs.
- **Critério de Aceite:**
  - `cargo package --list` confirma a inclusão de `.cargo/hide-libm-shadow.map`.
  - `cargo check --no-default-features` avisa amigavelmente caso o target-cpu esteja desconfigurado, sem quebrar compilações genéricas.

---

## 🎨 Sprint 2 — E2: Saneamento da API Pública & Documentação

**Objetivo:** Organizar os exportes do crate no `src/lib.rs`, fechar módulos de implementação interna e enriquecer os doc-comments e representações `Debug`, simplificando radicalmente a descoberta da API para desenvolvedores externos.

---

### Task 2.1: Substituir `pub use common::*` por re-exports seletivos

- **Referência:** `TODO-crate-publica.md` — Finding F3
- **Arquivos impactados:** `src/lib.rs`, `src/common/mod.rs`, e referências internas em `src/` e `tests/`
- **Descrição detalhada:**
  1. Remover `pub use common::*` e `pub use standalone::*` da raiz de `src/lib.rs`.
  2. Manter a declaração `pub mod common;` e `pub mod standalone;`.
  3. Criar re-exports explícitos e bem documentados na raiz de `src/lib.rs` apenas para tipos indispensáveis a integradores terceiros, como:
     - `pub use common::diagnostics::SystemSnapshot;`
     - `pub use common::spsc::RtStatusFlags;` (se necessário)
  4. Atualizar os imports internos no projeto (`src/main.rs`, `src/clap/`, etc.) para apontar para caminhos qualificados (ex: `crate::common::diagnostics::...`) onde o autocomplete dependia do re-export glob.
- **Critério de Aceite:**
  - `nam_rs::` expõe apenas os tipos de primeira classe no nível raiz.
  - `cargo check --all-targets --all-features` compila perfeitamente sem warnings de importação quebrada.

---

### Task 2.2: Restringir a visibilidade dos submódulos internos (`loader` e `models`)

- **Referência:** `TODO-crate-publica.md` — Findings F8 e F9
- **Arquivos impactados:** `src/loader/mod.rs`, `src/models/slimmable.rs`
- **Descrição detalhada:**
  1. Em `src/loader/mod.rs`, alterar os módulos internos `dispatcher`, `transpose`, `namb_encoder` de `pub mod` para `pub(crate) mod`. Manter expostos apenas os utilitários de carregamento (`load_and_build_model`, `LoadOptions`, `LoadedModelPair`).
  2. Em `src/models/slimmable.rs`, garantir que o submodule `slicing` permaneça visível apenas internamente como `pub(crate) mod slicing;`.
  3. Executar o suite de testes de integração (`cargo test --test '*'`) para validar que nenhum teste externo dependia desses submódulos internos.
- **Critério de Aceite:**
  - Módulos internos não aparecem na documentação gerada por `cargo doc --no-default-features`.

---

### Task 2.3: Implementar `Debug` descritivo em `StaticModel` e refatorar `LoadedModelPair`

- **Referência:** `TODO-crate-publica.md` — Findings F5 e F7
- **Arquivos impactados:** `src/models/static_model.rs`, `src/loader/loaded_model_pair.rs`
- **Descrição detalhada:**
  1. Implementar o trait `std::fmt::Debug` para `StaticModel` em `src/models/static_model.rs`, utilizando a chamada `class_label()` e o número de canais/camadas para gerar uma string explicativa (ex: `StaticModel::WaveNetStandard { channels: 16, receptive_field: 2047 }`).
  2. Atualizar a implementação manual de `std::fmt::Debug` em `LoadedModelPair` para formatar os campos `model_l` e `model_r` com a representação detalhada do modelo, e adicionar um doc-comment na struct explicando a arquitetura de ponteiros para swaps atômicos via SPSC.
- **Critério de Aceite:**
  - `println!("{:?}", loaded_pair)` exibe os detalhes estruturais da variante ativa em vez do marcador genérico `"StaticModel"`.

---

### Task 2.4: Atualizar documentação do `lib.rs` (Tom, Exemplos de Cargo.toml e Quick Start)

- **Referência:** `TODO-crate-publica.md` — Findings F4, F6 e F10
- **Arquivos impactados:** `src/lib.rs`
- **Descrição detalhada:**
  1. Ajustar a mensagem de aviso inicial no doc-comment de `src/lib.rs` para adotar um tom estritamente técnico e profissional, corrigindo pequenas rasuras/typos.
  2. Atualizar todos os trechos de `Cargo.toml` documentados nos exemplos de `3.0.2` para `3` (ou a versão semver flexível).
  3. Enriquecer o exemplo de Quick Start documentando explicitamente a origem e finalidade do `SystemSnapshot::capture()`.
- **Critério de Aceite:**
  - O teste do doctest (`cargo test --doc`) roda e passa integralmente.
  - A documentação gerada possui leitura profissional e fluida.

---

## 🛠 Sprint 3 — E3: Projeto Exemplo & Developer Experience (DX)

**Objetivo:** Oferecer um ponto de entrada prático, completo e imediatamente executável por qualquer desenvolvedor interessado em integrar o engine DSP/Inference do NAM-rs em sua própria aplicação.

---

### Task 3.1: Criar o projeto de exemplo público `examples/basic_inference.rs`

- **Referência:** `TODO-crate-publica.md` — Finding F13
- **Arquivos impactados:** `examples/basic_inference.rs` (Novo arquivo)
- **Descrição detalhada:**
  1. Criar a pasta `examples/` na raiz do projeto.
  2. Implementar `examples/basic_inference.rs` cobrindo o ciclo de vida completo de consumo da crate:
     - Captura de capacidades do sistema (`SystemSnapshot::capture()`).
     - Definição de `LoadOptions` (com e sem prewarm).
     - Carregamento de um modelo `.nam` de teste a partir dos fixtures existentes (ex: `tests/fixtures/models/linear_test.nam` ou construído em memória/string JSON sintética para total autonomia).
     - Inspeção de metadados (`architecture`, `topology`, `loudness`, `input_level_dbu`, `model_info()`).
     - Instanciação e execução do loop de inferência áudio bloco a bloco via trait `NamModel::process()`.
     - Demonstração do uso das rotinas SIMD de `math::activations` (`tanh`, `sigmoid`, `tanh_slice`).
     - Alternância de precisão via TLS `set_activation_tls(ActivationPrecision::Standard)`.
- **Critério de Aceite:**
  - O arquivo inclui o cabeçalho de licença Apache-2.0 / Copyright 2026.
  - `cargo run --example basic_inference --no-default-features` compila rapidamente e executa exibindo relatórios limpos no terminal.

---

### Task 3.2: Registrar a pasta `examples/` no `Cargo.toml` e realizar Smoke-Test Final

- **Referência:** `TODO-crate-publica.md` — Finding F13
- **Arquivos impactados:** `Cargo.toml`
- **Descrição detalhada:**
  1. Adicionar `"examples/"` ao array `include` em `Cargo.toml`.
  2. Executar o comando `cargo package --list` e inspecionar se todos os arquivos essenciais (`src/`, `examples/`, `README.md`, `LICENSE.txt`, `NOTICE.txt`, `.cargo/hide-libm-shadow.map`) estão presentes no tarball distribuído, e que nenhum arquivo temporário foi incluído.
  3. Executar a validação completa de verificação de higiene: `utils/lints.sh` e `utils/tests-quick.sh`.
- **Critério de Aceite:**
  - `cargo package` gera o pacote sem erros.
  - Os scripts de lint e teste rápido (`utils/lints.sh` e `utils/tests-quick.sh`) rodam e passam com 100% de sucesso.
