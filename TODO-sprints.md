<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# TODO — Sprints de Integração CLAP (NAM-rs v1.5.0-alpha)

---

## Sprint 0 — Fundação Estrutural e Documentação

**Objetivo:** Reorganizar o repositório em três camadas (`common/`, `standalone/`, `clap/`), atualizar documentação com decisões confirmadas, garantir regressão zero.

---

### Épico 0.1 — Reorganização de Diretórios

- [x] **Tarefa 0.1.1** — Criar `src/common/mod.rs` e mover módulos compartilhados
  
  - Criar diretório `src/common/`
  - Mover `src/audio_host.rs` → `src/common/audio_host.rs`
  - Mover `src/diagnostics.rs` + `src/diagnostics_test.rs` → `src/common/`
  - Mover `src/params.rs` → `src/common/params.rs`
  - Mover `src/spsc.rs` + `src/spsc_test.rs` → `src/common/`
  - Criar `src/common/mod.rs` com re-exports públicos de todos os sub-módulos
  - **Aceite:** `cargo check --features standalone` passa sem erros

- [x] **Tarefa 0.1.2** — Criar `src/standalone/mod.rs` e mover módulos PipeWire
  
  - Criar diretório `src/standalone/`
  - Mover `src/pw_host.rs` + `src/pw_host_test.rs` → `src/standalone/`
  - Mover `src/rt_setup.rs` + `src/rt_setup_test.rs` → `src/standalone/`
  - Mover `src/cli.rs` → `src/standalone/cli.rs`
  - Mover `src/colors.rs` → `src/standalone/colors.rs` (usado apenas no terminal)
  - Criar `src/standalone/mod.rs` com `#[cfg(feature = "standalone")]` e re-exports
  - **Aceite:** `cargo check --features standalone` passa sem erros

- [x] **Tarefa 0.1.3** — Criar `src/clap/mod.rs` (stub)
  
  - Criar diretório `src/clap/`
  - Criar `src/clap/mod.rs` com stub protegido por `#[cfg(feature = "clap-plugin")]`
  - Conteúdo mínimo: docstring do módulo + placeholder
  - **Aceite:** `cargo check --no-default-features --features clap-plugin` passa

- [x] **Tarefa 0.1.4** — Refatorar `src/lib.rs` como hub mínimo de re-exports
  
  - Expor `pub mod common;` (sempre)
  - Expor `pub mod standalone;` sob `#[cfg(feature = "standalone")]`
  - Expor `pub mod clap;` sob `#[cfg(feature = "clap-plugin")]`
  - Manter `pub mod dsp;`, `pub mod models;`, `pub mod math;`, `pub mod loader;` (sempre)
  - Adicionar `pub use common::*;` para re-exports de conveniência (manter compatibilidade)
  - Atualizar docstring do `lib.rs` para refletir a nova estrutura tripartida
  - **Aceite:** `cargo check --features standalone` passa sem erros

- [x] **Tarefa 0.1.5** — Refatorar `src/main.rs` para entry-point mínimo
  
  - Atualizar imports: `use nam_rs::standalone::{cli, pw_host, rt_setup};`
  - Ajustar `use nam_rs::colors::Colorize` → `use nam_rs::standalone::colors::Colorize`
  - Garantir que `main.rs` permaneça enxuto (apenas orquestração e delegação)
  - **Aceite:** `cargo build --features standalone` produz binário funcional

- [x] **Tarefa 0.1.6** — Atualizar todos os imports internos do crate
  
  - Varrer `src/standalone/*.rs`: ajustar `use crate::` para nova estrutura
  - Varrer `src/dsp/*.rs`: ajustar refs a `diagnostics`, `spsc`, etc.
  - Varrer `src/models/*.rs`: ajustar imports se necessário
  - Varrer `src/loader/*.rs`: ajustar imports se necessário
  - **Aceite:** `cargo check --features standalone` sem erros, sem warnings

- [x] **Tarefa 0.1.7** — Atualizar imports em testes e benchmarks
  
  - Varrer `tests/*.rs`: ajustar `use nam_rs::` para nova estrutura (ou validar que re-exports funcionam)
  - Varrer `benches/*.rs`: ajustar imports
  - Varrer `fuzz/`: ajustar imports se aplicável
  - **Aceite:** `cargo test` — todos os 138+ testes passam

- [x] **Tarefa 0.1.8** — Atualizar `utils/run-standalone.sh`
  
  - Ajustar quaisquer referências a paths ou módulos que tenham mudado
  - Garantir que o script continua funcional ponta-a-ponta
  - **Aceite:** `utils/run-standalone.sh` executa e processa áudio normalmente

- [x] **Tarefa 0.1.9** — Validação final do Épico 0.1
  
  - Executar `utils/lints.sh` — zero erros, zero warnings
  - Executar `cargo test` — todos os testes passam
  - Executar `cargo build --release --features standalone` — binário funcional
  - Executar `cargo build --no-default-features --features clap-plugin --lib` — compila sem PipeWire
  - **Aceite:** Todos os 4 comandos acima passam com sucesso absoluto

> **📋 CHECKPOINT DE REVISÃO — Épico 0.1 concluído**
> Validar: Estrutura de diretórios, compilação dual-feature, todos os testes. "utils/*.sh" inteiro passando com sucesso.

---

### Épico 0.2 — Atualização de Documentação

> **Contexto:** O Épico 0.1 produziu mudanças arquiteturais significativas (estrutura tripartida `common/`+`standalone/`+`clap/`) e consolidou decisões técnicas chave (`clack-plugin`, `egui`+`baseview`). A documentação atual está **parcialmente desatualizada** em vários pontos críticos — alguns documentos referenciam módulos em paths antigos, e `clap_integration.md` ainda contém seção de decisão "Pendente" que contradiz o roadmap consolidado.
>
> **Ordem de execução:** 0.2.1 → 0.2.2 → 0.2.3 → 0.2.4 → 0.2.5 (sequencial — cada tarefa depende das anteriores para consistência cross-referencial).

- [x] **Tarefa 0.2.1** — Atualizar `README.md` (estado do projeto e modos de operação)
  
  **Contexto:** O README é a "porta de entrada" pública. Precisa refletir com clareza o estado atual: standalone estável + CLAP em alpha.
  
  **Alterações mandatórias:**
  
  - **Seção de Status** (topo, logo após o header): Adicionar badge/aviso bilíngue (PT-BR/EN):
    - PT-BR: `⚠️ Standalone PipeWire: ESTÁVEL (v1.4.3) | Plugin CLAP: EM DESENVOLVIMENTO (alpha)`
    - EN: `⚠️ Standalone PipeWire: STABLE (v1.4.3) | CLAP Plugin: IN DEVELOPMENT (alpha)`
  - **Nova Seção "Modos de Operação" / "Operation Modes"**: Explicar as duas features:
    - `standalone` (padrão): binário Linux com PipeWire, uso musical imediato
    - `clap-plugin`: biblioteca `.so` para DAWs, em desenvolvimento ativo
    - Incluir os dois comandos de build exatos com suas diferenças
  - **Seção Roadmap**: Substituir qualquer referência vaga a "futuro plugin" pelo roadmap concreto das Sprints 1-4 (scaffolding → áudio bypass → parâmetros CLAP → GUI egui)
  - **Seção Changelog / CHANGELOG.md**: Adicionar entrada `v1.5.0-alpha` com:
    - Reorganização tripartida de módulos (common/standalone/clap)
    - Decisão confirmada: `clack-plugin` como framework CLAP
    - 157 testes passando com zero regressões
  - **Verificar e corrigir paths de módulos** em exemplos de código dentro do README (se houver referências antigas como `nam_rs::spsc::`, atualizar para `nam_rs::common::spsc::`)
  
  **Aceite:** `grep -n "Pendente\|pending\|TODO\|FIXME" README.md` retorna zero resultados; documento bilíngue revisado.

- [x] **Tarefa 0.2.2** — Atualizar `docs/architecture.md` (estrutura tripartida e decisões ADR)
  
  **Contexto:** A Seção 4 já foi atualizada com a tabela tripartida durante o Épico 0.1. Faltam: diagrama visual de camadas, registro formal das decisões de framework, e seção de estratégia de feature flags.
  
  **Alterações mandatórias:**
  
  - **Seção 4 — Adicionar Diagrama Mermaid de Camadas** (logo após a tabela existente): Diagrama `graph TD` mostrando:
    - `src/clap/` e `src/standalone/` dependem de `src/common/`
    - `src/common/`, `src/dsp/`, `src/math/`, `src/models/`, `src/loader/` são agnósticos ao host
    - Seta indicando que `pipewire` é dependência **exclusiva** do caminho `standalone`
    - Seta indicando que `clack-plugin` é dependência **exclusiva** do caminho `clap`
  - **Nova Seção 4.1 — Estratégia de Compilação Condicional (Feature Flags)**:
    - Tabela com 3 builds: `standalone` (padrão), `clap-plugin`, e `no-default-features` (lib pura DSP)
    - Para cada build: comando exato, saída esperada (binário vs `.so`), dependências incluídas/excluídas
  - **Nova Seção 11 — Decisões de Arquitetura (ADR)**:
    - **ADR-002:** Framework CLAP: `clack-plugin` confirmado, `nih-plug` rejeitado. Justificativa: controle granular, overhead zero, sem VST3, sem GUI embutida forçada.
    - **ADR-003:** GUI: `egui` + `baseview` confirmado para Sprint 4. Justificativa: puro Rust, sem dependências C++, integração nativa com CLAP via `baseview`.
    - **ADR-004:** DAW primária de desenvolvimento: REAPER. Justificativa: ferramenta de debug sem igual (plugin scan rápido, hot-reload, análise de buffers variáveis). Bitwig e Studio One: validação premium pós-Sprint 4.
  - **Verificar Seção 10** ("Suporte a Plugins"): A referência `src/params.rs` está desatualizada — o arquivo foi movido para `src/common/params.rs`. Corrigir todas as referências de módulos para os novos paths.
  
  **Aceite:** `grep -n "src/params.rs\|src/spsc.rs\|src/diagnostics.rs\|src/audio_host.rs" docs/architecture.md` retorna zero resultados (todos os paths antigos eliminados).

- [x] **Tarefa 0.2.3** — Reescrever `docs/clap_integration.md` (decisão confirmada + extensões CLAP)
  
  **Contexto:** Este é o documento mais crítico a atualizar. A Seção 4 atual ("Decisão de Framework — Pendente") está **diretamente contraditória** com o roadmap consolidado e pode confundir qualquer colaborador. Deve ser reescrita, não apenas atualizada.
  
  **Alterações mandatórias:**
  
  - **Remover completamente a Seção 4 atual** ("Decisão de Framework (Pendente)") — incluindo toda a análise comparativa `nih-plug` vs `clack`
  - **Substituir por nova Seção 4 — Framework: `clack-plugin` (Decisão Confirmada)**:
    - Registrar a decisão: `clack-plugin`.
    - Motivo: controle granular sem overhead, zero dependências C++, mapeamento direto ao spec CLAP
    - `nih-plug` **descartado**: adiciona VST3, GUI embutida e abstração opinativa incompatíveis com RT do NAM-rs
    - Link para crate: `https://github.com/prokopyl/clack`
  - **Nova Seção 5 — Extensões CLAP Planejadas** (via `clack-extensions`):
    - `clap-ext-params`: Automação de parâmetros (`input_gain_db`, `output_gain_db`, `gate_threshold_db`, `bypass`)
    - `clap-ext-state`: Save/load de estado de projeto (serializa `model_path` + params)
    - `clap-ext-gui`: GUI nativa via `egui` + `baseview` (Sprint 4, diferida)
    - `clap-ext-thread-pool`: Pool de threads para pré-aquecimento de modelos sem bloquear a audio thread (Sprint 3)
    - Para cada extensão: Sprint alvo + flag de feature necessária
  - **Nova Seção 6 — Plugin Descriptor**:
    - Plugin ID: `"br.eti.fabiolima.nam-rs"` (RFC inversão de domínio)
    - Nome: `"NAM-rs Neural Amp Modeler"`
    - Vendor: `"Fabio Lima"`
    - URL: `"https://github.com/fabiohl/nam-rs"`
    - Features CLAP: `["audio-effect", "distortion", "gate", "simulator", "stereo"]`
  - **Atualizar Seção 5 atual → Seção 7 — DAWs Alvo de Validação**:
    - REAPER: **Primário** durante desenvolvimento (debug de performance, buffers variáveis)
    - Bitwig Studio: Validação premium.
    - Studio One: Validação premium.
    - CLAP-info / CLAP-host: Ferramentas CLI de contrato.
  - **Atualizar referências de paths** nas Seções 1 e 2:
    - `src/params.rs` → `src/common/params.rs`
    - `src/loader.rs` → `src/loader/` (é um módulo, não arquivo único)
    - `src/dsp/pipeline.rs` — verificar se ainda correto
  
  **Aceite:** `grep -n "Pendente\|pending\|nih-plug" docs/clap_integration.md` retorna zero resultados; documento não contém seções incompletas ou contradições com o backlog.

- [x] **Tarefa 0.2.4** — Atualizar `docs/dependencies.md` (dependências CLAP planejadas)
  
  **Contexto:** O `dependencies.md` cobre apenas as dependências existentes no `Cargo.toml` atual. As dependências CLAP ainda não foram adicionadas ao `Cargo.toml` (isso é feito na Sprint 1), mas devem ser **pré-documentadas** aqui como "planejadas", seguindo o padrão da tabela existente, para que a Tarefa 1.1.1 tenha guia de referência técnica.
  
  **Alterações mandatórias:**
  
  - **Adicionar nova Seção 4 — Dependências Planejadas (Sprint 1+)**:

    - Tabela com o mesmo formato da Seção 2, com coluna adicional "Sprint Alvo":

    | Crate              | Versão Planejada           | Feature Flag  | Sprint   | Justificativa                                                                                                                                                   |
    | ------------------ | -------------------------- | ------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `clack-plugin`     | `^0.3` (verificar última)  | `clap-plugin` | Sprint 1 | API Rust para implementação de plugins CLAP. Abstração tipada sobre `clap-sys` sem overhead de runtime. Escolhido sobre `nih-plug` por não forçar VST3 nem GUI. |
    | `clack-extensions` | `^0.3` (verificar última)  | `clap-plugin` | Sprint 1 | Extensões do spec CLAP (params, state, gui, thread-pool). Crate separado do `clack-plugin` para modularidade.                                                   |
    | `egui`             | `^0.31` (verificar última) | `clap-plugin` | Sprint 4 | Framework GUI imediato, puro Rust. Renderização de GPU via `wgpu`. Integrado ao CLAP via `baseview`.                                                            |
    | `baseview`         | `git (mão)`                | `clap-plugin` | Sprint 4 | Janela nativa multiplataforma para `egui` em contexto de plugin. Não publicado no crates.io — usar Git dep.                                                     |
  
  - **Atualizar Seção 1** (Dependências do Sistema): Adicionar nota que `clack-plugin` e `clack-extensions` **não requerem** dependências de sistema adicionais (puro Rust via `clap-sys` interno).
  
  **Aceite:** Tabela presente e formatada corretamente; ao executar `cargo search clack-plugin`, versão documentada é compatível com a disponível.

- [x] **Tarefa 0.2.5** — Validação final do Épico 0.2
  
  **Checklist obrigatório:**
  
  - [x] `grep -rn "src/params.rs\|src/spsc.rs\|src/diagnostics.rs\|src/audio_host.rs" docs/` → zero resultados (sem paths antigos)
  - [x] `grep -rn "Pendente\|pending\|nih-plug\|TODO\|FIXME" docs/` → zero resultados (sem incompletos)
  - [x] Cada documento tem cabeçalho SPDX correto
  - [x] `utils/lints.sh` — zero erros (nenhum arquivo .rs foi alterado, mas validar de qualquer forma)
  - [x] Revisar manualmente cross-referências entre `README.md`, `architecture.md`, `clap_integration.md` e `dependencies.md` — sem contradições
  
  **Aceite:** Todos os 5 itens do checklist marcados; documentação 100% sincronizada com o código e com o roadmap.

> **📋 CHECKPOINT DE REVISÃO — Sprint 0 concluída**
> Revisão completa: estrutura de diretórios, documentação, compilação dual-feature.

---

## Sprint 1 — Scaffolding CLAP e Esqueleto cdylib

**Objetivo:** Plugin CLAP mínimo que é detectado e carregado pelo REAPER sem crash. Bypass puro funcional (copia input → output sem processar).

**Estado atual do código (pré-Sprint 1):**

- `Cargo.toml`: sem `clack-plugin` / `clack-extensions`; feature `clap-plugin = []` vazia; sem `crate-type = ["cdylib"]`
- `src/clap/mod.rs`: stub vazio (apenas docstring + `#![cfg(feature = "clap-plugin")]`)
- `src/lib.rs`: já expõe `pub mod clap` sob `#[cfg(feature = "clap-plugin")]` — **não modificar**
- `src/common/params.rs`: `NamPluginParams` já existe — usar para estado compartilhado futuramente

**Referência da API:** `clack-plugin = "0.1"` — usar `clack_plugin::prelude::*`

---

### Épico 1.1 — Configuração de Build

- [x] **Tarefa 1.1.1** — Adicionar dependências CLAP ao `Cargo.toml`

  **Contexto:** O `Cargo.toml` atual tem `clap-plugin = []` sem dependências. Precisamos adicionar `clack-plugin` e `clack-extensions` como opcionais e vinculá-los à feature.

  **Alterações em `Cargo.toml`:**

  1. Na seção `[dependencies]`, adicionar após `rustfft`:

     ```toml
     clack-plugin = { version = "0.1", optional = true }
     clack-extensions = { version = "0.1", optional = true }
     ```

  2. Na seção `[features]`, alterar:

     ```toml
     # antes:
     clap-plugin = []
     # depois:
     clap-plugin = ["dep:clack-plugin", "dep:clack-extensions"]
     ```

  **Aceite:** `cargo check --no-default-features --features clap-plugin` compila sem erros.

- [ ] **Tarefa 1.1.2** — Configurar `crate-type = ["rlib", "cdylib"]` no `Cargo.toml`

  **Contexto:** Para gerar um `.so` carregável pelo REAPER, o crate precisa ser compilado como `cdylib`. O `rlib` deve ser mantido para que `cargo test` continue funcionando.

  **Alteração em `Cargo.toml`:** Após a seção `[[bin]]`, adicionar:

  ```toml
  [lib]
  name = "nam_rs"
  crate-type = ["rlib", "cdylib"]
  ```

  **Aceite:** `cargo build --no-default-features --features clap-plugin --lib` gera `target/debug/libnam_rs.so` sem erros.

> **📋 CHECKPOINT — Build CLAP funcional**

---

### Épico 1.2 — Implementação do Plugin Skeleton

- [ ] **Tarefa 1.2.1** — Criar `src/clap/descriptor.rs`

  **Contexto:** O descriptor é a identidade do plugin. O REAPER lê estas strings durante o scan.

  **Criar o arquivo `src/clap/descriptor.rs`** com o seguinte conteúdo:

  ```rust
  // SPDX-License-Identifier: MIT OR Apache-2.0
  // Copyright (c) 2026 Fábio Henrique de Lima Silva.

  //! Descriptor de identidade do plugin NAM-rs no formato CLAP.

  use clack_plugin::prelude::*;

  /// Retorna o descritor imutável do plugin.
  /// Lido pelo host durante scan — deve ser determinístico e sem alocações.
  pub fn nam_descriptor() -> PluginDescriptor {
      PluginDescriptor::new("br.eti.fabiolima.nam-rs", "NAM-rs Neural Amp Modeler")
          .with_vendor("Fabio Lima")
          .with_url("https://github.com/fabiohl/nam-rs")
          .with_description("Real-time Neural Amp Modeler plugin (CLAP)")
          .with_features([
              CLAP_PLUGIN_FEATURE_AUDIO_EFFECT,
              CLAP_PLUGIN_FEATURE_DISTORTION,
              CLAP_PLUGIN_FEATURE_GATE,
              "simulator",
              CLAP_PLUGIN_FEATURE_STEREO,
          ])
  }
  ```

  **Atenção:** Verificar nomes exatos das constantes em `clack_plugin::prelude` (podem ser strings literais se a versão 0.1 não exportar constantes). Alternativa segura:

  ```rust
  .with_features(["audio-effect", "distortion", "gate", "simulator", "stereo"])
  ```

  **Aceite:** `cargo check --no-default-features --features clap-plugin` compila sem erros.

- [ ] **Tarefa 1.2.2** — Criar structs `NamClapPlugin`, `NamClapShared`, `NamClapMainThread` em `src/clap/plugin.rs`

  **Contexto:** A trait `Plugin` define os três componentes do ciclo de vida. Por ora, `Shared` e `MainThread` são structs unitárias — a lógica real vem na Sprint 2.

  **Criar o arquivo `src/clap/plugin.rs`** com o seguinte conteúdo:

  ```rust
  // SPDX-License-Identifier: MIT OR Apache-2.0
  // Copyright (c) 2026 Fábio Henrique de Lima Silva.

  //! Definição do plugin NAM-rs e seus componentes de ciclo de vida CLAP.

  use clack_plugin::prelude::*;
  use crate::clap::descriptor::nam_descriptor;
  use crate::clap::processor::NamClapProcessor;

  /// Estado compartilhado entre a audio thread e a main thread (lock-free).
  /// Sprint 1: vazio — será preenchido com AtomicF32 para parâmetros na Sprint 2.
  pub struct NamClapShared;

  /// Estado exclusivo da main thread (carregamento de modelos, state save/load).
  /// Sprint 1: vazio — será preenchido na Sprint 2.
  pub struct NamClapMainThread;

  /// Plugin NAM-rs: ponto de entrada principal do ciclo de vida CLAP.
  pub struct NamClapPlugin;

  impl Plugin for NamClapPlugin {
      type AudioProcessor<'a> = NamClapProcessor;
      type Shared<'a> = NamClapShared;
      type MainThread<'a> = NamClapMainThread;
  }

  impl DefaultPluginFactory for NamClapPlugin {
      fn get_descriptor() -> PluginDescriptor {
          nam_descriptor()
      }

      fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
          Ok(NamClapShared)
      }

      fn new_main_thread<'a>(
          _host: HostMainThreadHandle<'a>,
          _shared: &'a Self::Shared<'a>,
      ) -> Result<Self::MainThread<'a>, PluginError> {
          Ok(NamClapMainThread)
      }
  }
  ```

  **Aceite:** `cargo check --no-default-features --features clap-plugin` compila sem erros.

- [ ] **Tarefa 1.2.3** — Criar `src/clap/processor.rs` (bypass puro)

  **Contexto:** O `PluginAudioProcessor` é chamado pela audio thread do host. Por ora, implementa bypass puro: copia input para output. **Não há alocações aqui.**

  **Criar o arquivo `src/clap/processor.rs`** com o seguinte conteúdo:

  ```rust
  // SPDX-License-Identifier: MIT OR Apache-2.0
  // Copyright (c) 2026 Fábio Henrique de Lima Silva.

  //! Processador de áudio CLAP — Sprint 1: bypass puro (input → output).

  use clack_plugin::prelude::*;
  use crate::clap::plugin::{NamClapShared, NamClapMainThread};

  /// Processador de áudio RT-safe. Executa na audio thread do host.
  /// Sprint 1: bypass — copia cada amostra de input para output sem processar.
  pub struct NamClapProcessor;

  impl<'a> PluginAudioProcessor<'a, NamClapShared, NamClapMainThread> for NamClapProcessor {
      fn activate(
          _host: HostAudioProcessorHandle<'a>,
          _main_thread: &mut NamClapMainThread,
          _shared: &'a NamClapShared,
          _audio_config: PluginAudioConfiguration,
      ) -> Result<Self, PluginError> {
          Ok(Self)
      }

      fn process(
          &mut self,
          _process: Process,
          mut audio: Audio,
          _events: Events,
      ) -> Result<ProcessStatus, PluginError> {
          for mut port_pair in &mut audio {
              let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
                  continue;
              };
              for channel_pair in channel_pairs {
                  match channel_pair {
                      ChannelPair::InputOnly(_) => {}
                      ChannelPair::OutputOnly(buf) => buf.fill(0.0),
                      ChannelPair::InputOutput(input, output) => {
                          output.copy_from_slice(input);
                      }
                      ChannelPair::InPlace(_) => {} // in-place: nada a fazer no bypass
                  }
              }
          }
          Ok(ProcessStatus::Continue)
      }
  }
  ```

  **Aceite:** `cargo check --no-default-features --features clap-plugin` compila sem erros.

- [ ] **Tarefa 1.2.4** — Atualizar `src/clap/mod.rs` e exportar entry point em `src/lib.rs`

  **Parte A — Reescrever `src/clap/mod.rs`:**

  ```rust
  // SPDX-License-Identifier: MIT OR Apache-2.0
  // Copyright (c) 2026 Fábio Henrique de Lima Silva.

  //! Integração do NAM-rs como plugin no formato CLAP (CLever Audio Plug-in).
  //!
  //! Ativado via feature flag `clap-plugin`. Totalmente isolado do PipeWire.

  pub mod descriptor;
  pub mod plugin;
  pub mod processor;

  use clack_plugin::prelude::*;
  use plugin::NamClapPlugin;

  clack_export_entry!(SinglePluginEntry<NamClapPlugin>);
  ```

  **Parte B — `src/lib.rs` já está correto** (expõe `pub mod clap` sob feature). Não modificar.

  **Aceite:** `cargo build --no-default-features --features clap-plugin --lib` gera `target/debug/libnam_rs.so` sem erros nem warnings.

- [ ] **Tarefa 1.2.5** — Criar script `utils/build-clap.sh`

  **Criar `utils/build-clap.sh`** com o seguinte conteúdo:

  ```bash
  #!/bin/bash
  # SPDX-License-Identifier: MIT OR Apache-2.0
  # Copyright (c) 2026 Fábio Henrique de Lima Silva.
  #
  # Build e instalação do plugin NAM-rs no formato CLAP.
  # Gera libnam_rs.so e copia para ~/.clap/nam-rs.clap

  set -euo pipefail

  TARGET_DIR="target/release"
  CLAP_DIR="$HOME/.clap"
  PLUGIN_NAME="nam-rs.clap"

  echo "🔨 Building NAM-rs CLAP plugin (release)..."
  cargo build --release --no-default-features --features clap-plugin --lib

  echo "📁 Instalando em $CLAP_DIR/$PLUGIN_NAME ..."
  mkdir -p "$CLAP_DIR"
  cp "$TARGET_DIR/libnam_rs.so" "$CLAP_DIR/$PLUGIN_NAME"

  echo "✅ Plugin instalado: $CLAP_DIR/$PLUGIN_NAME"
  echo "   Reabra o REAPER e faça um novo scan de plugins CLAP."
  ```

  Tornar executável: `chmod +x utils/build-clap.sh`

  **Aceite:** `./utils/build-clap.sh` executa sem erros e `~/.clap/nam-rs.clap` existe.

> **📋 CHECKPOINT — Plugin skeleton pronto para teste**

---

### Épico 1.3 — Validação e Estabilidade

> OBS1: Ótimo momento para leitura humana geral do estado do código em `src/`.
> OBS2: Hora de instalar e configurar o REAPER para os testes funcionais.

- [ ] **Tarefa 1.3.1** — Validar detecção no REAPER

  1. Executar `./utils/build-clap.sh`
  2. Abrir REAPER → `Options` → `Preferences` → `Plug-ins` → `CLAP` → `Re-scan`
  3. Verificar que `NAM-rs Neural Amp Modeler` (vendor: `Fabio Lima`) aparece na lista
  4. Inserir numa faixa de áudio via `FX → Add`

  **Aceite:** REAPER detecta e instancia o plugin sem segfault ou mensagem de erro no log de plugins.

- [ ] **Tarefa 1.3.2** — Validar bypass no REAPER

  1. Com plugin inserido, reproduzir áudio na faixa
  2. Comparar áudio com e sem plugin (usar Track FX bypass do REAPER)
  3. O áudio deve ser idêntico (bypass puro)

  **Aceite:** Áudio limpo, sem artefatos, sem dropouts, sem crashes. Latência reportada pelo REAPER: 0 amostras.

- [ ] **Tarefa 1.3.3** — Validar que standalone não regrediu

  ```bash
  cargo build --release --features standalone
  cargo test
  utils/lints.sh
  ```

  **Aceite:** Build, testes (157+) e lints passam sem erros ou regressões.

- [ ] **Tarefa 1.3.4** — Executar suíte de validação completa

  ```bash
  utils/lints.sh   # zero erros
  cargo test       # todos os testes passam
  ```

  **Aceite:** Suíte completa sem falhas.

> **📋 CHECKPOINT DE REVISÃO — Sprint 1 concluída**
> Plugin CLAP bypass funcional no REAPER. Standalone intacto.

---

## Sprint 2 — Roteamento DSP e Params CLAP

**Objetivo:** Plugin processa áudio real (inferência neural) e expõe parâmetros automáveis na DAW.

---

### Épico 2.1 — Integração do Pipeline DSP

- [ ] **Tarefa 2.1.1** — Adaptar buffers CLAP → Pipeline DSP
  
  - No `process()`, converter `ChannelPair` para slices `&[f32]` / `&mut [f32]`
  - Invocar `DspPipeline::process()` com buffers convertidos
  - Respeitar restrições RT: zero-alloc, zero-lock, zero-I/O
  - **Aceite:** Compilação sem erros

- [ ] **Tarefa 2.1.2** — Carregar modelo no `activate()`
  
  - No `activate()` (cold-path): carregar modelo default ou do state salvo
  - Instanciar `NamModel` + `NamResampler` (se sample rate ≠ 48kHz)
  - Pré-alocar todos os buffers internos
  - **Aceite:** Plugin carrega modelo .nam e processa áudio no REAPER

- [ ] **Tarefa 2.1.3** — Validar inferência neural no REAPER
  
  - Inserir plugin, carregar modelo real (.nam)
  - Processar guitarra em tempo real
  - Verificar qualidade do áudio (sem artefatos, sem clipping inesperado)
  - **Aceite:** Áudio processado é perceptualmente idêntico ao standalone

> **📋 CHECKPOINT — DSP funcional no CLAP**

---

### Épico 2.2 — Parâmetros CLAP

- [ ] **Tarefa 2.2.1** — Implementar `src/clap/params_clap.rs`
  
  - Definir parâmetros CLAP via `clack-extensions`:
    - `input_gain_db` (ID=0, range: -24..+24 dB, default: 0.0)
    - `output_gain_db` (ID=1, range: -24..+24 dB, default: 0.0)
    - `gate_threshold_db` (ID=2, range: -96..-20 dB, default: -70.0)
    - `bypass` (ID=3, stepped 0/1, default: 0)
  - Mapear ↔ `NamPluginParams`
  - **Aceite:** Parâmetros aparecem na interface do REAPER

- [ ] **Tarefa 2.2.2** — Sincronização de params via eventos CLAP (sample-accurate)
  
  - No `process()`, iterar `Events::input()` para `CLAP_EVENT_PARAM_VALUE`
  - Aplicar mudanças via atomics (análogo ao SPSC existente)
  - **Aceite:** Automação de parâmetros funcional no REAPER

- [ ] **Tarefa 2.2.3** — Implementar State (Save/Load)
  
  - Usar `clack-extensions` State para serializar/desserializar
  - Persistir: caminho do modelo + valores de parâmetros
  - Formato: JSON compacto via `serde_json`
  - **Aceite:** Salvar/reabrir projeto no REAPER preserva estado completo

- [ ] **Tarefa 2.2.4** — Validação final Sprint 2
  
  - Automação funcional de todos os 4 parâmetros
  - Save/load de estado persistente
  - `utils/lints.sh` + `cargo test` — zero falhas
  - **Aceite:** Suíte completa sem regressões

> **📋 CHECKPOINT DE REVISÃO — Sprint 2 concluída**
> Plugin com DSP real + parâmetros automáveis + state.

---

## Sprint 3 — Audio Ports, Latência e Estabilidade

**Objetivo:** Plugin robusto com declaração correta de portas, latência reportada, e estabilidade de longa duração.

---

### Épico 3.1 — Extensions de Portas e Latência

- [ ] **Tarefa 3.1.1** — Implementar Audio Ports Extension
  
  - Declarar: 1 porta stereo de entrada + 1 porta stereo de saída
  - Suporte in-place processing
  - **Aceite:** REAPER exibe portas corretamente

- [ ] **Tarefa 3.1.2** — Implementar `clap.latency` Extension
  
  - Latência = samples do resampler (se ativo) + pipeline DSP
  - Notificar host quando latência mudar (ex: troca de modelo com sample rate diferente)
  - **Aceite:** REAPER compensa latência automaticamente

> **📋 CHECKPOINT — Portas e latência corretos**

---

### Épico 3.2 — Estabilidade e Testes CLAP

- [ ] **Tarefa 3.2.1** — Teste de zero-allocation no process callback CLAP
  
  - Adaptar `CountingAllocator` para contexto CLAP
  - Garantir zero heap-allocs no hot-path
  - **Aceite:** Teste passa com zero alocações

- [ ] **Tarefa 3.2.2** — Teste de hot-swap de modelo durante processamento
  
  - Trocar modelo enquanto plugin está processando áudio
  - Verificar transição suave sem crashes
  - **Aceite:** Troca funcional sem artefatos audíveis

- [ ] **Tarefa 3.2.3** — Teste com block sizes variáveis (1–4096 samples)
  
  - Processar com diferentes buffer sizes configurados no REAPER
  - **Aceite:** Funcional em todos os tamanhos de bloco testados

- [ ] **Tarefa 3.2.4** — Testes de integração automatizados (`tests/clap_integration_test.rs`)
  
  - Usar `clack-host` (dev-dependency) para carregar plugin programaticamente
  - Validar: processamento, params, state — tudo sem DAW real
  - **Aceite:** Testes passam no CI local

- [ ] **Tarefa 3.2.5** — Validação final Sprint 3
  
  - `utils/lints.sh` + `cargo test` — zero falhas
  - Soak test manual no REAPER (1h+ de processamento contínuo)
  - **Aceite:** Estabilidade comprovada

> **📋 CHECKPOINT DE REVISÃO — Sprint 3 concluída**
> Plugin robusto, estável, com portas e latência corretos.

---

## Sprint 4 — Interface Gráfica (egui + baseview)

**Objetivo:** GUI funcional embarcada na janela do REAPER, desacoplada do hot-path DSP.

---

### Épico 4.1 — Infraestrutura GUI

- [ ] **Tarefa 4.1.1** — Adicionar dependências GUI via `cargo add`
  
  - `cargo add egui --optional`
  - `cargo add baseview --git https://github.com/RustAudio/baseview --optional`
  - Avaliar `egui-baseview` ou integração manual
  - Atualizar feature `clap-plugin` para incluir deps GUI
  - **Aceite:** `cargo check --no-default-features --features clap-plugin` compila

- [ ] **Tarefa 4.1.2** — Implementar `src/clap/gui.rs` — Extension CLAP GUI
  
  - Implementar `clap.gui` extension via clack-extensions
  - Criar janela via `baseview` com handle do host (`set_parent` / `raw-window-handle`)
  - Renderizar loop egui na janela embarcada
  - Comunicação GUI ↔ DSP via SPSC existente (`Ordering::Relaxed`)
  - **Aceite:** Janela abre no REAPER com conteúdo egui visível

> **📋 CHECKPOINT — Janela GUI embarcada funcional**

---

### Épico 4.2 — Design e Interação

- [ ] **Tarefa 4.2.1** — Implementar painel de controle
  
  - Seletor de modelo (.nam/.namb) com botão de browse
  - Knobs rotativos: Input Gain, Output Gain, Gate Threshold
  - Toggle: Bypass
  - **Aceite:** Controles funcionais e sincronizados com params CLAP

- [ ] **Tarefa 4.2.2** — Implementar visualizadores
  
  - Indicador de nível (VU meter) — leitura via SPSC atômico
  - Telemetria DSP (latência mediana/P99) — leitura via SPSC
  - **Aceite:** Visualizadores atualizam em tempo real

- [ ] **Tarefa 4.2.3** — Sincronização bidirecional GUI ↔ Host
  
  - Mudanças na GUI → notificam host (param change)
  - Automação do host → reflete na GUI
  - **Aceite:** Parâmetros sincronizados em ambas direções

- [ ] **Tarefa 4.2.4** — Validação final Sprint 4
  
  - GUI abre e fecha sem crash
  - Redimensionamento funcional (se aplicável)
  - Zero impacto no hot-path DSP (verificar com telemetria)
  - `utils/lints.sh` + `cargo test` — zero falhas
  - **Aceite:** GUI completa e estável

> **📋 CHECKPOINT DE REVISÃO — Sprint 4 concluída**
> GUI egui funcional e integrada no REAPER.

---

## Sprint 5 — Thread Pool e Otimização Multi-Instância

**Objetivo:** Suporte eficiente a múltiplas instâncias simultâneas via `clap.thread-pool`.

---

### Épico 5.1 — Extension Thread Pool

- [ ] **Tarefa 5.1.1** — Implementar `clap.thread-pool` Extension
  
  - Detectar suporte do host via `HostSharedHandle`
  - Delegar cálculos pesados (GEMV WaveNet, Conv1D) ao thread pool do host
  - Implementar fallback: single-thread se host não suporta a extension
  - **Aceite:** Plugin utiliza thread pool quando disponível

- [ ] **Tarefa 5.1.2** — Stress Test multi-instância no REAPER
  
  - Teste com 8 instâncias simultâneas — verificar estabilidade
  - Teste com 16 instâncias — medir CPU total
  - Teste com 30 instâncias — stress extremo, verificar ausência de glitches
  - Comparar CPU vs modo standalone equivalente
  - **Aceite:** 16 instâncias estáveis sem glitches audíveis

- [ ] **Tarefa 5.1.3** — Otimização de memória por instância
  
  - Avaliar compartilhamento de pesos entre instâncias (mesmo modelo carregado)
  - Reduzir footprint por instância quando possível
  - **Aceite:** Footprint de memória documentado e otimizado

- [ ] **Tarefa 5.1.4** — Validação final Sprint 5
  
  - `utils/lints.sh` + `cargo test` — zero falhas
  - Stress test aprovado
  - **Aceite:** Multi-instância estável e eficiente

> **📋 CHECKPOINT DE REVISÃO — Sprint 5 concluída**
> Thread pool funcional, multi-instância estável.

---

## Sprint 6 — Polish, Validação Final e Release Alpha

**Objetivo:** Preparar para release pública v2.0.0.

---

### Épico 6.1 — Validação em DAWs

- [ ] **Tarefa 6.1.1** — Validação final no REAPER
  
  - Teste completo end-to-end: load, process, automate, save/load, GUI, multi-instância
  - Regressão: comparar qualidade de áudio com standalone
  - **Aceite:** 100% funcional sem ressalvas

- [ ] **Tarefa 6.1.2** — Validação com `clap-info` / `clap-validator`
  
  - Executar ferramentas CLI de validação de contrato CLAP
  - Corrigir quaisquer non-conformances
  - **Aceite:** Zero erros no validator

- [ ] **Tarefa 6.1.3** — Validação no Bitwig Studio (Linux)
  
  - Instalar e configurar Bitwig Studio
  - Testar: detecção, instanciação, processamento, params, state, GUI
  - Foco: conformidade CLAP (Bitwig é referência de implementação CLAP)
  - **Aceite:** Plugin funcional no Bitwig

- [ ] **Tarefa 6.1.4** — Validação no Presonus Studio One (Linux)
  
  - Instalar e configurar Studio One
  - Testar: detecção, instanciação, processamento, params, state, GUI
  - Foco: compatibilidade com DAW líder de mercado
  - **Aceite:** Plugin funcional no Studio One

> **📋 CHECKPOINT — Validação cross-DAW aprovada**

---

### Épico 6.2 — Documentação e Release

- [ ] **Tarefa 6.2.1** — Documentação final do plugin
  
  - Guia de instalação do plugin CLAP (usuário final)
  - Guia de build para contribuidores
  - Screenshots da GUI no REAPER
  - **Aceite:** Documentação completa e acessível

- [ ] **Tarefa 6.2.2** — Atualizar `README.md` para v2.0.0
  
  - Documentar modo plugin CLAP como funcionalidade alpha
  - Atualizar instruções de build (dual-target)
  - Atualizar Changelog
  - **Aceite:** README reflete estado real do projeto

- [ ] **Tarefa 6.2.3** — Release v2.0.0
  
  - Tag git: `v2.0.0`
  - Changelog completo
  - Binários: standalone + plugin `.clap`
  - **Aceite:** Release publicado no GitHub

> **📋 CHECKPOINT DE REVISÃO — Sprint 6 concluída**
> Release v2.0.0 publicado. Projeto pronto para feedback da comunidade.
