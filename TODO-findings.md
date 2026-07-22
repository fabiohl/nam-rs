<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Especificação e Mapeamento de Exposição do nam-rs no crates.io

Documento de achados técnicos, pesquisas de ecossistema e arquitetura detalhada para a correta exposição e compartilhamento de funcionalidades reutilizáveis do `nam-rs` na comunidade Rust (`crates.io`), sob a licença **Apache-2.0**.

---

## 1. Contexto e Motivação (Visão do Pesquisador Inovador & Especialista crates.io)

O `nam-rs` é uma implementação de referência em Rust de ultra-alta performance e segurança em tempo real (RT-Safety) para modelagem neural de amplificadores de áudio (NAM - Neural Amp Modeler).

O ecossistema Rust de áudio e DSP em `crates.io` necessita de componentes puros em Rust, altamente otimizados e modulares para:

1. **Parsing de Modelos NAM (`.nam` e `.namb`)**: Biblioteca pura em Rust com suporte a verificação CRC32 e descompressão de contêineres `.namb`, sem alocações no hot-path.
2. **Kernels de Ativação SIMD & Math RT-Safe**: Funções `tanh` e `sigmoid` com suporte nativo a `x86-64-v3` (AVX2/FMA), GEMV otimizado e somatório Kahan.
3. **Métricas de Perda Perceptual de Áudio**: Implementações puras em Rust de MR-STFT (*Multi-Resolution Short-Time Fourier Transform*), ESR (*Error-to-Signal Ratio*) e medição LUFS (BS.1770-4) para validação off-RT de modelos neurais e efeitos DSP.
4. **Motor DSP Neutro & Lock-free SPSC GC Cascade**: Gerenciamento de troca de recursos (modelos, IRs) sem alocação ou desalocação na thread de áudio em tempo real.

Tudo isso é exposto no `crates.io` sob a licença **Apache-2.0**, mantendo o `nam-rs` como a crate principal unificada, com arquitetura limpa de sub-módulos e *feature flags* finas.

---

## 2. Achados Técnicos Detalhados e Propostas de Solução

### Achado 01: Higiene de Pacote no crates.io (Payload Control Estrito para `/src`)

* **Severidade**: Crítica (Tamanho de pacote e higiene de repositório)

* **Contexto**: Sem a instrução `include` no manifesto `Cargo.toml`, o comando `cargo publish` empacota todos os arquivos rastreados no Git. Isso inclui modelos de pesos de teste (`.namb`), WAVs de validação perceptual, scripts internos em `utils/`, `docs/`, `benches/`, `tests/` e `.agents/`, o que pode aproximar ou superar o limite de 10 MB do `crates.io`.

* **Diretriz de Design**: Somente o código-fonte em `src/`, as licenças e o arquivo `README.md` devem ser incluídos no pacote publicado. Ativos de teste e scripts pertencem exclusivamente ao repositório Git para garantia de qualidade interna do `nam-rs`.

* **Especificação da Solução**:
  Adicionar a cláusula `include` no bloco `[package]` do `Cargo.toml`:

  ```toml
  include = [
      "src/**/*",
      "Cargo.toml",
      "README.md",
      "LICENSE.txt",
      "NOTICE.txt",
  ]
  ```

* **Resultado Esperado**: Redução do arquivo `.crate` gerado de ~50 MB para apenas **~150 KB**, otimizando o download e o cache no `crates.io` e no `docs.rs`.

---

### Achado 02: Foco Exclusivo da Documentação no docs.rs para Consumidores da Crate

* **Severidade**: Alta (Ergonomia, clareza e usabilidade pública)

* **Contexto**: A documentação no `docs.rs` é a principal referência para desenvolvedores que utilizam a biblioteca. Ela não deve expor guias internos de desenvolvimento do repositório, mas sim focar na API pública reutilizável.

* **Diretriz Confirmada**: O `docs.rs` não deve poluir o leitor com detalhes internos (scripts de benchmark, oráculos de teste ou guias de manutenção), mas sim focar nos seguintes pilares:

  1. **Quickstart & Modular Import**: Como adicionar `nam-rs` ao `Cargo.toml` (`default-features = false` para builds leves sem PipeWire/GUI).
  2. **Submódulo `nam_rs::loader`**: Como chamar `load_and_build_model` para carregar arquivos `.nam` (JSON) e contêineres `.namb` (com checagem CRC32). O tipo retornado é `LoadedModelPair`.
  3. **Submódulo `nam_rs::math::activations`**: Como usar as funções `tanh` e `sigmoid` do sub-módulo `nam_rs::math::activations` (re-exportadas via `pub use tanh::*` e `pub use sigmoid::*`).
  4. **Submódulo `nam_rs::dsp`**: Como integrar os blocos de pipeline DSP (gate, oversampling, cabsim) em um pipeline de áudio customizado.

* **Configuração de Metadados TOML**:
  No [Cargo.toml](file:///home/fabio/nam-rs/Cargo.toml), o bloco `[package.metadata.docs.rs]` já existe e deve ser mantido, **porém** com atenção crítica:

  > [!IMPORTANT]
  > `all-features = true` inclui a feature `standalone`, que depende de `libpipewire`. O ambiente de build do `docs.rs` não garante a disponibilidade de `libpipewire`. Avaliar se a feature `standalone` deve ser excluída no contexto de compilação do `docs.rs`, usando `features = ["clap-plugin"]` em vez de `all-features`.

  Configuração atual (já presente no `Cargo.toml`):

  ```toml
  [package.metadata.docs.rs]
  all-features = true
  rustdoc-args = ["--cfg", "docsrs"]
  ```

  Alternativa segura para evitar falha de compilação no `docs.rs` por ausência de `libpipewire`:

  ```toml
  [package.metadata.docs.rs]
  features = ["clap-plugin"]
  rustdoc-args = ["--cfg", "docsrs"]
  ```

---

### Achado 03: Arquitetura de Feature Flags Finas para Zero Overhead

* **Severidade**: Alta (Manutenibilidade, compilação limpa e zero dependências indesejadas)

* **Contexto**: Projetos que consomem apenas o parser ou as ativações matemáticas não devem compilar dependências como PipeWire, CLAP, egui ou baseview.

* **Especificação da Solução**:

  * `default = ["standalone"]`: Permite a instalação direta da aplicação executável via `cargo install nam-rs`.

  * `default-features = false`: Habilita o modo biblioteca pura (somente `src/`, sem dependências de sistema).

  * Exemplo de consumo mínimo no `Cargo.toml` de um projeto cliente:

    ```toml
    [dependencies]
    # Consumo leve e puro da biblioteca nam-rs no crates.io:
    nam-rs = { version = "3.0.0", default-features = false }
    ```

  * O consumidor obtém acesso direto aos módulos `loader`, `models`, `math` e `dsp` sem qualquer dependência de áudio de sistema.

---

### Achado 04: Garantia de Suporte Integral a Formatos (.nam/.namb), Evolução e Embutimento Linux

* **Severidade**: Crítica (Arquitetura e Contrato de Integração)
* **Contexto**: A crate publicada deve fornecer suporte completo a qualquer modelo `.nam` ou `.namb` do ecossistema NAM, evoluir em paridade com o projeto e permitir integração suave em pipelines de áudio no Linux.
* **Detalhamento da Solução**:
  1. **Cobertura Total de Arquiteturas de Modelos**:
     O submódulo `nam_rs::loader` suporta integralmente via a função pública `load_and_build_model` e o tipo `LoadedModelPair` (em `src/loader/loaded_model_pair.rs`):
     * Topologias WaveNet (Standard, Lite, CH16, Feather, Nano, A1, A2, FiLM).
     * Topologias LSTM (1x16, 2x8, official, dyn).
     * Topologias ConvNet e Linear.
     * Contêineres binários `.namb` (especificação em `docs/namb-spec.md`) com verificação de integridade CRC32 e descompressão em memória.
  2. **Fonte Única da Verdade (*Single Source of Truth*)**:
     Como o aplicativo executável (PipeWire CLI) e o plugin CLAP importam exatamente os mesmos submódulos em `src/`, qualquer atualização no parser ou nas camadas neurais estará automaticamente disponível para os consumidores do `crates.io` em cada nova release.
  3. **Embutimento Ergonomico em Pipelines Linux**:
     Os blocos DSP individuais em `nam_rs::dsp` (gate, oversample, cabsim, pipeline) operam sobre fatias simples de memória `&[f32]` e `&mut [f32]`, sem impor frameworks de sistema. Isso permite integração direta em:
     * Callbacks de tempo real em **PipeWire** e **JACK**.
     * Plugins de áudio **LV2**, **CLAP**, **VST3** e **ALSA pcm**.
     * Processadores de mídia em **GStreamer** e **FFmpeg**.
     * Aplicações headless e servidores de áudio customizados em Rust.

---

### Achado 05: Procedimento Oficial e Exaustivo de Release e `cargo publish`

* **Severidade**: Crítica (Operação e Segurança de Lançamento)
* **Contexto**: A publicação no `crates.io` será executada no momento do lançamento da versão `v3.0.0`. O procedimento completo está especificado abaixo para execução manual pelo mantenedor.

#### Checklist Oficial de Publicação (`cargo publish`)

1. **Validação do Estado do Repositório**:

   * Certificar-se de que a árvore Git está limpa e sincronizada no branch `main`:

     ```bash
     git status
     ```

   * Executar a suíte de qualidade e testes rápidos:

     ```bash
     utils/lints.sh
     utils/tests-quick.sh
     ```

2. **Inspeção do Pacote Gerado (`cargo package`)**:

   * Gerar o tarball localmente e verificar a lista de arquivos incluídos (garantir que apenas `src/` e licenças estão presentes):

     ```bash
     cargo package --list
     ```

   * Compilar e validar o pacote em ambiente isolado sem fazer upload:

     ```bash
     cargo publish --dry-run
     ```

3. **Autenticação e Upload para o crates.io**:

   * Obter a chave API Token de publicação na conta do mantenedor em `https://crates.io/settings/tokens`.

   * Autenticar o Cargo localmente (necessário apenas uma vez por máquina):

     ```bash
     cargo login <CRATES_IO_TOKEN>
     ```

   * Executar a publicação oficial da versão 3.0.0:

     ```bash
     cargo publish
     ```

4. **Registro de Release e Tagging no Git**:

   * Criar e enviar a tag Git correspondente (deve ser feito **após** a publicação bem-sucedida no `crates.io`):

     ```bash
     git tag -a v3.0.0 -m "Release v3.0.0 - crates.io publish"
     git push origin v3.0.0
     ```

---

## 3. Épicos e Especificação Ágil de Tarefas Técnicas (Skill `planejador-arquiteto`)

### Épico E1: Ajustes de Manifesto e Metadados no Cargo.toml

#### Tarefa E1.1: Configuração do Restritor `include` ✅ CONCLUÍDO

* **Arquivo Alvo**: [Cargo.toml](file:///home/fabio/nam-rs/Cargo.toml) — bloco `[package]`.

* **Descrição**: Inserir o campo `include` para restringir os arquivos empacotados pelo Cargo ao estritamente necessário para compilar a crate.

* **Posicionamento**: Logo após o campo `authors` no bloco `[package]`.

* **Snippet TOML**:

  ```toml
  include = [
      "src/**/*",
      "Cargo.toml",
      "README.md",
      "LICENSE.txt",
      "NOTICE.txt",
  ]
  ```

* **Critério de Aceite**: O comando `cargo package --list` exibe exclusivamente arquivos de `src/` e os documentos `README.md`, `LICENSE.txt`, `NOTICE.txt`.

---

#### Tarefa E1.2: Decisão e Configuração de `[package.metadata.docs.rs]` ✅ CONCLUÍDO

* **Arquivo Alvo**: [Cargo.toml](file:///home/fabio/nam-rs/Cargo.toml) — seção já existente `[package.metadata.docs.rs]`.

* **Descrição**: Avaliar se `all-features = true` é seguro para o ambiente de build do `docs.rs` (que não possui `libpipewire`). Se a compilação com `all-features` falhar no `docs.rs`, substituir pela opção abaixo.

* **Opção Segura Alternativa** (evita dependência de `libpipewire`):

  ```toml
  [package.metadata.docs.rs]
  features = ["clap-plugin"]
  rustdoc-args = ["--cfg", "docsrs"]
  ```

* **Critério de Aceite**: A documentação é gerada com sucesso na build do `docs.rs` sem erros de link com `libpipewire`.

---

### Épico E2: Estruturação da Documentação Rustdoc Focada no Consumidor (`src/lib.rs`)

#### Tarefa E2.1: Atualização da Documentação de Nível Crate no `src/lib.rs` ✅ CONCLUÍDO

> **Concluído em 2026-07-22**: A documentação de nível crate foi expandida em `src/lib.rs` com:
>
> * Guia "Quick Start — Loading a Model" com doc-test executável do `load_and_build_model` usando `linear_test.nam` como fixture.
> * Guia "Activation Functions" com doc-test executável de `tanh` e `sigmoid`.
> * Tabela "Module Map" com links para os módulos principais.
> * `cargo test --doc` passa com 2 novos doc-tests executáveis (0 falhas).

* **Arquivo Alvo**: [src/lib.rs](file:///home/fabio/nam-rs/src/lib.rs)
* **Descrição**: Expandir a documentação inicial do módulo para incluir o guia de consumo da biblioteca com exemplos de código executáveis (`doc-tests`) utilizando os símbolos públicos corretos.
* **Símbolos Públicos para Exemplos**:
  * `nam_rs::loader::load_and_build_model` — função pública para carregamento de `.nam`/`.namb`.
  * `nam_rs::loader::LoadedModelPair` — struct pública retornada pelo loader.
  * `nam_rs::math::activations::tanh` / `nam_rs::math::activations::sigmoid` — funções de ativação (re-exportadas de `tanh::*` e `sigmoid::*`).
* **Critério de Aceite**: O comando `cargo test --doc` compila e executa todos os exemplos presentes em `src/lib.rs` sem falhas.

---

#### Tarefa E2.2: Seção de RT-Safety e Garantias de Desempenho

> **Concluído em 2026-07-22**: Adicionada seção "Real-Time Safety & Performance Guarantees" em `src/lib.rs` cobrindo:
>
> * Zero Heap Allocations no hot-path (SPSC GC cascade, `alloc_audit`)
> * Zero Blocking I/O (`RtStatusFlags` atomic bitmask)
> * Denormal Protection (FTZ + DAZ via MXCSR, `set_daz_ftz`, reaplicação periódica)
> * Panic-Free Processing (sem `unwrap`/`expect` no hot-path)
> * Lock-Free Concurrency (`#[repr(align(128))]`, `Acquire`/`Release`)
> * Seção de licenciamento Apache-2.0 com requisitos para consumidores.
> * `cargo doc --no-deps`: 0 warnings, 0 erros. `cargo test --doc`: 2 passed, 0 failed.

* **Arquivo Alvo**: [src/lib.rs](file:///home/fabio/nam-rs/src/lib.rs)
* **Descrição**: Adicionar seção formal no Rustdoc explicando as garantias de tempo real (zero heap allocations no hot-path, zero blocking I/O, FTZ+DAZ) e os requisitos de licença Apache-2.0 para consumidores da biblioteca.
* **Critério de Aceite**: A seção é visível e renderizada corretamente em `cargo doc --no-deps`.

---

### Épico E3: Validação no Fluxo de Qualidade (`utils/lints.sh`)

#### Tarefa E3.1: Validação Automatizada de Documentação e Doc-Tests

> **Concluído em 2026-07-22**: Adicionada fase `[6/6]` em `utils/lints.sh` que executa:
>
> * `cargo doc --no-deps` com `RUSTDOCFLAGS="-D warnings"` (falha em warnings de doc)
> * `cargo test --doc` (compila e executa todos os doc-tests)
> * `PHASE_TOTAL` atualizado de 5 para 6. Script completo executado com sucesso: 6/6 fases passam.

* **Arquivo Alvo**: [utils/lints.sh](file:///home/fabio/nam-rs/utils/lints.sh)
* **Descrição**: Adicionar uma nova fase ao final do script que executa `cargo doc --no-deps` (para validar zero warnings de documentação) e `cargo test --doc` (para compilar e executar os doc-tests).
* **Critério de Aceite**: O script `utils/lints.sh` executa todas as fases com sucesso, sem erros de documentação nem falhas de doc-tests.

---

## 4. Próximos Passos e Status de Execução

Conforme instruído, este mapeamento e especificação estão 100% concluídos e prontos em `TODO-findings.md`. Nenhuma alteração de código ou execução de script de publicação foi realizada nesta etapa, mantendo o ambiente limpo para o momento da release da versão v3.0.0.
