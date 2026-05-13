<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# Dependências do Projeto NAM-rs

Esta documentação lista e detalha contextualmente as dependências de sistema operacional e de software englobadas no `Cargo.toml`. O objetivo principal é justificar a necessidade destas abstrações frente às rígidas regras de arquitetura e performance (em detrimento de libs mais ricas ou bloated).

## 1. Dependências do Sistema (Linux)

Os seguintes pacotes devem estar instalados no sistema para viabilizar a compilação e execução do NAM-rs em seus diferentes modos. O comando consolidado para instalação em sistemas baseados em Debian/Ubuntu é:

```bash
sudo apt install build-essential cmake pkg-config pipewire libpipewire-0.3-dev \
                 clang libclang-dev qpwgraph libgtk-3-dev \
                 libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
                 libxkbcommon-dev libssl-dev git curl linux-tools-generic
```

### Detalhamento por Função

* **Ferramentas Base e Build**:

  * `build-essential` e `cmake`: Compiladores e utilitários de build fundamentais para o ecossistema C/C++ do qual algumas dependências Rust derivam.
  * `pkg-config`: Essencial para que o `cargo` localize os caminhos de headers e bibliotecas compartilhadas (`.so`) no sistema.
  * `clang` e `libclang-dev`: Requisito do *rust-bindgen* para transcrever os headers C do PipeWire para bindings Rust em tempo de compilação.
  * `libssl-dev`, `git` e `curl`: Necessários para ferramentas de suporte, controle de versão e instalação de componentes do ecossistema Rust.
  * `linux-tools-generic`: Provê o `perf`, essencial para profiling de baixo nível e otimização do *Hot Path* DSP.

* **Backend de Áudio e Testes (PipeWire)**:

  * `pipewire` e `libpipewire-0.3-dev`: Headers do core de processamento. Necessários apenas para a feature `standalone`.
  * `qpwgraph`: Utilitário recomendado para roteamento visual do grafo de áudio (opcional, mas altamente sugerido para usuários).

* **Interface Gráfica e Janelamento (Sprint 4 — Planejado)**:

  * `libgtk-3-dev`, `libxcb-*`, `libxkbcommon-dev`: Bibliotecas de sistema para suporte a janelas nativas, renderização via X11/Wayland e gerenciamento de teclado. Exigidas pelas crates `egui` e `baseview` para a interface do plugin CLAP.

## 2. Abstrações de Software (Crates - Cargo.toml)

| Crate                          | Versão Fixada | Função Primordial e Justificativa de Arquitetura                                                                                                                                                                                                                                                 | Alternativas Consideradas                                                                                   |
| ------------------------------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **`anyhow`**                   | `^1.0`        | Transição enxuta das sinalizações de erro nas threads de preparação e CLI, limitando uso de unwrap()/panic() passíveis de falhas fatais.                                                                                                                                                         | Rejeitado erro padronizado *Result* com Enums própios em prol da sintaxe ágil.                              |
| **`libc`**                     | `^0.2`        | Vinculação à C Standard Library para chamadas POSIX: `pthread_setschedparam` (SCHED_FIFO), `pthread_setaffinity_np` (core affinity), `mlockall`, `prctl` (THP disable), `sigaction` (SIGINT handler). Acesso direto sem wrappers intermediários.                                                 | Crates encapsuladores adicionam overhead de dependência sem benefício para chamadas POSIX bem documentadas. |
| **`pipewire`**                 | `0.9.x`       | Bindings Rust para `libpipewire 0.3`. Fornece o backend de áudio nativo para Linux moderno (low-latency). **Condicional à feature `standalone`** (padrão). Em builds para plugins (`--features clap-plugin`), esta dependência é totalmente removida do binário final, garantindo portabilidade. | *jack*: preterido para priorizar o ecossistema de áudio nativo do Linux moderno.                            |
| **`rtrb`**                     | `0.3.x`       | Ring buffer SPSC lock-free para comunicação CLI→DSP (parâmetros, modelos, resamplers). Usado em todo payload que transita entre threads sem travar a thread RT.                                                                                                                                  | *crossbeam*: overhead desnecessário para SPSC puro; *ringbuf*: API menos ergonômica.                        |
| **`serde`** e **`serde_json`** | `^1.0`        | Deserialização do formato `.nam` (JSON) em startup. Parseia campos aninhados (`config`, `weights`, `metadata`) de forma robusta.                                                                                                                                                                 | Parser manual: rejeitado pela complexidade de manutenção com campos opcionais e matrizes de pesos.          |
| **`lexopt`**                   | `0.3.x`       | Parser CLI minimalista e zero-alloc. Extrai `--model`, `--input-gain`, `--output-gain`, `--buffer-size` sem macros ou dependências pesadas.                                                                                                                                                      | *clap*: aumenta significativamente o tamanho do binário para uma CLI simples.                               |
| **`log`**                      | `0.4.x`       | Fachada de logging padrão do ecossistema Rust. Permite `log::info!`, `log::warn!`, `log::error!` com overhead zero quando desativado.                                                                                                                                                            | Logging manual via `eprintln!`: não oferece filtragem por nível (`RUST_LOG`).                               |
| **`env_logger`**               | `0.11.x`      | Backend de logging configurável via variável de ambiente `RUST_LOG`. Inicializado uma vez no `main()` com padrão `info`. `default-features = false` para minimizar dependências (sem regex, sem formatação de cores nativa).                                                                     | *tracing*: overhead desnecessário para aplicação CLI sem instrumentação distribuída.                        |
| **`half`**                     | `^2.7`        | Suporte a floats de precisão simples `f16`. Essencial para compressão de pesos (Weight Compression F16C) visando ocupação da Cache L1.                                                                                                                                                           | *f32*: Consome o dobro de memória e causa gargalos de Cache L1 na WaveNet Standard.                         |
| **`minstant`**                 | `^0.1`        | Telemetria de alta precisão baseada em RDTSC. Utilizado para medir latência por bloco na thread RT com overhead desprezível.                                                                                                                                                                     | `std::time::Instant`: inconsistente no SCHED_FIFO e pode incorrer em syscalls indesejadas.                  |
| **`rustfft`**                  | `^6.4`        | Algoritmo FFT de alta performance. Utilizado exclusivamente offline (fora da thread RT) no `NamResampler::new()` para transformação de fase mínima via Cepstrum Real.                                                                                                                            | *realfft*: Wrapper sobre rustfft; preferimos rustfft direto com `default-features = false`.                 |

## 3. Dependências de Build e Testes Automáticos (Dev-Dependencies)

Acessadas secundariamente via `cargo bench` e `cargo test`, não impactam no footprint binário release final:

| Crate           | Versão Fixada | Função Primordial e Justificativa de Arquitetura                                                                                                                                                                                                                                    |
| --------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`criterion`** | `^0.8`        | Avaliação da performance métrica estatística rigorosa. A flag `html_reports` encontra-se inativa para atenuar tempo de compilação semântica em esteiras iterativas onde avalia-se *FMA Latency per Vector* (inferiores a micro-segundos).                                           |
| **`proptest`**  | `^1.11`       | Property-based testing para validação exaustiva dos limites algorítmicos das funções FastMath (`simd_tanh`, `simd_sigmoid`). Gera 10.000+ vetores aleatórios por execução para varrer buracos aritméticos causados pelo Newton-Raphson de recíproca quadrática (`_mm256_rsqrt_ps`). |

## 4. Utilitários Adicionais do Cargo (QA & Dev)

Estes componentes devem ser instalados via `cargo install` para habilitar rotinas avançadas de manutenção e garantia de qualidade:

| Utilitário       | Função no Projeto                                                                                                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`cargo-edit`** | Gerenciamento de dependências. Utilizado no script `utils/mod-update.sh` para o comando `cargo upgrade`, garantindo que as bibliotecas permaneçam seguras e atualizadas.            |
| **`cargo-fuzz`** | Fuzz testing (libFuzzer). Utilizado para testar a robustez dos parsers JSON e NAMB contra entradas malformadas ou adversárias. Localizado no diretório `/fuzz`.                     |
| **`clippy`**     | Linter estático para Rust. Utilizado no script `utils/lints.sh` para garantir a adesão às melhores práticas e evitar antipadrões que possam comprometer a performance ou segurança. |
| **`rustfmt`**    | Formatador de código. Garante consistência visual em todo o repositório, essencial para revisão de código e manutenção por múltiplos colaboradores.                                 |

## 5. Dependências Planejadas

As seguintes dependências serão introduzidas a partir da Sprint 1 para viabilizar o suporte a plugins CLAP e a interface gráfica embarcada:

| Crate              | Versão Planejada | Feature Flag  | Sprint   | Justificativa                                                                                                                                                   |
| ------------------ | ---------------- | ------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clack-plugin`     | `^0.1`           | `clap-plugin` | Sprint 1 | API Rust para implementação de plugins CLAP. Abstração tipada sobre `clap-sys` sem overhead de runtime. Escolhido sobre `nih-plug` por não forçar VST3 nem GUI. |
| `clack-extensions` | `^0.1`           | `clap-plugin` | Sprint 1 | Extensões do spec CLAP (params, state, gui, thread-pool). Crate separado do `clack-plugin` para modularidade.                                                   |
| `egui`             | `^0.34`          | `clap-plugin` | Sprint 4 | Framework GUI imediato, puro Rust. Renderização de GPU via `wgpu`. Integrado ao CLAP via `baseview`.                                                            |
| `baseview`         | `git`            | `clap-plugin` | Sprint 4 | Janela nativa multiplataforma para `egui` em contexto de plugin. Não publicado no crates.io — usar dependência via Git.                                         |
