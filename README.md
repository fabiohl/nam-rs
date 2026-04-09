# 🎸 NAM-rs

![License](https://img.shields.io/badge/License-MIT_OR_Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg) ![Platform](https://img.shields.io/badge/platform-Linux%20x86__64-lightgrey.svg) ![PipeWire](https://img.shields.io/badge/PipeWire-1.6%2B-green.svg)

O **NAM-rs** é um cliente standalone de inferência neural em tempo real para emulação de amplificadores, pedais de guitarra e equipamentos de estúdio - baseado no [Neural Amp Modeler (NAM)](https://www.tone3000.com/guides/neural-amp-modeler). Escrito inteiramente em Rust nativo e idiomático, opera exclusivamente como cliente PipeWire standalone no Linux.

O motor de inferência é uma transposição otimizada da biblioteca C++ [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio) de Mike Oliphant, substituindo Eigen por instruções SIMD explícitas via `core::arch::x86_64` (AVX2/FMA).

## ✨ Arquitetura

O NAM-rs adota uma arquitetura opinativa e focada em três pilares:

1. **PipeWire Standalone Nativo:** Integra diretamente com o servidor PipeWire via `pipewire-rs` como cliente nativo - sem abstrações VST/LV2/CLAP. O processo gerencia suas portas de áudio diretamente no *Graph Engine* do PipeWire.
2. **Inferência SIMD de Ultra-Baixa Latência:** A linha de base é x86-64-v3 (AVX2 + FMA obrigatórios). Funções de ativação (tanh, sigmoid) usam aproximações FastMath (Padé + rsqrt Newton-Raphson) em registradores de 256 bits. Multiversioning AVX-512 implementado via `#[target_feature(enable = "avx512f,avx512vl")]` para hardware que suporta ZMM (Intel Xeon, AMD Zen 4+), processando 16 floats por instrução.
3. **Determinismo de Tempo Real:** A thread DSP é promovida a `SCHED_FIFO` com afinidade de núcleo rígida (*Core Affinity*), impedindo migrações e falhas de cache. Comunicação CLI ↔ DSP via ring buffer SPSC alinhado a 128 bytes. **Zero alocações** na heap durante processamento de áudio.
4. **CLI Interativa e Hot-Swap de Modelos:** Interface de linha de comando com `lexopt` para configuração inicial (`--model`, `--input-gain`, `--output-gain`) e loop `stdin` interativo em runtime. Troca de modelos (.nam/.namb) sem interromper o PipeWire, com delegação do `Drop` do modelo antigo para thread GC (lock-free Drop-Delegation).

## 🚀 Guia Rápido

### Pré-requisitos

* Ubuntu 26.04+ / Linux kernel 7.0+ com PipeWire 1.6+.
* Processador x86-64 com suporte AVX2 e FMA (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015).
* Toolchain do Rust 1.94+ (`cargo`).
* Pacotes de desenvolvimento: `sudo apt install build-essential pkg-config pipewire libpipewire-0.3-dev clang libclang-dev`

### Build e Execução

```bash
git clone https://github.com/fabiohl/nam-rs.git
cd nam-rs
cargo build --release
```

Para iniciar o processamento:

```bash
target/release/nam-rs --model tests/fixtures/models/BossWN-standard.nam --input-gain -3.0 --output-gain 0.0
```

Para que o motor execute inabalável sob modelos NAM severos ("Standard"), é fundamental conceder a autorização de políticas SCHED mais avançadas ao binário.
Adicione seu usuário ao grupo de áudio do sistema e edite limits:

1. `sudo usermod -aG audio $USER`
2. Crie ou edite o arquivo de limites (ex: `sudo nano /etc/security/limits.d/audio.conf`):

```text
@audio   -  rtprio     95
@audio   -  memlock    unlimited
```

Após a inicialização, o nodo aparece na matriz PipeWire. Use `qpwgraph` ou `pw-link` para conectar a entrada de instrumento e a saída para monitores.

## 📚 Documentação

* [docs/architecture.md](docs/architecture.md) — Topologia, módulos e decisões de design
* [docs/dependencies.md](docs/dependencies.md) — Dependências sistêmicas e crates Rust
* [docs/NAM-rs-referência.md](docs/NAM-rs-referência.md) — Documento de referência arquitetural
* [docs/NAM-rs-sprints.md](docs/NAM-rs-sprints.md) — Sprints e tarefas técnicas detalhadas

## 🧠 Modelos Suportados

O NAM-rs suporta nativamente a inferência de modelos convencionais e experimentais da comunidade, oferecendo dois níveis de operação de parsing:

* **Modo Estático (Altíssima Performance):** Construções de *Const Generics* dimensionadas em tempo de compilação.
  * **WaveNet:** Standard (16×8), Lite (12×8), Feather (8×4) e Nano (4×2)
  * **LSTM:** 1 e 2 Camadas (8 a 24 de Hidden Size: `1×8`, `1×12`, `1×16`, `1×24`, `2×8`, `2×12`, `2×16`)
* **Modo Dinâmico (Flexibilidade Absoluta):** Fallback ativado automaticamente ao carregar arranjos `.nam` com geometrias extrassensoriais não catalogadas (`num_layers` e `channels` arbitrários), operando sem *loop unrolling*.

## 🤝 Contribuindo

Contribuições são bem-vindas! O projeto está em fase ativa de desenvolvimento. Áreas de interesse:

* Testes + testes + testes + testes....
* Apesar do suporte a AVX-512, eu não tenho uma CPU suportado pra testar. Se alguém tiver alguma por ai, seria muito útil..

Sinta-se à vontade para abrir *Issues* ou enviar *Pull Requests*.

## ⚖️ Licença e Transparência (Vibe Coding)

Este projeto é duplamente licenciado sob **MIT** ou **Apache License, Version 2.0**, à sua escolha. Veja os arquivos `LICENSE-MIT` e `LICENSE-APACHE` para mais detalhes.

**Nota de Transparência sobre Inteligência Artificial:** A arquitetura, as decisões rigorosas de engenharia, a documentação e a curadoria deste projeto são de autoria intelectual do mantenedor. Contudo, o código-fonte em si foi iterado e gerado integralmente com o auxílio de Inteligência Artificial (*Vibe Coding*). Mais especificamente, usando a IDE Google Antigravity.

O uso da licença Apache 2.0 visa justamente oferecer maior segurança jurídica aos contribuidores e usuários em relação a patentes, dado este modelo moderno de desenvolvimento de software. A estrutura está aberta para a comunidade refatorar, auditar e expandir livremente.
