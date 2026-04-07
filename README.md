# 🎸 NAM-rs

![License](https://img.shields.io/badge/License-MIT_OR_Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg) ![Platform](https://img.shields.io/badge/platform-Linux%20x86__64-lightgrey.svg) ![PipeWire](https://img.shields.io/badge/PipeWire-1.6%2B-green.svg)

O **NAM-rs** é um cliente standalone de inferência neural em tempo real para emulação de amplificadores, pedais de guitarra e equipamentos de estúdio - baseado no [Neural Amp Modeler (NAM)](https://www.tone3000.com/guides/neural-amp-modeler). Escrito inteiramente em Rust nativo e idiomático, opera exclusivamente como cliente PipeWire standalone no Linux.

O motor de inferência é uma transposição otimizada da biblioteca C++ [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio) de Mike Oliphant, substituindo Eigen por instruções SIMD explícitas via `core::arch::x86_64` (AVX2/FMA).

## ✨ Arquitetura

O NAM-rs adota uma arquitetura opinativa e focada em três pilares:

1. **PipeWire Standalone Nativo:** Integra diretamente com o servidor PipeWire via `pipewire-rs` como cliente nativo - sem abstrações VST/LV2/CLAP. O processo gerencia suas portas de áudio diretamente no *Graph Engine* do PipeWire.
2. **Inferência SIMD de Ultra-Baixa Latência:** A linha de base é x86-64-v3 (AVX2 + FMA obrigatórios). Funções de ativação (tanh, sigmoid) usam aproximações FastMath (Padé + rsqrt Newton-Raphson) em registradores de 256 bits, eliminando o custo das funções transcendentais da stdlib. Suporte futuro a AVX-512 via multiversioning.
3. **Determinismo de Tempo Real:** A thread DSP é promovida a `SCHED_FIFO` com afinidade de núcleo rígida (*Core Affinity*), impedindo migrações e falhas de cache. Comunicação CLI ↔ DSP via ring buffer SPSC alinhado a 128 bytes. **Zero alocações** na heap durante processamento de áudio.

## 🎯 Status do Projeto

| Sprint       | Descrição                                                            | Status       |
| ------------ | -------------------------------------------------------------------- | ------------ |
| **Sprint 1** | Purga do I/O legado, SPSC lock-free, Host PipeWire nativo            | ✅ Concluída |
| **Sprint 2** | Infraestrutura SIMD (dot product AVX2/FMA), FastMath (tanh, sigmoid) | ✅ Concluída |
| **Sprint 3** | Redes inferenciais (WaveNet + LSTM), integração no callback PipeWire | ✅ Concluída |
| **Sprint 4** | Carregamento de modelos (.nam JSON / .namb binário), gain staging    | 🔲 Planejada |
| **Sprint 5** | Resampling FIR Sinc, validação MSE contra referência C++             | 🔲 Planejada |

## 🚀 Guia Rápido

### Pré-requisitos

* Ubuntu 26.04+ / Linux kernel 7.0+ com PipeWire 1.6+.
* Processador x86-64 com suporte AVX2 e FMA (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015).
* Toolchain do Rust 1.94+ (`cargo`).

Para dependências de compilação, consulte [docs/dependencies.md](docs/dependencies.md).

### Build e Execução

```bash
git clone https://github.com/fabiohl/nam-rs.git
cd nam-rs
cargo build --release
```

Para iniciar o processamento:

```bash
target/release/nam-rs
```

Após a inicialização, o nodo aparece na matriz PipeWire. Use `qpwgraph` ou `pw-link` para conectar a entrada de instrumento e a saída para monitores.

## 📚 Documentação

* [docs/architecture.md](docs/architecture.md) — Topologia, módulos e decisões de design
* [docs/dependencies.md](docs/dependencies.md) — Dependências sistêmicas e crates Rust
* [docs/NAM-rs-referência.md](docs/NAM-rs-referência.md) — Documento de referência arquitetural
* [docs/NAM-rs-sprints.md](docs/NAM-rs-sprints.md) — Sprints e tarefas técnicas detalhadas

## 🤝 Contribuindo

Contribuições são bem-vindas! O projeto está em fase ativa de desenvolvimento. Áreas de interesse:

* Parser do formato binário .namb (Sprint 4)
* Gain staging SIMD (Sprint 4)
* Resampling FIR de fase linear (Sprint 5)
* Suporte AVX-512 via multiversioning

Sinta-se à vontade para abrir *Issues* ou enviar *Pull Requests*.

## ⚖️ Licença e Transparência (Vibe Coding)

Este projeto é duplamente licenciado sob **MIT** ou **Apache License, Version 2.0**, à sua escolha. Veja os arquivos `LICENSE-MIT` e `LICENSE-APACHE` para mais detalhes.

**Nota de Transparência sobre Inteligência Artificial:** A arquitetura, as decisões rigorosas de engenharia, a documentação e a curadoria deste projeto são de autoria intelectual do mantenedor. Contudo, o código-fonte em si foi iterado e gerado integralmente com o auxílio de Inteligência Artificial (*Vibe Coding*). Mais especificamente, usando a IDE Google Antigravity.

O uso da licença Apache 2.0 visa justamente oferecer maior segurança jurídica aos contribuidores e usuários em relação a patentes, dado este modelo moderno de desenvolvimento de software. A estrutura está aberta para a comunidade refatorar, auditar e expandir livremente.
