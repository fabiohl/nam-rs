# 🎧 AudioRip

![License](https://img.shields.io/badge/License-MIT_OR_Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg) ![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)

O **AudioRip** é um utilitário de linha de comando altamente otimizado para captura de áudio *bit-perfect* e salvamente em disco no formato wav.

Sou apenas um entusiasta de áudio amador. Partindo de uma necessidade pessoal de capturar áudio gerado no computador, eu uso este projeto como um playground de experimentação de técnicas de processamento de áudio.

Um profissional de áudio pode estranhar certas decisões arquiteturais do projeto para algo tão mundano. Porém, além do playground, eu estou usando o AudioRip como uma fundação para os meus "brinquedos" futuros que já estão no "forno".

## ✨ Ideias de Arquitetura

O ecossistema de áudio no Linux possui ótimas ferramentas, mas o AudioRip propõe uma abordagem arquitetural pouco ortodoxa e ainda que extremamente eficiente, baseada em três pilares:

1. **A "Hack" do `nih_plug` em Standalone:** Em vez de depender de bindings complexos de ALSA ou PipeWire puro, o AudioRip sequestra o backend JACK do framework `nih_plug` (rodando via `pw-jack`). Isso obriga o PipeWire a acionar o binário sob regras rígidas de baixa latência, herdando toda a estabilidade de um plugin DSP profissional.
2. **Zero-Blocking I/O com `io_uring`:** A gravação do arquivo WAV (disco) é delegada ao `tokio-uring`. Isso garante que a thread de gravação não engasgue com syscalls bloqueantes tradicionais do POSIX, permitindo um *throughput* maciço sem interromper a captura.
3. **Otimizações para baixa latência em hardware moderno:** A linha de base de compilação é x86-64-v3 (processadores de 2018 para cá). O objetivo é poder usar os recursos de SIMD dos processadores modernos, permitindo mais operações em menos ciclos de clock. A comunicação entre a thread DSP (crítica) e a thread de disco é feita via Ring Buffers SPSC (*Single-Producer Single-Consumer*) lock-free. Além disso, aplicamos alinhamento manual de memória (`#[repr(align(128))]`) para mitigar severamente o *False Sharing* nos caches L1/L2 da CPU.

Como disse, não sou profissional de áudio dsp. Certamente há abordagens melhores para isso. Este foi apenas o caminho que encontrei. Ideias são bem-vindas!

## 🚀 Guia Rápido

### Pré-requisitos

* Ambiente Linux moderno com suporte a `io_uring`.
* PipeWire configurado com suporte a JACK (`pw-jack`).
* Toolchain do Rust instalada (`cargo`).

### Build e Execução

Clone o repositório e compile em modo release para garantir a performance de tempo real:

```bash
git clone [https://github.com/fabiohl/audiorip.git](https://github.com/fabiohl/audiorip.git)
cd audiorip
cargo build --release
```

Para iniciar a captura (exemplo básico):

```bash
pw-jack target/release/audiorip --output gravacao.wav
```

Por padrão o AudioRip não "captura" nenhuma fonte de áudio que esteja rodando no momento. Você pode usar um utilitário como o `qpwgraph` para "conectar" uma fonte de áudio nas entradas de áudio do AudioRip.

O AudioRip tem um mecanismo para que ele ignore os silêncios inevitáveis nos processos de conexão pipewire, pausa de reprodução, etc. Só será salvo o áudio puro.

### 📚 Documentação Aprofundada

Este README é apenas a ponta do iceberg. Para entender profundamente as decisões de design, fluxos de dados, esquemas de lock-free e o graceful shutdown do arquivo WAV, **[leia a documentação completa na pasta `docs/`](docs/)**.

* `docs/architecture.md` - Topologia e decisões de design.
* `docs/development.md` - Como configurar o ambiente e desenvolver.

## 🤝 Contribuindo

Contribuições são extremamente bem-vindas\! Se você é apaixonado por DSP, Rust e sistemas de baixo nível, há muito espaço para evoluir o AudioRip.

No momento não tenho nada em particular que gostaria de implementar - Apesar de vislumbrar algumas áreas de interesse que as pessoas podem ter:

* Suporte a codificação FLAC *on-the-fly* (assíncrona, pós-ring buffer).
* Expansão da interface de linha de comando (CLI) baseada nos padrões da indústria de áudio profissional.
* Exportação da lógica de graceful shutdown do WAV para uma micro-crate independente.

Sinta-se à vontade para abrir *Issues* ou enviar *Pull Requests*.

## ⚖️ Licença e Transparência (Vibe Coding)

Este projeto é duplamente licenciado sob **MIT** ou **Apache License, Version 2.0**, à sua escolha. Veja os arquivos `LICENSE-MIT` e `LICENSE-APACHE` para mais detalhes.

**Nota de Transparência sobre Inteligência Artificial:** A arquitetura, as decisões rigorosas de engenharia (como a topologia DSP vs. io\_uring), a documentação e a curadoria deste projeto são de autoria intelectual do mantenedor. Inclusive foi um trabalho de que me orgulho muito e considerto tendo sido muito bem feito. Contudo, o código-fonte em si foi iterado e gerado integralmente com o auxílio de Inteligência Artificial (*Vibe Coding*). Mais especificamente, usando a IDE Google Antigravity.

O uso da licença Apache 2.0 visa justamente oferecer maior segurança jurídica aos contribuidores e usuários em relação a patentes, dado este modelo moderno de desenvolvimento de software. A estrutura está aberta para a comunidade refatorar, auditar e expandir livremente.
