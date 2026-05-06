<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# 🎸 NAM-rs 1.4.0

![License](https://img.shields.io/badge/License-MIT_OR_Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/Linux%20x86__64-lightgrey.svg) ![PipeWire](https://img.shields.io/badge/PipeWire-green.svg)

O **NAM-rs** é um cliente [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) em tempo real para simulação de, por exemplo, amplificadores, pedais de guitarra e equipamentos de estúdio. Ele tenta manter paridade com a implementação padrão do NAM, mas com muitas melhorias e otimizações.

Nesta versão 1.4 ele foca em rodar em modo standalone, mas já prepara o terreno para plugins. Ou seja, ele é um executável que captura qualquer cadeia de sinal de áudio do seu computador e manda processado para a sua saida de áudio desejada. Minha intenção com isto foi testar a tecnologia de forma rápida, sem perder tempo com abstrações mais complexas.

O motor de inferência é muito baseado na biblioteca C++ [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio) de Mike Oliphant, porém re-escrito inteiramente em Rust nativo e idiomático e com muitas otimizações feitas sob medida. Em muitos locais do _Hot Path_ praticamente alcançando a meta teórica de throughput da microarquitetura.

Está totalmente otimizado para extrair o máximo de performarce e baixa latência possível - possibilitado por uma base de código limpo e organizado, pelo uso intensivo de instruções SIMD modernas (AVX2/FMA/x86-64-v3) e pelos recursos modernos do Pipewire/Linux.

Caprichei na busca intensiva pelo estado da arte em matéria de otimização! Isto totalmente se pagou se você considerar que as cadeias de áudio no computador costumam ser stereo. Então, na verdade, são DOIS canais de áudio sendo processados simultâneamente em baixíssima latência e usando muito pouca CPU.

## **🇺🇲 English Version (International)**

**NAM-rs** is a real-time [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) client for simulating guitar amplifiers, pedals, and studio gear. It aims to maintain parity with the standard NAM implementation while introducing several performance improvements and optimizations.

In version 1.x, the focus is on standalone mode. It runs as an executable that captures any audio signal from your computer and sends the processed result to your desired output. My intention was to test the technology quickly without spending time on more complex abstractions.

The inference engine is heavily based on Mike Oliphant's [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio) C++ library, but entirely rewritten in native and idiomatic Rust with numerous tailor-made optimizations. In many parts of the _Hot Path_, it practically achieves the theoretical microarchitecture throughput target.

It is fully optimized for maximum performance and ultra-low latency. This is achieved through a clean and organized codebase, intensive use of modern SIMD instructions (AVX2/FMA/x86-64-v3), and modern PipeWire/Linux features.

I worked hard to achieve state-of-the-art optimization! This effort truly pays off when you consider that computer audio signals are usually stereo. As a result, two audio channels are processed simultaneously with extremely low latency and very low CPU usage.

_Note: A full translation of NAM-rs is planned. For now, our primary focus remains on the development of the project itself._

## ✨ Arquitetura

O NAM-rs adota uma arquitetura opinativa e focada em três pilares:

1. **PipeWire Standalone Nativo:** Integra diretamente com o servidor PipeWire como cliente nativo - sem abstrações VST/LV2/CLAP (no momento). O processo gerencia suas portas de áudio diretamente no _Graph Engine_ do PipeWire.
2. **Inferência SIMD ultra-rápidas:** A linha de base é x86-64-v3 (AVX2 + FMA obrigatórios). Funções de ativação (tanh, sigmoid) usam aproximações FastMath (Padé + rsqrt Newton-Raphson) em registradores de 256 bits. Multiversioning AVX-512 implementado via `Avx512Math` para hardware ZMM (Intel Xeon, AMD Zen 4+), processando 16 floats por instrução. WaveNet opera em **Batch GEMM** (bloco de até 64 frames por invocação).
3. **Determinismo de Tempo Real:** A thread DSP é promovida a `SCHED_FIFO` com afinidade de CPU rígida (_Core Affinity_), impedindo migrações e falhas de cache. Comunicação CLI ↔ DSP via ring buffer SPSC alinhado a 128 bytes. **Zero alocações** na heap durante processamento de áudio.
4. **Rust puro:** A escolha do Rust não foi por "hype". Há motivos muito fortes que compelem a ele. Além de alta performance, por ser uma linguagem compilada similar arquiteturalmente aos tradicionais C/C++, trata-se de uma linguagem com sintaxe moderna, muito expressiva e rica em recursos. Sintaxe que oferece garantias de segurança e de performance já em tempo de compilação. Por exemplo, versões estáticas (wavenet.rs) onde o tamanho do kernel e os canais são conhecidos em tempo de compilação permitem otimizações agressivas de loop unrolling pelo LLVM.

## 🚀 Guia Rápido

### Pré-requisitos

* Kernel Linux e servidor de áudio Pipewire relativamente recentes devem ser suficientes. Eu testei usando Ubuntu 25.10.

* Processador x86-64-v3 com suporte AVX2 e FMA (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015). CPUs de 2019 para cá são muito recomendáveis para redes neurais NAM.

* Toolchain Rust recente (`rustup`/`cargo`). Eu usei a versão 1.94 durante a maior parte do desenvolvimento.

* Pacotes de desenvolvimento: `sudo apt install build-essential cmake pkg-config pipewire libpipewire-0.3-dev clang libclang-dev qpwgraph`

* Utilitários do Cargo (necessários para QA): `cargo install cargo-mutants cargo-edit`

* Para que o motor execute inabalável sob modelos NAM realistas (especialamente "Lite" e "Standard"), é fundamental conceder a autorização de políticas SCHED mais avançadas ao binário. Adicione seu usuário ao grupo de áudio do sistema e edite limits:

  1. `sudo usermod -aG audio $USER`
  2. Crie ou edite o arquivo de limites (ex: `sudo nano /etc/security/limits.d/audio.conf`):

```text
@audio   -  rtprio     95
@audio   -  memlock    unlimited
```

* Crie uma regra no _udev_ para permitir que o grupo `audio` bloqueie a latência de despertar da CPU (C-states),
  1. `sudo nano /etc/udev/rules.d/99-audio-dma-latency.rules`
  2. Reinicie a máquina ou execute `sudo udevadm control --reload-rules && sudo udevadm trigger` para aplicar as regras

```text
KERNEL=="cpu_dma_latency", GROUP="audio", MODE="0664"
```

* Também é muito recomendável configurar o governador de performence da sua CPU (`intel_pstate` ou `amd_pstate`) para o modo **Performance**.
  * Desktop modernos (como GNOME no Ubuntu/Fedora ou KDE Plasma), o sistema já gerencia isso nativamente via power-profiles-daemon.
  * Caso você prefira o `tlp`, pode editar o /etc/tlp.conf

```text
CPU_SCALING_GOVERNOR_ON_AC=performance
CPU_SCALING_GOVERNOR_ON_BAT=powersave
```

### Build e Execução

```bash
git clone https://github.com/fabiohl/nam-rs.git
cd nam-rs
cargo build --release
```

Por ser um projeto em intensa evolução, não é aconselhável usar a branch "main" para trabalhos em ambiente de produção. Recomenda-se uma uma das versões release disponíveis em <https://github.com/fabiohl/nam-rs/tags>.

**Obs:** O .cargo/config.toml permite configurar uma compilação ainda mais otimizada para a sua CPU atual ("march=native"). O script utils/switchMarch.sh automatiza esta troca.

Você pode usar o `qpwgraph &` como um editor visual de conexões pipewire.
Para iniciar o processamento:

```bash
target/release/nam-rs --model tests/nam_files/NEVE1073-Standard.nam
target/release/nam-rs --model tests/fixtures/models/BossWN-standard.nam --input-gain -3.0 --output-gain 0.0
# Em máquinas fracas ou setups de desktop, aumente o buffer para aliviar a CPU:
target/release/nam-rs --model HeavyModel.nam --buffer-size 512
```

### Telemetria e Monitoramento

A cada 10 segundos, o NAM-rs exibe um relatório de performance no terminal para garantir a saúde do processamento:

`📊 Telemetria DSP (10s): 262µs (Mediana) | 524µs (P99) | 1048µs (Máx) [938 blocos]`

* **Mediana**: Indica o custo típico de processamento por bloco. Valores próximos de 0µs indicam que o `Silence Bypass` está poupando CPU.
* **P99 (Estabilidade)**: O indicador mais crítico. Mostra que 99% dos blocos foram processados abaixo deste tempo. Se o P99 se aproximar do tempo do seu buffer (ex: 5333µs para buffer 256 - o padrão do nam-rs), o risco de estalos aumenta.
* **Máx**: O pior caso registrado no intervalo, útil para detectar picos causados por interferências do sistema operacional.
* **Blocos**: O volume total de dados processados que compõem a estatística do intervalo.

Após a inicialização, o nodo aparece na matriz PipeWire. O NAM-rs deve estar disponível como um dispositivo de saída no seu player de áudio.
Se necessário, use `qpwgraph` ou `pw-link` para fazer os devidos roteamentos.

## 📚 Documentação

* [docs/architecture.md](docs/architecture.md) — Topologia, módulos e decisões de design
* [docs/dependencies.md](docs/dependencies.md) — Dependências sistêmicas e crates Rust
* [docs/benchmarks.md](docs/benchmarks.md) — Como interpretar as métricas de performance do Criterion
* [docs/clap_integration.md](docs/clap_integration.md) — Estratégia de integração CLAP (Clever Audio Plug-in)

## 🧠 Modelos Suportados

O NAM-rs suporta nativamente arquivos Neural Amp Modeler (.nam ou .namb). Arquivos de Impulse Response (.wav) não são suportados (ao menos não no momento).
No momento é suportado nativamente a "Arquitetura A1" do NAM. O suporte à "Arquitetura A2" está em fase de **staging** (scaffolding e loader prontos na v1.4).
Oferece dois níveis de operação de parsing:

* **Modo Estático (Altíssima Performance):** Construções de _Const Generics_ dimensionadas em tempo de compilação.
  * **WaveNet:** Standard (16×8), Lite (12×6), Feather (8×4) e Nano (4×2)
  * **LSTM:** 1 e 2 Camadas (8 a 24 de Hidden Size: `1×8`, `1×12`, `1×16`, `1×24`, `2×8`, `2×12`, `2×16`)
* **Modo Dinâmico (Flexibilidade Absoluta):** Fallback ativado automaticamente ao carregar arranjos `.nam` com geometrias extrassensoriais não catalogadas (`num_layers` e `channels` arbitrários), operando sem _loop unrolling_.

## 🧪 Testes e Validação

O NAM-rs mantém uma suíte de aproximandamente **138 verificações** automatizadas distribuídas em cinco camadas:

```bash
# Testes unitários + integração + proptest + fuzz + E2E PipeWire
cargo test

# Apenas testes unitários inline
cargo test --lib

# Apenas testes de integração (incluindo zero-alloc, block sizes, golden vectors Feather/Nano, modelos comunitários)
cargo test --test nam_infer_test

# Fuzz testing dos parsers (9 testes × 5000 cases = ~45.000 inputs)
cargo test --test proptest_parsers

# Benchmarks de latência (17 funções benchmark, 23 medições criterion)
cargo bench --bench inference_bench

# Lint rigoroso de código fonte (formatação + clippy)
utils/lints.sh

# Bateria de Estabilidade (Soak Test) - Manutenção de Longa Duração
utils/soak-test.sh
```

### Teste de Estabilidade (Soak Test)

Para garantir que o motor permaneça estável durante horas de uso contínuo, o NAM-rs inclui uma suíte de **Soak Test** que processa milhões de frames (ex: 10M+ de silêncio/ruído, 100M+ ciclos de buffer circular). Estes testes são projetados para detectar:

* **Drift Numérico**: Acúmulo de erro de arredondamento em filtros e resamplers.
* **Estabilidade de FSM**: Integridade de contadores de Gate e transições de fade.
* **Resiliência de Memória**: Estresse de fronteiras em `VirtualRingBuffer`.

Para executar a bateria completa (pode levar vários minutos/horas):
`bash utils/soak-test.sh`

Para executar os benchmarks de longa duração (estatística de 30s):
`bash utils/long-bench.sh`

Categorias de teste incluem: parsing JSON e NAMB, **fuzz testing via proptest** (bytes adversários, JSON malformado, NAMB corrompido), **verificação zero-allocation** no hot path (counting allocator), estabilidade numérica de longa duração, auto-consistência (determinismo), golden vectors C++ ↔ Rust, pipeline E2E SPSC, paridade estático/dinâmico, estabilidade sob silêncio (denormals/DAZ/FTZ), rejeição de JSON malformado, gain staging roundtrip, hot-swap rápido de modelos, **block sizes variáveis** (1–512 amostras), **modelos comunitários** (5 modelos .nam) e **rejeição de formatos não-suportados** (Keras Legacy, ativações não-Tanh).

## ⏲️ Changelog

* 1.0 (28/04/2026): Release inicial. Suporte completo à "Arquitetura A1" do Neural Amp Modeler.
* 1.1 (30/04/2026): Otimizações de performance sortidas.
* 1.2 (02/05/2026): Algoritimo de resampling próprio.
* 1.3 (04/05/2026): Otimizações intensivas de SIMD e refatoração de telemetria.
* 1.4 (05/05/2026): Staging para A2 (scaffolding) e CLAP. Toneladas de otimizações de performance.

## 🛣️ Próximos Passos (Roadmap)

O NAM-rs está em transição para a v1.4+, focada em expandir as fronteiras de compatibilidade e portabilidade:

* **Arquitetura A2:** Implementação do kernel de inferência para modelos A2 (v0.6+), suportando FiLM, Gating dinâmico e 11 novas funções de ativação. Atualmente em fase de _staging_ (scaffolding e loader compatível prontos).
* **Integração CLAP:** Transformação do motor DSP em um plugin nativo CLAP (Clever Audio Plug-in), permitindo o uso do NAM-rs dentro de qualquer DAW (Bitwig, REAPER, etc.) sem dependência de PipeWire.
* **Interface Gráfica (GUI):** Planejamento de uma interface visual minimalista para carregamento de modelos e ajuste de parâmetros em tempo real.
* **Otimizações A2 SIMD:** Portar as novas ativações e camadas FiLM para kernels AVX2/AVX-512 altamente otimizados.

## 🤝 Contribuindo

Contribuições são bem-vindas! O projeto está em fase ativa de desenvolvimento.

* Testes + testes + testes + testes....
* Apesar do suporte a AVX-512, eu não tenho uma CPU capacitada pra testar. Se alguém tiver alguma por ai, seria muito útil..
* Antes de submeter PRs execute a bateria completa de testes citada acima. Use e abuse das skills de IA agênticas em .agents/.

## 🙏 Créditos e Agradecimentos

Este projeto herda lógica, inspiração e ciência de trabalhos notáveis na comunidade de áudio e inteligência artificial que pavimentaram este caminho.
Impossível mapear todos os que contribuíram para o avanço desta tecnologia. Mas no escopo dos nosso projeto, impossível não mencionar:

* **Steven Atkinson** — Pelo desenvolvimento pioneiro do [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler), sua pesquisa original sobre modelagem de amplificadores com deep learning e por compartilhar o ecossistema abertamente.
* **Mike Oliphant** — Pela excepcional biblioteca [NeuralAudio (C++)](https://github.com/mikeoliphant/NeuralAudio), que serviu de base direta e fonte de pesquisa para a transposição das lógicas de inferência para este motor estrutural.

## ⚖️ Licença e Transparência (Vibe Coding)

**Nota de Transparência sobre Inteligência Artificial:** A arquitetura, as decisões rigorosas de engenharia, a documentação, a atenciosa regência dos agentes e a curadoria deste projeto são de autoria intelectual do mantenedor. Contudo, o código-fonte em si foi e gerado iterado com o auxílio de Inteligência Artificial (_Vibe Coding_). Mais especificamente, usando a IDE Google Antigravity.

Este projeto é duplamente licenciado sob **MIT** ou **Apache License, Version 2.0**, à sua escolha. Veja os arquivos `LICENSE-MIT` e `LICENSE-APACHE` para mais detalhes.
O uso da licença Apache 2.0 visa justamente oferecer maior segurança jurídica aos contribuidores e usuários em relação a patentes, dado este modelo moderno de desenvolvimento de software. A estrutura está aberta para a comunidade refatorar, auditar e expandir livremente.
