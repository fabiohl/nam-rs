# 🎸 NAM-rs

![License](https://img.shields.io/badge/License-MIT_OR_Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg) ![Platform](https://img.shields.io/badge/platform-Linux%20x86__64-lightgrey.svg) ![PipeWire](https://img.shields.io/badge/PipeWire-1.6%2B-green.svg)

O **NAM-rs** é um cliente [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) em tempo real para simulação de, por exemplo, amplificadores, pedais de guitarra e equipamentos de estúdio. Na medida do possível, ele tenta manter paridade com a implementação padrão do NAM, mas com algumas melhorias e otimizações.

O motor de inferência é uma transposição otimizada da biblioteca C++ [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio) de Mike Oliphant, porém re-escrito inteiramente em Rust nativo e idiomático.

Está totalmente otimizado para extrair o máximo de performarce e baixa latência possível - possibilitado por uma base de código limpo e organizado, pelo uso intensivo de instruções SIMD modernas (AVX2/FMA/x86-64-v3) e pelos recursos modernos do Pipewire/Linux. Caprichei na busca intensiva pelo estado de arte em matéria de otimização!

## ✨ Arquitetura

O NAM-rs adota uma arquitetura opinativa e focada em três pilares:

1. **PipeWire Standalone Nativo:** Integra diretamente com o servidor PipeWire via `pipewire-rs` como cliente nativo - sem abstrações VST/LV2/CLAP (no momento). O processo gerencia suas portas de áudio diretamente no *Graph Engine* do PipeWire.
2. **Inferência SIMD ultra-rápidas:** A linha de base é x86-64-v3 (AVX2 + FMA obrigatórios). Funções de ativação (tanh, sigmoid) usam aproximações FastMath (Padé + rsqrt Newton-Raphson) em registradores de 256 bits. Multiversioning estático no WaveNet via trait `SimdMath` com `#[target_feature]` (zero v-table no hot-loop); AVX-512 implementado via `Avx512Math` para hardware ZMM (Intel Xeon, AMD Zen 4+), processando 16 floats por instrução. WaveNet opera em **Batch GEMM** (bloco de até 64 frames por invocação).
3. **Determinismo de Tempo Real:** A thread DSP é promovida a `SCHED_FIFO` com afinidade de CPU rígida (*Core Affinity*), impedindo migrações e falhas de cache. Comunicação CLI ↔ DSP via ring buffer SPSC alinhado a 128 bytes. **Zero alocações** na heap durante processamento de áudio.
4. **CLI Interativa e Hot-Swap de Modelos:** Interface de linha de comando com `lexopt` para configuração inicial (`--model`, `--input-gain`, `--output-gain`, `--buffer-size`) e loop `stdin` interativo em runtime. Troca de modelos (.nam/.namb) sem interromper o PipeWire, com delegação do `Drop` do modelo antigo para thread GC (lock-free Drop-Delegation).
5. **Rust puro:** A escolha do rusto não foi por "hype". Há motivos muito fortes que compelem a ele. Além de garantias de segurança e de performance, por ser uma linguagem compilada similar arquiteturalmente aos tradicionais C/C++ - trata-se de uma linguagem com sintaxe moderna, muito expressiva e rica em recursos. Sintaxe que oferece garantias de segurança e de performance já em tempo de compilação. Por exemplo, versões estáticas (wavenet.rs) onde o tamanho do kernel e os canais são conhecidos em tempo de compilação permitem otimizações agressivas de loop unrolling pelo LLVM.

## 🚀 Guia Rápido

### Pré-requisitos

* Linha de base: Ubuntu 26.04+ / Linux kernel 7.0+ com PipeWire 1.6+ (possivelmente funciona em versões anteriores também).

* Processador x86-64-v3 com suporte AVX2 e FMA (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015).

* Toolchain do Rust 1.94+ (`cargo`).

* Pacotes de desenvolvimento: `sudo apt install build-essential cmake pkg-config pipewire libpipewire-0.3-dev clang libclang-dev`

* Para que o motor execute inabalável sob modelos NAM realistas (especialamente "Lite" e "Standard"), é fundamental conceder a autorização de políticas SCHED mais avançadas ao binário. Adicione seu usuário ao grupo de áudio do sistema e edite limits:
  
  1. `sudo usermod -aG audio $USER`
  2. Crie ou edite o arquivo de limites (ex: `sudo nano /etc/security/limits.d/audio.conf`):

```text
@audio   -  rtprio     95
@audio   -  memlock    unlimited
```

1. Crie uma regra no *udev* para permitir que o grupo `audio` bloqueie a latência de despertar da CPU (C-states), (ex: `sudo nano /etc/udev/rules.d/99-audio-dma-latency.rules`):
   *(Reinicie a máquina ou execute `sudo udevadm control --reload-rules && sudo udevadm trigger` para aplicar as regras)*

```text
KERNEL=="cpu_dma_latency", GROUP="audio", MODE="0664"
```

1. Também é muito recomendável configurar o governador da sua CPU (`intel_pstate` ou `amd_pstate`) para o modo **Performance**.
   * Desktop modernos (como GNOME no Ubuntu/Fedora ou KDE Plasma), o sistema já gerencia isso nativamente via power-profiles-daemon.
   * Caso você prefira o `tlp`, pode editar o /etc/tlp.conf
     `CPU_SCALING_GOVERNOR_ON_AC=performance`
     `CPU_SCALING_GOVERNOR_ON_BAT=powersave`

### Build e Execução

```bash
git clone https://github.com/fabiohl/nam-rs.git
cd nam-rs
cargo build --release
```

Você pode usar o `qpwgraph &` como um editor visual de conexões pipewire.
Para iniciar o processamento:

```bash
target/release/nam-rs --model tests/Neve31102-Pre30-R.nam
target/release/nam-rs --model tests/fixtures/models/BossWN-standard.nam --input-gain -3.0 --output-gain 0.0

# Em máquinas fracas ou setups de desktop, aumente o buffer para aliviar a CPU:
target/release/nam-rs --model tests/HeavyModel.nam --buffer-size 512
```

Após a inicialização, o nodo aparece na matriz PipeWire. Use `qpwgraph` ou `pw-link` para conectar a entrada de instrumento e a saída para monitores.

## 📚 Documentação

* [docs/architecture.md](docs/architecture.md) — Topologia, módulos e decisões de design
* [docs/dependencies.md](docs/dependencies.md) — Dependências sistêmicas e crates Rust

## 🧠 Modelos Suportados

O NAM-rs suporta nativamente arquivos Neural Amp Modeler (.nam ou .namb). Arquivos de Impulse Response (.wav) não são suportados (ao menos não no momento). Oferece dois níveis de operação de parsing:

* **Modo Estático (Altíssima Performance):** Construções de *Const Generics* dimensionadas em tempo de compilação.
  * **WaveNet:** Standard (16×8), Lite (12×8), Feather (8×4) e Nano (4×2)
  * **LSTM:** 1 e 2 Camadas (8 a 24 de Hidden Size: `1×8`, `1×12`, `1×16`, `1×24`, `2×8`, `2×12`, `2×16`)
* **Modo Dinâmico (Flexibilidade Absoluta):** Fallback ativado automaticamente ao carregar arranjos `.nam` com geometrias extrassensoriais não catalogadas (`num_layers` e `channels` arbitrários), operando sem *loop unrolling*.

## 🧪 Testes e Validação

O NAM-rs mantém uma suíte de **138 verificações** automatizadas distribuídas em cinco camadas:

```bash
# Testes unitários + integração + proptest + fuzz + E2E PipeWire
cargo test

# Apenas testes unitários inline (95 testes)
cargo test --lib

# Apenas testes de integração (29 testes, incluindo zero-alloc, block sizes, golden vectors Feather/Nano, modelos comunitários)
cargo test --test nam_infer_test

# Fuzz testing dos parsers (9 testes × 5000 cases = ~45.000 inputs)
cargo test --test proptest_parsers

# Benchmarks de latência (15 funções benchmark, 21 medições criterion)
cargo bench --bench inference_bench

# Lint rigoroso de código fonte (formatação + clippy)
utils/lints.sh
```

Categorias de teste incluem: parsing JSON e NAMB, **fuzz testing via proptest** (bytes adversários, JSON malformado, NAMB corrompido), **verificação zero-allocation** no hot path (counting allocator), estabilidade numérica de longa duração, auto-consistência (determinismo), golden vectors C++ ↔ Rust, pipeline E2E SPSC, paridade estático/dinâmico, estabilidade sob silêncio (denormals/DAZ/FTZ), rejeição de JSON malformado, gain staging roundtrip, hot-swap rápido de modelos, **block sizes variáveis** (1–512 amostras), **modelos comunitários** (5 modelos .nam) e **rejeição de formatos não-suportados** (Keras Legacy, ativações não-Tanh).

## 🤝 Contribuindo

Contribuições são bem-vindas! O projeto está em fase ativa de desenvolvimento.

* Testes + testes + testes + testes....
* Apesar do suporte a AVX-512, eu não tenho uma CPU suportado pra testar. Se alguém tiver alguma por ai, seria muito útil..
* Para validar alterações: execute `cargo test` e `utils/lints.sh` antes de submeter PRs.

## 🙏 Créditos e Agradecimentos

Este projeto herda lógica, inspiração e ciência de trabalhos notáveis na comunidade de áudio e inteligência artificial que pavimentaram este caminho.
Impossível mapear todos os que contribuíram para o avanço desta tecnologia. Mas no escopo dos nosso projeto, impossível não mencionar:

* **Steven Atkinson** — Pelo desenvolvimento pioneiro do [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler), sua pesquisa original sobre modelagem de amplificadores com deep learning e por compartilhar o ecossistema abertamente.
* **Mike Oliphant** — Pela excepcional biblioteca [NeuralAudio (C++)](https://github.com/mikeoliphant/NeuralAudio), que serviu de base direta e fonte de pesquisa para a transposição das lógicas de inferência e FastMath vetorial (SIMD) portadas para este motor estrutural.

## ⚖️ Licença e Transparência (Vibe Coding)

Este projeto é duplamente licenciado sob **MIT** ou **Apache License, Version 2.0**, à sua escolha. Veja os arquivos `LICENSE-MIT` e `LICENSE-APACHE` para mais detalhes.

**Nota de Transparência sobre Inteligência Artificial:** A arquitetura, as decisões rigorosas de engenharia, a documentação, a atenciosa regência dos agentes e a curadoria deste projeto são de autoria intelectual do mantenedor. Contudo, o código-fonte em si foi iterado e gerado integralmente com o auxílio de Inteligência Artificial (*Vibe Coding*). Mais especificamente, usando a IDE Google Antigravity.

O uso da licença Apache 2.0 visa justamente oferecer maior segurança jurídica aos contribuidores e usuários em relação a patentes, dado este modelo moderno de desenvolvimento de software. A estrutura está aberta para a comunidade refatorar, auditar e expandir livremente.
