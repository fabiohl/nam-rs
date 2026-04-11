---
name: pesquisador-inovador
description: Use esta habilidade para pensar além, inovar e criar soluções proativas além do óbvio, visando modernização e alta performance.
---

# Skill: Pesquisador Inovador

## When to use this skill

Use quando as tarefas englobarem inovação pesada sobre algoritmos DSP, redes LSTM/WaveNet e contornos macro do tempo de execução PipeWire para o autômato independente NAM-rs. Proponha implementações sub-milissegundo engajando os vetores microarquiteturais da CPU em escala massiva.

## Estado Atual do Projeto (referência obrigatória antes de propor inovações)

O NAM-rs está na **Fase Alpha**. A infraestrutura RT-safe está funcional:

- **Motor de inferência:** LSTM e WaveNet portados do NeuralAmpModelerCore em C++, rodando a 48 kHz.
- **Resampling bidirecional:** `rubato 0.16` com `SincFixedIn<f32>` (Kaiser Window); converte entre o rate do PipeWire e os 48 kHz internos. O resampler é construído fora do callback RT e enviado via canal SPSC dedicado (`rtrb::Producer<NamResampler>`).
- **Concorrência lock-free:** `rtrb` (SPSC Ring Buffer) transporta `ParamPayload` (InputGain, OutputGain, LoadModel, SetSampleRate) e `DynamicModel` para GC. `RtStatusFlags` (AtomicU32 + AtomicBool) comunica status RT→Main sem I/O.
- **CLI interativa:** implementada com `lexopt`, rodando em thread separada, envia payloads via SPSC. Zero interação com a thread DSP por qualquer outro meio.
- **Thread DSP:** configurada como `SCHED_FIFO` prioridade 90, fixada em CPU via `pthread_setaffinity_np`. **ZERO** alocação heap, **ZERO** I/O, **ZERO** locks no `process()` callback.
- **Gain staging:** `apply_gain_simd` usa `std::simd` (x86-64-v3 / AVX2+FMA). Ajuste automático dBu via metadados `.namb` (`input_level_dbu`, `loudness`).
- **Formatos de modelo:** `.nam` (JSON) e `.namb` (binário com CRC32). Dispatcher em `loader::dispatcher`.

## Instructions

### 1. Foco em Instruções Paralelas e Matrizes Extremas

- Implementações neurais em software áudio restrito dependem do engajamento denso SIMD em dot-products. Proponha resoluções baseadas nas extensões `std::simd` (Fused Multiply-Add — FMA), operando sobre estruturas SoA pré-alocadas com `const generics`.
- Identifique possíveis desdobramentos temporais vetoriais via `std::simd` otimizando instâncias polinomiais (FastMath Minimax) com o menor desvio preditivo na CPU sem comprometer fidelidade numérica.
- Ao propor melhorias no resampler, respeite que o `NamResampler` (baseado em `rubato SincFixedIn`) já é RT-safe: a pesquisa deve incidir em otimizar a qualidade FIR ou explorar alternativas que mantenham o padrão suporte a CPUs antigas'.
- Para AVX-512: use multiversioning via `#[target_feature(enable = "avx512f")]` quando identificar oportunidade de ganho de performance mensurável - mas sem quebrar o suporte a CPUs antigas'.

### 2. Aderência Operacional de Tempo Real Linux/Host

- Sugestões técnicas atrelam-se integralmente na soberania do Core Affinity (`pthread_setaffinity_np`) para prevenir Core Migrations e preservar o cache L1/L2, aliados ao escalonador `SCHED_FIFO`.
- Parâmetros (metadados `.namb`: `input_level_dbu`, `loudness`) fluem exclusivamente via SPSC `rtrb` sob o enum `ParamPayload`, permitindo reescalonamento dBu/dBFS sem qualquer acesso concorrente não-atômico.
- O projeto **não usa** e **não deve adotar** `io_uring`, gravação em disco ou qualquer E/S bloqueante no caminho RT. Toda pesquisa de I/O pertence à thread CLI ou à thread principal.

### 3. Minimalismo Lock-Free de Eventos do Rust

- A CLI em Rust opera assincronamente com os comandos do usuário, transacionando parâmetros exclusivamente via `rtrb::Producer<ParamPayload>` (Ring Buffer SPSC), com alinhamento a 128 bytes via `#[repr(align(128))]` no enum `ParamPayload` — garantindo isenção total de locks (Mutex, RwLock, Spinlock).
- Ao pesquisar novas formas de comunicação RT→Main, o padrão de `RtStatusFlags` (campos `AtomicU32`/`AtomicBool` em `Arc`) é o modelo consolidado: **sem canais adicionais** salvo justificativa de throughput demonstrável.
- Inovações que exijam novos canais SPSC devem ser dimensionadas como potências de 2 e documentadas em `docs/architecture.md`.

### 4. Acionar o planejamento e a execução

- Acione a skill `planejador-arquiteto` para planejar a melhor formar de executar as ideias levantadas.
