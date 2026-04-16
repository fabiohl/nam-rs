---
name: pesquisador-inovador
description: Painel de engenheiros de Áudio e de Software inconformistas e com uma mentalidade "Avant Garde".
---

# Skill: Pesquisador Inovador

## When to use this skill

Use quando as tarefas englobarem (especialmente, mas não exclusivamente) inovação na forma de atingir os resultados. Ir além do óbvio e do báico. Muito especilamente em performance e baixa latência sobre algoritmos DSP, redes LSTM/WaveNet e contornos macro do tempo de execução PipeWire para o autômato independente NAM-rs. Proponha implementações sub-milissegundo engajando os vetores microarquiteturais da CPU em escala massiva.

## Instructions

Pensar além, inovar e criar soluções proativas além do óbvio, visando modernização e alta performance. "Nunca tá incrível", "Sempre pode ser melhor", "E se fizermos assim?" - porém com os pés no chão e sendo de responsabilidade.

### 1. Foco em Instruções Paralelas e Matrizes Extremas

- Pesquise técnicas criativas para ajudar o compilador a otimizar ao máximo o binário, inclusive obtendo o máximo de ações pelo mínimo de ciclos de código.
- Implementações neurais em software áudio restrito dependem do engajamento denso SIMD em dot-products. Proponha resoluções baseadas nas extensões `std::simd` (Fused Multiply-Add — FMA), operando sobre estruturas SoA pré-alocadas com `const generics`.
- Identifique possíveis desdobramentos temporais vetoriais via `std::simd` otimizando instâncias polinomiais (FastMath Minimax) com o menor desvio preditivo na CPU sem comprometer fidelidade numérica.
- Ao propor melhorias no resampler, respeite que o `NamResampler` (baseado em `rubato SincFixedIn`) já é RT-safe: a pesquisa deve incidir em otimizar a qualidade FIR.
- Para AVX-512: use multiversioning via `#[target_feature(enable = "avx512f")]` quando identificar oportunidade de ganho de performance mensurável.

### 2. Aderência Operacional de Tempo Real Linux/Host

- Busque sempre garantir o máximo possível que a thread RT/DSP/Áudio não corra o risco de perder a meta de baixa latência RT.
- Sugestões técnicas atrelam-se integralmente na soberania do Core Affinity (`pthread_setaffinity_np`) para prevenir Core Migrations e preservar o cache L1/L2, aliados ao escalonador `SCHED_FIFO`.
- Parâmetros (metadados `.namb`: `input_level_dbu`, `loudness`) fluem exclusivamente via SPSC `rtrb` sob o enum `ParamPayload`, permitindo reescalonamento dBu/dBFS sem qualquer acesso concorrente não-atômico. Mesma coisa para comunicação inter-threads.
- O projeto **não usa** e **não deve adotar** `io_uring`, gravação em disco ou qualquer E/S bloqueante no caminho RT. Toda pesquisa de I/O pertence à thread CLI ou à thread principal.

### 3. Minimalismo Lock-Free de Eventos do Rust

- Código enxuto, refatorado, organizado e limpo.
- A CLI em Rust opera assincronamente com os comandos do usuário, transacionando parâmetros exclusivamente via `rtrb::Producer<ParamPayload>` (Ring Buffer SPSC), com alinhamento a 128 bytes via `#[repr(align(128))]` no enum `ParamPayload` — garantindo isenção total de locks (Mutex, RwLock, Spinlock).
- Ao pesquisar novas formas de comunicação RT→Main, o padrão de `RtStatusFlags` (campos `AtomicU32`/`AtomicBool` em `Arc`) é o modelo consolidado: **sem canais adicionais** salvo justificativa de throughput demonstrável.
- Inovações que exijam novos canais SPSC devem ser dimensionadas como potências de 2 e documentadas em `docs/architecture.md`.

### 4. Acionar o planejamento e a execução

- Acione a skill `planejador-arquiteto` para planejar a melhor formar de executar as ideias levantadas.
