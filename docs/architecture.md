<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# Arquitetura NAM-rs: Cliente Standalone de Inferência Neural

A arquitetura do NAM-rs é projetada para processamento DSP de baixa latência e inferência neural focada em simulação de equipamentos de áudio (Neural Amp Modeler). Operando como cliente PipeWire standalone no Linux, utiliza Rust idiomático com foco em segurança RT (Real-Time).

## 1. Topologia PipeWire: Dual-Stream (Capture + Playback)

- **Virtual Sink (Audio/Sink):** O NAM-rs declara-se como saída de som padrão via `pw_stream`. Apps conectam-se automaticamente via WirePlumber.
- **Playback Stream (Stream/Output):** Uma segunda stream lê o áudio processado e o entrega ao hardware real, contornando limitações de monitor ports de sinks virtuais.
- **DspBridge (Lock-Free Double-Buffer):** Estrutura alinhada (128B) que isola as streams. O capture escreve no buffer inativo (Release); o playback lê do ativo (Acquire), sincronizado por `AtomicU64` (generation).
- **True Stereo e Bypass:** Inferência simétrica L/R. Se R for silêncio ou idêntico a L, o sistema pula a inferência de R (poupando ~50% de CPU).

> **Nota:** A topologia Dual-Stream é mantida em preferência ao `pw_filter` por garantir roteamento "plug-and-play" automático pelo WirePlumber e pela maturidade dos wrappers safe no crate `pipewire`.

## 2. Inferência & Microarquitetura (SIMD x86-64-v3/v4)

- **Multiversioning via Trait `SimdMath`:** Despacho estático via monomorfização genérica (`Avx2Math`, `Avx512Math`). O código é emitido com intrinsics nativos (`fma`, `avx2`, `avx512f`) sem overhead de v-table no hot-path.
- **FastMath Activations:** `simd_tanh` e `simd_sigmoid` usam polinômios Minimax de grau 7 com refinamento Newton-Raphson. Erro máximo ~1.2e-5, otimizado para o intervalo [-8, 8].
- **Dot Product ILP:** Implementação com 4 acumuladores independentes que saturam o throughput de portas FMA, quebrando cadeias de dependência.
- **Weight Compression F16C:** Pesos são armazenados em `f16` (Half-Precision) para "morar" na Cache L1 (32KB). A descompressão para `f32` ocorre on-the-fly via `_mm256_cvtph_ps`, eliminando gargalos de banda de memória (Memory-Bound).
- **Gate-Major Layout (LSTM):** Transposição de pesos para layout `[Gate][Input][Hidden]`, permitindo que cada porta seja calculada via GEMV (General Matrix-Vector multiplication) contíguo, maximizando o prefetch de cache e o throughput VNNI/FMA.
- **BF16 Nativo (AVX-512 BF16):** Suporte a kernels nativos via `_mm512_dpbf16_ps` para CPUs modernas (Sapphire Rapids/Zen5), dobrando o throughput em relação ao fallback.
- **Lazy BF16 Conditioning:** Mecanismo de cache stateful que detecta mudanças no vetor de condicionamento. A conversão de `f32` para `bf16` e a mixagem inicial só ocorrem se o parâmetro mudar, reduzindo o custo computacional de modelos com muitos frames e condicionamento estático.

## 3. Gestão e Isolação Temporais (Strict RT)

- **Thread DSP (SCHED_FIFO):** Afixada via Core Affinity (`select_optimal_cpu`) preferindo cores com menor carga de IRQs.
- **Zero-Allocation:** Proibição estrita de alocações heap, `Vec`, `Box` ou panics na thread DSP.
- **Isolamento de Jitter:**
  - `mlockall` para evitar page faults.
  - PM QoS Lock (`/dev/cpu_dma_latency`) para desabilitar C-States profundos.
  - Desabilitação de THP (Transparent Huge Pages) via `prctl` para evitar spikes de compactação do kernel.
- **Telemetria de Alta Precisão (RDTSC):** Substituição de `Instant::now()` (syscall vDSO) por leitura direta do TSC calibrado no callback RT. Garante precisão de ~1ns com overhead de ~1 ciclo de CPU, eliminando jitter induzido pelo kernel na medição de carga de DSP.
- **Canais SPSC (rtrb):** Comunicação lock-free entre CLI (async) e DSP (RT). Payload alinhado (128B) para evitar False Sharing.

## 4. Estrutura de Módulos

| Módulo | Responsabilidade |
| :--- | :--- |
| `pw_host` / `rt_setup` | Orquestração PipeWire, afinidade, RT priority e isolamento. |
| `models/` | Implementações WaveNet (CNN) e LSTM recorrente (Static/Dynamic). |
| `math/` | `SimdMath` trait, FastMath, dot products e suporte BF16. |
| `dsp/resampler` | Resampler Sinc Polifásico nativo (Minimum Phase). |
| `dsp/gate` | FSM de Histerese Dinâmica e rampa linear SIMD. |
| `loader/` | Parsing de modelos `.nam` (JSON) e `.namb` (binário). |

## 5. DSP & Resampling Nativo

O NAM-rs utiliza um **Resampler Sinc Polifásico de Fase Mínima** nativo, substituindo dependências externas.

- **Vantagens:** Elimina pré-ringing (energia concentrada no início), reduz latência algorítmica de ~1.5ms para ~0.1ms e oferece performance ~9x superior via convolução AVX2/AVX-512 dedicada.
- **Gate FSM:** Implementa histerese temporal e de amplitude (Schmitt Trigger) para evitar oscilações em níveis de ruído. Inclui rampa linear SIMD para transições suaves (fade-in/out).

### Fluxo DSP Bidirecional

```text
PipeWire Input (Nk Hz)
    │
    ▼ Gate FSM: Detecção de Silêncio/Mono + Rampa de Ganho
    │
    ▼ Input Gain (SIMD)
    │
    ▼ NamResampler::process_input (Nk → 48 kHz)
    │
    ▼ NamModel::process (Inferência Neural @ 48 kHz)
    │
    ▼ NamResampler::process_output (48 kHz → Nk)
    │
    ▼ Output Gain (SIMD) + Clipping
    │
    ▼ DspBridge (Lock-Free Write) → Playback Stream (Read) → Hardware
```

## 6. Estratégia de Testes & Qualidade

- **Unitários Inline:** Testes de lógica interna e matemática rápida.
- **Integração (`tests/`):** Validação de inferência E2E, parsing de modelos reais e paridade estático-dinâmico.
- **Zero-Allocation Guard:** Uso de um `CountingAllocator` customizado em testes para garantir rigorosamente que o hot-path não aloca heap.
- **Fuzz Testing (`proptest`):** ~45.000 inputs adversários para garantir robustez absoluta dos parsers contra dados malformados.
- **Golden Vectors:** Comparação bit-a-bit ou MSE contra referências C++ (NeuralAudio/NAMCore).

## 8. Formato Binário NAMB (Native Audio Model Binary)

O formato `.namb` é uma evolução otimizada do JSON original para uso em tempo-real.

- **NAMB v1:** Encapsula o JSON de metadados e os pesos em `f32` (Little-Endian) em um único bloco binário com CRC32.
- **NAMB v2 (Pre-Transposed):** Armazena os pesos diretamente no layout final do kernel (Gate-Major para LSTM ou Interleaved-4 para WaveNet).
  - Elimina a necessidade de transposição de memória durante o carregamento.
  - Reduz latência de swap de modelo de ~50ms para <1ms (cold-path de carregamento).
  - Identificado por flag no header `NambHeader::layout_type`.

## 9. Referências

Os seguintes repositórios GitHub são a principal base de referência para a implementação do NAM-rs:

- [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
- [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio)
