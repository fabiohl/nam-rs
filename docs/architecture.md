# Arquitetura NAM-rs: Cliente Standalone de Inferência Neural em Tempo Real

A arquitetura do NAM-rs é meticulosamente projetada para processamento DSP de baixa latência e aprendizado de máquina neural estritamente focado na emulação de amplificadores e pedais de guitarra (NAM - Neural Amp Modeler). O projeto é concebido em Rust nativo e idiomático.

## 1. Topologia PipeWire e Standalone

- **PipeWire-First:** O servidor abandona abstrações tradicionais VST/LV2 para interagir como um cliente Standalone através da biblioteca `pipewire-rs` (via `nih_plug` standalone). O processo assume imediatamente a jurisdição percussiva.
- **Integração Lock-Free:** O carregamento do áudio é submetido a uma cadência rígida. A aplicação atua em ciclo de vida de conexão passiva; somente emitindo status contínuo após validação do arranjo matemático (Buffer Inicial Latente).
- **Pass-Through Transitório:** O callback `process()` opera em modo pass-through enquanto o motor de inferência não estiver implementado (Sprints 2-5). O áudio de entrada flui inalterado para a saída.

## 2. Inferência FastMath e Microarquitetura (AVX2 / AVX-512)

- **Supressão Algorítmica Rápida (FMA):** A base abandona o custo letárgico nas Unidades Lógicas (`std::math`) usando `std::simd` para processamento paralelo do polinômio Minimax nas portas lógicas LSTM. Multiplicadores vetoriais processados em "Fused Multiply-Add" reduzem a operação por ciclo massivamente.
- **Paralelismo Avançado Via Multiversioning:** Como vetores espessos ocultos necessitam de banda estendida, a thread dispõe despachos antecipados por meio do branch multiversioning com base numa verificação inicial de capacidade, garantindo total suporte a processamento monumental usando os ZMM Registers estendidos no `AVX-512`.
- **Arrays Genéricos:** Utiliza SoA com const generics, unrolling loops para resolver gargalos no Branch Predictor (BTB).

## 3. Gestão e Isolação Temporais

- **SCHED_FIFO e Affinity:** A Thread principal é ancorada por _Core Affinity_ no _SCHED_FIFO_ (se disponível), desativando a jurisdição do escalonador _CFS_ do SO. O silício não sofrerá instâncias de _Cache Misses_ no L1/L2.
- **SPSC Ring Buffers (planejado):** Comunicação parametrizada entre CLI e DSP (input/output gain, troca de modelo .namb) viajará via canais SPSC, blindados para _False Sharing_ a 128 bytes. A infraestrutura será implementada na Tarefa 1.2.
- **Purga de I/O de Disco Concluída (Tarefa 1.1):** Toda a infraestrutura de gravação herdada do AudioRip (io_uring, fallocate, headers WAV, tokio-uring) foi extirpada. O NAM-rs é um injetor de modelo, não um capturador multimídia.
- **Tempo Linear de Sinc FIR:** Efetuará superamostragem de Fase Linear endógena (Sprint 5).

## 4. Módulos Fonte Atuais

| Módulo          | Responsabilidade                                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `src/main.rs`   | Ponto de entrada: handler ctrlc, thread coordenadora de shutdown, inicialização do `nih_plug` standalone          |
| `src/audio.rs`  | Callback DSP: configuração RT (Core Affinity + SCHED_FIFO), seleção dinâmica de CPU, process() pass-through       |
| `src/buffer.rs` | Flag global de shutdown (`SHUTDOWN: AtomicBool`). Será expandido na Tarefa 1.2 com estruturas SPSC para CLI → DSP |
