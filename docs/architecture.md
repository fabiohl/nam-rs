# Arquitetura NAM-rs: Cliente Standalone de Inferência Neural em Tempo Real

A arquitetura do NAM-rs é meticulosamente projetada para processamento DSP de baixa latência e aprendizado de máquina neural estritamente focado na emulação de amplificadores, pedais de guitarra e equipamentos de áudio simulados (NAM - Neural Amp Modeler). O projeto é concebido em Rust nativo e idiomático, operando como cliente PipeWire standalone no Linux.

## 1. Topologia PipeWire Standalone

- **PipeWire-First:** O servidor abandona abstrações tradicionais VST/LV2/CLAP para interagir como um cliente Standalone nativo através da biblioteca `pipewire-rs`, sem intermediários. O processo assume imediatamente a jurisdição percussiva.
- **Integração Lock-Free:** O carregamento do áudio é submetido a uma cadência rígida. A aplicação atua em ciclo de vida de conexão passiva; somente emitindo status contínuo após validação do arranjo matemático (Buffer Inicial Latente).
- **Pass-Through Transitório:** O callback `process()` opera em modo pass-through enquanto o motor de inferência neural não estiver integrado (pós-Sprint 3). Após integração, o dispatch para WaveNet/LSTM substituirá o pass-through.

## 2. Inferência FastMath e Microarquitetura (AVX2 / AVX-512)

- **Supressão Algorítmica Rápida (FMA):** A base abandona o custo letárgico nas Unidades Lógicas (`std::math`) usando intrinsics `core::arch::x86_64` para processamento paralelo do polinômio Minimax/Padé nas portas lógicas LSTM e ativações WaveNet. Multiplicadores vetoriais processados em "Fused Multiply-Add" (FMA) reduzem a operação por ciclo massivamente.
- **Funções de Ativação FastMath (implementado — Sprint 2):** `simd_tanh` usa polinômio Padé de grau 5 + `_mm256_rsqrt_ps` com refinamento Newton-Raphson. `simd_sigmoid` deriva via identidade `0.5 * (1 + tanh(0.5*x))`. Erro máximo ~5e-3 validado por testes MSE.
- **Dot Product SIMD (implementado — Sprint 2):** `dot_product_avx2` processa 8 floats por iteração usando `_mm256_fmadd_ps`, com loop tail escalar para fatias irregulares.
- **Paralelismo Avançado Via Multiversioning (futuro):** Despacho antecipado via `#[target_feature(enable = "avx512f,avx512vl")]` planejado para quando as redes WaveNet/LSTM (Sprint 3) criarem consumidores reais das operações SIMD.
- **Arrays Genéricos (futuro — Sprint 3):** Utilizará SoA com const generics para unrolling de loops, resolvendo gargalos no Branch Predictor (BTB).

## 3. Gestão e Isolação Temporais

- **SCHED_FIFO e Affinity (implementado — Sprint 1):** A thread principal DSP é ancorada por _Core Affinity_ no _SCHED_FIFO_ (se disponível), desativando a jurisdição do escalonador _CFS_ do SO. O silício não sofrerá instâncias de _Cache Misses_ no L1/L2.
- **SPSC Ring Buffers (implementado — Sprint 1):** Comunicação parametrizada entre CLI e DSP (input/output gain, troca de modelo .namb) transita via canais SPSC (`rtrb`), com payload `ParamPayload` alinhado a 128 bytes (`#[repr(align(128))]`) para blindar contra _False Sharing_.
- **Purga de I/O de Disco Concluída (Sprint 1):** Toda a infraestrutura de gravação herdada do AudioRip (io_uring, fallocate, headers WAV, tokio-uring) foi extirpada. O NAM-rs é um injetor de modelo, não um capturador multimídia.
- **Tempo Linear de Sinc FIR (futuro — Sprint 5):** Efetuará superamostragem de Fase Linear endógena para conversão de sample rate.

## 4. Módulos Fonte Atuais (pós-Sprint 2)

| Módulo                 | Responsabilidade                                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `src/main.rs`          | Ponto de entrada: detecção AVX2/FMA, inicialização PipeWire nativo (`pipewire::init`), handler CTRL+C, coordenação de shutdown |
| `src/pw_host.rs`       | Host PipeWire nativo: ThreadLoopBox, StreamBox Filter, callback DSP RT (Core Affinity + SCHED_FIFO), pass-through              |
| `src/spsc.rs`          | Flag SHUTDOWN (`AtomicBool`), payload SPSC `ParamPayload` (#[repr(align(128))]), setup do Ring Buffer (`rtrb`)                 |
| `src/math/mod.rs`      | Módulo raiz de operações matemáticas e inferência neural                                                                       |
| `src/math/simd.rs`     | `dot_product_avx2`: Dot product via AVX2+FMA sobre registradores YMM de 256 bits                                               |
| `src/math/fastmath.rs` | `simd_tanh` (Padé grau 5 + rsqrt Newton-Raphson), `simd_sigmoid` (via identidade com tanh)                                     |

## 5. Módulos Futuros (Sprints 3–5)

| Módulo Planejado         | Sprint | Responsabilidade                                                     |
| ------------------------ | ------ | -------------------------------------------------------------------- |
| `src/models/wavenet.rs`  | 3      | Rede CNN causal dilatada (WaveNet) — SoA estática com const generics |
| `src/models/lstm.rs`     | 3      | Rede LSTM recorrente — gates concatenadas com const generics         |
| `src/models/mod.rs`      | 3      | Trait `NamModel`, dispatch WaveNet/LSTM, Prewarm                     |
| `src/loader/nam_json.rs` | 4      | Parser do formato `.nam` (JSON) — serde_json                         |
| `src/loader/namb.rs`     | 4      | Parser do formato `.namb` (Tone3000 binário) — CRC32 + Little-Endian |
| `src/dsp/gain.rs`        | 4      | Gain staging SIMD baseado em metadados input/output_level_dbu        |
| `src/dsp/resampler.rs`   | 5      | Conversão de sample rate FIR Sinc de fase linear (rubato/resampler)  |
