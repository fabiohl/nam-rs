# Arquitetura NAM-rs: Cliente Standalone de Inferência Neural em Tempo Real

A arquitetura do NAM-rs é meticulosamente projetada para processamento DSP de baixa latência e aprendizado de máquina neural estritamente focado na simulação de, por exemplo, amplificadores, pedais de guitarra e equipamentos de áudio (NAM - Neural Amp Modeler). O projeto é concebido em Rust nativo e idiomático, operando como cliente PipeWire standalone no Linux.

## 1. Topologia PipeWire: Arquitetura Dual-Stream (Capture + Playback)

- **Processador Sistêmico (Estilo Easy Effects):** O NAM-rs declara-se como um **Virtual Sink** (`media.class = Audio/Sink`) via `pw_stream` (Direction::Input). O *WirePlumber* eleva-o a saída de som padrão. Apps (YouTube, VLC, etc.) enviam áudio automaticamente. Entretanto, **o monitor port do `Audio/Sink` copia os buffers antes do callback `process()`** — o que torna modificações in-place invisíveis ao monitor. Por isso, o NAM-rs usa uma **segunda stream de playback** (`Stream/Output/Audio`, Direction::Output) que lê o áudio processado de um `DspBridge` compartilhado e o entrega ao hardware.
- **DspBridge (Lock-Free Inter-Stream):** Buffer `#[repr(align(128))]` com arrays `[f32; 8192]` L/R + `AtomicU32` (n_samples) + `AtomicU64` (generation). O capture callback escreve com `fence(Release)`; o playback callback lê com `fence(Acquire)`. O `node.group = "nam-rs-dsp"` em ambas as streams garante clock sync. Alocação via `Box::leak` — nunca dropado em runtime.
- **Integração Lock-Free end-to-end:** `CLI → Parser (.nam/.namb) → Model Dispatcher → Prewarm → Injeção Lock-free Dual via SPSC → Thread DSP (SCHED_FIFO) → Gain Stage Input → Inferência Neural (Capture Stream) → DspBridge → Gain Stage Output → Playback Stream → Hardware Sink.`
- **True Stereo e Bypass Inteligente:** O callback `process()` negocia canais planares de 32 bits (`F32P` com canais = 2). Ele extrai os arrays e invoca inferência simétrica para L e R (`model_l`, `model_r`). Contudo, o sistema prevê uma mitigação vital de CPU: se o canal R for puro silêncio ou idênticamente replicado a L, a thread pula o processamento do R, inferindo apenas a percurso mono L e estampando bit-a-bit à saída R, poupando virtualmente 50% dos ciclos. Para instrumentos físicos (ex: Guitarra) tocarem por cima de "backing tracks", a rota do microfone deve ser conectada ao Sink do NAM-rs manualmente no `qpwgraph`.

## 2. Inferência FastMath e Microarquitetura (AVX2 / AVX-512)

- **Supressão Algorítmica Rápida (FMA):** A base abandona o custo letárgico nas Unidades Lógicas (`std::math`) usando intrinsics `core::arch::x86_64` para processamento paralelo do polinômio Minimax/Padé nas portas lógicas LSTM e ativações WaveNet. Multiplicadores vetoriais processados em "Fused Multiply-Add" (FMA) reduzem a operação por ciclo massivamente.
- **Funções de Ativação FastMath:** `simd_tanh` usa polinômio Padé de grau 5 + `_mm256_rsqrt_ps` com refinamento Newton-Raphson. `simd_sigmoid` deriva via identidade `0.5 * (1 + tanh(0.5*x))`. Erro máximo ~5e-3 validado por testes MSE. Variantes AVX-512 (`simd_tanh_avx512`, `simd_sigmoid_avx512`) operam sobre registradores ZMM de 512 bits com `_mm512_rsqrt14_ps`.
- **Dot Product SIMD — Multi-Accumulator ILP:** `dot_product_avx2` processa até 32 floats por iteração usando **4 acumuladores YMM independentes** (`sum0..sum3`) que quebram a cadeia de dependência FMA, saturando o throughput de 2 FMA ports em CPUs modernas. Loop graduado (32→16→8→escalar) otimiza tanto vetores longos quanto curtos (H=8..16 típicos de LSTM/WaveNet). `dot_product_avx512` usa 2 acumuladores ZMM (32 floats/iter). Ver `src/math/simd.rs`.
- **Multiversioning Estático via Trait `SimdMath` (Sprint 2):** O WaveNet estático utiliza uma **trait `SimdMath`** com implementações `Avx2Math` e `Avx512Math` que mapeiam diretamente para as funções intrínsecas correspondentes. O dispatch no hot-loop é resolvido em tempo de compilação via monomorphização genérica (`process_internal::<M: SimdMath>`), eliminando completamente a indireção de ponteiros de função (v-table). As funções públicas `process_avx2()` e `process_avx512()` são anotadas com `#[target_feature(enable = "...")]`, garantindo que o LLVM emita instruções nativas sem fallback. O mesmo padrão aplica-se a `prewarm_avx2()` / `prewarm_avx512()`. Ver `src/math/simd.rs` e `src/models/wavenet.rs`.
- **Multiversioning Dinâmico — LazyLock V-Table:** O WaveNet dinâmico (`wavenet_dyn.rs`) e o LSTM mantêm despacho via `SimdMathConfig::get()` que retorna `&'static SimdMathConfig` cacheado em `LazyLock`, eliminando a leitura atômica de `is_x86_feature_detected!` a cada bloco DSP. A inicialização ocorre **uma única vez** no primeiro acesso (overhead zero). Ver `src/math/simd.rs`.
- **Processamento Temporal em Bloco / Batch GEMM (Sprint 2):** O WaveNet estático processa múltiplos frames por invocação (`num_frames` até 64) em vez de sample-by-sample. `Conv1d::process_block`, `DenseLayer::process_block` e `WaveNetLayer::process_block_internal` iteram sobre o vetor `num_frames` inteiro de uma vez, mantendo os pesos estacionários em registradores YMM/ZMM. Isso transforma centenas de dot products rasos em multiplicações Small-GEMM hiperdensas, eliminando overhead de loop e evicções destrutivas na L1 Cache.
- **Arrays Genéricos:** Utiliza SoA com const generics para unrolling de loops, resolvendo gargalos no Branch Predictor (BTB). `WaveNetModel<CH, K, HEAD>` e `LstmModel1<H, H1_IH, H_H4>` com zero-allocation e trait `NamModel`.

> **Decisão Arquitetural (Lição Aprendida com as Sprints de LSTM vs WaveNet):** A tentativa de forçar paralelismo extremo (GEMV Fused Gate + Prefetch L1) no LSTM resultou em ganhos marginais (~1%) sob o teto de latência. O motor superscalar de CPUs modernas (*Out-of-Order Execution* e *Hardware Prefetcher*) já esgota a capacidade de vetorização inter-amostras no LSTM. Para latências sub-milisegundo, o projeto deslocou o foco de otimização pesada para as topologias **WaveNet**. Diferente da LSTM (recursividade IIR onde `t` depende estritamente de `t-1`), a convolução dilatada da WaveNet (estrutura FIR) permite processamento unificado em matriz (**Processamento Temporal em Bloco / Batch GEMM**). Agrupar dezenas de amostras num vetor denso destrava um teto matemático brutalmente mais alto, eliminando o *overhead* do loop interno, evicções destrutivas na L1 Cache e saturando portas FMA reais.

## 3. Gestão e Isolação Temporais

- **SCHED_FIFO e Affinity:** A thread principal DSP é ancorada por *Core Affinity* no *SCHED_FIFO* (se disponível), desativando a jurisdição do escalonador *CFS* do SO. O silício não sofrerá instâncias de *Cache Misses* no L1/L2.
- **SPSC Ring Buffers:** Comunicação parametrizada entre CLI e DSP (input/output gain, troca de modelo .nam/.namb, taxa de amostragem) transita via canais SPSC (`rtrb`), com payload `ParamPayload` alinhado a 128 bytes (`#[repr(align(128))]`) para blindar contra *False Sharing*. O campo `LoadModel` trafega com dois slots tipados (`model_l: Option<Box<DynamicModel>>`, `model_r: Option<Box<DynamicModel>>`). Canal SPSC dedicado para `NamResampler` garante que a construção de resamplers não trave o callback RT. `RtStatusFlags` substitui prints.
- **Hot-Swap Lock-Free com GC Thread duplo:** Ao trocar o par de modelos ativo, os `Box<DynamicModel>` obsoletos são enviados para uma fila SPSC GC (produzida na thread DSP, consumida background). Essa fila tem volume dobrado (acompanhando o estéreo) para assegurar que a desalocação ocorra via thread coletora, longe do strict schedule `SCHED_FIFO`.
- **Purga de I/O de Disco Concluída:** Toda a infraestrutura de gravação herdada do AudioRip (io_uring, fallocate, headers WAV, tokio-uring) foi extirpada. O NAM-rs é um injetor de modelo, não um capturador multimídia.
- **Tempo Linear de Sinc FIR:** `NamResampler` realiza conversão bidirecional de sample rate (`pw_rate→48 kHz` na entrada, `48 kHz→pw_rate` na saída) usando filtro Sinc Kaiser-BlackmanHarris2 com `sinc_len=256` e **interpolação Hermite cúbica** (upgrade de linear→cubic: +6 dB SNR no resampling com overhead <5%). Bypass automático quando `pw_rate == 48000`. Ver `src/dsp/resampler.rs` — `sinc_params()`.
- **Detecção Reativa de Sample Rate (RT-safe):** O callback `param_changed` do stream PipeWire detecta alterações de `AudioInfoRaw::rate` e as comunica via `AtomicU32` compartilhado. No `process()`, a detecção seta flags atômicas (`needs_resampler_rebuild`, `requested_rate`) em `RtStatusFlags` — **sem alocar**. A thread principal (loop `while !SHUTDOWN`) verifica as flags, constrói `NamResampler::new()` fora do RT, e envia via canal SPSC dedicado `Producer<NamResampler>`. O callback drena o canal e substitui o resampler ativo (zero-alloc swap). Suporta hot-plug de hardware externo sem falhas de transição.

## 4. Módulos Fonte Atuais

| Módulo                      | Responsabilidade                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/main.rs`               | Ponto de entrada: parser CLI (`lexopt`), detecção AVX2/FMA, inicialização PipeWire nativo, handler CTRL+C, thread GC de drop-delegation, stdin loop interativo, diagnósticos estruturados                                                                                                                                                                                                                                                                                       |
| `src/diagnostics.rs`        | Sistema de diagnósticos estruturados: `NamErrorCode` (catálogo E1xxx–E5xxx), `NamDiagnostic` (mensagem amigável + bloco de suporte copiável), `SystemSnapshot` (captura de ambiente). Zero dependências novas — opera exclusivamente fora da thread RT                                                                                                                                                                                                                          |
| `src/pw_host.rs`            | Host PipeWire dual-stream: Capture stream (Audio/Sink, Direction::Input) + Playback stream (Stream/Output/Audio, Direction::Output), DspBridge (#[repr(align(128))]) inter-stream lock-free, callback DSP RT (Core Affinity + SCHED_FIFO), resampling planar, dispatch True Stereo/Bypass Mono, gain staging.                                                                                                                                                                   |
| `src/spsc.rs`               | Flag SHUTDOWN (`AtomicBool`), payload SPSC `ParamPayload` (#[repr(align(128))]) com LoadModel (par `model_l`, `model_r`); fila GC suportando dupla descarga paralela; `RtStatusFlags` atômicas e canal SPSC de resampler.                                                                                                                                                                                                                                                       |
| `src/math/mod.rs`           | Módulo raiz de operações matemáticas e inferência neural                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `src/math/simd.rs`          | `dot_product_avx2` (YMM 256-bit, 4 acumuladores ILP), `dot_product_avx512` (ZMM 512-bit, 2 acumuladores ILP), `SimdMathConfig` (v-table `LazyLock` estática — zero overhead de dispatch via `get()`), trait `SimdMath` + structs `Avx2Math`/`Avx512Math` (dispatch estático por monomorphização para WaveNet), `set_daz_ftz` (DAZ+FTZ via intrínsecos + inline asm)                                                                                                             |
| `src/math/fastmath.rs`      | `simd_tanh`/`simd_sigmoid` AVX2 (YMM), `simd_tanh_avx512`/`simd_sigmoid_avx512` (ZMM), helpers `tanh_slice_avx2/512`, `sigmoid_slice_avx2/512`                                                                                                                                                                                                                                                                                                                                  |
| `src/models/mod.rs`         | Trait `NamModel`, dispatch WaveNet/LSTM, DynamicModel, type aliases LSTM (Lstm1x8..Lstm2x16)                                                                                                                                                                                                                                                                                                                                                                                    |
| `src/models/wavenet.rs`     | WaveNet CNN causal dilatada — SoA com const generics, Conv1d + DenseLayer + prewarm copy_buffer. Multiversioning estático via trait `SimdMath` (Avx2Math/Avx512Math) com `#[target_feature]` eliminando v-table no hot-loop. Processamento em bloco (Batch GEMM) via `process_block`/`process_block_internal` operando `num_frames` simultaneamente. Software prefetch proativo (`_mm_prefetch`) nos taps dilatados do kernel Conv1d para eliminar L1 cache misses previsíveis. |
| `src/models/wavenet_dyn.rs` | Implementação de fallback dinâmico (no-unrolling) para convoluções causais dilatadas, suportando modelos comunitários de geometria arbitrária não tabelada.                                                                                                                                                                                                                                                                                                                     |
| `src/models/lstm.rs`        | LSTM recorrente — gates [i\|f\|g\|o], macro `define_lstm_process!` unificando AVX2 e AVX-512, 1 e 2 camadas com const generics, despacho runtime via `SimdMathConfig::get()` (LazyLock v-table)                                                                                                                                                                                                                                                                                 |
| `src/models/lstm_dyn.rs`    | Implementação de fallback dinâmico para redes recorrentes LSTM submetidas em matrizes extensas heterogêneas operando iteradores sem *loop unrolling* explícito.                                                                                                                                                                                                                                                                                                                 |
| `src/loader/mod.rs`         | Módulo raiz de carregamento de modelos NAM (fora da thread RT)                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `src/loader/dispatcher.rs`  | Construtor responsável por inferir topologia a partir do `NamModelData` bruto, alocando `DynamicModel` via matrizes de const generics estáticas ou acionando fallback para modelos dinâmicos comunitários flexíveis. Valida `activation` (rejeita não-Tanh) e propaga `gated` para o path dinâmico.                                                                                                                                                                             |
| `src/loader/nam_json.rs`    | Parser do formato `.nam` (JSON) — `serde_json`, classificação de topologia WaveNet/LSTM                                                                                                                                                                                                                                                                                                                                                                                         |
| `src/loader/namb.rs`        | Parser do formato `.namb` — CRC32 IEEE 802.3 + Little-Endian                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `src/dsp/mod.rs`            | Módulo raiz DSP para operações pré/pós motor neural                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `src/dsp/gain.rs`           | Gain staging SIMD (AVX2 `_mm256_mul_ps`) baseado em metadados `input/output_level_dbu`. `detect_clipping_stereo_simd` detecta saturação estéreo via AVX2 (8 samples/iteração) substituindo loop escalar. Fórmulas de calibração: `input_db_adj = input_level_dbu − 12.0` (positivo = boost; referência NAM = 12 dBu); `output_db_adj = −18.0 − loudness` (alvo = −18 LUFS). Multiplicador linear: `10^((user_db + model_db_adj) / 20)`.                                         |
| `src/dsp/resampler.rs`      | `NamResampler`: wrapper multi-channel Planar (2 channels), resampler bidirecional FIR Sinc (rubato 0.16), interpolação Hermite **cúbica** (+6 dB SNR vs linear), bypass automático para frequências de 48 kHz cravadas.                                                                                                                                                                                                                                                         |

## 5. Gestão de Dependências DSP

### rubato 0.16.x — Decisão de Versão

O crate `rubato` é usado para resampling FIR Sinc de fase linear em `src/dsp/resampler.rs`.
A versão foi **fixada em `0.16.x`** (não `2.0.0`) pelos seguintes motivos técnicos:

| Critério               | rubato 0.16.x                                                | rubato 2.0.0                                   |
| ---------------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| **API de buffer**      | `Vec<Vec<f32>>` (planares x2 canais) — compatível zero-alloc | Requer `audioadapter` + `audioadapter-buffers` |
| **Superfície de dep.** | Mínima (sem dependências de wrapper)                         | +2 crates obrigatórios                         |
| **RT-safe**            | `process_into_buffer()` multi-canal sempre disponível        | Idem, mas API diferente                        |
| **Feature FFT**        | Desabilitada (`default-features = false`)                    | Idem                                           |

**Configuração no Cargo.toml:**

```toml
rubato = { version = "0.16", default-features = false }
# fft_resampler desabilitado: remove dependência RustFFT do binário final
```

### Fluxo DSP Bidirecional

```text
PipeWire Input (Nk Hz) — L/R (F32P)
    │
    ▼ apply_gain_simd(input_gain_mult)          — SIMD AVX2 dual-channel
    │
    ▼ NamResampler::process_input(Nk → 48k)    — Planar L e R processados sincronamente
    │
    ▼ NamModel::process(48 kHz)                — True Stereo: if R==L { bypass Mono() + L->R }
    │
    ▼ NamResampler::process_output(48k → Nk)   — Planar reconvertido simetricamente
    │
    ▼ apply_gain_simd(output_gain_mult)         — SIMD AVX2 dual-channel
    │
    ▼ DspBridge (fence Release)                 — Buffer compartilhado entre streams
    │
    ▼ Playback Stream (fence Acquire)           — Copia do bridge para output buffer
    │
    ▼ PipeWire Output (Nk Hz)                   — Entrega ao hardware (Direction::Output)
```

**Parâmetros do filtro Sinc:**

- `sinc_len = 256` — comprimento do filtro (maior qualidade, menor aliasing)
- `f_cutoff = 0.95` — corte a 95% do Nyquist (0.95 × fs/2)
- `oversampling_factor = 128` — superamostragem interna
- `window = BlackmanHarris2` — atenuação de stop-band > −100 dB
- `interpolation = Cubic` — Hermite cúbica (+6 dB SNR vs linear, overhead <5%)

**Plano de migração futura:**

- `rubato 2.0.0`: considerar quando `audioadapter` estabilizar; a API pública de `NamResampler` não muda.

## 6. Política de Testes (`cargo test` e `cargo bench`)

O projeto adota a convenção idiomática do Rust, com três camadas complementares:

### 6.1. Testes Unitários Inline (`#[cfg(test)]`) — 83 testes

Cada módulo em `src/` contém um bloco `#[cfg(test)] mod tests { ... }` no final do arquivo, testando funções e structs **privadas** com acesso direto. Estes testes são compilados apenas em modo test e não afetam o binário de produção.

| Módulo                      | Testes | Cobertura                                                                                                            |
| --------------------------- |:------:| -------------------------------------------------------------------------------------------------------------------- |
| `src/loader/dispatcher.rs`  | 16     | Build Standard/Feather/LSTM, rejeição arq./topologia, exaustão pesos, overflow, validação activation, gated dispatch |
| `src/dsp/gain.rs`           | 14     | Gain staging SIMD, true-bypass bitwise, extremos ±60dB/+24dB/±96dB, roundtrip 6dB, -0.0, silêncio, mono detect       |
| `src/loader/nam_json.rs`    | 11     | Parse WaveNet/LSTM/Feather, topologia Standard/Lite/Nano, rejeição JSON malformado                                   |
| `src/diagnostics.rs`        | 8      | Catálogo de códigos, unicidade numérica, formatação suporte, timestamp ISO 8601, snapshot, days_to_date              |
| `src/dsp/resampler.rs`      | 7      | Bypass 48 kHz, up/down/roundtrip 44k↔48k↔96k, impulse response input/output                                          |
| `src/loader/namb.rs`        | 5      | Parse binário, CRC32, header, magic, version                                                                         |
| `src/models/lstm.rs`        | 5      | Alocação, process zeros, determinismo, gate order, 2-layer                                                           |
| `src/math/fastmath.rs`      | 4      | MSE de `simd_tanh`/`simd_sigmoid` AVX2 e AVX-512 vs. `std::f32`                                                      |
| `src/models/wavenet.rs`     | 4      | Alocação, prewarm NaN-free, process zeros, determinismo                                                              |
| `src/math/simd.rs`          | 3      | `dot_product_avx2`/`dot_product_avx512`, `set_daz_ftz` MXCSR bits                                                    |
| `src/models/wavenet_dyn.rs` | 3      | Gated activation `tanh⊙sigmoid`, non-gated fallback, block_size 2×ch                                                 |
| `src/spsc.rs`               | 3      | RtStatusFlags default, canais SPSC, concorrência multi-thread                                                        |

> **Nota:** `src/main.rs` contém 0 testes. Isto é esperado — o `main.rs` é apenas bootstrapping (CLI parser, PipeWire init, stdin loop). Toda a lógica testável está em `src/lib.rs` e submódulos.
>
> Os testes estruturais recentes (ex: rejeição JSON malformado e gain staging roundtrip) consolidam o hardening da base para uso em cenários empacotados em releases mais maduros.

### 6.2. Testes de Integração (`tests/`) — 21 testes

O diretório `tests/` contém `nam_infer_test.rs`, que consome a API pública `nam_rs::*` como um usuário externo. Os 18 testes cobrem **11 categorias** distintas:

#### Parsing e Topologia (1 teste)

- **`test_wavenet_model_json_parsing`** — Parseia `BossWN-standard.nam`, valida arquitetura "WaveNet" e topologia `Standard`.

#### Estabilidade Numérica (1 teste)

- **`test_wavenet_computational_stability`** — Processa ~5.4s de onda senoidal pelo WaveNet sintético (4096 blocos×64 amostras em release, 512 em debug). Verifica finitude de todas as saídas e RMS ≤ 10.0.

#### Dispatcher com Modelo Real (2 testes)

- **`test_dispatcher_build_model_real_json`** — `build_model()` com `BossWN-standard.nam` produz `DynamicModel` funcional.
- **`test_dispatcher_build_model_real_lstm`** — `build_model()` com `BossLSTM-1x16.nam` produz `DynamicModel` funcional.

#### Auto-Consistência / Determinismo (2 testes)

- **`test_auto_consistency_wavenet`** — Dois `DynamicModel` idênticos geram saída bitwise identical (MSE = 0.0).
- **`test_auto_consistency_lstm`** — Idem para LSTM 1×16.

#### Golden Vectors C++ ↔ Rust (2 testes)

- **`test_golden_vectors_wavenet`** — Compara saída Rust vs. referência C++ (validação dual: MSE < 5e-2 + SNR ≥ 9 dB).
- **`test_golden_vectors_lstm`** — Compara saída Rust vs. referência C++ (validação dual: MSE < 1e-3 + SNR ≥ 22 dB).

#### Pipeline End-to-End SPSC (1 teste)

- **`test_end_to_end_spsc_pipeline`** — Cadeia completa CLI→SPSC→DSP sem PipeWire: parse + dispatcher + envio pela fila `rtrb` como `ParamPayload::LoadModel` + drainagem + inferência + validação.

#### Paridade Numérica Estático ↔ Dinâmico (2 testes)

- **`test_parity_lstm_static_vs_dynamic`** — MSE = 0.0 entre LSTM const-generic e fallback dinâmico.
- **`test_parity_wavenet_static_vs_dynamic`** — MSE = 0.0 entre WaveNet estático e fallback dinâmico.

#### NAMB Roundtrip (1 teste)

- **`test_namb_roundtrip_dispatcher_e2e`** — Cadeia `.namb` binário → `parse_namb` → `build_model` → inferência.

#### Estabilidade de Topologias Adicionais (3 testes)

- **`test_wavenet_stability_feather`** — Estabilidade WaveNet Feather (CH=8).
- **`test_wavenet_stability_nano`** — Estabilidade WaveNet Nano (CH=4).
- **`test_lstm_stability_2x8`** — Estabilidade LSTM 2×8.

#### Auto-Consistência Adicional (1 teste)

- **`test_auto_consistency_lstm_2x8`** — Determinismo absoluto LSTM 2×8.

#### Estabilidade sob Silêncio / Denormals (1 teste)

- **`test_denormal_stability_silence`** — Processa 4096 blocos de silêncio total por WaveNet e LSTM, validando finitude, ausência de subnormais na saída, estabilidade de magnitude e timing por bloco (< 500μs em release).

#### Hot-Swap Rápido via SPSC (1 teste)

- **`test_rapid_hot_swap_spsc`** — Carrega 3 modelos diferentes (WaveNet Standard, LSTM 1×16, WaveNet Feather), envia sequencialmente pela fila SPSC, drena e processa cada um verificando finitude, magnitude razoável e descarte correto do modelo anterior (ownership transfer `Box<DynamicModel>`).

### 6.3. Benchmarks `criterion` (`cargo bench`) — 6 benchmarks

O arquivo `benches/inference_bench.rs` mede a latência de processamento com o framework `criterion` (harness=false). O deadline de tempo-real a 48 kHz com buffer de 64 amostras é **1.33 ms**.

| Benchmark                               | Descrição                                      | Referência prática                           |
| --------------------------------------- | ---------------------------------------------- | -------------------------------------------- |
| `WaveNet_Standard_CH16_64samp_48kHz`    | Inferência WaveNet Standard (modelo real .nam) | 1 bloco DSP completo — deve caber em 1.33 ms |
| `LSTM_2x16_64samp_48kHz`                | Inferência LSTM 2×16 (sintético, 3345 pesos)   | Topologia recorrente mais pesada suportada   |
| `FastMath_tanh_AVX2_256elem`            | Ativação tanh Padé×rsqrt sobre 256 f32         | Kernel chamado N×layers/bloco no WaveNet     |
| `FastMath_sigmoid_AVX2_256elem`         | Ativação sigmoid derivada de tanh              | Kernel chamado N×gates/bloco no LSTM         |
| `WaveNet_Dynamic_Standard_64samp_48kHz` | Inferência WaveNet Dynamic (fallback dinâmico) | Mede overhead do path sem const generics     |
| `LSTM_Dynamic_1x16_64samp_48kHz`        | Inferência LSTM Dynamic 1×16 (fallback)        | Mede overhead do path sem const generics     |

> **Nota:** Durante `cargo bench`, os 76 testes unitários aparecem como `ignored` — isto é o comportamento normal do criterion, que re-roda o binário com harness desabilitado.

Execução: `cargo bench --bench inference_bench`

### 6.4. Golden Vectors e Infraestrutura C++

Os golden vectors são arquivos binários (`.golden.bin`) contendo input e output de referência gerados pelo motor C++ (NeuralAudio Internal mode). Servem para validar que o motor Rust produz resultados numericamente compatíveis com a implementação C++ de referência.

**Formato `.golden.bin`:**

```text
[u32 num_samples LE]
[f32×N input samples LE]       — senoidal 440 Hz a 48 kHz
[f32×N expected output LE]     — output do C++ NeuralAudio
```

**Regeneração:** `./tests/fixtures/golden_gen_build.sh`

### 6.5. Layout de `tests/fixtures/`

```text
tests/fixtures/
├── models/                         ← Modelos .nam (commitados no repo)
│   ├── BossWN-standard.nam
│   ├── BossWN-feather.nam
│   ├── BossWN-nano.nam
│   ├── BossLSTM-1x16.nam
│   └── BossLSTM-2x8.nam
├── unsupported/                    ← Formatos Keras/Legacy não suportados
│   ├── tw40_blues_deluxe_deerinkstudios.json
│   └── README.md
├── golden_wavenet_standard.bin     ← Golden vectors (gerados pelo C++)
├── golden_lstm_1x16.bin
├── golden_gen.cpp                  ← Gerador C++ de golden vectors
├── golden_gen_build.sh             ← Script de build do gerador
└── README.md
```

### 6.6. Convenções e Guardas

- **Guarda SIMD por runtime detection:** Testes que exercitam kernels AVX2/AVX-512 envolvem o corpo em `if std::is_x86_feature_detected!("avx2") && ...`, garantindo que máquinas sem suporte não sofram `SIGILL`.
- **Modelos de teste opcionais:** Testes que dependem de arquivos `.nam` reais fazem `if !path.exists() { eprintln!("SKIP: ..."); return; }`, permitindo execução parcial sem falsos positivos.
- **Comando de execução:** `cargo test` dispara todas as camadas. `cargo test --lib` executa apenas os 83 unitários inline; `cargo test --test nam_infer_test` apenas os 18 de integração.

## 7. Referências

Os seguintes repositórios GitHub são a principal base de referência para a implementação do NAM-rs:

- [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
- [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio)
