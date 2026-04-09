# Arquitetura NAM-rs: Cliente Standalone de Inferência Neural em Tempo Real

A arquitetura do NAM-rs é meticulosamente projetada para processamento DSP de baixa latência e aprendizado de máquina neural estritamente focado na emulação de amplificadores, pedais de guitarra e equipamentos de áudio simulados (NAM - Neural Amp Modeler). O projeto é concebido em Rust nativo e idiomático, operando como cliente PipeWire standalone no Linux.

## 1. Topologia PipeWire Standalone

- **PipeWire-First:** O servidor abandona abstrações tradicionais VST/LV2/CLAP para interagir como um cliente Standalone nativo através da biblioteca `pipewire-rs`, sem intermediários. O processo assume imediatamente a jurisdição percussiva.
- **Integração Lock-Free end-to-end:** A arquitetura do projeto obedece ao seguinte fluxo determinístico: `CLI → Parser (.nam/.namb) → Model Dispatcher (determinação de arquitetura estática/dinâmica) → Prewarm (Thread CLI) → Injeção Lock-free via SPSC → Thread DSP (SCHED_FIFO) → Gain Stage Input → Inference Nativa (WaveNet/LSTM) → Gain Stage Output → PipeWire Output.`
- **Pass-Through Transitório → Dispatch Ativo:** O callback `process()` extrai amostras `f32` do buffer PipeWire e as despacha para o modelo neural ativo (`WaveNet` ou `LSTM`) via trait `NamModel`. Quando nenhum modelo está carregado, opera em pass-through nativo (Injeção Nula).

## 2. Inferência FastMath e Microarquitetura (AVX2 / AVX-512)

- **Supressão Algorítmica Rápida (FMA):** A base abandona o custo letárgico nas Unidades Lógicas (`std::math`) usando intrinsics `core::arch::x86_64` para processamento paralelo do polinômio Minimax/Padé nas portas lógicas LSTM e ativações WaveNet. Multiplicadores vetoriais processados em "Fused Multiply-Add" (FMA) reduzem a operação por ciclo massivamente.
- **Funções de Ativação FastMath:** `simd_tanh` usa polinômio Padé de grau 5 + `_mm256_rsqrt_ps` com refinamento Newton-Raphson. `simd_sigmoid` deriva via identidade `0.5 * (1 + tanh(0.5*x))`. Erro máximo ~5e-3 validado por testes MSE. Variantes AVX-512 (`simd_tanh_avx512`, `simd_sigmoid_avx512`) operam sobre registradores ZMM de 512 bits com `_mm512_rsqrt14_ps`.
- **Dot Product SIMD:** `dot_product_avx2` processa 8 floats por iteração usando `_mm256_fmadd_ps`; `dot_product_avx512` processa 16 floats com `_mm512_fmadd_ps` + `_mm512_reduce_add_ps`. Loop tail escalar preserva corretude em fatias irregulares.
- **Multiversioning AVX-512:** Despacho em tempo de execução via `SimdMathConfig::current()` que inspeciona CPUID e injeta ponteiros de função definitivos (`dot_product`, `tanh_slice`, `sigmoid_slice`). Quando AVX-512F + VL presentes, todos os kernels LSTM e FastMath operam sobre ZMM de 512 bits. Macro unificada `define_lstm_process!` parametriza ambas as vias sem duplicação de código.
- **Arrays Genéricos:** Utiliza SoA com const generics para unrolling de loops, resolvendo gargalos no Branch Predictor (BTB). `WaveNetModel<CH, K, HEAD>` e `LstmModel1<H, H1_IH, H_H4>` com zero-allocation e trait `NamModel`.

## 3. Gestão e Isolação Temporais

- **SCHED_FIFO e Affinity:** A thread principal DSP é ancorada por _Core Affinity_ no _SCHED_FIFO_ (se disponível), desativando a jurisdição do escalonador _CFS_ do SO. O silício não sofrerá instâncias de _Cache Misses_ no L1/L2.
- **SPSC Ring Buffers:** Comunicação parametrizada entre CLI e DSP (input/output gain, troca de modelo .nam/.namb, taxa de amostragem) transita via canais SPSC (`rtrb`), com payload `ParamPayload` alinhado a 128 bytes (`#[repr(align(128))]`) para blindar contra _False Sharing_. O campo `LoadModel.model` trafega como `Option<Box<DynamicModel>>` tipado (sem ponteiro opaco `*mut ()`). Canal SPSC dedicado para `NamResampler` (Main→RT) garante que a construção de resamplers (com alocações heap) ocorra exclusivamente fora do callback RT. `RtStatusFlags` (struct atômica) substitui todos os `println!`/`eprintln!` no callback por comunicação silenciosa RT→Main.
- **Hot-Swap Lock-Free com GC Thread:** Ao trocar o modelo ativo, o `Box<DynamicModel>` obsoleto é enviado para uma segunda fila SPSC GC (produzida na thread DSP, consumida em thread background). O `Drop` ocorre fora do limite `SCHED_FIFO`, garantindo ausência de alocações e liberações de heap no hot path.
- **Purga de I/O de Disco Concluída:** Toda a infraestrutura de gravação herdada do AudioRip (io_uring, fallocate, headers WAV, tokio-uring) foi extirpada. O NAM-rs é um injetor de modelo, não um capturador multimídia.
- **Tempo Linear de Sinc FIR:** `NamResampler` realiza conversão bidirecional de sample rate (`pw_rate→48 kHz` na entrada, `48 kHz→pw_rate` na saída) usando filtro Sinc Kaiser-BlackmanHarris2 com `sinc_len=256`, garantindo isolação total entre o rate do PipeWire e o rate interno dos modelos NAM (48 kHz). Bypass automático quando `pw_rate == 48000`.
- **Detecção Reativa de Sample Rate (RT-safe):** O callback `param_changed` do stream PipeWire detecta alterações de `AudioInfoRaw::rate` e as comunica via `AtomicU32` compartilhado. No `process()`, a detecção seta flags atômicas (`needs_resampler_rebuild`, `requested_rate`) em `RtStatusFlags` — **sem alocar**. A thread principal (loop `while !SHUTDOWN`) verifica as flags, constrói `NamResampler::new()` fora do RT, e envia via canal SPSC dedicado `Producer<NamResampler>`. O callback drena o canal e substitui o resampler ativo (zero-alloc swap). Suporta hot-plug de hardware externo sem falhas de transição.

## 4. Módulos Fonte Atuais

| Módulo                      | Responsabilidade                                                                                                                                                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/main.rs`               | Ponto de entrada: parser CLI (`lexopt`), detecção AVX2/FMA, inicialização PipeWire nativo, handler CTRL+C, thread GC de drop-delegation, stdin loop interativo                                                                                                                  |
| `src/pw_host.rs`            | Host PipeWire nativo: ThreadLoopBox, StreamBox Filter, callback DSP RT (Core Affinity + SCHED_FIFO), resampling bidirecional, dispatch NamModel, gain staging. **Sprint 8:** zero alocações e zero I/O no callback — resampler via SPSC, status via `RtStatusFlags`             |
| `src/spsc.rs`               | Flag SHUTDOWN (`AtomicBool`), payload SPSC `ParamPayload` (#[repr(align(128))]) com InputGain, OutputGain, LoadModel (tipado `Box<DynamicModel>`), SetSampleRate; fila GC secundária (`rtrb`); `RtStatusFlags` (atômicas RT→Main); canal SPSC dedicado `NamResampler` (Main→RT) |
| `src/math/mod.rs`           | Módulo raiz de operações matemáticas e inferência neural                                                                                                                                                                                                                        |
| `src/math/simd.rs`          | `dot_product_avx2` (YMM 256-bit), `dot_product_avx512` (ZMM 512-bit), `SimdMathConfig` (v-table de despacho dinâmico multiversioning)                                                                                                                                           |
| `src/math/fastmath.rs`      | `simd_tanh`/`simd_sigmoid` AVX2 (YMM), `simd_tanh_avx512`/`simd_sigmoid_avx512` (ZMM), helpers `tanh_slice_avx2/512`, `sigmoid_slice_avx2/512`                                                                                                                                  |
| `src/models/mod.rs`         | Trait `NamModel`, dispatch WaveNet/LSTM, DynamicModel, type aliases LSTM (Lstm1x8..Lstm2x16)                                                                                                                                                                                    |
| `src/models/wavenet.rs`     | WaveNet CNN causal dilatada — SoA com const generics, Conv1d + DenseLayer + prewarm copy_buffer                                                                                                                                                                                 |
| `src/models/wavenet_dyn.rs` | Implementação de fallback dinâmico (no-unrolling) para convoluções causais dilatadas, suportando modelos comunitários de geometria arbitrária não tabelada.                                                                                                                     |
| `src/models/lstm.rs`        | LSTM recorrente — gates [i\|f\|g\|o], macro `define_lstm_process!` unificando AVX2 e AVX-512, 1 e 2 camadas com const generics, despacho runtime via `is_x86_feature_detected!`                                                                                                 |
| `src/models/lstm_dyn.rs`    | Implementação de fallback dinâmico para redes recorrentes LSTM submetidas em matrizes extensas heterogêneas operando iteradores sem _loop unrolling_ explícito.                                                                                                                 |
| `src/loader/mod.rs`         | Módulo raiz de carregamento de modelos NAM (fora da thread RT)                                                                                                                                                                                                                  |
| `src/loader/dispatcher.rs`  | Construtor responsável por inferir topologia a partir do `NamModelData` bruto, alocando `DynamicModel` via matrizes de const generics estáticas ou acionando fallback para modelos dinâmicos comunitários flexíveis.                                                            |
| `src/loader/nam_json.rs`    | Parser do formato `.nam` (JSON) — `serde_json`, classificação de topologia WaveNet/LSTM                                                                                                                                                                                         |
| `src/loader/namb.rs`        | Parser do formato `.namb` (Tone3000 binário) — CRC32 IEEE 802.3 + Little-Endian                                                                                                                                                                                                 |
| `src/dsp/mod.rs`            | Módulo raiz DSP para operações pré/pós motor neural                                                                                                                                                                                                                             |
| `src/dsp/gain.rs`           | Gain staging SIMD (AVX2 `_mm256_mul_ps`) baseado em metadados `input/output_level_dbu`                                                                                                                                                                                          |
| `src/dsp/resampler.rs`      | `NamResampler`: resampler FIR Sinc Kaiser bidirecional (rubato 0.16), RT-safe, bypass auto em 48 kHz                                                                                                                                                                            |

## 5. Gestão de Dependências DSP

### rubato 0.16.x — Decisão de Versão

O crate `rubato` é usado para resampling FIR Sinc de fase linear em `src/dsp/resampler.rs`.
A versão foi **fixada em `0.16.x`** (não `2.0.0`) pelos seguintes motivos técnicos:

| Critério               | rubato 0.16.x                                               | rubato 2.0.0                                   |
| ---------------------- | ----------------------------------------------------------- | ---------------------------------------------- |
| **API de buffer**      | `Vec<Vec<f32>>` pré-alocados — compatível com zero-alloc RT | Requer `audioadapter` + `audioadapter-buffers` |
| **Superfície de dep.** | Mínima (sem dependências de wrapper)                        | +2 crates obrigatórios                         |
| **RT-safe**            | `process_into_buffer()` sempre disponível                   | Idem, mas API diferente                        |
| **Feature FFT**        | Desabilitada (`default-features = false`)                   | Idem                                           |

**Configuração no Cargo.toml:**

```toml
rubato = { version = "0.16", default-features = false }
# fft_resampler desabilitado: remove dependência RustFFT do binário final
```

### Fluxo DSP Bidirecional

```text
PipeWire Input (Nk Hz)
    │
    ▼ apply_gain_simd(input_gain_mult)          — SIMD AVX2, antes do NAM
    │
    ▼ NamResampler::process_input(Nk → 48k)    — FIR Sinc Kaiser, ou bypass se Nk=48k
    │
    ▼ NamModel::process(48 kHz)                — WaveNet / LSTM inference
    │
    ▼ NamResampler::process_output(48k → Nk)   — FIR Sinc Kaiser, ou bypass se Nk=48k
    │
    ▼ apply_gain_simd(output_gain_mult)         — SIMD AVX2, após o NAM
    │
    ▼ PipeWire Output (Nk Hz)                  — placa de som recebe no rate original
```

**Parâmetros do filtro Sinc:**

- `sinc_len = 256` — comprimento do filtro (maior qualidade, menor aliasing)
- `f_cutoff = 0.95` — corte a 95% do Nyquist (0.95 × fs/2)
- `oversampling_factor = 128` — superamostragem interna
- `window = BlackmanHarris2` — atenuação de stop-band > −100 dB
- `interpolation = Linear` — equilíbrio custo/qualidade para RT

**Plano de migração futura:**

- `rubato 2.0.0`: considerar quando `audioadapter` estabilizar; a API pública de `NamResampler` não muda.

## 6. Política de Testes (`cargo test` e `cargo bench`)

O projeto adota a convenção idiomática do Rust, com três camadas complementares:

### 6.1. Testes Unitários Inline (`#[cfg(test)]`) — 51 testes

Cada módulo em `src/` contém um bloco `#[cfg(test)] mod tests { ... }` no final do arquivo, testando funções e structs **privadas** com acesso direto. Estes testes são compilados apenas em modo test e não afetam o binário de produção.

| Módulo                     | Testes | Cobertura                                                                      |
| -------------------------- |:------:| ------------------------------------------------------------------------------ |
| `src/dsp/gain.rs`          | 4      | Gain staging SIMD, true-bypass bitwise, extremos ±60dB/+24dB                   |
| `src/dsp/resampler.rs`     | 7      | Bypass 48 kHz, up/down/roundtrip 44k↔48k↔96k, impulse response                 |
| `src/loader/dispatcher.rs` | 11     | Build Standard/Feather/LSTM, rejeição arq./topologia, exaustão pesos, overflow |
| `src/loader/nam_json.rs`   | 6      | Parse WaveNet/LSTM/Feather, topologia Standard/Lite/Nano, validação            |
| `src/loader/namb.rs`       | 5      | Parse binário Tone3000, CRC32, header, magic, version                          |
| `src/math/fastmath.rs`     | 4      | MSE de `simd_tanh`/`simd_sigmoid` AVX2 e AVX-512 vs. `std::f32`                |
| `src/math/simd.rs`         | 2      | `dot_product_avx2`/`dot_product_avx512`                                        |
| `src/models/wavenet.rs`    | 4      | Alocação, prewarm NaN-free, process zeros, determinismo                        |
| `src/models/lstm.rs`       | 5      | Alocação, process zeros, determinismo, gate order, 2-layer                     |
| `src/spsc.rs`              | 3      | RtStatusFlags default, canais SPSC, concorrência multi-thread                  |

> **Nota:** `src/main.rs` contém 0 testes. Isto é esperado — o `main.rs` é apenas bootstrapping (CLI parser, PipeWire init, stdin loop). Toda a lógica testável está em `src/lib.rs` e submódulos.

### 6.2. Testes de Integração (`tests/`) — 9 testes

O diretório `tests/` contém `nam_infer_test.rs`, que consome a API pública `nam_rs::*` como um usuário externo. Os 9 testes cobrem **5 categorias** distintas:

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

- **`test_golden_vectors_wavenet`** — Compara saída Rust vs. referência C++ (MSE < 1e-4).
- **`test_golden_vectors_lstm`** — Compara saída Rust vs. referência C++ (MSE < 1e-5).

#### Pipeline End-to-End SPSC (1 teste)

- **`test_end_to_end_spsc_pipeline`** — Cadeia completa CLI→SPSC→DSP sem PipeWire: parse + dispatcher + envio pela fila `rtrb` como `ParamPayload::LoadModel` + drainagem + inferência + validação.

### 6.3. Benchmarks `criterion` (`cargo bench`) — 4 benchmarks

O arquivo `benches/inference_bench.rs` mede a latência de processamento com o framework `criterion` (harness=false). O deadline de tempo-real a 48 kHz com buffer de 64 amostras é **1.33 ms**.

| Benchmark                            | Descrição                                      | Referência prática                           |
| ------------------------------------ | ---------------------------------------------- | -------------------------------------------- |
| `WaveNet_Standard_CH16_64samp_48kHz` | Inferência WaveNet Standard (modelo real .nam) | 1 bloco DSP completo — deve caber em 1.33 ms |
| `LSTM_2x16_64samp_48kHz`             | Inferência LSTM 2×16 (sintético, 3345 pesos)   | Topologia recorrente mais pesada suportada   |
| `FastMath_tanh_AVX2_256elem`         | Ativação tanh Padé×rsqrt sobre 256 f32         | Kernel chamado N×layers/bloco no WaveNet     |
| `FastMath_sigmoid_AVX2_256elem`      | Ativação sigmoid derivada de tanh              | Kernel chamado N×gates/bloco no LSTM         |

> **Nota:** Durante `cargo bench`, os 51 testes unitários aparecem como `ignored` — isto é o comportamento normal do criterion, que re-roda o binário com harness desabilitado.

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
│   ├── BossLSTM-2x8.nam
│   └── tw40_blues_deluxe_deerinkstudios.json
├── golden_wavenet_standard.bin     ← Golden vectors (gerados pelo C++)
├── golden_lstm_1x16.bin
├── golden_gen.cpp                  ← Gerador C++ de golden vectors
├── golden_gen_build.sh             ← Script de build do gerador
└── README.md
```

### 6.6. Convenções e Guardas

- **Guarda SIMD por runtime detection:** Testes que exercitam kernels AVX2/AVX-512 envolvem o corpo em `if std::is_x86_feature_detected!("avx2") && ...`, garantindo que máquinas sem suporte não sofram `SIGILL`.
- **Modelos de teste opcionais:** Testes que dependem de arquivos `.nam` reais fazem `if !path.exists() { eprintln!("SKIP: ..."); return; }`, permitindo execução parcial sem falsos positivos.
- **Comando de execução:** `cargo test` dispara ambas as camadas. `cargo test --lib` executa apenas os 51 unitários inline; `cargo test --test nam_infer_test` apenas os 9 de integração.
