# Arquitetura NAM-rs: Cliente Standalone de Inferência Neural em Tempo Real

A arquitetura do NAM-rs é meticulosamente projetada para processamento DSP de baixa latência e aprendizado de máquina neural estritamente focado na emulação de amplificadores, pedais de guitarra e equipamentos de áudio simulados (NAM - Neural Amp Modeler). O projeto é concebido em Rust nativo e idiomático, operando como cliente PipeWire standalone no Linux.

## 1. Topologia PipeWire Standalone

- **PipeWire-First:** O servidor abandona abstrações tradicionais VST/LV2/CLAP para interagir como um cliente Standalone nativo através da biblioteca `pipewire-rs`, sem intermediários. O processo assume imediatamente a jurisdição percussiva.
- **Integração Lock-Free:** O carregamento do áudio é submetido a uma cadência rígida. A aplicação atua em ciclo de vida de conexão passiva; somente emitindo status contínuo após validação do arranjo matemático (Buffer Inicial Latente).
- **Pass-Through Transitório → Dispatch Ativo:** O callback `process()` extrai amostras `f32` do buffer PipeWire e as despacha para o modelo neural ativo (`WaveNet` ou `LSTM`) via trait `NamModel`. Quando nenhum modelo está carregado, opera em pass-through nativo (Injeção Nula).

## 2. Inferência FastMath e Microarquitetura (AVX2 / AVX-512)

- **Supressão Algorítmica Rápida (FMA):** A base abandona o custo letárgico nas Unidades Lógicas (`std::math`) usando intrinsics `core::arch::x86_64` para processamento paralelo do polinômio Minimax/Padé nas portas lógicas LSTM e ativações WaveNet. Multiplicadores vetoriais processados em "Fused Multiply-Add" (FMA) reduzem a operação por ciclo massivamente.
- **Funções de Ativação FastMath:** `simd_tanh` usa polinômio Padé de grau 5 + `_mm256_rsqrt_ps` com refinamento Newton-Raphson. `simd_sigmoid` deriva via identidade `0.5 * (1 + tanh(0.5*x))`. Erro máximo ~5e-3 validado por testes MSE.
- **Dot Product SIMD:** `dot_product_avx2` processa 8 floats por iteração usando `_mm256_fmadd_ps`, com loop tail escalar para fatias irregulares.
- **Paralelismo Avançado Via Multiversioning (futuro):** Despacho antecipado via `#[target_feature(enable = "avx512f,avx512vl")]` planejado para quando as redes WaveNet/LSTM criarem consumidores reais das operações SIMD.
- **Arrays Genéricos:** Utiliza SoA com const generics para unrolling de loops, resolvendo gargalos no Branch Predictor (BTB). `WaveNetModel<CH, K, HEAD>` e `LstmModel1<H, H1_IH, H_H4>` com zero-allocation e trait `NamModel`.

## 3. Gestão e Isolação Temporais

- **SCHED_FIFO e Affinity:** A thread principal DSP é ancorada por _Core Affinity_ no _SCHED_FIFO_ (se disponível), desativando a jurisdição do escalonador _CFS_ do SO. O silício não sofrerá instâncias de _Cache Misses_ no L1/L2.
- **SPSC Ring Buffers:** Comunicação parametrizada entre CLI e DSP (input/output gain, troca de modelo .namb, taxa de amostragem) transita via canais SPSC (`rtrb`), com payload `ParamPayload` alinhado a 128 bytes (`#[repr(align(128))]`) para blindar contra _False Sharing_.
- **Purga de I/O de Disco Concluída:** Toda a infraestrutura de gravação herdada do AudioRip (io_uring, fallocate, headers WAV, tokio-uring) foi extirpada. O NAM-rs é um injetor de modelo, não um capturador multimídia.
- **Tempo Linear de Sinc FIR:** `NamResampler` realiza conversão bidirecional de sample rate (`pw_rate→48 kHz` na entrada, `48 kHz→pw_rate` na saída) usando filtro Sinc Kaiser-BlackmanHarris2 com `sinc_len=256`, garantindo isolação total entre o rate do PipeWire e o rate interno dos modelos NAM (48 kHz). Bypass automático quando `pw_rate == 48000`.

## 4. Módulos Fonte Atuais

| Módulo                   | Responsabilidade                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `src/main.rs`            | Ponto de entrada: detecção AVX2/FMA, inicialização PipeWire nativo (`pipewire::init`), handler CTRL+C, coordenação de shutdown                                     |
| `src/pw_host.rs`         | Host PipeWire nativo: ThreadLoopBox, StreamBox Filter, callback DSP RT (Core Affinity + SCHED_FIFO), resampling bidirecional, dispatch NamModel, gain staging      |
| `src/spsc.rs`            | Flag SHUTDOWN (`AtomicBool`), payload SPSC `ParamPayload` (#[repr(align(128))]) com InputGain, OutputGain, LoadModel, SetSampleRate; setup do Ring Buffer (`rtrb`) |
| `src/math/mod.rs`        | Módulo raiz de operações matemáticas e inferência neural                                                                                                           |
| `src/math/simd.rs`       | `dot_product_avx2`: Dot product via AVX2+FMA sobre registradores YMM de 256 bits                                                                                   |
| `src/math/fastmath.rs`   | `simd_tanh` (Padé grau 5 + rsqrt Newton-Raphson), `simd_sigmoid` (via identidade com tanh)                                                                         |
| `src/models/mod.rs`      | Trait `NamModel`, dispatch WaveNet/LSTM, DynamicModel, type aliases LSTM (Lstm1x8..Lstm2x16)                                                                       |
| `src/models/wavenet.rs`  | WaveNet CNN causal dilatada — SoA com const generics, Conv1d + DenseLayer + prewarm copy_buffer                                                                    |
| `src/models/lstm.rs`     | LSTM recorrente — gates concatenadas [i\|f\|g\|o], 1 e 2 camadas com const generics                                                                                |
| `src/loader/mod.rs`      | Módulo raiz de carregamento de modelos NAM (fora da thread RT)                                                                                                     |
| `src/loader/nam_json.rs` | Parser do formato `.nam` (JSON) — `serde_json`, classificação de topologia WaveNet/LSTM                                                                            |
| `src/loader/namb.rs`     | Parser do formato `.namb` (Tone3000 binário) — CRC32 IEEE 802.3 + Little-Endian                                                                                    |
| `src/dsp/mod.rs`         | Módulo raiz DSP para operações pré/pós motor neural                                                                                                                |
| `src/dsp/gain.rs`        | Gain staging SIMD (AVX2 `_mm256_mul_ps`) baseado em metadados `input/output_level_dbu`                                                                             |
| `src/dsp/resampler.rs`   | `NamResampler`: resampler FIR Sinc Kaiser bidirecional (rubato 0.16), RT-safe, bypass auto em 48 kHz                                                               |

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

``` text
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
- Detecção automática de rate: callback `param_changed` do PipeWire → `SetSampleRate` SPSC → recriação do `NamResampler`.
