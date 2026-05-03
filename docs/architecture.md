<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# Arquitetura NAM-rs: Cliente Standalone de Inferência Neural em Tempo Real

A arquitetura do NAM-rs é meticulosamente projetada para processamento DSP de baixa latência e aprendizado de máquina neural estritamente focado na simulação de, por exemplo, amplificadores, pedais de guitarra e equipamentos de áudio (NAM - Neural Amp Modeler). O projeto é concebido em Rust nativo e idiomático, operando como cliente PipeWire standalone no Linux.

## 1. Topologia PipeWire: Arquitetura Dual-Stream (Capture + Playback)

- **Processador Sistêmico (Estilo Easy Effects):** O NAM-rs declara-se como um **Virtual Sink** (`media.class = Audio/Sink`) via `pw_stream` (Direction::Input). O *WirePlumber* eleva-o a saída de som padrão. Apps (YouTube, VLC, etc.) enviam áudio automaticamente. Entretanto, **o monitor port do `Audio/Sink` copia os buffers antes do callback `process()`** — o que torna modificações in-place invisíveis ao monitor. Por isso, o NAM-rs usa uma **segunda stream de playback** (`Stream/Output/Audio`, Direction::Output) que lê o áudio processado de um `DspBridge` compartilhado e o entrega ao hardware.
- **DspBridge (Lock-Free Inter-Stream Double-Buffer):** Estrutura `#[repr(align(128))]` contendo dois `BridgeBuffer`s (front/back), cada um com arrays `[f32; 8192]` L/R e `n_samples`. Um `AtomicUsize` indica o índice de leitura ativo, garantindo isolamento total entre leitura e escrita. O capture callback escreve no buffer inativo e troca o índice com `fence(Release)`; o playback callback lê do buffer ativo com `fence(Acquire)`, validando atualizações por um `AtomicU64` (generation). O `node.group = "nam-rs-dsp"` em ambas as streams garante clock sync, enquanto o `node.link-group = "nam-rs-link-group"` evita que o gerenciador de sessão (WirePlumber) crie loops de feedback automáticos entre os nós do mesmo processo. Alocação via `Box::leak` — nunca dropado em runtime.
- **Integração Lock-Free end-to-end:** `CLI → Parser (.nam/.namb) → Model Dispatcher → Prewarm → Injeção Lock-free Dual via SPSC → Thread DSP (SCHED_FIFO) → Gain Stage Input → Inferência Neural (Capture Stream) → DspBridge → Gain Stage Output → Playback Stream → Hardware Sink.`
- **True Stereo e Bypass Inteligente:** O callback `process()` negocia canais planares de 32 bits (`F32P` com canais = 2). Ele extrai os arrays e invoca inferência simétrica para L e R (`model_l`, `model_r`). Contudo, o sistema prevê uma mitigação vital de CPU: se o canal R for puro silêncio ou idênticamente replicado a L, a thread pula o processamento do R, inferindo apenas a percurso mono L e estampando bit-a-bit à saída R, poupando virtualmente 50% dos ciclos. Para instrumentos físicos (ex: Guitarra) tocarem por cima de "backing tracks", a rota do microfone deve ser conectada ao Sink do NAM-rs manualmente no `qpwgraph`.

> **Decisão Arquitetural — Adiamento da Migração para `pw_filter` (2026-04-30):**
>
> A migração da topologia Dual-Stream (Audio/Sink + Stream/Output + DspBridge) para `pw_filter` (`Audio/Duplex`, processamento in-place) foi **avaliada e adiada** pelos seguintes motivos:
>
> 1. **Ausência de wrapper safe no crate `pipewire` 0.9.2:** O `pw_filter` possui FFI disponível em `pipewire-sys` (27 funções geradas automaticamente), mas o crate de alto nível não expõe nenhuma abstração segura. Toda a implementação exigiria FFI unsafe direto, criando uma superfície de manutenção frágil sem suporte upstream.
> 2. **Risco de regressão no roteamento WirePlumber:** Com `Audio/Sink`, o WirePlumber elege o NAM-rs como default sink automaticamente — apps (YouTube, VLC) conectam sem intervenção. Com `Audio/Duplex`, o comportamento do gerenciador de sessão é incerto e poderia exigir configuração manual do usuário, quebrando a experiência "plug-and-play".
> 3. **Estabilidade comprovada da arquitetura atual:** A topologia Dual-Stream com DspBridge está operacional, testada (inclusive sob concorrência em `test_dsp_bridge_concurrent_access`) e performando dentro das margens RT (WaveNet Standard 64samp ≈ 198 µs, 85% de margem). O overhead do bridge (~4 cópias `copy_nonoverlapping` + 5 operações atômicas por ciclo) é desprezível frente ao custo da inferência neural.
>
> **Condições para revisita futura:** (a) O crate `pipewire-rs` expor wrappers safe para `pw_filter`, ou (b) validação experimental isolada confirmando que `Audio/Duplex` preserva o roteamento automático do WirePlumber para Virtual Sinks com `priority.session` elevada.

## 2. Inferência FastMath e Microarquitetura (AVX2 / AVX-512)

- **Supressão Algorítmica Rápida (FMA):** A base abandona o custo letárgico nas Unidades Lógicas (`std::math`) usando intrinsics `core::arch::x86_64` para processamento paralelo do polinômio Minimax/Padé nas portas lógicas LSTM e ativações WaveNet. Multiplicadores vetoriais processados em "Fused Multiply-Add" (FMA) reduzem a operação por ciclo massivamente.
- **Funções de Ativação FastMath:** `simd_tanh` usa polinômio Minimax de grau 7 (3 coeficientes: `c0`, `c1`, `c2`) + `_mm256_rsqrt_ps` com refinamento Newton-Raphson. `simd_sigmoid` deriva via identidade `0.5 * (1 + tanh(0.5*x))`. Erro máximo **~1.2e-5** por ativação, validado por testes MSE com threshold `1e-4`. Os coeficientes foram derivados via Diferencial Evolution Minimax otimizando o intervalo `[-8, 8]`. Variantes AVX-512 (`simd_tanh_avx512`, `simd_sigmoid_avx512`) operam sobre registradores ZMM de 512 bits com `_mm512_rsqrt14_ps`, usando os mesmos coeficientes de grau 7.
- **Dot Product SIMD — Multi-Accumulator ILP:** `dot_product_avx2` processa até 32 floats por iteração usando **4 acumuladores YMM independentes** (`sum0..sum3`) que quebram a cadeia de dependência FMA, saturando o throughput de 2 FMA ports em CPUs modernas. Loop graduado (32→16→8→escalar) otimiza tanto vetores longos quanto curtos (H=8..16 típicos de LSTM/WaveNet). `dot_product_avx512` usa 2 acumuladores ZMM (32 floats/iter). Ver `src/math/simd.rs`.
- **Multiversioning Estático via Trait `SimdMath`:** O WaveNet estático utiliza uma **trait `SimdMath`** com implementações `Avx2Math` e `Avx512Math` que mapeiam diretamente para as funções intrínsecas correspondentes. O dispatch no hot-loop é resolvido em tempo de compilação via monomorphização genérica (`process_internal::<M: SimdMath>`), eliminando completamente a indireção de ponteiros de função (v-table). As funções públicas `process_avx2()` e `process_avx512()` são anotadas com `#[target_feature(enable = "...")]`, garantindo que o LLVM emita instruções nativas sem fallback. O mesmo padrão aplica-se a `prewarm_avx2()` / `prewarm_avx512()`. Ver `src/math/simd.rs` e `src/models/wavenet.rs`.
- **Multiversioning Dinâmico — LazyLock V-Table:** O WaveNet dinâmico (`wavenet_dyn.rs`) e o LSTM mantêm despacho via `SimdMathConfig::get()` que retorna `&'static SimdMathConfig` cacheado em `LazyLock`, eliminando a leitura atômica de `is_x86_feature_detected!` a cada bloco DSP. A inicialização ocorre **uma única vez** no primeiro acesso (overhead zero). Ver `src/math/simd.rs`.
- **Processamento Temporal em Bloco / Batch GEMM:** Tanto o WaveNet estático (`wavenet.rs`) quanto o WaveNet dinâmico (`wavenet_dyn.rs`) processam múltiplos frames por invocação (`num_frames` até 64) em vez de sample-by-sample. `Conv1d::process_block`, `DenseLayer::process_block` e `WaveNetLayer::process_block_internal` iteram sobre o vetor `num_frames` inteiro de uma vez, mantendo os pesos estacionários em registradores YMM/ZMM. Isso transforma centenas de dot products rasos em multiplicações Small-GEMM hiperdensas, eliminando overhead de loop e evicções destrutivas na L1 Cache. O path dinâmico (`WaveNetDynModel::process`) chunka o input em blocos de `WAVENET_MAX_NUM_FRAMES` e propaga `num_frames` real para todas as camadas — eliminando o antigo overhead sample-by-sample.
- **Fusão de Operadores no WaveNet Layer:** O `process_block_internal` do WaveNet estático opera com loop frame-first: para cada frame, os passos Conv1d → Input Mixin → Tanh → Head Accumulate → One-by-One → Residual executam sobre um array temporário `[0.0; CH]` que vive em registradores YMM (~64 bytes para CH=16), reduzindo o tráfego L1 de ~6 passes a ~2 passes. O `block_buffer` intermediário foi eliminado no path estático.
- **FMA Multi-Channel / Dot Product 4×:** `dot_product_4x` (trait `SimdMath` + `SimdMathConfig`) calcula 4 dot products simultaneamente reutilizando o mesmo carregamento do vetor de entrada, reduzindo loads L1 em ~75%. Utilizado em `Conv1d::process_single_frame`, `DenseLayer::process_single_frame`, `DenseLayer::process_acc_single_frame`, `DenseLayer::process_block`, e nos equivalentes dinâmicos (`Conv1dDyn::process_block`, `DenseLayerDyn::process_block`, etc.). Tail handling escalar para `OUT % 4 != 0`.
- **Arrays Genéricos:** Utiliza SoA com const generics para unrolling de loops, resolvendo gargalos no Branch Predictor (BTB). `WaveNetModel<CH, K, HEAD>` e `LstmModel1<H, H1_IH, H_H4>` com zero-allocation e trait `NamModel`.
- **Monomorfização de Modelos (Static Dispatch via Enum):** O núcleo de inferência repudia o uso de *Trait Objects* (`Box<dyn NamModel>`). Para evadir o custo computacional atrelado a *vtables*, toda arquitetura suportada é encapsulada num `enum DynamicModel` fechado. A implementação do `NamModel` sobre este enum usa rotinas `match` anotadas com `#[inline(always)]`. Esta decisão permite ao otimizador LLVM fundir os saltos dinâmicos do *hot-path* em código estritamente inline e manter total conformidade com *zero-allocation* do buffer SPSC à thread de DSP PipeWire. Qualquer nova topologia (WaveNet ou LSTM customizada) introduzida no projeto **deverá** ser compulsoriamente agregada a este `enum` como uma variante.
- **Compressão de Pesos F16C (VNNI-like):** A maior topologia suportada (`WaveNet Standard`) armazena cerca de 80 KB de pesos. Este volume ultrapassa a capacidade típica da Cache L1 de Dados (32 KB por núcleo nas arquiteturas Zen/Skylake), forçando o processador a despejar dados para a Cache L2, incorrendo num atraso de ~14 ciclos de clock para re-buscar cada linha de cache. Para contornar este gargalo de Memory-Bound, as matrizes de pesos são carregadas na heap convertidas de `f32` para `f16` (Half-Precision). A expansão de `f16` para `f32` ocorre "on-the-fly" nos próprios registradores L1 através das instruções nativas `_mm256_cvtph_ps` (AVX2) e `_mm512_cvtph_ps` (AVX-512). A latência da descompressão SIMD custa apenas ~4 a 5 ciclos (com throughput de 0.5 ciclo) e é plenamente sobreposta/mascarada pelo agendador superscalar (Out-of-Order Execution). A redução pela metade do consumo de memória RAM permite aos pesos do WaveNet "morarem" permanentemente no L1, superando imensamente o custo computacional da descompressão. A perda de precisão (10 bits de mantissa) degrada minimamente a fidelidade do áudio — os vetores de referência C++ acusam erros numéricos (MSE) confinados a escalas imperceptíveis (< 1e-4), verificados sistematicamente via `cargo test`.

> **Decisão Arquitetural (Lição Aprendida com LSTM vs WaveNet):** A tentativa de forçar paralelismo extremo (GEMV Fused Gate + Prefetch L1) no LSTM resultou em ganhos marginais (~1%) sob o teto de latência. O motor superscalar de CPUs modernas (*Out-of-Order Execution* e *Hardware Prefetcher*) já esgota a capacidade de vetorização inter-amostras no LSTM. Para latências sub-milisegundo, o projeto deslocou o foco de otimização pesada para as topologias **WaveNet**. Diferente da LSTM (recursividade IIR onde `t` depende estritamente de `t-1`), a convolução dilatada da WaveNet (estrutura FIR) permite processamento unificado em matriz (**Processamento Temporal em Bloco / Batch GEMM**). Agrupar dezenas de amostras num vetor denso destrava um teto matemático brutalmente mais alto, eliminando o *overhead* do loop interno, evicções destrutivas na L1 Cache e saturando portas FMA reais.

## 3. Gestão e Isolação Temporais

- **SCHED_FIFO e Affinity:** A thread principal DSP é ancorada por *Core Affinity* no *SCHED_FIFO* (se disponível), desativando a jurisdição do escalonador *CFS* do SO. O silício não sofrerá instâncias de *Cache Misses* no L1/L2. Além disso, a nomeação explícita da thread (`pthread_setname_np`) como `nam_rs_dsp` garante fácil observabilidade e profiling pelo SO (via `htop`/`perf`). A seleção do núcleo é feita por `select_optimal_cpu()`, que ranqueia CPUs por capacidade (`cpu_capacity`) e menor carga de interrupções (`/proc/interrupts`), preferindo cores com menos IRQs despachadas — minimizando preempção involuntária na thread DSP.
- **Desabilitação de THP (Transparent Huge Pages):** Antes do `mlockall`, a função `configure_realtime_thread` desabilita THP **exclusivamente para o processo** via `prctl(PR_SET_THP_DISABLE, 1)`. O thread `khugepaged` do kernel executa compactação e promoção de páginas em background, podendo causar latency spikes de 50–500 µs ao tocar memória na região do processo. O `prctl` elimina esse vetor de jitter sem afetar outros processos do sistema. Ver `src/pw_host.rs` — `configure_realtime_thread()`.
- **Advisory de Isolamento de IRQs:** No startup, `SystemSnapshot::emit_irq_advisory()` detecta o core afixado pela thread DSP e emite uma sugestão informativa ao usuário com o comando exato para isolar IRQs do sistema daquele core (via `/proc/irq/default_smp_affinity`). A máscara bitwise é calculada dinamicamente invertendo o bit do core afixado (`0xFFFFFFFF ^ (1 << core)`). Não é automatizado (requer `sudo`), mas fornece um one-liner pronto para cópia. Ver `src/diagnostics.rs` — `emit_irq_advisory()`.
- **Isolamento via Mlockall e PM QoS (C-States):** O motor previne falhas de paginação (page faults) na RAM com chamadas diretas para `mlockall` (exigindo limits `memlock` ilimitado no sistema). Para garantir que transistores da CPU respondam em *zero milissegundos*, um PM QoS Lock é acionado gravando `0` no fd `/dev/cpu_dma_latency` (mantido aberto durante a execução). Isto restringe o processador de descer a estados de economia profunda (como C6/C7). Além disso, o próprio DspBridge é blindado via `madvise` usando flags `MADV_DONTFORK | MADV_DONTDUMP`, bloqueando eventuais acessos concorrentes imprevistos pelo kernel (como serviços core dumpers ou duplicação de forks imprevistos).
- **SPSC Ring Buffers:** Comunicação parametrizada entre CLI e DSP (input/output gain, troca de modelo .nam/.namb, taxa de amostragem) transita via canais SPSC (`rtrb`), com payload `ParamPayload` alinhado a 128 bytes (`#[repr(align(128))]`) para blindar contra *False Sharing*. O campo `LoadModel` trafega com dois slots tipados (`model_l: Option<Box<DynamicModel>>`, `model_r: Option<Box<DynamicModel>>`). Canal SPSC dedicado para `NamResampler` garante que a construção de resamplers não trave o callback RT. `RtStatusFlags` substitui prints.
- **Hot-Swap Lock-Free com GC Thread duplo:** Ao trocar o par de modelos ativo, os `Box<DynamicModel>` obsoletos são enviados para uma fila SPSC GC (produzida na thread DSP, consumida background). Essa fila tem volume dobrado (acompanhando o estéreo) para assegurar que a desalocação ocorra via thread coletora, longe do strict schedule `SCHED_FIFO`.
- **Purga de I/O de Disco Concluída:** Toda a infraestrutura de gravação herdada do AudioRip (io_uring, fallocate, headers WAV, tokio-uring) foi extirpada. O NAM-rs é um injetor de modelo, não um capturador multimídia.
- **Tempo Linear de Sinc FIR:** `NamResampler` realiza conversão bidirecional de sample rate (`pw_rate→48 kHz` na entrada, `48 kHz→pw_rate` na saída) usando filtro Sinc Kaiser-BlackmanHarris2 com `sinc_len=256` e **interpolação Hermite cúbica** (upgrade de linear→cubic: +6 dB SNR no resampling com overhead <5%). Bypass automático quando `pw_rate == 48000`. Ver `src/dsp/resampler.rs` — `sinc_params()`.
- **Detecção Reativa de Sample Rate (RT-safe):** O callback `param_changed` do stream PipeWire detecta alterações de `AudioInfoRaw::rate` e as comunica via `AtomicU32` compartilhado. No `process()`, a detecção seta flags atômicas (`needs_resampler_rebuild`, `requested_rate`) em `RtStatusFlags` — **sem alocar**. A thread principal (loop `while !SHUTDOWN`) verifica as flags, constrói `NamResampler::new()` fora do RT, e envia via canal SPSC dedicado `Producer<NamResampler>`. O callback drena o canal e substitui o resampler ativo (zero-alloc swap). Suporta hot-plug de hardware externo sem falhas de transição.

## 4. Módulos Fonte Atuais

| Módulo                      | Responsabilidade                                                                                                                                                                                                                                                                                                                                            |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/main.rs`               | Ponto de entrada: orquestração do startup, detecção de features SIMD, inicialização PipeWire, handler de sinal CTRL+C (async-signal-safe), spawn da thread de GC (Drop-Delegation) e execução do host PipeWire.                                                                                                                                             |
| `src/cli.rs`                | Parser de linha de comando robusto baseado em `lexopt`: interpretação de ganhos, caminhos de modelo (com expansão de `~`) e tamanho de buffer.                                                                                                                                                                                                              |
| `src/colors.rs`             | Sistema de estilização ANSI para o terminal (Trait `Colorize`): fornece feedback visual premium para logs e diagnósticos.                                                                                                                                                                                                                                   |
| `src/diagnostics.rs`        | Sistema de diagnósticos estruturados: `NamErrorCode` (catálogo E1xxx–E5xxx), `NamDiagnostic` (mensagem amigável + bloco de suporte copiável), `SystemSnapshot` (captura de ambiente + advisory de IRQ isolation). Zero dependências novas — opera exclusivamente fora da thread RT.                                                                         |
| `src/pw_host.rs`            | Host PipeWire dual-stream: Capture stream (Audio/Sink) + Playback stream (Stream/Output), DspBridge (#[repr(align(128))]) lock-free, callback DSP RT com pipeline modularizado (`capture_dsp_pipeline` → `apply_input_stage` → `run_inference` → `apply_output_stage` → `write_bridge`), gain staging, resampling planar, dispatch True Stereo/Bypass Mono. |
| `src/rt_setup.rs`           | Utilitários de configuração e monitoramento de tempo real: Core Affinity (`select_optimal_cpu`), SCHED_FIFO (`configure_realtime_thread`), PM QoS (`lock_cpu_c_states`), DAZ/FTZ, THP disable, mlockall, polling de status RT (`poll_rt_status`), cálculo de multiplicadores de ganho, detecção de hardware sink.                                           |
| `src/spsc.rs`               | Flag SHUTDOWN, payload SPSC `ParamPayload` (#[repr(align(128))]) com LoadModel (par `model_l`, `model_r`); fila GC suportando dupla descarga paralela; `RtStatusFlags` atômicas e canal SPSC de resampler.                                                                                                                                                  |
| `src/math/mod.rs`           | Módulo raiz de operações matemáticas e inferência neural.                                                                                                                                                                                                                                                                                                   |
| `src/math/simd.rs`          | `dot_product_avx2/512` (acumuladores ILP), `SimdMathConfig` (v-table `LazyLock` estática), trait `SimdMath` + structs `Avx2Math`/`Avx512Math` (dispatch estático por monomorphização), `set_daz_ftz` (DAZ+FTZ via intrínsecos).                                                                                                                             |
| `src/math/fastmath.rs`      | `simd_tanh`/`simd_sigmoid` AVX2/AVX-512 via polinômios Minimax de grau 7, helpers de slice-processing para WaveNet e LSTM.                                                                                                                                                                                                                                  |
| `src/models/mod.rs`         | Trait `NamModel`, dispatch WaveNet/LSTM, DynamicModel, type aliases para perfis comuns.                                                                                                                                                                                                                                                                     |
| `src/models/wavenet.rs`     | WaveNet CNN causal estática (SoA + Const Generics), Batch GEMM via `process_block`, software prefetch proativo. Multiversioning estático via monomorphização (zero v-table).                                                                                                                                                                                |
| `src/models/wavenet_dyn.rs` | WaveNet dinâmico (fallback) para topologias não tabeladas. Suporte a Gated Activation e Batch GEMM (num_frames até 64).                                                                                                                                                                                                                                     |
| `src/models/lstm.rs`        | LSTM recorrente estático (SoA + Const Generics), macro `define_lstm_process!` unificando kernels AVX2/AVX-512.                                                                                                                                                                                                                                              |
| `src/models/lstm_dyn.rs`    | Fallback dinâmico para redes LSTM de arquitetura comunitária flexível.                                                                                                                                                                                                                                                                                      |
| `src/loader/mod.rs`         | Módulo raiz de carregamento (fora da thread RT).                                                                                                                                                                                                                                                                                                            |
| `src/loader/dispatcher/`    | Construtor modularizado: converte `NamModelData` em `DynamicModel`. Possui sub-módulos `wavenet.rs` e `lstm.rs` para especialização da lógica de construção. Valida `activation` e gerencia o `WeightCursor` para leitura sequencial de pesos.                                                                                                              |
| `src/loader/nam_json.rs`    | Parser do formato `.nam` (JSON) via `serde_json`, classificação automática de topologias (Standard/Lite/Feather/Nano).                                                                                                                                                                                                                                      |
| `src/loader/namb.rs`        | Parser do formato `.namb` (binário): validação de CRC32 IEEE 802.3 e mapeamento Little-Endian.                                                                                                                                                                                                                                                              |
| `src/dsp/mod.rs`            | Módulo raiz DSP.                                                                                                                                                                                                                                                                                                                                            |
| `src/dsp/gain.rs`           | Gain staging SIMD baseado em metadados dBu/LUFS. Fórmulas de calibração NAM (12 dBu ref), detecção de clipping estéreo SIMD e rampa linear SIMD AVX2 (`apply_ramp_simd`) para fade-in/out do Gate FSM.                                                                                                                                                      |
| `src/dsp/resampler.rs`      | `NamResampler`: motor nativo de resampling FIR Sinc Polifásico de Fase Mínima, convolução SIMD AVX2+FMA com path opcional AVX-512 (`convolve_avx512`, dispatch dinâmico via `SimdMathConfig`), bypass automático a 48 kHz.                                                                                                                                  |
| `src/dsp/sinc_kernel.rs`    | Geração offline de kernels FIR: Sinc+Kaiser (β=12), transformação de fase mínima via Cepstrum Real (f64), partição polifásica com alinhamento 32B.                                                                                                                                                                                                          |

## 5. Gestão de Dependências DSP

### Resampler FIR Sinc Polifásico Nativo — Fase Mínima

O NAM-rs utiliza um motor de resampling nativo implementado em `src/dsp/resampler.rs` e `src/dsp/sinc_kernel.rs`,
substituindo o antigo crate `rubato 0.16`. O novo resampler é um filtro FIR Sinc Polifásico de **Fase Mínima**
com convolução SIMD AVX2+FMA.

**Vantagens sobre a implementação anterior (rubato, fase linear):**

| Aspecto                  | rubato 0.16 (fase linear)             | Nativo (fase mínima)                                 |
| ------------------------ | ------------------------------------- | ---------------------------------------------------- |
| **Pré-ringing**          | Presente (simétrico ao redor do pico) | Eliminado (energia concentrada no início do impulso) |
| **Latência algorítmica** | ~1.5 ms                               | ~0.1 ms                                              |
| **Dependência externa**  | Crate `rubato` + `realfft`            | Apenas `rustfft` (FFT offline no `new()`)            |
| **SIMD**                 | Genérico (sem intrínsecas)            | AVX2+FMA nativo, coeficientes alinhados 32B          |
| **Performance**          | ~90 µs/bloco (1024 samp)              | ~10 µs/bloco (1024 samp) — **~9× mais rápido**       |

**Arquitetura interna:**

- **256 fases polifásicas** sobreabundantes com **32 taps/fase** (janela Kaiser, β=12, >120 dB stop-band).
- **Transformação de fase mínima** via Cepstrum Real em f64 (Oppenheim & Schafer), executada offline em `NamResampler::new()`.
- **Interpolação linear** entre fases adjacentes para razões de conversão arbitrárias.
- **Double-buffer delay line** para acesso SIMD contíguo sem lógica de wrap.
- **Zero alocações** no `process_input()` / `process_output()` — RT-safe.

> **Código-fonte:** `src/dsp/sinc_kernel.rs` — geração de kernel e partição polifásica.
> `src/dsp/resampler.rs` — motor de convolução SIMD e API pública `NamResampler`.

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

- Suporte a taxas adicionais (88.2 kHz, 176.4 kHz, 192 kHz) pode ser adicionado sem alteração da API pública de `NamResampler`.

## 6. Política de Testes (`cargo test` e `cargo bench`)

O projeto adota a convenção idiomática do Rust, com três camadas complementares:

### 6.1. Testes Unitários Inline (`#[cfg(test)]`) — 81 testes

Cada módulo em `src/` contém um bloco `#[cfg(test)] mod tests { ... }` no final do arquivo, testando funções e structs **privadas** com acesso direto. Estes testes são compilados apenas em modo test e não afetam o binário de produção.

| Módulo                      | Testes | Cobertura                                                                                                                                   |
| --------------------------- |:------:| ------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/dsp/gain.rs`           | 14     | Gain staging SIMD, true-bypass bitwise, extremos ±60dB/+24dB/±96dB, roundtrip 6dB, -0.0, silêncio, mono detect                              |
| `src/models/wavenet.rs`     | 12     | Alocação, prewarm NaN-free, process zeros, determinismo, Conv1d (identity/bias/dilation/zero/known), DenseLayer (identity/bias/rectangular) |
| `src/loader/nam_json.rs`    | 11     | Parse WaveNet/LSTM/Feather, topologia Standard/Lite/Nano, rejeição JSON malformado                                                          |
| `src/diagnostics.rs`        | 8      | Catálogo de códigos, unicidade numérica, formatação suporte, timestamp ISO 8601, snapshot, days_to_date                                     |
| `src/models/lstm.rs`        | 8      | Alocação, process zeros, determinismo, gate order, 2-layer, state evolution, variable block sizes, reset on prewarm                         |
| `src/dsp/resampler.rs`      | 7      | Bypass 48 kHz, up/down/roundtrip 44k↔48k↔96k, impulse response input/output                                                                 |
| `src/loader/namb.rs`        | 5      | Parse binário, CRC32, header, magic, version                                                                                                |
| `src/math/fastmath.rs`      | 4      | MSE de `simd_tanh`/`simd_sigmoid` AVX2 e AVX-512 vs. `std::f32`                                                                             |
| `src/math/simd.rs`          | 3      | `dot_product_avx2`/`dot_product_avx512`, `set_daz_ftz` MXCSR bits                                                                           |
| `src/models/wavenet_dyn.rs` | 3      | Gated activation `tanh⊙sigmoid`, non-gated fallback, block_size 2×ch                                                                        |
| `src/spsc.rs`               | 3      | RtStatusFlags default, canais SPSC, concorrência multi-thread                                                                               |
| `src/pw_host.rs`            | 1      | DspBridge concorrência lock-free double-buffer (fence Acquire/Release, 1000 buffers, coerência temporal)                                    |
| `src/loader/dispatcher/`    | 0      | Lógica de construção e dispatch testada via testes de integração em `tests/nam_infer_test.rs`.                                              |

> **Nota:** `src/main.rs` contém 0 testes. Isto é esperado — o `main.rs` é apenas bootstrapping (CLI parser, PipeWire init, stdin loop). Toda a lógica testável está em `src/lib.rs` e submódulos.

### 6.2. Testes de Integração (`tests/`) — 43 testes

O diretório `tests/` contém quatro arquivos de teste que consomem a API pública `nam_rs::*` como um usuário externo:

- **`nam_infer_test.rs`** (29 testes) — inferência neural, parsing, estabilidade, determinismo, golden vectors (Standard, LSTM, Feather, Nano), SPSC E2E, verificação zero-allocation, block sizes variáveis, modelos comunitários, rejeição de formatos.
- **`proptest_parsers.rs`** (9 testes) — fuzz testing via `proptest` (5000 cases cada) para `parse_nam_json()` e `parse_namb()`: bytes arbitrários, JSON semi-válido, truncamento, weight overflow, magic corrompido, CRC inválido, buffer truncado, offsets fora de limites. Nenhum panic em ~45.000 inputs adversários.
- **`proptest_math.rs`** (4 testes) — validação estocástica via `proptest` (10.000 cases cada): `prop_simd_tanh_avx2_rmse` e `prop_simd_sigmoid_avx2_rmse` verificam que FastMath AVX2 mantém RMSE < threshold contra `std::f32`; `prop_dot_product_avx2_vs_scalar` e `prop_dot_product_avx512_vs_scalar` verificam exatidão numérica do dot product SIMD contra acumulação f64 escalar com tolerância L1-norm.
- **`pw_integration_test.rs`** (1 teste) — `test_pipewire_headless_integration`: validação headless do PipeWire (init/connect/shutdown) sem hardware de áudio.

Os 29 testes de `nam_infer_test.rs` cobrem **16 categorias** distintas:

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

#### Golden Vectors C++ ↔ Rust (4 testes)

- **`test_golden_vectors_wavenet`** — Compara saída Rust vs. referência C++ (validação dual: MSE < 5e-2 + SNR ≥ 9 dB).
- **`test_golden_vectors_lstm`** — Compara saída Rust vs. referência C++ (validação dual: MSE < 1e-3 + SNR ≥ 22 dB).
- **`test_golden_vectors_wavenet_feather`** — Golden vectors WaveNet Feather (CH=8) — MSE < 5e-2, SNR ≥ 9 dB.
- **`test_golden_vectors_wavenet_nano`** — Golden vectors WaveNet Nano (CH=4) — MSE < 5e-2, SNR ≥ 9 dB.

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

#### Verificação Zero-Allocation no Hot Path (3 testes)

- **`test_zero_alloc_process_wavenet`** — Counting Allocator (`#[global_allocator]` condicional `#[cfg(test)]`) prova que `WaveNetModel::process()` estático completa com 0 alocações heap.
- **`test_zero_alloc_process_lstm`** — Idem para `LstmModel::process()` estático (1×16).
- **`test_zero_alloc_process_wavenet_dynamic`** — Idem para WaveNet dinâmico (Feather). Documenta exceção conhecida se o path dinâmico alocar (uso de `Vec` interno).

#### Block Sizes Variáveis (3 testes)

- **`test_wavenet_variable_block_sizes`** — WaveNet Standard com block sizes {1, 16, 32, 64, 128, 256, 512}. Verifica finitude, RMS ≤ 10.0, e consistência entre block_size=1 e demais (MSE < 1e-7).
- **`test_lstm_variable_block_sizes`** — LSTM 1×16 com mesmos block sizes e critérios.
- **`test_wavenet_dynamic_variable_block_sizes`** — WaveNet Dynamic (fallback) com mesmos block sizes e critérios.

#### Modelos Comunitários (1 teste)

- **`test_community_models_inference`** — Exercita 5 modelos de `tests/nam_files/` (ChandlerRedd47, EVH-5150, NEVE1073, UA610B, little-bear-t7). Valida parse, build, prewarm, process, finitude, magnitude < 100.0 e topologia (Standard/Lite) detectada corretamente.

#### Rejeição de Formatos Não-Suportados (2 testes)

- **`test_reject_keras_legacy_format`** — Verifica que Keras Legacy JSON é rejeitado graciosamente por `parse_nam_json()` ou `build_model()`.
- **`test_reject_activation_non_tanh`** — Verifica que ativação ReLU (não-Tanh) é rejeitada por `build_model()` com `Err`.

### 6.3. Benchmarks `criterion` (`cargo bench`) — 17 funções benchmark (23 medições individuais)

O arquivo `benches/inference_bench.rs` mede a latência de processamento com o framework `criterion` (harness=false). O deadline de tempo-real a 48 kHz com buffer de 64 amostras é **1.33 ms**.

| Benchmark                                          | Descrição                                      | Referência prática                              |
| -------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- |
| `WaveNet_Standard_CH16_64samp_48kHz`               | Inferência WaveNet Standard (modelo real .nam) | 1 bloco DSP completo — deve caber em 1.33 ms    |
| `WaveNet_Standard_CH16_{32,128,256,512}samp_48kHz` | WaveNet Standard com buffers variáveis         | Perfil de latência para block sizes maiores     |
| `LSTM_2x16_64samp_48kHz`                           | Inferência LSTM 2×16 (sintético, 3345 pesos)   | Topologia recorrente mais pesada suportada      |
| `LSTM_2x16_{32,128,256,512}samp_48kHz`             | LSTM 2×16 com buffers variáveis                | Perfil de latência para block sizes maiores     |
| `FastMath_tanh_AVX2_256elem`                       | Ativação tanh Minimax×rsqrt sobre 256 f32      | Kernel chamado N×layers/bloco no WaveNet        |
| `FastMath_sigmoid_AVX2_256elem`                    | Ativação sigmoid derivada de tanh              | Kernel chamado N×gates/bloco no LSTM            |
| `WaveNet_Dynamic_Standard_64samp_48kHz`            | Inferência WaveNet Dynamic (fallback dinâmico) | Mede overhead do path sem const generics        |
| `LSTM_Dynamic_1x16_64samp_48kHz`                   | Inferência LSTM Dynamic 1×16 (fallback)        | Mede overhead do path sem const generics        |
| `DotProduct_AVX2_256elem`                          | Dot product SIMD sobre 256 f32                 | Kernel central de DenseLayer/Conv1d             |
| `DotProduct_AVX2_64elem`                           | Dot product SIMD sobre 64 f32                  | Tamanho típico de hidden layer LSTM             |
| `Resampler_44100_to_48k_1024samp`                  | NamResampler 44.1 kHz → 48 kHz (1024 amostras) | Conversão FIR Sinc para hardware 44.1 kHz       |
| `Resampler_96000_to_48k_1024samp`                  | NamResampler 96 kHz → 48 kHz (1024 amostras)   | Conversão FIR Sinc para hardware 96 kHz         |
| `Resampler_48000_bypass_1024samp`                  | NamResampler 48 kHz bypass (overhead mínimo)   | Mede overhead da branch bypass                  |
| `FastMath_tanh_AVX512_256elem`                     | Ativação tanh AVX-512 sobre 256 f32            | Condicional: requer `avx512f`+`avx512vl`        |
| `FastMath_sigmoid_AVX512_256elem`                  | Ativação sigmoid AVX-512 sobre 256 f32         | Condicional: requer `avx512f`+`avx512vl`        |
| `Prewarm_WaveNet_Standard_2048samp`                | Benchmark de `prewarm(2048)` para WaveNet      | Tempo gasto na inicialização/alocação de buffer |
| `Prewarm_LSTM_2x16_2048samp`                       | Benchmark de `prewarm(2048)` para LSTM         | Tempo gasto na estabilização dos estados LSTM   |

> **Nota:** Durante `cargo bench`, os 79 testes unitários aparecem como `ignored` — isto é o comportamento normal do criterion, que re-roda o binário com harness desabilitado. Os benchmarks AVX-512 são condicionais: em hardware sem suporte, imprimem `SKIP` e retornam sem falha.

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

**Modelos cobertos:** WaveNet Standard, LSTM 1×16, WaveNet Feather (CH=8), WaveNet Nano (CH=4).

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
├── golden_wavenet_feather.bin      ← Golden vectors WaveNet Feather (CH=8)
├── golden_wavenet_nano.bin         ← Golden vectors WaveNet Nano (CH=4)
├── golden_lstm_1x16.bin
├── golden_gen.cpp                  ← Gerador C++ de golden vectors
├── golden_gen_build.sh             ← Script de build do gerador
└── README.md
```

### 6.6. Counting Allocator para Verificação Zero-Allocation

O test binary (`cargo test --test nam_infer_test`) inclui um **Counting Allocator** condicional (`#[cfg(test)] #[global_allocator]`) que intercepta todas as chamadas `malloc` durante intervalos de medição. O allocator é thread-aware via `SYS_gettid` no Linux, garantindo que apenas alocações da thread sob teste são contabilizadas.

**Componentes:**

- `CountingAllocator` — wrapper sobre `System` allocator que incrementa `AtomicUsize` quando `TRACKING_ENABLED` está ativo.
- `TrackingGuard` — RAII guard que ativa o tracking no `new()` e desativa automaticamente no `drop()`, garantindo cleanup mesmo em panics.
- `TRACKING_THREAD` — `AtomicI32` com o TID da thread atual, isolando a contagem de alocações de outras threads do test harness.

**Impacto:** Zero no binário de produção (`nam-rs`). O `#[global_allocator]` existe **apenas** no test binary. O binário release compila sem o counting allocator.

> **Código-fonte:** `tests/nam_infer_test.rs` — seção "Counting Allocator para Verificação Zero-Allocation" (linhas 40–96).

### 6.7. Fuzz Testing via Proptest

O arquivo `tests/proptest_parsers.rs` exercita os parsers de entrada com **~45.000 inputs adversários** (9 testes × 5.000 cases cada), garantindo que `parse_nam_json()` e `parse_namb()` **nunca** façam panic em dados malformados.

**Estratégias de fuzzing para `parse_nam_json()`:**

- Bytes arbitrários (0–4096 bytes) convertidos lossy para string.
- JSON semi-válido com campos `architecture`/`config`/`weights` randomizados.
- JSON válido de modelo real truncado em posição aleatória.
- Pesos contendo `f32::MAX`, `f32::INFINITY`, `f32::NAN`, `f32::MIN_POSITIVE`.

**Estratégias de fuzzing para `parse_namb()`:**

- Bytes arbitrários (0–8192 bytes).
- NAMB válido com magic number corrompido.
- NAMB válido com CRC32 alterada em 1 bit.
- NAMB válido truncado em posição aleatória.
- NAMB com `weights_offset` apontando além do buffer.

**Auditoria de `unwrap()`:** Os parsers `parse_nam_json()` e `parse_namb()` contêm **zero** chamadas `unwrap()` ou `expect()` em dados de input externo — todo acesso usa `?` operator ou `.ok_or_else(|| anyhow!(...))`, garantindo retorno gracioso de `Err` sem panic.

> **Código-fonte:** `tests/proptest_parsers.rs`.

### 6.8. Convenções e Guardas

- **Guarda SIMD por runtime detection:** O nam-rs compila estritamente para `x86-64-v3`, portanto verificações de AVX2/FMA foram removidas. Testes que exercitam kernels experimentais de **AVX-512** continuam a usar guardas como `if std::is_x86_feature_detected!("avx512f")`, garantindo fallback seguro nestes casos.
- **Modelos de teste opcionais:** Testes que dependem de arquivos `.nam` reais fazem `if !path.exists() { eprintln!("SKIP: ..."); return; }`, permitindo execução parcial sem falsos positivos.
- **Comando de execução:** `cargo test` dispara todas as camadas (124 verificações). `cargo test --lib` executa apenas os 81 unitários inline; `cargo test --test nam_infer_test` os 29 de inferência; `cargo test --test proptest_parsers` os 9 de fuzz testing; `cargo test --test proptest_math` os 4 estocásticos; `cargo test --test pw_integration_test` o headless PipeWire.

## 8. Sistema de Logging e Gestão de Erros

O NAM-rs adota uma arquitetura de erros em três camadas que separa a infraestrutura de transporte da experiência do usuário e do controle de fluxo interno:

- **Transporte Unificado (`env_logger` + `log` crate):** A "espinha dorsal" de comunicação. O `env_logger` é inicializado no startup e atua como o coletor final de todas as mensagens. Ele respeita a variável `RUST_LOG`, permitindo auditoria técnica sem alteração de código.
- **Camada de Apresentação e UX (`NamDiagnostic`):** Um sistema de diagnósticos estruturados em texto legível para o usuário final. Ele encapsula erros técnicos, adicionando mensagens amigáveis, dicas de solução (*hints*) e um bloco de suporte técnico formatado (com versão, kernel, features e timestamp). Quando um `NamDiagnostic` é emitido via `.emit()`, ele despacha a mensagem formatada através da macro `log::error!`, convergindo para o sistema de logging unificado.
- **Controle de Fluxo e Propagação (`anyhow`):** Utilizado internamente para gerenciar o "borbulhamento" de erros e o curto-circuito de pipelines (ex: falha na leitura de arquivo impede o parse). Ao contrário de um `panic!`, o uso de `anyhow` permite que a aplicação capture a falha no limite de uma função (como `load_and_send_model`), reporte ao usuário via `NamDiagnostic` e continue operando normalmente (contenção de falha).

### Fluxo Típico de Erro (Ex: Carga de Modelo)

1. **Falha de I/O:** `std::fs::read` retorna `std::io::Error`.
2. **Diagnóstico:** Um `NamDiagnostic` é construído com o erro capturado e emitido. O usuário vê a falha estilizada no terminal.
3. **Curto-circuito:** O erro é convertido para `anyhow::Error`, interrompendo a cadeia de parsing.
4. **Recuperação:** A função chamadora recebe o `Err`, ignora-o (pois o diagnóstico já foi emitido) e mantém o loop principal do programa ativo.

## 9. Referências

Os seguintes repositórios GitHub são a principal base de referência para a implementação do NAM-rs:

- [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
- [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio)
