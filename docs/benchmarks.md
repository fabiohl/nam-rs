<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Performance Benchmarks (Criterion)

The NAM-rs project uses **Criterion.rs** as its official performance benchmarking suite. Given the latency-sensitive nature of a real-time audio engine (DSP), conducting measurements with statistical rigor is essential to avoid being misled by operating system variations (noise, context switches, clock fluctuations).

## How to Run the Benchmarks

To execute the performance suite:

```bash
cargo bench --bench inference_bench
```

### Long-Duration Benchmarks (Soak Bench)

To evaluate performance under constant pressure and identify jitter caused by cache misses or TLB misses in large blocks, the project offers a long-duration benchmarking suite (30s+ per function):

```bash
cargo bench --features long_bench
```

Or via the recommended manual trigger script:

```bash
bash utils/tests-long.sh
```

These benchmarks use blocks of **4096 samples** (~85ms), reducing the relative weight of invocation overhead and focusing purely on the DSP engine's throughput.

## How to Interpret Criterion Output

When you run a benchmark, Criterion reports output similar to this:

```text
WaveNet_Standard_CH16_64samp_48kHz
                        time:   [107.03 µs 107.32 µs 107.61 µs]
                        change: [−9.3273% −6.2506% −3.5233%] (p = 0.00 < 0.05)
                        Performance has improved.
                        Found 5 outliers among 50 measurements (10.00%)
  ...
```

### Understanding the Metrics

1. **`time: [A B C]` (Confidence Interval)**
   Shows the execution time per iteration, expressed through a **95% confidence interval**.
   * The central number (`B`, e.g., `107.32 µs`) is the best point estimate of the mean time.
   * The outer numbers (`A` and `C`) define the lower and upper bounds, statistically guaranteeing (with 95% certainty) that the true performance lies within this margin.
2. **`change: [...]` and `(p = X < 0.05)` (Statistical Significance)**
   * `change` displays the percentage difference compared to the last run on the same machine (negative values indicate faster code).
   * The **p-value** (`p`) indicates the probability that this variation occurred by chance. If `p < 0.05` (5% significance level), Criterion certifies that the observed variation is real and not just operating system noise.
3. **Textual Conclusions**
   Based on mathematical calculations, the software summarizes the conclusions:
   * **Performance has improved / regressed**: The p-value confirmed that the source code change caused a measurable statistical difference (positive or negative).
   * **Change within noise threshold**: The p-value is high, the error margins overlap, or the variation is negligible. The detected change is noise.
4. **Outliers (Jitter)**
   Samples are run hundreds of times, and anomalies are reported. In a critical real-time system like NAM-rs, occurrences of `high severe` are usually linked to *jitter* (processing glitches, audio thread preemption by the OS kernel, cache misses, etc.). Running benchmarks in shielded environments (SCHED_FIFO and CPU affinity enabled) mitigates outliers.

## Temporal History (Baselines)

You do not need to compare times mentally. **Criterion automatically saves the baseline of your last run**.

All historical tracking metrics are recorded in local files within your project under: `target/criterion/`

*(Note: NAM-rs intentionally disables HTML report generation with temporal charts in `Cargo.toml` (`default-features = false`) to omit downloading extensive visual dependencies, limiting evaluation to the console).*

## Comparative Results: Scalar LSTM vs. SIMD (Fused Gates T3)

Optimizations introduced gate fusion and SIMD activations (AVX2/AVX-512) into the recurrent networks' hot-path. Below are the measured gains on an x86-64-v3 (AVX2/FMA) architecture for 64-sample blocks:

| Topology      | Implementation      | Latency (Average) | Speedup   |
|:------------- |:------------------- |:----------------- |:--------- |
| **LSTM 1x8**  | Scalar (Baseline)   | ~22.45 µs         | -         |
| **LSTM 1x8**  | **SIMD Fused (T3)** | **~6.36 µs**      | **3.53x** |
| **LSTM 2x16** | Scalar (Baseline)   | ~83.66 µs         | -         |
| **LSTM 2x16** | **SIMD Fused (T3)** | **~20.29 µs**     | **4.12x** |

### Technical Conclusion

The performance gain exceeding **4x** on complex models (2x16) validates the kernel fusion strategy. By processing the 4 LSTM gates simultaneously via SIMD vectors and keeping data in registers between the Sigmoid and Tanh activations, we drastically reduce CPU cycles wasted on redundant loads/stores and memory latency.

## Cycle Budget (WaveNet Hot-Path)

To guide future optimizations, we performed granular instrumentation of the WaveNet hot-path (`WaveNetLayer::process_block_internal`) using hardware cycle counters (**RDTSC**). This measurement identifies where the CPU spends most of its time during audio block processing.

### Cycle Distribution per Stage (Per Layer)

Below is the average percentage distribution of cycles on an x86-64-v3 (AVX2) architecture for a Standard model (CH=16):

| Operational Stage          | Operations Involved                 | Budget (%) | Technical Justification                                              |
|:-------------------------- |:----------------------------------- |:---------- |:-------------------------------------------------------------------- |
| **Conv1D (SIMD GEMV)**     | Causal convolution, MACs, dilation  | **~45%**   | Most computationally intensive phase (matrix-vector multiplication). |
| **1x1 & Residual (Fused)** | Dense projection, residual addition | **~25%**   | High memory pressure (read-modify-write) and channel projection.     |
| **Mixin (Conditioning)**   | Timbre metadata injection           | **~15%**   | Dense operation applied to the input of each layer.                  |
| **Act & Head (Fused)**     | Tanh/Sigmoid, Skip-Connections      | **~15%**   | Cost of transcendental functions (approximated via SIMD).            |

### Data Flow Analysis (Array Level)

At the `WaveNetLayerArray` level, the layer cascade dominates processing (**>90% of total time**). Interface stages (input **Rechannel** and output **Head Rechannel**) represent a negligible fixed overhead as the number of layers increases, validating the scalability of the NAM-rs architecture for complex models.

> [!TIP]
> Fusing **Tanh** with **Head Accumulation** was the most impactful optimization of Epic E, reducing the activation stage budget from ~30% to ~15% by eliminating redundant passes through L1 Cache memory.

## Experiment Report: Temporal Tiling (Dual-Frame) on Conv1D

In the hot-path optimization Epic, a **Temporal Tiling** variant ("Dual-Frame" processing) was designed and tested for `Conv1D` kernels, aiming to maximize L1 Cache weight reuse by processing two frames simultaneously in WaveNet inference.

### Measurement Results (64 samples, 48kHz, CH=16, AVX2)

* **Single-Frame (Baseline):** ~84 µs
* **Dual-Frame Tiling:** ~100 µs (Regression of ~19%)

### Analysis and Architectural Decision

Although theory suggested that loading weights from memory half as often would save bandwidth (L1 cache), in practice the x86-64 architecture (AVX2/FMA) proved to be limited by **Register Pressure**.
To process two frames in parallel:

1. The number of required SIMD accumulators doubled (from 4 YMM to 8 YMM per channel).
2. Instruction overhead in the frontend (e.g., broadcasts and blends) outweighed the savings on loads.
3. The compiler was forced to use register spilling or hit execution port bottlenecks for blend/shuffle instructions (Port 5).

**Conclusion:** The primary bottleneck of `Conv1D` in NAM-rs is not tied to L1 Cache bandwidth, but rather to computational throughput and register contention in the backend (FMA). Because of this, while the kernel implementation has been kept in the `SimdMath` trait for portability and testing on architectures with more registers (e.g., AVX-512 or ARM NEON), the main loop in `WaveNetLayer` continues to use **Single-Frame processing** to ensure the lowest latency and highest real-time stability.

## Experiment Report: Stereo Fusion in the Output Stage (T3.2)

Task T3.2 aimed to eliminate redundant memory passes in the final output stage by fusing the gain (Hysteresis/Gate) operations of the L and R channels into a single stereo SIMD call.

### Measurement Results (64 samples, 48kHz, AVX2)

| Topology        | Before Fusion | After Fusion (T3.2) | Gain (%)  |
|:--------------- |:------------- |:------------------- |:--------- |
| **WaveNet Std** | ~107.3 µs     | ~101.4 µs           | **~5.5%** |
| **LSTM 2x16**   | ~15.4 µs      | ~14.7 µs            | **~4.5%** |

### Conclusion

Stereo fusion reduces memory traffic in the L1 Cache by reading the L and R channels simultaneously and applying the gain/ramp weights in a single loop. The gain is more pronounced in smaller blocks (e.g., 32 samples, where a **~8.5%** improvement was measured), where dispatch overhead and partial cache misses have a higher relative weight.

## Criterion A2 Architecture (Epic 2)

The A2 architecture introduces per-layer conditioning (FiLM + Gating) and a configurable channel count (CH=3 Lite, CH=8 Full). Epic 2 focused on a SIMD-heavy hot-path for CH=8 (`A2Conv1dCh8`) with col-major-per-tap weight layout, enabling AVX2 T=4 broadcast-FMA convolution.

### A2-Full (CH=8) — Optimized SIMD Path

A2-Full uses the `A2Conv1dCh8` fast path with f32 weights in col-major layout (`w[k * 64 + in * 8 + out]`), where 8 output-channel weights are contiguous per `(tap, input)` pair. This layout feeds directly into AVX2 broadcast-FMA without transposition.

| Block Size   | Latency (µs) | Per-Sample (ns) | CPU % at 48kHz |
|:------------ |:------------ |:--------------- |:-------------- |
| **64 samp**  | **~30.9 µs** | ~483            | ~2.3%          |
| **128 samp** | ~30.8 µs     | ~241            | ~1.2%          |
| **256 samp** | ~31.5 µs     | ~123            | ~0.6%          |

### A2-Lite (CH=3) — f32 Native GEMV Path

A2-Lite uses the dedicated `A2Conv1dCh3` fast path (`src/models/a2/conv1d_ch3.rs`), mirroring the CH=8 kernel design: f32 native weights in col-major-per-tap layout (one `_mm_loadu_ps` load, one `_mm_fmadd_ps` FMA per input channel — no f16 decode). The kernel is a fully unrolled GEMV (18 FMAs for K=6, 45 FMAs for K=15), with post-conv operations (Mixin, LeakyReLU, head, l1x1) batched via AVX2. Despite having ~6.5x fewer weights (1,871 vs 12,146), the smaller CH=3 vector width (needing only 128-bit XMM registers vs 256-bit YMM for CH=8) results in a latency profile closer to the CH=8 SIMD path.

| Block Size   | Latency (µs) | Per-Sample (ns) | CPU % at 48kHz |
|:------------ |:------------ |:--------------- |:-------------- |
| **64 samp**  | **~48.7 µs** | ~761            | ~3.7%          |
| **128 samp** | ~48.7 µs     | ~381            | ~1.8%          |
| **256 samp** | ~48.8 µs     | ~191            | ~0.9%          |

### Comparative Analysis

| Variant | Weights | Channels | Conv Path                   | 64-samp Latency |
|:------- |:------- |:-------- |:--------------------------- |:--------------- |
| A2-Full | 12,146  | 8        | f32 col-major SIMD          | **~30.9 µs**    |
| A2-Lite | 1,871   | 3        | f32 col-major unrolled GEMV | ~48.7 µs        |

The CH=8 SIMD path is ~58% faster than CH=3 despite processing ~6.5x more weights, validating the architectural decision to invest in a dedicated col-major `A2Conv1dCh8` kernel. The CH=3 kernel operates on 128-bit XMM registers (3 channels + 1 zero pad) versus 256-bit YMM for CH=8, reducing SIMD throughput per instruction.

### Key Findings

1. **Near-constant per-block latency** across block sizes (64-256) indicates that fixed overhead (function dispatch, buffer management) is minimal; the engine scales almost perfectly with block size.
2. **A2-Full at 30.9 µs for 64 samples** is ~3.5x faster than WaveNet Standard CH=16 (~107 µs), despite A2-Full having twice the layers (23 vs 10+10).
3. **Both variants stay well under the 1.33 ms real-time deadline** at 48 kHz with a 64-sample buffer, leaving ample headroom for other DSP processing.
4. **Golden tests confirm zero regression** in A1 models (WaveNet Standard, Feather, Nano, LSTM) — all 34 integration tests pass.

## Gate FSM (Dynamic Hysteresis)

The gate FSM (`DynamicHysteresis`) runs in the DSP hot-path on every audio callback to decide whether to open or close the noise gate based on detected volume. The benchmark measures `update()` (state machine tick) + `multiplier()` (current gain read) across three steady-state scenarios at realistic DSP block sizes.

### Results (64, 128, 256 samples — x86-64-v3 AVX2/FMA)

| Scenario             | 64 samp  | 128 samp | 256 samp | Steady Path                                  |
|:-------------------- |:-------- |:-------- |:-------- |:-------------------------------------------- |
| **Open**             | ~2.11 ns | ~2.03 ns | ~1.89 ns | Volume above open threshold, gate stays open |
| **Closed**           | ~1.66 ns | ~1.71 ns | ~1.70 ns | Gate already closed, volume stays below      |
| **FadingOut (ramp)** | ~22.7 ns | ~22.8 ns | ~22.7 ns | Gate actively ramping multiplier toward zero |

### Analysis

* The gate FSM overhead is negligible — even the most expensive path (FadingOut at ~22.7 ns) represents **~0.0017%** of the 1.33 ms audio deadline at 48 kHz with 64-sample blocks.
* Open and Closed steady states are essentially single-branch operations (~1.7–2.1 ns), confirming that the gate imposes no measurable latency in the hot-path.
* FadingOut includes the ramp step arithmetic (`fade_counter -= n_samples`, `current_multiplier = fade_counter * inv_fade_frames`, `ramp_samples = n_samples`) and remains constant across block sizes because only the numeric subtraction and multiplication are block-size-independent.
* The gate's actual computational cost is in `apply_gain_rt` / `apply_gain_rt_stereo` (SIMD gain application), not in the FSM decision logic measured here.

### Running Gate_FSM bench

```sh
cargo bench --bench inference_bench -- "Gate_FSM"
```

## IR Cabsim Convolution (Epic 4)

The cabsim engine uses UPOLS (Uniform-Partitioned Overlap-Save) frequency-domain convolution. All FFTs of the kernel partitions are pre-computed at construction time; the `ConvEngine::process()` hot-path performs zero allocations and operates on pre-allocated buffers exclusively.

### Benchmarks (64-sample blocks at 48 kHz)

IR lengths correspond to realistic cabinet impulse response durations:

* **Short** (64 samples, 1.3 ms): 1 partition, minimal overhead
* **Medium** (2048 samples, 42.7 ms): 32 partitions — typical medium-length guitar cabinet IR
* **Long** (16384 samples, 341.3 ms): 256 partitions — full-length ambient/reverb IR

| Benchmark                 | IR Samples | Partitions | Latency (µs) | CPU % at 48kHz |
|:------------------------- |:---------- |:---------- |:------------ |:-------------- |
| ShortIR_64samp            | 64         | 1          | ~1.5         | ~0.1%          |
| MediumIR_2048_64          | 2,048      | 32         | ~8.7         | ~0.8%          |
| LongIR_16384_64           | 16,384     | 256        | ~62.1        | ~5.8%          |
| MediumIR_2048_256samp     | 2,048      | 8          | ~13.0        | ~0.2%          |
| Engine_Construction_2048  | 2,048      | 32         | ~20.5        | — (load-time)  |
| Engine_Construction_16384 | 16,384     | 256        | ~142.3       | — (load-time)  |

> [!NOTE]
> Values measured on x86-64-v3 (AVX2/FMA). For comparison, neural inference (WaveNet Standard CH=16) consumes ~107 µs per 64-sample block.
> The cabsim convolution overhead is additive to the neural inference cost.
> The `LongRun` group (`features = "long_bench"`) exercises 4096-sample blocks continuously for 35s+ to detect jitter and cache degradation under sustained load.

### RT-Safety Validation

* **Heap-audit tests** (`tests/cabsim_heap_audit.rs`) confirm zero allocations on the `ConvEngine::process()` hot-path for short (64), medium (512), long (4096) IRs and passthrough mode.
* **Golden convolution tests** (`tests/cabsim_golden.rs`) verify UPOLS output against direct convolution reference using deterministic synthetic IRs (ESR < 1e-5), with `#[ignore]`-gated long-run (8k–32k sample IR) stress tests.
* **Conv engine construction** (including all FFT pre-computation) is performed outside the audio thread; its cost is measurable but irrelevant to RT deadlines.

### Running

```sh
# Standard benchmarks (Short, Medium, Long IR at 64-sample blocks)
cargo bench --bench inference_bench -- "Cabsim"

# 256-sample block variant
cargo bench --bench inference_bench -- "Cabsim_MediumIR_2048_256"

# Construction cost benchmarks
cargo bench --bench inference_bench -- "Cabsim_Engine_Construction"

## Investigação T13.2: Custo do Kahan por-tap no Conv1d

### Contexto

A implementação estática do conv1d (`src/models/wavenet/conv1d.rs:192-205` e `conv1d_dual.rs:197-220`)
executa Kahan compensated summation **dentro** do laço per-tap, serializando a redução SIMD→escalar
a cada tap. Para K ≤ 3 (todos os modelos A1 WaveNet), o erro de soma simples é O(3·ε) — desprezível
para áudio — tornando o Kahan por-tap potencialmente superdimensionado.

O próprio módulo `kahan.rs` documenta K ≤ 3 como um caso de "Quando NÃO usar" (linha 16-17).

### Metodologia

**Benchmark 1: Loop interno isolado** (`kahan_inner_loop_isolated`)

Isola o padrão exato do laço per-tap: `dot_product_4x_interleaved` + acumulação Kahan vs plain `+=`.
Testa K ∈ {1, 2, 3, 6, 15, 32} com IN=16 para medir o custo marginal por tap.

**Benchmark 2: Conv1d completo** (`conv1d_kahan_full`)

Compara `Conv1d` estático (Kahan, const-generics) vs `Conv1dDyn` (sem Kahan, dimensões runtime)
para configurações típicas de WaveNet A1 (K=3, IN/OUT ∈ {8, 12, 16}), processando 64 frames.

Arquivo: `benches/kahan_conv1d_bench.rs`.

### Resultados

#### Loop interno isolado (custo por tap)

| K  | Kahan (ns) | Plain (ns) | Overhead | Overhead % |
|:---|-----------:|-----------:|---------:|-----------:|
| 1  | 8.51       | 8.09       | 0.43 ns  | +5.3%      |
| 2  | 16.42      | 15.12      | 1.30 ns  | +8.6%      |
| 3  | 24.11      | 22.80      | 1.31 ns  | +5.7%      |
| 6  | 47.00      | 44.77      | 2.23 ns  | +5.0%      |
| 15 | 115.79     | 107.64     | 8.15 ns  | +7.6%      |
| 32 | 249.72     | 231.84     | 17.88 ns | +7.7%      |

**Custo marginal por kahan_add:** ~0.43 ns por chamada (4 chamadas/tap = 4 canais).
O overhead relativo estabiliza em ~5–8% independente de K — a maior parte do tempo
é dominada pelo `dot_product_4x_interleaved` SIMD.

#### Conv1d completo (64 frames)

| Config              | Static Kahan (µs) | Dyn No-Kahan (µs) | Ratio |
|:--------------------|------------------:|------------------:|------:|
| IN=8,  OUT=8,  K=3 | 1.128             | 2.688             | 2.38× |
| IN=8,  OUT=16, K=3 | 2.024             | 4.755             | 2.35× |
| IN=16, OUT=16, K=3 | 3.474             | 6.941             | 2.00× |
| IN=12, OUT=12, K=3 | 2.024             | 4.466             | 2.21× |

**Observação:** O `Conv1d` estático (com Kahan) é **2× mais rápido** que o `Conv1dDyn`
(sem Kahan). A vantagem estrutural do caminho const-generic (unrolling, stack arrays,
eliminação de indireção de ponteiros) domina completamente o custo do Kahan — o Kahan é
"grátis" no contexto do caminho estático versus a alternativa dinâmica.

### Análise numérica

**Para K = 3 taps com f32 (ε ≈ 1.19×10⁻⁷):**

- Erro worst-case por canal: 3 × ε ≈ 3.6×10⁻⁷ absoluto
- Em dBFS (sinal de magnitude 1.0): 20×log₁₀(3.6×10⁻⁷) ≈ **−129 dB**
- Limiar de percepção humana: ~0.1 dB a −80 dBFS
- Ruído de quantização 16-bit: −96 dBFS
- **Conclusão:** O erro de soma simples para K=3 está 33 dB abaixo do ruído de 16-bit
  e 49 dB abaixo do limiar perceptivo.

**Cadeia completa de 10 camadas WaveNet A1 (IN=16, K=3, 300 adições sequenciais):**

- Erro worst-case sem Kahan: 300 × ε ≈ 3.6×10⁻⁵
- Em dBFS: 20×log₁₀(3.6×10⁻⁵) ≈ **−89 dB**
- **Ainda abaixo do ruído de 16-bit (−96 dB)**, mas com margem reduzida (7 dB).
- Com Kahan: erro limitado a ε ≈ 1.2×10⁻⁷ (−138 dBFS) — 42 dB de margem extra,
  completamente irrelevante para áudio.

> [!NOTE]
> O worst-case acima assume acumulação monotônica (todos os termos com mesmo sinal),
> que nunca ocorre em sinais de áudio reais (alternam polaridade). Na prática, o erro
> real é ordens de grandeza menor por cancelamento parcial.

### Decisão

**Remover Kahan do caminho estático (Conv1d, Conv1dDual) para K ≤ 3.** Justificativa:

1. **Numérica:** O erro de soma simples para K=3 é −129 dBFS por camada — irrisório mesmo
   após 10 camadas (−89 dB worst-case teórico, << −100 dB na prática com áudio real).
2. **Performance:** O ganho de remover Kahan (~5–8% no loop interno) é modesto mas
   real. No contexto do caminho estático, o custo é absorvido pela vantagem estrutural
   const-generic, mas a simplificação do hot-path reduz pressão de registrador e
   melhora a previsibilidade do compilador.
3. **Consistência:** O caminho dinâmico (`conv1d_dyn_kernels.rs`) já usa plain `+=` sem
   Kahan. Alinhar os caminhos elimina um delta de precisão entre modos de compilação.
4. **Documentação:** O próprio módulo `kahan.rs:16-17` já lista "Single-digit additions
   (K ≤ 3 taps)" como caso de não-uso.

**Antes de efetivar a remoção**, executar a suíte completa de goldens (`cargo test --test golden_vectors`)
para confirmar que a saída do modelo se mantém dentro da banda de tolerância.

### Como executar o benchmark

```sh
cargo bench --bench kahan_conv1d_bench
```

# Long-duration soak (35s+ measurement, 4096-sample blocks)
cargo bench --features long_bench --bench inference_bench -- "Cabsim_LongRun"
```
