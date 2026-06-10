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

| Block Size  | Latency (µs)  | Per-Sample (ns) | CPU % at 48kHz |
|:----------- |:------------- |:--------------- |:-------------- |
| **64 samp** | **~30.9 µs**  | ~483            | ~2.3%          |
| **128 samp** | ~30.8 µs      | ~241            | ~1.2%          |
| **256 samp** | ~31.5 µs      | ~123            | ~0.6%          |

### A2-Lite (CH=3) — u16 Interleaved Path

A2-Lite uses the generic `A2Conv1d<3>` path with u16 interleaved weights that require dequantization and transposition in the hot-path. Despite having ~6.5x fewer weights (1,871 vs 12,146), the dequantization overhead makes it slower than the CH=8 SIMD path.

| Block Size  | Latency (µs)  | Per-Sample (ns) | CPU % at 48kHz |
|:----------- |:------------- |:--------------- |:-------------- |
| **64 samp** | **~48.7 µs**  | ~761            | ~3.7%          |
| **128 samp** | ~48.7 µs      | ~381            | ~1.8%          |
| **256 samp** | ~48.8 µs      | ~191            | ~0.9%          |

### Comparative Analysis

| Variant      | Weights | Channels | Conv Path            | 64-samp Latency |
|:------------ |:------- |:-------- |:-------------------- |:--------------- |
| A2-Full      | 12,146  | 8        | f32 col-major SIMD   | **~30.9 µs**    |
| A2-Lite      | 1,871   | 3        | u16 interleaved GEMV | ~48.7 µs         |

The CH=8 SIMD path is ~58% faster than CH=3 despite processing ~6.5x more weights, validating the architectural decision to invest in a dedicated col-major `A2Conv1dCh8` kernel. The u16 dequantization and transposition overhead in CH=3 dominates the arithmetic savings.

### Key Findings

1. **Near-constant per-block latency** across block sizes (64-256) indicates that fixed overhead (function dispatch, buffer management) is minimal; the engine scales almost perfectly with block size.
2. **A2-Full at 30.9 µs for 64 samples** is ~3.5x faster than WaveNet Standard CH=16 (~107 µs), despite A2-Full having twice the layers (23 vs 10+10).
3. **Both variants stay well under the 1.33 ms real-time deadline** at 48 kHz with a 64-sample buffer, leaving ample headroom for other DSP processing.
4. **Golden tests confirm zero regression** in A1 models (WaveNet Standard, Feather, Nano, LSTM) — all 34 integration tests pass.
