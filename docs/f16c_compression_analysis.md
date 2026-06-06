<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Performance Analysis: F16C Weight Compression

This document details the architectural motivation, performance benefits, and sonic impact of storing neural network weights in half-precision (`f16`/`u16`) in NAM-rs.

## 1. The Problem: Cache-Bound Inference

The largest supported topology (WaveNet Standard) stores approximately **80 KB** of weights as `f32` floats. Modern CPU L1 data caches are typically **32 KB per core** (AMD Zen / Intel Core).

During the inference hot-loop, those 80 KB exceed L1 capacity. The processor is forced to fetch weights from L2 (and in some cases L3), incurring approximately **14 idle cycles per L1 miss**. At hundreds of thousands of invocations per second, this stall is a meaningful bottleneck.

## 2. The Solution: `f16` Compression via F16C Hardware

NAM-rs converts all static weight matrices (Conv1d, DenseLayer, LSTM projections) to **half-precision (16-bit)** during model loading. This halves the in-memory footprint:

- WaveNet Standard: 80 KB → **40 KB** (fits comfortably in L1)

### Decompression Cost

Decompression from `f16` to `f32` is performed by dedicated hardware via F16C + AVX2/AVX-512 instructions:

1. `_mm_loadu_si128` loads 8 compressed `u16` values (128 bits).
2. `_mm256_cvtph_ps` converts them to 8 `f32` values in a 256-bit YMM register.
3. The instruction takes ~4–5 cycles, but modern Out-of-Order CPUs hide this latency almost entirely behind concurrent FMA operations.

## 3. Practical Impact

Eliminating L1 cache misses in the weight fetch path enables:

- **Lower buffer sizes without XRUNs:** the audio thread finishes each block faster and more predictably. Running at 32–64 samples (~0.7–1.3 ms) becomes viable even on mid-range hardware.
- **Lower CPU utilization:** leaves headroom for other DAW tracks and processing.

## 4. Precision Trade-off

Converting `f32` → `f16` truncates the mantissa to 10 bits (vs. 23 bits in f32), introducing quantization error per weight of approximately **~3.9e-3** (the dominant drift source — see `docs/fastmath-approximations.md` §5).

### Psychoacoustic Assessment

The quantization error sits at approximately **−80 to −100 dBFS** — well below:

- The noise floor of any typical single-coil pickup (~−60 dB).
- The output distortion of any high-gain amplifier emulation.
- The 16-bit digital noise floor (~−96 dBFS).

Cross-validation against NeuralAmpModelerCore (7 reference models) confirms that WaveNet SNR with BF16/F16 weights remains in the **−43 to −50 dB** range vs. the C++ reference — perceptually indistinguishable from the `f32` path.

## 5. BF16 vs F16 Selection

At runtime, the dispatcher (`src/math/common/dispatch.rs`) selects between:

- **F16 (`_mm256_cvtph_ps`):** 10-bit mantissa, ~3.9e-3 error. Used on AVX2 hardware.
- **BF16 (`_mm512_dpbf16_ps`):** 7-bit mantissa, ~8× larger error, but enables native VNNI dot-product on Sapphire Rapids+ without a separate conversion step.

See `docs/architecture.md §2` and `src/math/common/dispatch.rs` for dispatch logic.
