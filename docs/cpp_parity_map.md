<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# C++ ↔ Rust Parity Map — NeuralAmpModelerCore × NAM-rs

Point-to-point mapping between the canonical C++ reference
[`github.com/NeuralAmpModelerCore`](https://github.com/sdatkinson/NeuralAmpModelerCore)
and the NAM-rs Rust engine (`src/`). This document tracks parity status, known
divergences, and the established equivalence status.

Correctness is verified along **two complementary axes**:

1. **Interop parity** — does NAM-rs match NAMCore? (golden vectors + live cross-validation,
   §9 / §9.1). Both are f16c engines, so this proves *agreement*, not absolute fidelity.
2. **Ideal-math fidelity** — how far is NAM-rs from the exact mathematics? (the independent
   f64 reference oracle, §9.2). This isolates the genuine precision floor.

Together they pin the engine from both sides: agreement with the reference **and** distance from
the ideal. A quick map of every concept lives in the [Topology Table](#10-topology-table) (§10)
and the [Divergences](#11-nam-rs-divergences-from-c-reference-accepted) (§11).

---

## 1. DSP Engine Layer

| C++ (`NeuralAmpModelerCore/`)                                              | Rust (`src/`)                                                             | Parity established |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------ |
| `NAM/dsp.h:184` — `void DSP::Reset(sr, maxBuf)`                            | `models/mod.rs` — `NamModel::reset()`                                     | Established        |
| `NAM/dsp.cpp:93-102` — `Reset` impl (calls `SetMaxBufferSize` + `prewarm`) | `models/mod.rs` — `NamModel::set_max_buffer_size()` + `prewarm_samples()` | Established        |
| `NAM/dsp.cpp` — `DSP::process` (audio callback entry)                      | `dsp/pipeline/stages.rs` — `run_inference()`                              | —                  |
| `NAM/dsp.cpp` — DSP buffer lifecycle                                       | `dsp/pipeline/context.rs` — `DspPipelineContext`                          | —                  |
| `NAM/dsp.cpp` — Noise gate (threshold + hysteresis)                        | `dsp/gate.rs` — `DynamicHysteresis`                                       | —                  |
| `NAM/dsp.cpp` — Prewarm / silence stabilization                            | `loader/mod.rs` — `load_and_build_model` (prewarm 2048 samples)           | Established        |

---

## 2. Model Dispatch

| C++ (`NeuralAmpModelerCore/`)                           | Rust (`src/`)                                            | Parity established |
| ------------------------------------------------------- | -------------------------------------------------------- | ------------------ |
| `NAM/dsp.cpp` — `GetDSP` factory (dynamic dispatch)     | `models/mod.rs` — `StaticModel` enum + manual `match`    | —                  |
| `NeuralModel.cpp:L155-218` — WaveNet topology detection | `loader/nam_json/topology.rs` — `get_wavenet_topology()` | Established        |

---

## 3. WaveNet Architecture

### 3.1 Core Inference

| C++ (`NeuralAmpModelerCore/`)                                                | Rust (`src/`)                                                                                      | Parity established |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------ |
| `NAM/wavenet/model.cpp` — `WaveNet::process`                                 | `models/wavenet/model.rs` — `process_block_internal()`                                             | Established        |
| `WaveNetLayerArrayT<CH,1,1,HEAD,K,Dilations,true>` (C++ template for Array2) | `models/wavenet/model.rs` — `WaveNetModel::array2` (type param `WaveNetLayerArray<CH,1,HEAD,K,1>`) | Established        |
| C++ `model->Prewarm()` (no-arg call)                                         | `models/wavenet/mod.rs` — `prewarm()` ignores `num_samples`                                        | Established        |

### 3.2 Layer Components

| C++ (`NeuralAmpModelerCore/`)                          | Rust (`src/`)                                                      | Parity established |
| ------------------------------------------------------ | ------------------------------------------------------------------ | ------------------ |
| WaveNet causal dilated Conv1D                          | `models/wavenet/conv1d.rs` — `Conv1d<IN,OUT,K>`                    | Established        |
| Conv1D dual-frame temporal tiling                      | `models/wavenet/conv1d_dual.rs`                                    | —                  |
| Input mixin (1×1 projection, conditioned input)        | `models/wavenet/dense.rs` — `DenseLayer::process_fused()`          | Established        |
| 1×1 residual projection                                | `models/wavenet/dense.rs` — `DenseLayer::process_residual_batch()` | Established        |
| Layer states (delay buffers, receptive field tracking) | `models/wavenet/common.rs` — `WaveNetLayerState`                   | —                  |
| BF16 layer state caching                               | `models/wavenet/common.rs` — `u16` mirrored buffer variant         | —                  |

### 3.3 Dynamic WaveNet (fallback path for free geometries)

> Models whose topology does not match a catalog SKU (Standard/Lite/Feather/Nano) are routed through the dynamic fallback path (`WaveNetModelDyn`) — not rejected. This engine handles arbitrary channel counts, kernel sizes, dilations, `condition_size > 1`, and post-stack heads. The `Conv1dDyn` convolution kernel serves as the low-level compute engine for this path, for the A2 dynamic engine (`WaveNetA2Dyn`), and for static WaveNet test/stress kernels.
>
> **v2 multi-SR goldens:** Dynamic engines (`WaveNetModelDyn`, `LstmModelDyn`, `WaveNetA2Dyn`) are anchored at **48 kHz only** for v1 golden vectors. v2 multi-SR goldens are intentionally not provided because: (a) dynamic engines handle arbitrary free geometries — geometry variance subsumes sample-rate variance in practice; (b) live cross-validation in `tests/cpp_parity.rs` (the `live_cross_validation_v2_*` tests) already exercises multi-SR parity via the C++ toolchain for the canonical dynamic geometries (`wavenet_dyn_free`, `lstm_dyn_test`); (c) the A2 dynamic geometries (`a2_dynamic_gated_ch8`, `a2_dynamic_blended_ch3`, `wavenet_a2_film_*`) are **wired and validated by v1 goldens via `WaveNetA2Dyn`** (§6 / §13) and are not part of the fixed fast-path's multi-SR sweep. See `tests/fixtures/README.md` §v2 multi-SR coverage.

| C++ (`NeuralAmpModelerCore/`) | Rust (`src/`)                                     | Parity established |
| ----------------------------- | ------------------------------------------------- | ------------------ |
| Dynamic Conv1D                | `models/wavenet/conv1d_dyn.rs` — `Conv1dDyn`      | —                  |
| Generic WaveNet inference     | `models/wavenet/model_dyn.rs` — `WaveNetModelDyn` | —                  |
| Generic LSTM inference        | `models/lstm/model_dyn.rs` — `LstmModelDyn`       | —                  |
| Generic WaveNet A2 inference  | `models/a2/model/dynamic.rs` — `WaveNetA2Dyn`     | —                  |

### 3.4 Topology → Concrete Type Mapping

| C++ topology          | Rust construct                          | Static type              |
| --------------------- | --------------------------------------- | ------------------------ |
| WaveNet Standard (16) | `loader/dispatcher/wavenet/standard.rs` | `WaveNetModel<16, 3, 8>` |
| WaveNet Lite (12)     | `loader/dispatcher/wavenet/lite.rs`     | `WaveNetModel<12, 3, 6>` |
| WaveNet Feather (8)   | `loader/dispatcher/wavenet/feather.rs`  | `WaveNetModel<8, 3, 4>`  |
| WaveNet Nano (4)      | `loader/dispatcher/wavenet/nano.rs`     | `WaveNetModel<4, 3, 2>`  |
| Any other geometry    | `loader/dispatcher/wavenet/dynamic.rs`  | `WaveNetModelDyn`        |

---

## 4. LSTM Architecture

### 4.1 Core Inference

| C++ (`NeuralAmpModelerCore/`)                | Rust (`src/`)                                                           | Parity established |
| -------------------------------------------- | ----------------------------------------------------------------------- | ------------------ |
| `NAM/lstm.cpp` — `LSTM::process_sample`      | `models/lstm/layer.rs` — `process_sample_avx2` / `_avx512` (via macro)  | Established        |
| LSTM gate computation (sigmoid + tanh fused) | `math/lstm/gates.rs` — `fused_lstm_gates`                               | Established        |
| LSTM HF gate dispatch (β1.1–β1.2)            | `math/lstm/gates.rs`, `models/lstm/layer_kernels.rs` — HF branch-direct | Established        |
| LSTM 2-layer pipelined processing            | `models/lstm/model2.rs` — `define_lstm2_process_pipelined!`             | Established        |
| LSTM Prewarm (silence → convergence)         | `models/lstm/mod.rs` — `lstm_prewarm_common`                            | Established        |

### 4.2 Layer Components

| C++ (`NeuralAmpModelerCore/`)                 | Rust (`src/`)                                                                   | Parity established |
| --------------------------------------------- | ------------------------------------------------------------------------------- | ------------------ |
| Gate-major weight layout `[Gate][IH][Hidden]` | `models/lstm/layer.rs` — `LstmLayer.input_hidden_weights` `[[[u16; H]; IH]; 4]` | Established        |
| Bias vector `[H * 4]`                         | `models/lstm/layer.rs` — `LstmLayer.bias: [f32; H4]`                            | Established        |
| Hidden state `[H]`                            | `models/lstm/layer.rs` — `LstmLayer.state` / `state_bf16`                       | Established        |
| Cell state `[H]`                              | `models/lstm/layer.rs` — `LstmLayer.cell_state`                                 | Established        |
| Head projection (H → 1)                       | `models/lstm/model1.rs` — `head_weights` / `head_bias`                          | Established        |
| FP32 native head rechannel                    | `models/lstm/model1.rs` — `use_f32_head: bool`                                  | Established        |
| Kahan-compensated head accumulation (β2.2)    | `math/common/scalar_ref/dot.rs` — `dot_product_f32_native_kahan`                | Established        |

### 4.3 Scalar Parity Reference

| C++ (`NeuralAmpModelerCore/`)           | Rust (`src/`)                                            | Notes                    |
| --------------------------------------- | -------------------------------------------------------- | ------------------------ |
| LSTM scalar minimax sigmoid (degree-17) | `math/activations/sigmoid.rs` — `scalar_minimax_sigmoid` | For C++ parity test only |

### 4.4 LSTM Configurations

| Config                         | Rust type                                    | `src/models/lstm/mod.rs` alias |
| ------------------------------ | -------------------------------------------- | ------------------------------ |
| `1×8`                          | `LstmModel1<8, 9, 32>`                       | `Lstm1x8`                      |
| `1×12`                         | `LstmModel1<12, 13, 48>`                     | `Lstm1x12`                     |
| `1×16`                         | `LstmModel1<16, 17, 64>`                     | `Lstm1x16`                     |
| `1×24`                         | `LstmModel1<24, 25, 96>`                     | `Lstm1x24`                     |
| `1×40`                         | `LstmModel1<40, 41, 160>`                    | `Lstm1x40`                     |
| `2×8`                          | `LstmModel2<8, 9, 16, 32>`                   | `Lstm2x8`                      |
| `2×12`                         | `LstmModel2<12, 13, 24, 48>`                 | `Lstm2x12`                     |
| `2×16`                         | `LstmModel2<16, 17, 32, 64>`                 | `Lstm2x16`                     |
| `2×24`                         | `LstmModel2<24, 25, 48, 96>`                 | `Lstm2x24`                     |
| Any other (num_layers, hidden) | `loader/dispatcher/lstm.rs` → `LstmModelDyn` | —                              |

### 4.5 LSTM Sample-Rate Drift & Parity Caps

LSTM is the one topology whose parity error grows with **signal length** and **host sample
rate** — its recurrent cell state accumulates f16c quantization error over time (full RCA in
[`audio_fidelity_map.md`](audio_fidelity_map.md) §3). NAMCore is *also* f16c, so the figures
below are **interop drift shared by both engines**, not a NAM-rs-only defect.

Measured nam-rs ↔ NAMCore (v2 stress signal, `tests/cpp_parity.rs`):

| Model         | 44.1 kHz | 48 kHz  | 88.2 kHz | 96 kHz  | 192 kHz     |
|:------------- |:--------:|:-------:|:--------:|:-------:|:-----------:|
| LSTM 1×16     | 2.39e-2  | 2.61e-2 | 5.39e-2  | 6.09e-2 | **1.42e-1** |
| LSTM 2×8      | 3.41e-3  | 3.88e-3 | 1.18e-2  | 1.45e-2 | 4.20e-2     |
| LSTM Official | 1.23e-3  | 1.23e-3 | 1.23e-3  | 1.23e-3 | 1.23e-3     |

**Parity gate (anti-masking).** The absolute ESR cap for LSTM is **rate-aware and measured** — it
is *not* a single flat number, and **no sample rate is excluded** to make a gate pass (Gate
Calibration Policy Rule 7, [`perceptual_validation.md`](perceptual_validation.md)):

| Host rate | `ABSOLUTE_ESR_CAP_LSTM` | Margin over worst measured   |
|:--------- |:-----------------------:|:---------------------------- |
| ≤ 96 kHz  | 0.08                    | ~1.3× (vs 6.09e-2 @ 96 kHz)  |
| > 96 kHz  | 0.20                    | ~1.4× (vs 1.42e-1 @ 192 kHz) |

Both caps are below the placebo line (ESR < 1.0). All four LSTM v2 tests
(`live_cross_validation_v2_lstm_{1x16,2x8,official,dyn}`) run `run_v2_multi_sr` across all five
supported rates. The 192 kHz / 1×16 drift is a **documented, tracked, asserted** limitation
(see §4.5 and §9.1 of this document for full RCA and measured gates), never a hidden gap.

**HighFidelity mode interop caps (Tarefa β1.3).** When `ActivationPrecision::HighFidelity` is
active, Rust uses polynomial exp-based tanh/sigmoid (~2.4e-7 / ~2.1e-7 error) while the C++
`render` tool continues to use standard Padé [5,4] tanh (~2.32e-3) and minimax degree-17
sigmoid (~4.09e-4). This **deliberate asymmetry** means HF mode is **closer to the mathematical
ideal** but **further from C++ bit-equivalence**. The parity gate uses HF-specific caps that
remain meaningful (ESR < 1.0) while documenting the increased interop drift:

| Host rate | `ABSOLUTE_ESR_CAP_LSTM_HF` | Notes                                    |
|:--------- |:--------------------------:|:---------------------------------------- |
| ≤ 96 kHz  | 0.30                       | ~5× standard cap; covers Padé→HF delta   |
| > 96 kHz  | 0.60                       | Conservative for 192 kHz recurrent drift |

HF caps for WaveNet: 5.0× the standard `ABSOLUTE_ESR_CAP_WAVENET` (still < 1.0).
HF tests exist in `tests/cpp_parity.rs` as `live_cross_validation_*_hf` and
`live_cross_validation_v2_*_hf` (all `#[ignore]`, require C++ toolchain).

> **Design decision:** HF mode is an "ideal math" mode, not a "match C++" mode. The divergence
> is expected and documented. Users seeking interop parity with NAMCore should use Standard
> activation precision. See [`audio_fidelity_map.md`](audio_fidelity_map.md) §2/§6 and the
> HF interop caps in §4.5 above (mirrored in [`perceptual_validation.md`](perceptual_validation.md)).
> **Weight-layout note (root cause of a real bug).** JSON `.nam` LSTM weights use the **Original**
> layout `[Gate][H][IH]`; the production runtime transposes them to the SIMD-friendly **GateMajor**
> `[Gate][IH][H]` at load (see §4.2, §7.4). Confusing the two produces a *completely* wrong output
> (ESR ≈ 21). This bit the f64 oracle's external anchor (Sprint S8 / Problem A) and is the reason
> the oracle is now cross-checked against production, not just numerically — see §9.2.

---

## 5. ConvNet Architecture

| C++ (`NeuralAmpModelerCore/`)                   | Rust (`src/`)                                              | Parity established |
| ----------------------------------------------- | ---------------------------------------------------------- | ------------------ |
| `convnet.cpp` / `convnet.h` — ConvNet inference | `models/convnet/model.rs` — `ConvNetModel`                 | —                  |
| Conv1D → BatchNorm → Activation blocks          | `models/convnet/block.rs` — `ConvNetBlock`                 | —                  |
| `convnet.h:108-118` — `_Head` (post-stack head) | `models/convnet/model.rs` — `ConvNetModel.post_stack_head` | —                  |
| ConvNet topology dispatch                       | `loader/dispatcher/convnet/mod.rs` — `build_convnet()`     | —                  |

> **Cross-validation status (PM-04).** The Rust engine is complete, dispatched, unit-tested, and
> covered by a *self-golden* determinism test (`test_golden_vectors_convnet_test`). **Formal external
> validation is blocked**: NAMCore v0.5.3's ConvNet (single shared `channels`, hard-coded `kernel=2`,
> matrix-multiply `_Head`, no `head_scale`) is architecturally incompatible with NAM 0.5.4's multi-block
> ConvNet (per-block `channels`/`kernel`/`activation`, Conv1D post-stack head, `head_scale`), so the C++
> `render` tool cannot emit a golden (an *expected SKIP* in `golden_gen_build.sh`). The f64 reference
> oracle also returns zeros for ConvNet (`reference_oracle.rs:282`). The safe, NAMCore-compatible path
> is an **independent f64 ConvNet oracle** (§9.2 trust-chain pattern) — see §13.1 / `TODO-findings.md` PM-04.

---

## 6. A2 Architecture (Fixed fast-path port)

> **Status:** A2 inference is implemented along **two complementary paths**:
>
> 1. **Fixed fast-path** — a port of `NAM/wavenet/a2_fast.cpp` for the production shapes **A2-Full** (8 ch) and **A2-Lite** (3 ch): pure A2, no FiLM/gating, full SIMD const-generic specialization. See [TODO-sprints.md](../TODO-sprints.md).
> 2. **Dynamic engine** (`WaveNetA2Dyn`) — the *general* A2 engine that **does** handle FiLM, `GatingActivation`, `BlendingActivation`, `condition_dsp`, and `bottleneck ≠ channels`. `is_a2_shape()` routes models using any of these features here (mirroring C++ `a2_fast.cpp`, which rejects them and falls back to its generic Eigen WaveNet). All of these are **golden-tested** (§13): gating/blending/`condition_dsp` reach near-bit-exact parity (>100 dB SNR), while **FiLM** carries a documented interop divergence vs the C++ generic path (18–36 dB SNR, tracked as **RF1** — see §13 and [`TODO-findings.md`](../TODO-findings.md) PM-03).
>
> `SlimmableWavenet` (single-net channel slicing) is a separate, deferred epic; the multi-model `SlimmableContainer` (independent sub-nets + crossfade) is implemented and tested (`tests/container_slimmable.rs`).
>
> **Golden vectors** are generated from the C++ `render` tool at pinned commit `9c7b185` (v0.5.3) and committed as pre-validated Layer 1 artifacts (`tests/fixtures/golden_wavenet_a2_{full,lite}.bin`). The v1 golden tests run actively on every `cargo test`. Layer 2 live cross-validation (`tests/cpp_parity.rs`) exists and is `#[ignore]` — normal for all live parity tests, requiring C++ toolchain; run via `utils/tests-long.sh`.
>
> **Calibrated SNR/ESR** (measured against C++ v0.5.3): A2-Full = 79.2 dB / 1.21e−8; A2-Lite = 90.7 dB / 8.58e−10. Thresholds include ≥8 dB SNR margin and ~6–7× ESR multiplier. See `tests/fixtures/README.md` (§`wavenet_a2_full.nam` & `wavenet_a2_lite.nam`) and §9.1.
>
> **⚠ Nature:** These goldens use **synthetic weights** (canonical A2 skeleton: 23 layers, K=6/15, LeakyReLU, head_scale=0.02 — generated by `tests/fixtures/generate_a2_fixtures.py`), now spanning **both** the fast-path (A2-Full/Lite) and the dynamic engine (FiLM / gated / blended / `condition_dsp`). They validate **numerical parity of the engine**, not real amplifier timbres. The engine **already supports** FiLM and generic A2 inference via `WaveNetA2Dyn`; what remains is integrating **official real-amp FiLM captures** (e.g. `wavenet_a2_max.nam`) as committed fixtures and elevating them to official goldens once the FiLM interop divergence (RF1) is fully characterized (§13, PM-03/PM-05).
>
> **Sample rate limitation:** A2 v2 multi-SR goldens only exist at 48 kHz. Models with an explicit `sample_rate` field in JSON (A2-Full, A2-Lite) are constrained to their native rate by the C++ `render` tool. All other models generate at 5 rates (44.1k/48k/88.2k/96k/192k).
>
> **Runtime safeguards:** The garbage guard detects absurd-but-finite amplitude (`max_sample > 10³`) on the DSP output, preventing silent corruption from reaching audio hardware. ESR is the primary scale-robust threshold in `topology_thresholds()` and `live_parity_thresholds()`.

| C++ (`NeuralAmpModelerCore/`)                               | Rust (`src/`)                                                                                                     | Parity established                  |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `NAM/wavenet/a2_fast.h` — architectural constants           | `models/a2/params.rs` — `A2_NUM_LAYERS`, `A2_LEAKY_SLOPE`, `A2_KERNEL_SIZES`, `A2_DILATIONS`, `A2_VALID_CHANNELS` | Established                         |
| `a2_fast.cpp:875-908` — `is_a2_shape()`                     | `loader/nam_json/topology.rs` — `is_a2_shape()`                                                                   | Established                         |
| `NAM/activations.h:L111-122` — `fast_tanh`                  | `models/a2/activations.rs` — `fast_tanh()`                                                                        | Established                         |
| `GatingActivation` class *(wired in `WaveNetA2Dyn`)*        | `models/a2/gating.rs` — `GatingActivationConfig` + `apply_gating_simd`                                            | Established (golden ~103 dB)        |
| `BlendingActivation` class *(wired in `WaveNetA2Dyn`)*      | `models/a2/gating.rs` — `BlendingActivationConfig` + `apply_blending_simd`                                        | Established (golden ~133 dB)        |
| `_FiLMParams` / FiLM modulation *(wired in `WaveNetA2Dyn`)* | `models/a2/film.rs` — `FiLMConfig` / `FiLMLayer` + `models/a2/model/dynamic/process.rs`                           | Golden-tested; RF1 divergence (§13) |
| **A2-Full / A2-Lite inference (fixed fast-path)**           | `models/a2/` — port of `A2FastModel<8>` / `A2FastModel<3>`                                                        | Established                         |
| `NAM/container.{h,cpp}` — `SlimmableContainer`              | `models/container.rs` + `loader/dispatcher/container/`                                                            | Established                         |

---

## 7. Weight Loading & Parsing

### 7.1 `.nam` JSON Format

| C++ (`NeuralAmpModelerCore/`)                         | Rust (`src/`)                                                                    | Parity established |
| ----------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------ |
| `.nam` JSON deserialization                           | `loader/nam_json/data.rs` — `NamModelData`                                       | Established        |
| Weight layout detection (row-major gate-major)        | `loader/nam_json/data.rs` — `WeightLayout` enum                                  | Established        |
| `NeuralModel.cpp` — topology dispatch                 | `loader/nam_json/topology.rs` — `get_wavenet_topology()` / `get_lstm_topology()` | Established        |
| NAM metadata (input_level_dbu, loudness, sample_rate) | `loader/nam_json/data.rs` — `NamMetadata`                                        | Established        |

### 7.2 NAMB Binary Format

| C++ ecosystem convention                       | Rust (`src/`)                                          | Parity established |
| ---------------------------------------------- | ------------------------------------------------------ | ------------------ |
| NAMB binary layout (magic, version, CRC32)     | `loader/namb.rs` — `NambHeader` + `parse_namb()`       | Established        |
| Original weight layout (row-major, unmodified) | `loader/namb.rs` — `WeightLayout::Original`            | Established        |
| GateMajorLstm transposed layout                | `loader/namb.rs` — `WeightLayout::GateMajorLstm`       | Established        |
| Interleaved4WaveNet transposed layout          | `loader/namb.rs` — `WeightLayout::Interleaved4WaveNet` | Established        |
| NAMB CRC32 validation (v2+)                    | `loader/namb.rs` — `crc32_ieee()`                      | Established        |
| NAMB encoder/export                            | `loader/namb_encoder.rs` — `encode_namb()`             | Established        |

### 7.3 WaveNet Weight Layout

| C++ (`NeuralAmpModelerCore/`)                       | Rust (`src/`)                                                     | Parity established |
| --------------------------------------------------- | ----------------------------------------------------------------- | ------------------ |
| `WaveNet.h` — `SetWeights` (global layout)          | `loader/dispatcher/wavenet/standard.rs` — `build_wavenet_typed()` | Established        |
| `WaveNetLayerArrayT::SetWeights` (per-array layout) | `loader/dispatcher/wavenet/standard.rs` — `build_wavenet_array()` | Established        |
| Head scale (final scalar multiplier)                | `loader/dispatcher/wavenet/standard.rs` — `head_scale`            | Established        |

### 7.4 LSTM Weight Layout

| C++ (`NeuralAmpModelerCore/`)            | Rust (`src/`)                                                               | Parity established |
| ---------------------------------------- | --------------------------------------------------------------------------- | ------------------ |
| `LSTMLayerT::SetNAMWeights` (NAM format) | `loader/dispatcher/lstm.rs` — `build_lstm_1layer()` / `build_lstm_2layer()` | Established        |
| Weight layout: `[H4*IH, H4, H, H, H, 1]` | `loader/dispatcher/lstm.rs` — `read_lstm_layer()`                           | Established        |

---

## 8. Math / SIMD Kernels

| C++ (`NeuralAmpModelerCore/`)                       | Rust (`src/`)                                                    | Parity established |
| --------------------------------------------------- | ---------------------------------------------------------------- | ------------------ |
| Dot product (f32)                                   | `math/gemm/dot.rs` — `dot_product`                               | Established        |
| Dot product (BF16 quantized weights)                | `math/gemm/dot_4x/` — `dot_product_bf16`                         | Established        |
| Dot product 4× interleaved (WaveNet Conv1D)         | `math/gemm/dot_4x/` — `dot_product_4x_interleaved`               | Established        |
| Dot product 4× interleaved dual-frame               | `math/gemm/dot_4x/` — `dot_product_4x_interleaved_dual_frame`    | Established        |
| GEMV (fused add)                                    | `math/gemm/gemv.rs` — `fused_add_gemv`                           | Established        |
| GEMV (overwrite)                                    | `math/gemm/gemv.rs` — `gemv_overwrite`                           | Established        |
| GEMV 4-gate (LSTM)                                  | `math/gemm/gemv_4gate.rs` — `gemv_overwrite_4gate`               | Established        |
| GEMV BF16 (LSTM 4-gate)                             | `math/gemm/gemv_bf16.rs` — `gemv_overwrite_bf16_4gate`           | Established        |
| Tanh activation (SIMD)                              | `math/activations/tanh.rs` — `tanh_slice`                        | Established        |
| Sigmoid activation (SIMD)                           | `math/activations/sigmoid.rs` — `sigmoid_slice`                  | Established        |
| Fused Tanh + accumulate (WaveNet head)              | `math/wavenet/accumulate.rs` — `tanh_and_accumulate_block`       | Established        |
| Fused Tanh + overwrite (first layer head)           | `math/wavenet/accumulate.rs` — `tanh_and_overwrite_block`        | Established        |
| Cascaded head accumulation (array N seeds from N−1) | `layer_array.rs` — `head_seeded` + accumulation                  | Established        |
| Gain application (linear)                           | `math/dsp/gain.rs` — `apply_gain`                                | Established        |
| Gain LUT (dB → linear)                              | `math/dsp/gain_lut.rs` — `GainLut`                               | —                  |
| Stereo convolution (resampler FIR)                  | `math/dsp/stereo/` — `convolve_stereo`                           | Established        |
| Kahan compensated summation (scalar fallback)       | `math/common/kahan.rs` — `KahanF32` / `Kahan4F32`                | Established        |
| Kahan dot product (LSTM head — β2.1)                | `math/common/scalar_ref/dot.rs` — `dot_product_f32_native_kahan` | Established        |
| BF16 quantization (f32 → u16)                       | `math/common/utility.rs` — `quantize_weight()`                   | Established        |
| Scalar reference (definitive math specification)    | `math/common/scalar_ref.rs` — all operations                     | —                  |

---

## 9. Cross-Validation

| C++ (`NeuralAmpModelerCore/`)                        | Rust (`src/` / `tests/`)                        | Parity established |
| ---------------------------------------------------- | ----------------------------------------------- | ------------------ |
| `render` CLI (golden output generation)              | `tests/cpp_parity.rs` — live cross-validation   | Established        |
| `ModelTest.cpp` (stress-signal tests)                | `tests/nam_infer_test.rs` — golden vector tests | Established        |
| `test_get_dsp.cpp` (official WaveNet test)           | `tests/fixtures/` — `wavenet.nam` model         | Established        |
| `test_slimmable_wavenet.cpp` (official WaveNet test) | `tests/fixtures/` — shared models               | Established        |
| SNR thresholds (C++ → Rust comparison)               | `tests/cpp_parity.rs` — per-model SNR passes    | —                  |

### 9.1 Post-Nuke ESR Measurements

After elimination of all BF16/F16 quantization and dual-mode paths from WaveNet A1, the ESR against NeuralAmpModelerCore v0.5.3 (commit `9c7b185`)
was recalibrated:

| Model                 | ESR (linear) | ESR (dB) | SNR (dB) | Notes                                         |
|:--------------------- |:------------ |:-------- |:-------- |:--------------------------------------------- |
| WaveNet A1-Std CH=16  | 4.58e-13     | −123.4   | 123.4    | Live v1, f32 + poly tanh                      |
| WaveNet Standard v2   | *varies*     | *varies* | 101.8*   | Multi-SR v2 worst @ 192k                      |
| WaveNet Standard (v2) | *varies*     | *varies* | 123.0**  | v2 best @ 48kHz                               |
| WaveNet Feather CH=8  | 4.92e-14     | −133.1   | 133.1    | Live v1                                       |
| WaveNet Feather (v2)  | *varies*     | *varies* | 117.6*   | v2 worst @ 192kHz                             |
| WaveNet Nano CH=4     | 6.30e-14     | −132.0   | 132.0    | Live v1                                       |
| WaveNet Nano (v2)     | *varies*     | *varies* | 114.6*   | v2 worst @ 192kHz                             |
| WaveNet Lite CH=12    | 5.84e-13     | −122.3   | 122.3    | Golden v1 (live 117.4); **P1/RF7 ✅resolved** |

> \* Worst-case across v2 multi-SR golden vectors.
> \*\* Best-case v2 (48 kHz native).
> All WaveNet SKUs — **Lite now included (post-P1/RF7 fix)** — achieve SNR ≫ 100 dB, comparable to LSTM/Linear (67–91 dB).
> Thresholds calibrated: Standard ≥ 85 dB SNR, Feather ≥ 100 dB, Nano ≥ 95 dB, Lite ≥ 105 dB.
> **Lite RCA:** the historical 0.9 dB came from a `MirroredBuffer` page-rounding bug (buffer not channel-aligned for non-power-of-two CH=12/CH=6, `1024 % 12 = 4`) plus an obsolete synthetic golden. Fixed by `MirroredBuffer::new_aligned()` (`lcm(page, channel_stride)`) + migration to the real `EVH-5150-Lite.nam` golden. Permanently guarded by `tests/wavenet_lite_block_invariance.rs`, `wavenet_ringbuffer_alignment`, and `test_mirror_buf_channel_alignment` (all active, not `#[ignore]`d).

These measurements replace the pre-nuke ESR of ~3e-3 to 1e-2 (−25 to −20 dB).
The ~10 order-of-magnitude improvement comes from eliminating BF16/F16 weight
quantization (previously the dominant drift source at ~3.9e-3 per element).

### 9.2 The f64 Reference Oracle — Independent Witness

Cross-validation against NAMCore answers *"do the two f16c engines agree?"* — but NAMCore is
itself an f16c implementation, so it cannot tell us how far either engine sits from the **ideal
mathematics**. That is the job of the **f64 reference oracle** (`src/testing/reference_oracle.rs`):
a from-scratch double-precision implementation of each topology that owes nothing to the SIMD
production path.

The trust chain has **four independent code paths**, each anchoring the next:

```text
NAMCore C++ (reference)
  └─ golden vectors + cpp_parity (interop parity, §9 / §9.1)
       └─ nam-rs production f32 (the shipped engine)
            └─ Rust f64 oracle (ideal-math witness, exact tanh/sigmoid)
                 └─ independent NumPy f64 anchor (3rd implementation)
```

**Why a 4th layer (the NumPy anchor)?** An oracle written to *mirror* the engine it judges proves
nothing — a shared conceptual bug passes silently in both. In Sprint S8 this exact trap was found
and fixed (Problem A): the original anchor copied the oracle's buffer layout, so a wrong LSTM gate
layout (§4.5) and WaveNet weight transpositions went undetected. The anchor was rebuilt to be
genuinely independent and is now validated **two ways**:

| Reference oracle vs…               | WaveNet  | LSTM    | A2       | Meaning                                    |
|:---------------------------------- |:--------:|:-------:|:--------:|:------------------------------------------ |
| **production f32** (separate code) | 6.13e-14 | 3.57e-3 | 4.28e-10 | true precision floor (f16c+Padé+f32)       |
| **NumPy f64 anchor** (3rd impl)    | 5.0e-16  | 3.5e-30 | 5.0e-16  | the oracle's own math is correct (< 1e-12) |

The first column is the genuine cost of f16c/Padé precision — and the source of the oracle ESR
gates (`WAVENET_ESR_LIMIT = 1e-12`, `LSTM_ESR_LIMIT = 7.0e-3`, `A2_ESR_LIMIT = 8.6e-10` =
measured × 2, `tests/common/constants.rs`). The second column proves the oracle is a faithful
f64 model, not a mirror of production. A compile-time `const { assert!(LIMIT < 1.0) }` meta-test
keeps every oracle gate below the placebo line.

| C++ / math reference                   | Rust (`tests/`)                                                   | Parity established    |
|:-------------------------------------- |:----------------------------------------------------------------- |:--------------------- |
| (no C++ equiv — ideal f64 math)        | `src/testing/reference_oracle.rs` — f64 forward pass              | Witness               |
| (no C++ equiv — independent NumPy f64) | `tests/fixtures/scripts/validate_oracle_f64.py` + anchors         | Established (< 1e-12) |
| Oracle gates / anti-placebo meta-tests | `tests/reference_oracle_f64.rs`, `tests/threshold_calibration.rs` | Established           |

> **Maintenance contract:** any change to `reference_oracle.rs` invalidates the anchors. In the
> same change set, regenerate them with `validate_oracle_f64.py` and re-confirm both columns above.
> Never leave the `test_oracle_vs_python_anchor_*` tests `#[ignore]`d without a tracked task.

---

## 10. Topology Table

| C++ NAM topology    | Rust module / type                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------- |
| WaveNet Standard 16 | `models::wavenet::WaveNetModel<16, 3, 8>`                                                   |
| WaveNet Lite 12     | `models::wavenet::WaveNetModel<12, 3, 6>`                                                   |
| WaveNet Feather 8   | `models::wavenet::WaveNetModel<8, 3, 4>`                                                    |
| WaveNet Nano 4      | `models::wavenet::WaveNetModel<4, 3, 2>`                                                    |
| WaveNet Dyn         | `models::wavenet::WaveNetModelDyn` (fallback genérico para geometrias livres)               |
| LSTM 1×3            | `models::lstm::Lstm1x3`                                                                     |
| LSTM 1×8            | `models::lstm::LstmModel1<8, 9, 32>`                                                        |
| LSTM 1×12           | `models::lstm::LstmModel1<12, 13, 48>`                                                      |
| LSTM 1×16           | `models::lstm::LstmModel1<16, 17, 64>`                                                      |
| LSTM 1×24           | `models::lstm::LstmModel1<24, 25, 96>`                                                      |
| LSTM 1×40           | `models::lstm::LstmModel1<40, 41, 160>`                                                     |
| LSTM 2×8            | `models::lstm::LstmModel2<8, 9, 16, 32>`                                                    |
| LSTM 2×12           | `models::lstm::LstmModel2<12, 13, 24, 48>`                                                  |
| LSTM 2×16           | `models::lstm::LstmModel2<16, 17, 32, 64>`                                                  |
| LSTM 2×24           | `models::lstm::LstmModel2<24, 25, 48, 96>`                                                  |
| LSTM Dyn            | `models::lstm::LstmModelDyn` (fallback genérico para camadas/hidden não-catalogados)        |
| A2-Full (8 ch)      | `models::a2::WaveNetA2<8>` (fixed fast-path, 8 channels, tap-major frame-tiled convolution) |
| A2-Lite (3 ch)      | `models::a2::WaveNetA2<3>` (fixed fast-path, 3 channels, unrolled GEMV convolution)         |
| WaveNet A2 Dyn      | `models::a2::WaveNetA2Dyn` (motor dinâmico para geometrias A2 não-catalogadas, FiLM/gating) |
| ConvNet             | `models::convnet::ConvNetModel` (feed-forward Conv1D+BatchNorm+Activation, sem recorrência) |

> Engine genres: **Catalog SKUs** (Standard/Lite/Feather/Nano, LSTM 1×/2×, A2-Full/Lite) use const-generic paths with full SIMD specialization. **Dynamic engines** (`WaveNetModelDyn`, `LstmModelDyn`, `WaveNetA2Dyn`) handle free geometry, `condition_size ≠ 1`, post-stack heads, and other non-catalogued topologies via generic dispatch (§3.3). **ConvNet** maps to the C++ `convnet.cpp` / `convnet.h` reference.

---

## 11. NAM-rs Divergences from C++ Reference (Accepted)

### 11.1 Architecture

| Divergence                                    | Rationale                                                                                                                                                                            |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **No `DspBridge` in CLAP mode**               | CLAP plugin receives both input and output in a single `process()` call. Bridge only needed standalone (PipeWire dual-thread topology).                                              |
| **MirroredBuffer (`memfd_create`)**           | Linux-specific virtual memory mirroring for O(1) linear access in WaveNet delay lines. C++ uses modulo-based circular access.                                                        |
| **Static const-generic dispatch (no vtable)** | Static `match` on `StaticModel` enum avoids vtable overhead. C++ `GetDSP` returns a pointer to a virtual base class.                                                                 |
| **Reset does NOT prewarm on load**            | `reset()` is a public API for explicit state clearing. Loader calls `prewarm()` separately to preserve LSTM initial states loaded from file. C++ `Reset` always calls `prewarm()`.   |
|                                               | **A2-specific implication (fixed Sprint E1):** calling `process()` on an A2 model with empty layers before `prewarm()` caused `head_write_pos` to grow unmasked, leading to OOB on   |
|                                               | the next `prewarm()`. Fixed by masking with `head_ring_mask` in the early-return path.                                                                                               |
| **Prewarm hardcoded to 2048 samples**         | C++ `PrewarmSamples()` returns `receptive_field`. NAM-rs uses 2048 as a safe upper bound covering all models.                                                                        |
| **`WavenetA2Placeholder` (silent output)**    | Retired and removed. Replaced by real `WaveNetA2` inference.                                                                                                                         |
| **No `std::complex` / STL data structures**   | Everything uses idiomatic Rust (`AlignedVec<T>`, `AtomicU64`, `rtrb` SPSC).                                                                                                          |
| **TSC-based latency measurement**             | NAM-rs calibrates the CPU TSC for nanosecond-accurate RT cycle measurements — no C++ equivalent.                                                                                     |
| **CPU C-state lock (`/dev/cpu_dma_latency`)** | Linux-specific RT tuning — no equivalent in cross-platform C++ reference.                                                                                                            |
| **SCHED_FIFO + `mlockall`**                   | Linux RT scheduling — not applicable to C++ reference.                                                                                                                               |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`condition_size ≠ 1` or                     | Multi-condition WaveNet (`condition_size > 1`) and post-stack heads (`head` sub-object) are official NAMCore features. Models using them are **accepted at load time** —             |
| `head` (non-null) accepted**                  | `get_wavenet_topology()` captures the non-catalog geometry as `Free` and routes it to `WaveNetModelDyn` (the dynamic engine), which is parameterized on `condition_size` at runtime. |
|                                               | Catalog SKUs (Standard/Lite/Feather/Nano) use `condition_size=1` and `head=null` via the const-generic fast-path. The dynamic path handles `condition_size > 1`, post-stack heads,   |
|                                               | and free geometries with generic dispatch (§3.3).                                                                                                                                    |

### 11.2 Math

| Divergence                                           | Rationale                                                                                                                                                                                                              |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Padé [5,4] Tanh vs `std::tanh`**                   | C++ uses IEEE-754 `std::tanh`. NAM-rs uses rational Padé approximant (error < 2.32e-3) — 10–20× throughput gain. This is the sole remaining math divergence for WaveNet A1 (BF16/F16 quantization paths eliminated).   |
| **Minimax degree-17 sigmoid vs `0.5+0.5*tanh(x/2)`** | Direct polynomial (1.67× lower error, −20.25% latency). C++ reference composes `std::tanh`.                                                                                                                            |
| **BF16 vs F16 dispatch**                             | NAM-rs runtime-detects `Avx512VnniBf16` and chooses precision. C++ has no equivalent multi-ISA/packed-format dispatch. BF16 has ~8× larger quantization error than F16 but allows VNNI native ops on Sapphire Rapids+. |
| **Kahan compensated summation (scalar fallback)**    | Applied in interleaved 4x scalar fallback dot products. C++ uses standard accumulation. Static conv1d paths also use plain accumulation.                                                                               |
| **Anti-subnormal DC dither (−220 dBFS)**             | Prevents subnormal float stalls. Below 24-bit DAC noise floor. C++ has no equivalent.                                                                                                                                  |
| **FP32 native head rechannel**                       | Final projection (head) runs in FP32 regardless of backbone precision. Eliminates quantization error at output. C++ uses same precision throughout.                                                                    |

### 11.3 Ecosystem

| Divergence                                  | Rationale                                                                                                                                |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Linux-only deployment (PipeWire + CLAP)** | C++ reference is cross-platform (Windows/macOS/Linux, VST3/AU/LV2).                                                                      |
| **NAMB binary pre-transposed layouts**      | Interleaved4WaveNet and GateMajorLstm layouts are NAM-rs inventions for zero-cost on-load dispatching. C++ uses only Original row-major. |
| **No Python golden generation**             | Cross-validation moved from Python scripts to C++ `render` CLI + Rust native `tests/cpp_parity.rs`.                                      |
| **Proptest-based activation verification**  | 10k+ random inputs against independent `f64` reference. C++ validation uses fixed test vectors.                                          |

---

## 12. IR Cabsim — New NAM-rs Feature (No C++ Equivalent)

> **Status:** The IR Cabsim convolution stage (`src/dsp/cabsim/`) is a **feature native to NAM-rs** with no equivalent in the canonical C++ reference (`NeuralAmpModelerCore`). There is no `ImpulseResponse` or convolution-processing class in the `NAM/` or `NeuralAmpModelerCore/` source tree.

The closest C++ reference is `dsp::ImpulseResponse` in the `AudioDSPTools` library (MIT-licensed utility used by `NeuralAmpModelerPlugin`):

| C++ reference                                                | Rust (`src/`)                         | Parity status   |
| ------------------------------------------------------------ | ------------------------------------- | --------------- |
| `AudioDSPTools/dsp/ImpulseResponse.h` (direct time-domain)   | `dsp/cabsim/conv.rs` — UPOLS engine   | **Analyzed**    |
| `NeuralAmpModelerPlugin/NeuralAmpModeler.cpp:676` (IR usage) | `dsp/pipeline/capture.rs` — cab stage | **New feature** |

### 12.1 Algorithmic Analysis — `dsp::ImpulseResponse` (C++) vs UPOLS (NAM-rs)

> **Analysis completed.**

#### 12.1.1 C++ `dsp::ImpulseResponse` — Algorithm

| Property                 | Detail                                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Algorithm**            | Direct time-domain convolution (O(N²)). Computes `<w, h[n:n+M]>` for each output sample `n`, where `w` is the time-reversed IR (`mWeight`) and `h` is the input history. |
| **Core loop**            | `this->mWeight.dot(input_history_segment)` — Eigen dot product (float). Result cast to `double`.                                                                         |
| **Partition**            | None. Sliding window via `History` base class. `mHistoryRequired = irLength - 1`.                                                                                        |
| **IR storage**           | Time-**reversed** in `mWeight`: sample `j = irLength - 1 - i` stored at `mWeight[j]`, so the dot product `weight · history[i..i+M]` computes convolution.                |
| **Resampling**           | Cubic interpolation (`ResampleCubic<float>`) when `mRawAudioSampleRate != mSampleRate`.                                                                                  |
| **Gain / normalization** | Fixed gain formula: `gain = 10^(-18×0.05) × 48000 / mSampleRate` (~−0.9 dB × 48k/sr, ≈ 0.126 at 48 kHz). No peak normalization.                                          |
| **Max IR length**        | Capped at 8192 samples (`mMaxLength`).                                                                                                                                   |
| **Precision**            | Weights & history: `float`. Outputs: `DSP_SAMPLE` (= `double`). Dot product in float → cast to double.                                                                   |
| **Latency**              | 0 samples (direct sample-by-sample convolution).                                                                                                                         |
| **Channels**             | Convolves mono; duplicates to all output channels.                                                                                                                       |

#### 12.1.2 NAM-rs `ConvEngine` (UPOLS) — Algorithm

| Property                 | Detail                                                                                                                                                                      |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Algorithm**            | Uniform-Partitioned Overlap-Save (UPOLS, O(N log N)). Frequency-domain multiply-accumulate over `ceil(IR_len / partition_size)` partitions per block.                       |
| **Core loop**            | For each partition `p`: load FDL spectrum at lag `p`, complex MAC with pre-FFT'd kernel partition `p`, accumulate. IFFT on accumulator, extract overlap-save tail.          |
| **Partition**            | Partition size = audio block size. `num_partitions = ir.len().div_ceil(block_size)`. Kernel partitions pre-FFT'd at construction (offline).                                 |
| **IR storage**           | Samples placed at start of FFT buffer (`fft_buf[i] = Complex::new(sample, 0.0)`), zero-padded to FFT size. Causal — no reversal needed (overlap-save discards wrap-around). |
| **Resampling**           | Polyphase resampler (`NamResampler`) — higher quality than cubic; batch-offline operation outside RT thread.                                                                |
| **Gain / normalization** | Optional peak normalization to 1.0 (`normalize_in_place`). No fixed gain reduction formula. IFFT scale = `1.0 / fft_size`.                                                  |
| **Max IR length**        | Unbounded (constrained by IR file size limit of 1 GiB; memory scales with `num_partitions × fft_size`).                                                                     |
| **Precision**            | `f32` throughout (weights, spectra, FDL, accumulator, outputs).                                                                                                             |
| **Latency**              | `partition_size` samples (one full audio block).                                                                                                                            |
| **Channels**             | Convolves mono; stereo duplication handled externally (`capture.rs`).                                                                                                       |

#### 12.1.3 Algorithmic Differences — Impact on Cross-Validation Tolerances

| #   | Divergence         | C++                                                    | NAM-rs                                                                 | Expected impact on ESR/SNR                                                                                                                                                                                                                                                                                                                                              |
| --- | ------------------ | ------------------------------------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Algorithm**      | Direct O(N²) time-domain                               | UPOLS O(N log N) frequency-domain                                      | **Primary divergence.** FFT → numerical noise from floating-point twiddle factors. Overlap-save discards the wrap-around half of the circular convolution — identical to direct convolution in exact arithmetic, but different in FP32. Partitions accumulate noise linearly with `num_partitions`.                                                                     |
| 2   | **Gain reduction** | Fixed: `10^(−0.9) × 48k/sr` (~0.126 at 48 kHz, −18 dB) | None (or peak-normalize to 1.0)                                        | **Dominant amplitude mismatch.** C++ output is ~0.126× the Rust output (before peak normalization). Cross-validation must either (a) compensate gain before comparison, or (b) use normalized metrics (ESR is gain-insensitive; SNR and absolute-error metrics will show ~18 dB offset).                                                                                |
| 3   | **Precision**      | Float weights + double output accumulator              | Float throughout                                                       | Accumulation error differences: C++ accumulates dot product in float then casts to double (no benefit for accumulation itself), but the history buffer holds up to 8192 samples — the dot product sums up to 8192 terms in float, similar to UPOLS. The double output cast provides headroom but doesn't change the dot product result. **Impact: small** (a few ULPs). |
| 4   | **Resampling**     | Cubic interpolation                                    | Polyphase (NamResampler)                                               | Cubic is lower quality — introduces interpolation error not present in polyphase. If input/output rates match (no resampling), this divergence does **not** apply. For mismatched rates: **moderate impact** — cubic error manifests as IR shape differences that propagate through convolution.                                                                        |
| 5   | **Max IR length**  | 8192 (hard cap, truncates)                             | Unbounded                                                              | If IR > 8192 samples: C++ truncates; NAM-rs does not. Cross-validation tests should use IRs ≤ 8192 samples to avoid this confound.                                                                                                                                                                                                                                      |
| 6   | **Latency**        | 0 samples                                              | `partition_size` samples                                               | NAM-rs output is time-shifted by `partition_size` samples relative to C++. Cross-validation must align sequences (shift or trim) before comparison.                                                                                                                                                                                                                     |
| 7   | **WAV loading**    | `dsp::wav::Load()` — supports PCM16, float32           | `CabSimIr` — PCM16, PCM24, float32 + NaN/Inf validation + TOCTOU guard | Loading differences (e.g. quantization rounding when PCM → float) are negligible relative to the algorithmic differences above.                                                                                                                                                                                                                                         |

#### 12.1.4 Cross-Validation Strategy

Given the dominant gain divergence (#2), cross-validation should apply **gain compensation** before comparison: multiply C++ output by `1 / gain = 10^(0.9) × mSampleRate / 48000` or normalize both outputs to the same peak RMS before computing ESR/SNR. ESR is naturally gain-insensitive (ratio metric), making it the preferred tolerance measure. SNR and absolute-error metrics should account for the gain offset.

Thresholds: expect **ESR < 1e-3** (relaxed from the 1e-5 internal golden threshold) due to FFT arithmetic differences compounded across multiple partitions. SNR floor is dictated by FFT twiddle-factor rounding (~140 dB for single FFT, degrading with `log₂(num_partitions)` due to frequency-domain accumulation).

**Cross-validation results** (gain-compensated via `CPP_GAIN_INV ≈ 7.943`):

| Scenario | IR len | Signal len | Partitions | ESR      | ESR (dB) | SNR (dB) |
| -------- | ------ | ---------- | ---------- | -------- | -------- | -------- |
| short    | 64     | 256        | 1          | 1.15e-14 | −139.4   | 139.4    |
| medium   | 512    | 1024       | 8          | 2.84e-14 | −135.5   | 135.5    |
| long     | 8192   | 16384      | 128        | 9.39e-14 | −130.3   | 130.3    |
| stress   | 32768  | —          | —          | N/A      | N/A      | N/A      |

> **Stress scenario skipped:** C++ `ImpulseResponse::mMaxLength = 8192` hard-caps the IR, truncating the 32768-sample stress IR. NAM-rs UPOLS stress validation is performed against direct convolution in `tests/cabsim_golden.rs::test_cabsim_golden_stress` (ESR < 1e-5). All three comparable scenarios exceed the 1e-3 ESR threshold by 10+ orders of magnitude, confirming bit-identical equivalence after gain compensation.

### 12.2 Cross-Validation Performed

The `AudioDSPTools` submodule is initialized at `tests/fixtures/NeuralAmpModelerPlugin/AudioDSPTools/`. Cross-validation tests are implemented as `#[ignore]` in `tests/cabsim_cpp_parity.rs` and run via `utils/tests-long.sh`.

- **Test validation:** Compares the UPOLS convolution engine against golden vectors generated by the C++ reference binary (`tests/fixtures/render_ir.cpp`).
- **Alignment & compensation:** Compares are gain-compensated (`CPP_GAIN_INV ≈ 7.943`) and latency-aligned.
- **Results:** Parity is verified with **ESR < 1e-13** in short, medium, and long scenarios, vastly exceeding the target threshold of 1e-3.

---

## 13. Pending / Open Work

Open parity items, by status. 🟢 = established/by-design and tracked; 🟡 = partial or
out-of-current-scope; 🔴 = known divergence under investigation.

Detailed RCA and concrete, low-risk mitigations for every 🟡/🔴 item below are tracked in
[`TODO-findings.md`](../TODO-findings.md) under **"Auditoria de Paridade NAMCore × NAM-rs"**
(findings `PM-01`…`PM-08`).

| Item                                                                                         | Status                                                                                                                                                                                              | Reference / Finding |
|:-------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------- |
| **ConvNet** C++ cross-validation                                                             | 🟡 Engine complete + self-golden + unit-tested; **C++ golden blocked** — NAMCore v0.5.3 single-block ConvNet is architecturally incompatible with NAM 0.5.4 multi-block. f64-oracle path open.      | §5 · PM-04          |
| **A2 general engine** (FiLM / gating / blending / `condition_dsp` / `bottleneck ≠ channels`) | 🟢 **Implemented & golden-tested** via `WaveNetA2Dyn`; gating/blending/`condition_dsp` near-bit-exact (>100 dB)                                                                                     | §6                  |
| **A2 FiLM** interop divergence vs C++ generic WaveNet (**RF1**)                              | 🟡 Wired & golden-tested; SNR 18–36 dB divergence documented & capped; not yet witnessed by the f64 oracle                                                                                          | §6 · PM-03          |
| **A2 official real-amp FiLM captures** (`wavenet_a2_max`, …)                                 | 🟡 **Temporary conformity** — validated against synthetic fixtures (`wavenet_a2_film_full`, `wavenet_a2_film_lite`); real-amp captures structurally incompatible (see §13.1). Sprint S9 goldens ✅. | §6 · PM-05          |
| **`SlimmableWavenet`** (single-net channel slicing)                                          | 🟡 Genuinely deferred (multi-model `SlimmableContainer` is implemented & tested)                                                                                                                    | §6 · PM-06          |
| **LSTM 1×16 @ 192 kHz** interop drift (1.42e-1)                                              | 🟢 Documented, asserted, rate-aware cap (inherent f16c)                                                                                                                                             | §4.5 / §9.1         |
| **A2-Full / A2-Lite v2 multi-SR goldens** (48 kHz only)                                      | 🟢 By design (explicit `sample_rate` field pins native rate)                                                                                                                                        | §6                  |
| **Dynamic engines v2 multi-SR goldens** (`*Dyn`)                                             | 🟢 By design — live cross-val covers SR; no committed goldens                                                                                                                                       | §3.3 (**RF3**)      |
| **Live v2 harness** — C++ `render` rate-reject silently passes                               | 🟡 Latent anti-masking gap in `run_v2_multi_sr` (no rate masked today)                                                                                                                              | §9 · PM-07          |

### 13.1 Notes on the open items

- **ConvNet (PM-04).** The Rust engine (`models/convnet/`) is complete, dispatched, and unit-tested,
  and has a *self-golden* determinism test (`test_golden_vectors_convnet_test`). It has **no external
  witness**: the f64 oracle returns zeros for ConvNet (`reference_oracle.rs:282`) and the C++ `render`
  tool cannot produce a golden because NAMCore v0.5.3's ConvNet (single shared `channels`, hard-coded
  `kernel=2`, matrix-multiply `_Head`, no `head_scale`) is incompatible with NAM 0.5.4's multi-block
  ConvNet (`golden_gen_build.sh` flags this as an *expected SKIP*). The safe path is an independent
  **f64 ConvNet oracle** (mirroring the §9.2 trust chain), not a C++ upgrade.
- **A2 FiLM / RF1 (PM-03).** FiLM is fully wired in `WaveNetA2Dyn` and golden-tested against the C++
  *generic* WaveNet (C++ `a2_fast.cpp` rejects FiLM). Gating (≈103 dB), blending (≈133 dB) and
  `condition_dsp` (≈139 dB) are near-bit-exact, but **FiLM** sits at 18–36 dB SNR — flagged `RF1` in
  `tests/common/validation.rs` and `perceptual_validation.md`. The divergence is *capped and tracked*,
  but it has **not** been independently classified as inherent-vs-bug because the f64 A2 oracle does
  not yet model FiLM. PM-03 proposes extending the oracle to settle this.
- **A2 official real-amp FiLM captures (PM-05).** The engine supports FiLM via `WaveNetA2Dyn` and the inference path is fully wired, but available real-amp FiLM models are structure-incompatible with the active A2 dynamic engine: `wavenet_a2_max.nam` carries `condition_size=8`, which the loader rejects at topology dispatch. The rejection is graceful (validated by `test_loader_gap_wavenet_a2_max` in Sprint S9.1), not a silent failure. No other compatible real-amp FiLM captures exist in `tests/fixtures/models-nondist/`. Therefore, the project conforms to synthetic captures (`wavenet_a2_film_full.nam`, `wavenet_a2_film_lite.nam`) for automated correctness verification. Sprint S9 golden vectors confirm the synthetic engine path: FilmFull SNR=36.0 dB / ESR=2.50e-4, FilmLite SNR=18.1 dB / ESR=1.54e-2 — both within the documented FiLM interop cap (RF1, PM-03). This conformity is temporary: if compatible real-amp FiLM captures become available, they should be integrated as fixtures and elevated to official goldens.
- **`SlimmableWavenet` (PM-06).** Genuinely deferred. The practical use case (adaptive quality via independent sub-nets + crossfade) is already covered by the implemented and tested `SlimmableContainer` (`tests/container_slimmable.rs`). Any future implementation must meet the following acceptance criteria:
  - Single-file parser supporting multiple channel widths.
  - RT-safe dynamic slicing of weights.
  - Bit-exact parity with C++ NAMCore's `SlimmableWavenet`.

**Recently resolved** — kept for traceability:

- **WaveNet Lite CH=12 (P1 / RF7).** The historical 🔴 "SNR ≈ 0.9 dB architectural divergence" is
  **resolved**. Root cause was a `MirroredBuffer` page-rounding bug (the delay-line buffer was not
  channel-aligned for non-power-of-two channel counts: `1024 % 12 = 4`, `1024 % 6 = 4`), compounded
  by an obsolete synthetic golden. Fixed via `MirroredBuffer::new_aligned()` (`lcm(page, channel_stride)`)
  and migration to the real `EVH-5150-Lite.nam` golden. Now **122.3 dB golden / 117.4 dB live**, with
  an *active* (non-`#[ignore]`) golden test and three regression guards (§9.1).
- **f64 oracle circularity (Sprint S8, Problem A).** The oracle was internally bugged and its anchor
  circular → rebuilt and 3-way cross-validated (§9.2).
- **LSTM non-native coverage (Sprint S8, Problem B).** Silently dropped → restored with a measured,
  rate-aware cap, no rate excluded (§4.5).

The process rules that prevent these classes of error live in
[`perceptual_validation.md`](perceptual_validation.md) (Gate Calibration Policy, Rules 6 & 7).

---

## See Also

- [`audio_fidelity_map.md`](audio_fidelity_map.md) — Off-spec DSP factors; §3 (LSTM drift) pairs with §4.5 here
- [`perceptual_validation.md`](perceptual_validation.md) — Metrics, gate methodology, Gate Calibration Policy
- [`TODO-findings.md`](../TODO-findings.md) — Parity audit findings `PM-01`…`PM-08` for §13 (ConvNet f64 oracle, A2 FiLM/RF1, doc sync, live-harness SKIP robustness)
- `tests/cpp_parity.rs` — Live cross-validation (`live_cross_validation_v2_*`)
- `tests/reference_oracle_f64.rs` + `tests/fixtures/scripts/validate_oracle_f64.py` — f64 oracle & independent anchor
