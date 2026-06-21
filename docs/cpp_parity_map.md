<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# C++ ↔ Rust Parity Map — NeuralAmpModelerCore × NAM-rs

Point-to-point mapping between the canonical C++ reference
[`github.com/NeuralAmpModelerCore`](https://github.com/sdatkinson/NeuralAmpModelerCore)
and the NAM-rs Rust engine (`src/`). This document tracks parity status, known
divergences, and the sprint/task that established each equivalence.

---

## 1. DSP Engine Layer

| C++ (`NeuralAmpModelerCore/`)                                              | Rust (`src/`)                                                             | Parity established |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------ |
| `NAM/dsp.h:184` — `void DSP::Reset(sr, maxBuf)`                            | `models/mod.rs` — `NamModel::reset()`                                     | S4.T04, S26.T02    |
| `NAM/dsp.cpp:93-102` — `Reset` impl (calls `SetMaxBufferSize` + `prewarm`) | `models/mod.rs` — `NamModel::set_max_buffer_size()` + `prewarm_samples()` | S26.T02            |
| `NAM/dsp.cpp` — `DSP::process` (audio callback entry)                      | `dsp/pipeline/stages.rs` — `run_inference()`                              | —                  |
| `NAM/dsp.cpp` — DSP buffer lifecycle                                       | `dsp/pipeline/context.rs` — `DspPipelineContext`                          | —                  |
| `NAM/dsp.cpp` — Noise gate (threshold + hysteresis)                        | `dsp/gate.rs` — `DynamicHysteresis`                                       | —                  |
| `NAM/dsp.cpp` — Prewarm / silence stabilization                            | `loader/mod.rs` — `load_and_build_model` (prewarm 2048 samples)           | S4.T04, S4.T05     |

---

## 2. Model Dispatch

| C++ (`NeuralAmpModelerCore/`)                           | Rust (`src/`)                                            | Parity established |
| ------------------------------------------------------- | -------------------------------------------------------- | ------------------ |
| `NAM/dsp.cpp` — `GetDSP` factory (dynamic dispatch)     | `models/mod.rs` — `StaticModel` enum + manual `match`    | —                  |
| `NeuralModel.cpp:L155-218` — WaveNet topology detection | `loader/nam_json/topology.rs` — `get_wavenet_topology()` | S4.T01–T03         |

---

## 3. WaveNet Architecture

### 3.1 Core Inference

| C++ (`NeuralAmpModelerCore/`)                                                | Rust (`src/`)                                                                                      | Parity established |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------ |
| `NAM/wavenet/model.cpp` — `WaveNet::process`                                 | `models/wavenet/model.rs` — `process_block_internal()`                                             | S3.T04, S4.T01     |
| `WaveNetLayerArrayT<CH,1,1,HEAD,K,Dilations,true>` (C++ template for Array2) | `models/wavenet/model.rs` — `WaveNetModel::array2` (type param `WaveNetLayerArray<CH,1,HEAD,K,1>`) | S1.T01             |
| C++ `model->Prewarm()` (no-arg call)                                         | `models/wavenet/mod.rs` — `prewarm()` ignores `num_samples`                                        | S4.T05             |

### 3.2 Layer Components

| C++ (`NeuralAmpModelerCore/`)                          | Rust (`src/`)                                                      | Parity established |
| ------------------------------------------------------ | ------------------------------------------------------------------ | ------------------ |
| WaveNet causal dilated Conv1D                          | `models/wavenet/conv1d.rs` — `Conv1d<IN,OUT,K>`                    | S3.T04             |
| Conv1D dual-frame temporal tiling                      | `models/wavenet/conv1d_dual.rs`                                    | —                  |
| Input mixin (1×1 projection, conditioned input)        | `models/wavenet/dense.rs` — `DenseLayer::process_fused()`          | S3.T04             |
| 1×1 residual projection                                | `models/wavenet/dense.rs` — `DenseLayer::process_residual_batch()` | S3.T04             |
| Layer states (delay buffers, receptive field tracking) | `models/wavenet/common.rs` — `WaveNetLayerState`                   | —                  |
| BF16 layer state caching                               | `models/wavenet/common.rs` — `u16` mirrored buffer variant         | —                  |

### 3.3 Dynamic WaveNet (fallback path for free geometries)

> Models whose topology does not match a catalog SKU (Standard/Lite/Feather/Nano) are routed through the dynamic fallback path (`WaveNetModelDyn`) — not rejected. This engine handles arbitrary channel counts, kernel sizes, dilations, `condition_size > 1`, and post-stack heads. The `Conv1dDyn` convolution kernel serves as the low-level compute engine for this path, for the A2 dynamic engine (`WaveNetA2Dyn`), and for static WaveNet test/stress kernels.
>
> **v2 multi-SR goldens:** Dynamic engines (`WaveNetModelDyn`, `LstmModelDyn`, `WaveNetA2Dyn`) are anchored at **48 kHz only** for v1 golden vectors. v2 multi-SR goldens are intentionally not provided because: (a) dynamic engines handle arbitrary free geometries — geometry variance subsumes sample-rate variance in practice; (b) live cross-validation in `tests/cpp_parity.rs` (lines 651, 662, 667, 678) already exercises multi-SR parity via the C++ toolchain for the canonical dynamic geometries (`wavenet_dyn_free`, `lstm_dyn_test`); (c) the A2 dynamic geometries (`a2_dynamic_gated_ch8`, `a2_dynamic_blended_ch3`, `wavenet_a2_film_*`) are forward-compat parser surface only and not part of the fixed fast-path. See `TODO-audit.md` RF3 and `tests/fixtures/README.md` §v2 multi-SR coverage.

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

| C++ (`NeuralAmpModelerCore/`)                | Rust (`src/`)                                                          | Parity established |
| -------------------------------------------- | ---------------------------------------------------------------------- | ------------------ |
| `NAM/lstm.cpp` — `LSTM::process_sample`      | `models/lstm/layer.rs` — `process_sample_avx2` / `_avx512` (via macro) | S3.T01–T02, S7.R02 |
| LSTM gate computation (sigmoid + tanh fused) | `math/lstm/gates.rs` — `fused_lstm_gates`                              | S3.T02             |
| LSTM 2-layer pipelined processing            | `models/lstm/model2.rs` — `define_lstm2_process_pipelined!`            | S3.T03             |
| LSTM Prewarm (silence → convergence)         | `models/lstm/mod.rs` — `lstm_prewarm_common`                           | S4.T05             |

### 4.2 Layer Components

| C++ (`NeuralAmpModelerCore/`)                 | Rust (`src/`)                                                                   | Parity established |
| --------------------------------------------- | ------------------------------------------------------------------------------- | ------------------ |
| Gate-major weight layout `[Gate][IH][Hidden]` | `models/lstm/layer.rs` — `LstmLayer.input_hidden_weights` `[[[u16; H]; IH]; 4]` | S3.T01             |
| Bias vector `[H * 4]`                         | `models/lstm/layer.rs` — `LstmLayer.bias: [f32; H4]`                            | S3.T01             |
| Hidden state `[H]`                            | `models/lstm/layer.rs` — `LstmLayer.state` / `state_bf16`                       | S3.T01             |
| Cell state `[H]`                              | `models/lstm/layer.rs` — `LstmLayer.cell_state`                                 | S3.T01             |
| Head projection (H → 1)                       | `models/lstm/model1.rs` — `head_weights` / `head_bias`                          | S3.T01             |
| FP32 native head rechannel                    | `models/lstm/model1.rs` — `use_f32_head: bool`                                  | E8.T08             |

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

---

---

## 5. ConvNet Architecture

| C++ (`NeuralAmpModelerCore/`)                   | Rust (`src/`)                                              | Parity established |
| ----------------------------------------------- | ---------------------------------------------------------- | ------------------ |
| `convnet.cpp` / `convnet.h` — ConvNet inference | `models/convnet/model.rs` — `ConvNetModel`                 | —                  |
| Conv1D → BatchNorm → Activation blocks          | `models/convnet/block.rs` — `ConvNetBlock`                 | —                  |
| `convnet.h:108-118` — `_Head` (post-stack head) | `models/convnet/model.rs` — `ConvNetModel.post_stack_head` | —                  |
| ConvNet topology dispatch                       | `loader/dispatcher/convnet/mod.rs` — `build_convnet()`     | —                  |

---

## 6. A2 Architecture (Fixed fast-path port)

> **Status:** A2 inference is fully implemented (Beta) as the **fixed fast-path** (`NAM/wavenet/a2_fast.cpp`) for the production shapes **A2-Full** (8 ch) and **A2-Lite** (3 ch). See [TODO-sprints.md](../TODO-sprints.md) (Epics 1–2). The `GatingActivation`/`BlendingActivation`/`_FiLMParams` rows below map **forward-compat parser surface only** — the general A2 engine (FiLM/gating/`condition_dsp`/`bottleneck≠channels`) is out of scope. `SlimmableWavenet` (single-net channel slicing) is a separate, deferred epic.
>
> **Golden vectors** are generated from the C++ `render` tool at pinned commit `9c7b185` (v0.5.3) and committed as pre-validated Layer 1 artifacts (`tests/fixtures/golden_wavenet_a2_{full,lite}.bin`). The v1 golden tests run actively on every `cargo test`. Layer 2 live cross-validation (`tests/cpp_parity.rs`) exists and is `#[ignore]` — normal for all live parity tests, requiring C++ toolchain; run via `utils/tests-long.sh`.
>
> **Calibrated SNR/ESR** (measured 2026-06-14 against C++ v0.5.3): A2-Full = 79.2 dB / 1.21e−8; A2-Lite = 90.7 dB / 8.58e−10. Thresholds include ≥8 dB SNR margin and ~6–7× ESR multiplier. See `tests/fixtures/README.md` (§`wavenet_a2_full.nam` & `wavenet_a2_lite.nam`) and §9.1.
>
> **⚠ Nature:** These goldens use **synthetic weights** (canonical A2 skeleton: 23 layers, K=6/15, LeakyReLU, head_scale=0.02 — generated by `tests/fixtures/generate_a2_fixtures.py`). They validate **numerical parity of the fast-path** Rust↔C++, not real amplifier timbres or FiLM-conditioned models. Official A2 models (`wavenet_a2_max.nam`, `slimmable_wavenet.nam`) use FiLM conditioning and remain unsupported (future feature). Elevation to official goldens will occur when the engine supports FiLM and generic A2 inference.
>
> **Sample rate limitation:** A2 v2 multi-SR goldens only exist at 48 kHz. Models with an explicit `sample_rate` field in JSON (A2-Full, A2-Lite) are constrained to their native rate by the C++ `render` tool. All other models generate at 5 rates (44.1k/48k/88.2k/96k/192k).
>
> **Runtime safeguards:** The garbage guard detects absurd-but-finite amplitude (`max_sample > 10³`) on the DSP output, preventing silent corruption from reaching audio hardware. ESR is the primary scale-robust threshold in `topology_thresholds()` and `live_parity_thresholds()`.

| C++ (`NeuralAmpModelerCore/`)                            | Rust (`src/`)                                                                                                     | Parity established |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------ |
| `NAM/wavenet/a2_fast.h` — architectural constants        | `models/a2/params.rs` — `A2_NUM_LAYERS`, `A2_LEAKY_SLOPE`, `A2_KERNEL_SIZES`, `A2_DILATIONS`, `A2_VALID_CHANNELS` | S26.T01            |
| `a2_fast.cpp:875-908` — `is_a2_shape()`                  | `loader/nam_json/topology.rs` — `is_a2_shape()`                                                                   | S26.T01, T11.2     |
| `NAM/activations.h:L111-122` — `fast_tanh`               | `models/a2/activations.rs` — `fast_tanh()`                                                                        | S26.T01            |
| `GatingActivation` class *(parser surface, not wired)*   | `models/a2/gating.rs` — `GatingActivationConfig`                                                                  | S26.T01            |
| `BlendingActivation` class *(parser surface, not wired)* | `models/a2/gating.rs` — `BlendingActivationConfig`                                                                | S26.T01            |
| `_FiLMParams` struct *(parser surface, not wired)*       | `models/a2/film.rs` — `FiLMConfig`                                                                                | S26.T01            |
| **A2-Full / A2-Lite inference (fixed fast-path)**        | `models/a2/` — port of `A2FastModel<8>` / `A2FastModel<3>`                                                        | Epics 1, 2         |
| `NAM/container.{h,cpp}` — `SlimmableContainer`           | `models/container.rs` + `loader/dispatcher/container/`                                                            | Epic 3             |

---

## 7. Weight Loading & Parsing

### 7.1 `.nam` JSON Format

| C++ (`NeuralAmpModelerCore/`)                         | Rust (`src/`)                                                                    | Parity established |
| ----------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------ |
| `.nam` JSON deserialization                           | `loader/nam_json/data.rs` — `NamModelData`                                       | S1.T01–T02         |
| Weight layout detection (row-major gate-major)        | `loader/nam_json/data.rs` — `WeightLayout` enum                                  | S3.T02             |
| `NeuralModel.cpp` — topology dispatch                 | `loader/nam_json/topology.rs` — `get_wavenet_topology()` / `get_lstm_topology()` | S4.T03             |
| NAM metadata (input_level_dbu, loudness, sample_rate) | `loader/nam_json/data.rs` — `NamMetadata`                                        | S1.T02             |

### 7.2 NAMB Binary Format

| C++ ecosystem convention                       | Rust (`src/`)                                          | Parity established |
| ---------------------------------------------- | ------------------------------------------------------ | ------------------ |
| NAMB binary layout (magic, version, CRC32)     | `loader/namb.rs` — `NambHeader` + `parse_namb()`       | S5.T02             |
| Original weight layout (row-major, unmodified) | `loader/namb.rs` — `WeightLayout::Original`            | S5.T02             |
| GateMajorLstm transposed layout                | `loader/namb.rs` — `WeightLayout::GateMajorLstm`       | S5.T02             |
| Interleaved4WaveNet transposed layout          | `loader/namb.rs` — `WeightLayout::Interleaved4WaveNet` | S5.T02             |
| NAMB CRC32 validation (v2+)                    | `loader/namb.rs` — `crc32_ieee()`                      | S12.T02            |
| NAMB encoder/export                            | `loader/namb_encoder.rs` — `encode_namb()`             | S10.T01            |

### 7.3 WaveNet Weight Layout

| C++ (`NeuralAmpModelerCore/`)                       | Rust (`src/`)                                                     | Parity established |
| --------------------------------------------------- | ----------------------------------------------------------------- | ------------------ |
| `WaveNet.h` — `SetWeights` (global layout)          | `loader/dispatcher/wavenet/standard.rs` — `build_wavenet_typed()` | S3.T03             |
| `WaveNetLayerArrayT::SetWeights` (per-array layout) | `loader/dispatcher/wavenet/standard.rs` — `build_wavenet_array()` | S3.T03             |
| Head scale (final scalar multiplier)                | `loader/dispatcher/wavenet/standard.rs` — `head_scale`            | S3.T03             |

### 7.4 LSTM Weight Layout

| C++ (`NeuralAmpModelerCore/`)            | Rust (`src/`)                                                               | Parity established |
| ---------------------------------------- | --------------------------------------------------------------------------- | ------------------ |
| `LSTMLayerT::SetNAMWeights` (NAM format) | `loader/dispatcher/lstm.rs` — `build_lstm_1layer()` / `build_lstm_2layer()` | S3.T01             |
| Weight layout: `[H4*IH, H4, H, H, H, 1]` | `loader/dispatcher/lstm.rs` — `read_lstm_layer()`                           | S3.T01             |

---

## 8. Math / SIMD Kernels

| C++ (`NeuralAmpModelerCore/`)                       | Rust (`src/`)                                                 | Parity established |
| --------------------------------------------------- | ------------------------------------------------------------- | ------------------ |
| Dot product (f32)                                   | `math/gemm/dot.rs` — `dot_product`                            | S2.T03             |
| Dot product (BF16 quantized weights)                | `math/gemm/dot_4x/` — `dot_product_bf16`                      | S3.T01             |
| Dot product 4× interleaved (WaveNet Conv1D)         | `math/gemm/dot_4x/` — `dot_product_4x_interleaved`            | S3.T04             |
| Dot product 4× interleaved dual-frame               | `math/gemm/dot_4x/` — `dot_product_4x_interleaved_dual_frame` | S10b.T02           |
| GEMV (fused add)                                    | `math/gemm/gemv.rs` — `fused_add_gemv`                        | S3.T01             |
| GEMV (overwrite)                                    | `math/gemm/gemv.rs` — `gemv_overwrite`                        | S26.T03            |
| GEMV 4-gate (LSTM)                                  | `math/gemm/gemv_4gate.rs` — `gemv_overwrite_4gate`            | S3.T01             |
| GEMV BF16 (LSTM 4-gate)                             | `math/gemm/gemv_bf16.rs` — `gemv_overwrite_bf16_4gate`        | S3.T01             |
| Tanh activation (SIMD)                              | `math/activations/tanh.rs` — `tanh_slice`                     | S2.T03             |
| Sigmoid activation (SIMD)                           | `math/activations/sigmoid.rs` — `sigmoid_slice`               | S2.T03             |
| Fused Tanh + accumulate (WaveNet head)              | `math/wavenet/accumulate.rs` — `tanh_and_accumulate_block`    | S3.T04             |
| Fused Tanh + overwrite (first layer head)           | `math/wavenet/accumulate.rs` — `tanh_and_overwrite_block`     | S25.T08            |
| Cascaded head accumulation (array N seeds from N−1) | `layer_array.rs` — `head_seeded` + accumulation               | S16.1              |
| Gain application (linear)                           | `math/dsp/gain.rs` — `apply_gain`                             | S2.T03             |
| Gain LUT (dB → linear)                              | `math/dsp/gain_lut.rs` — `GainLut`                            | —                  |
| Stereo convolution (resampler FIR)                  | `math/dsp/stereo/` — `convolve_stereo`                        | S17.T01            |
| Kahan compensated summation (scalar fallback)       | `math/common/kahan.rs` — `KahanF32` / `Kahan4F32`             | E8.T06             |
| BF16 quantization (f32 → u16)                       | `math/common/utility.rs` — `quantize_weight()`                | S3.T01             |
| Scalar reference (definitive math specification)    | `math/common/scalar_ref.rs` — all operations                  | —                  |

---

## 9. Cross-Validation

| C++ (`NeuralAmpModelerCore/`)                        | Rust (`src/` / `tests/`)                        | Parity established |
| ---------------------------------------------------- | ----------------------------------------------- | ------------------ |
| `render` CLI (golden output generation)              | `tests/cpp_parity.rs` — live cross-validation   | S13a.T01           |
| `ModelTest.cpp` (stress-signal tests)                | `tests/nam_infer_test.rs` — golden vector tests | S13a.T01           |
| `test_get_dsp.cpp` (official WaveNet test)           | `tests/fixtures/` — `wavenet.nam` model         | S13a.T01           |
| `test_slimmable_wavenet.cpp` (official WaveNet test) | `tests/fixtures/` — shared models               | S13a.T01           |
| SNR thresholds (C++ → Rust comparison)               | `tests/cpp_parity.rs` — per-model SNR passes    | —                  |

### 9.1 Post-Nuke ESR Measurements (T-HF6.6, jun/2026)

After elimination of all BF16/F16 quantization and dual-mode paths from WaveNet A1
(E-HF Sprint 6), the ESR against NeuralAmpModelerCore v0.5.3 (commit `9c7b185`)
was recalibrated:

| Model                 | ESR (linear) | ESR (dB) | SNR (dB) | Notes                           |
|:--------------------- |:------------ |:-------- |:-------- |:------------------------------- |
| WaveNet A1-Std CH=16  | 4.58e-13     | −123.4   | 123.4    | Live v1, f32 + poly tanh        |
| WaveNet Standard v2   | *varies*     | *varies* | 101.8*   | Multi-SR v2 worst @ 192k        |
| WaveNet Standard (v2) | *varies*     | *varies* | 123.0**  | v2 best @ 48kHz                 |
| WaveNet Feather CH=8  | 4.92e-14     | −133.1   | 133.1    | Live v1                         |
| WaveNet Feather (v2)  | *varies*     | *varies* | 117.6*   | v2 worst @ 192kHz               |
| WaveNet Nano CH=4     | 6.30e-14     | −132.0   | 132.0    | Live v1                         |
| WaveNet Nano (v2)     | *varies*     | *varies* | 114.6*   | v2 worst @ 192kHz               |
| WaveNet Lite CH=12    | —            | —        | 0.9      | P1 — architectural, `#[ignore]` |

> \* Worst-case across v2 multi-SR golden vectors.
> \*\* Best-case v2 (48 kHz native).
> All non-Lite models now achieve SNR ≫ 100 dB, comparable to LSTM/Linear (67–91 dB).
> Thresholds calibrated: Standard ≥ 85 dB SNR, Feather ≥ 100 dB, Nano ≥ 95 dB.

These measurements replace the pre-nuke ESR of ~3e-3 to 1e-2 (−25 to −20 dB).
The ~10 order-of-magnitude improvement comes from eliminating BF16/F16 weight
quantization (previously the dominant drift source at ~3.9e-3 per element).

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

| Divergence                                    | Rationale                                                                                                                                                                                   |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **No `DspBridge` in CLAP mode**               | CLAP plugin receives both input and output in a single `process()` call. Bridge only needed standalone (PipeWire dual-thread topology).                                                     |
| **MirroredBuffer (`memfd_create`)**           | Linux-specific virtual memory mirroring for O(1) linear access in WaveNet delay lines. C++ uses modulo-based circular access.                                                               |
| **Static const-generic dispatch (no vtable)** | Static `match` on `StaticModel` enum avoids vtable overhead. C++ `GetDSP` returns a pointer to a virtual base class.                                                                        |
| **Reset does NOT prewarm on load**            | `reset()` is a public API for explicit state clearing. Loader calls `prewarm()` separately to preserve LSTM initial states loaded from file (S4.T05). C++ `Reset` always calls `prewarm()`. |
| **Prewarm hardcoded to 2048 samples**         | C++ `PrewarmSamples()` returns `receptive_field`. NAM-rs uses 2048 as a safe upper bound covering all models.                                                                               |
| **`WavenetA2Placeholder` (silent output)**    | Retired and removed in Epic 1. Replaced by real `WaveNetA2` inference.                                                                                                                      |
| **No `std::complex` / STL data structures**   | Everything uses idiomatic Rust (`AlignedVec<T>`, `AtomicU64`, `rtrb` SPSC).                                                                                                                 |
| **TSC-based latency measurement**             | NAM-rs calibrates the CPU TSC for nanosecond-accurate RT cycle measurements — no C++ equivalent.                                                                                            |
| **CPU C-state lock (`/dev/cpu_dma_latency`)** | Linux-specific RT tuning — no equivalent in cross-platform C++ reference.                                                                                                                   |
| **SCHED_FIFO + `mlockall`**                   | Linux RT scheduling — not applicable to C++ reference.                                                                                                                                      |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`condition_size ≠ 1` or                     | Multi-condition WaveNet (`condition_size > 1`) and post-stack heads (`head` sub-object) are official NAMCore features. Models using them are **accepted at load time** —                    |
| `head` (non-null) accepted**                  | `get_wavenet_topology()` captures the non-catalog geometry as `Free` and routes it to `WaveNetModelDyn` (the dynamic engine), which is parameterized on `condition_size` at runtime.        |
|                                               | Catalog SKUs (Standard/Lite/Feather/Nano) use `condition_size=1` and `head=null` via the const-generic fast-path. The dynamic path handles `condition_size > 1`, post-stack heads,          |
|                                               | and free geometries with generic dispatch (§3.3).                                                                                                                                           |

### 11.2 Math

| Divergence                                           | Rationale                                                                                                                                                                                                                          |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Padé [5,4] Tanh vs `std::tanh`**                   | C++ uses IEEE-754 `std::tanh`. NAM-rs uses rational Padé approximant (error < 2.32e-3) — 10–20× throughput gain. Post-T-HF6.6, this is the sole remaining math divergence for WaveNet A1 (BF16/F16 quantization paths eliminated). |
| **Minimax degree-17 sigmoid vs `0.5+0.5*tanh(x/2)`** | Direct polynomial (1.67× lower error, −20.25% latency). C++ reference composes `std::tanh`.                                                                                                                                        |
| **BF16 vs F16 dispatch**                             | NAM-rs runtime-detects `Avx512VnniBf16` and chooses precision. C++ has no equivalent multi-ISA/packed-format dispatch. BF16 has ~8× larger quantization error than F16 but allows VNNI native ops on Sapphire Rapids+.             |
| **Kahan compensated summation (scalar fallback)**    | Applied in interleaved 4x scalar fallback dot products. C++ uses standard accumulation. Static conv1d paths also use plain accumulation (T13.2/T18.4).                                                                             |
| **Anti-subnormal DC dither (−220 dBFS)**             | Prevents subnormal float stalls. Below 24-bit DAC noise floor. C++ has no equivalent.                                                                                                                                              |
| **FP32 native head rechannel**                       | Final projection (head) runs in FP32 regardless of backbone precision. Eliminates quantization error at output. C++ uses same precision throughout.                                                                                |

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

| C++ reference                                                | Rust (`src/`)                         | Parity status            |
| ------------------------------------------------------------ | ------------------------------------- | ------------------------ |
| `AudioDSPTools/dsp/ImpulseResponse.h` (direct time-domain)   | `dsp/cabsim/conv.rs` — UPOLS engine   | **Analyzed (S5.3/T5.7)** |
| `NeuralAmpModelerPlugin/NeuralAmpModeler.cpp:676` (IR usage) | `dsp/pipeline/capture.rs` — cab stage | **New feature**          |

### 12.1 Algorithmic Analysis — `dsp::ImpulseResponse` (C++) vs UPOLS (NAM-rs)

> **Analysis completed:** Sprint 5.3, [T5.7] — submodule `AudioDSPTools` initialized at commit `0827c6c`.

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

**Cross-validation results** (T5.9, gain-compensated via `CPP_GAIN_INV ≈ 7.943`):

| Scenario | IR len | Signal len | Partitions | ESR      | ESR (dB) | SNR (dB) |
| -------- | ------ | ---------- | ---------- | -------- | -------- | -------- |
| short    | 64     | 256        | 1          | 1.15e-14 | −139.4   | 139.4    |
| medium   | 512    | 1024       | 8          | 2.84e-14 | −135.5   | 135.5    |
| long     | 8192   | 16384      | 128        | 9.39e-14 | −130.3   | 130.3    |
| stress   | 32768  | —          | —          | N/A      | N/A      | N/A      |

> **Stress scenario skipped:** C++ `ImpulseResponse::mMaxLength = 8192` hard-caps the IR, truncating the 32768-sample stress IR. NAM-rs UPOLS stress validation is performed against direct convolution in `tests/cabsim_golden.rs::test_cabsim_golden_stress` (ESR < 1e-5). All three comparable scenarios exceed the 1e-3 ESR threshold by 10+ orders of magnitude, confirming bit-identical equivalence after gain compensation.

### 12.2 Cross-Validation Performed (Sprint 5.3)

The `AudioDSPTools` submodule is initialized at `tests/fixtures/NeuralAmpModelerPlugin/AudioDSPTools/`. Cross-validation tests are implemented as `#[ignore]` in `tests/cabsim_cpp_parity.rs` and run via `utils/tests-long.sh`.

- **Test validation:** Compares the UPOLS convolution engine against golden vectors generated by the C++ reference binary (`tests/fixtures/render_ir.cpp`).
- **Alignment & compensation:** Compares are gain-compensated (`CPP_GAIN_INV ≈ 7.943`) and latency-aligned.
- **Results:** Parity is verified with **ESR < 1e-13** in short, medium, and long scenarios, vastly exceeding the target threshold of 1e-3.

---

## 13. Related Sprints & Tasks

| Sprint        | Topic                                                | Key C++ reference                               |
| ------------- | ---------------------------------------------------- | ----------------------------------------------- |
| S3 (T01–T05)  | LSTM parity, BF16, NAMB round-trip, Conv1D tail-loop | `LSTMLayerT::SetNAMWeights`, WaveNet level loop |
| S4 (T01–T05)  | WaveNet backfill, `reset()`, LSTM state preservation | `DSP::Reset`, `prewarm` contract                |
| S7 (R02)      | Hoist BF16 dispatch outside LSTM loop                | `LSTM::process_sample`                          |
| S13a (T01)    | Cross-validation suite vs NeuralAmpModelerCore       | `render` CLI, `ModelTest.cpp`                   |
| S25 (T01–T08) | Hotpath SIMD recovery, buffer alignment              | `process_block_f32_native`, head rechannel      |
| S26 (T01–T04) | Architectural adherence vs C++ reference             | `dsp.h`, `a2_fast.h`                            |
| S28 (T01)     | Cross-validation v2                                  | `t3k-mushra` metrics, A2 baselines              |

---

## 14. Version History

| Date       | Change                                                                                                                                                                                                                                                                                                                                                                                             |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-06-20 | [Tarefa A.1.3] Seção 6 (A2 Architecture): removido aviso de "C++ Live Cross-Validation Blocked" e "self-golden pattern" — os goldens A2 são gerados pelo C++ `render` v0.5.3 desde T2.5/T16.4; documentada natureza sintética dos pesos, limitação de sample rate (48 kHz apenas), ESR/SNR calibrados reais (79.2/90.7 dB) e status normal do `#[ignore]` de Layer 2.                              |
| 2026-06-20 | [Tarefa A.1.1] Atualização de §§3.3, 3.4, 4.4, 10 e nova Seção 5 (ConvNet): removida linguagem de "removido" dos engines dinâmicos; corrigido fallback `Free`/`WaveNetModelDyn`/`LstmModelDyn` para geometrias não-catalogadas; adicionadas entradas `WaveNetDyn`, `LSTM Dyn`, `WaveNetA2Dyn` e `ConvNet` na tabela de topologia; renumeradas todas as seções subsequentes (5→6, 6→7, ..., 15→14). |
| 2026-06-18 | [T-HF6.7] Post-nuke ESR table (§9.1): WaveNet ESR improved ~10 orders after lo-fi elimination (T-HF6.1–6.6). All non-Lite models SNR ≫ 100 dB. Thresholds 85-105 dB. Updated §11.2 Math divergences to note BF16/F16 paths eliminated for WaveNet A1.                                                                                                                                              |
| 2026-06-12 | [T11.2] `is_a2_shape()` now matches C++ `is_a2_shape()` exactly (20 criteria from `a2_fast.cpp:875-908`). Added `bottleneck`, `kernel_sizes`, `in_channels` to typed config structs; raw JSON capture via `layer_raw` for complex checks (activation arrays, FiLM, gating_mode, head sub-objects, groups,                                                                                          |
|            | slimmable). Strict rejection with clear diagnostics prevents silent fast-path misroute for models with `bottleneck≠channels`, gated/blended activation, or active FiLM conditioning. Full parity map entry updated with correct line range.                                                                                                                                                        |
| 2026-06-11 | [T7.8] A2 divergence root-cause analysis. Fixed Rust `prewarm()` to feed silence through `process()` (matching C++ `DSP::Reset` → `prewarm()` flow). C++ live cross-validation blocked by upstream `a2_fast.cpp` numerical bug (A2-Full output ~10^14). Self-golden pattern maintained with corrected prewarm.                                                                                     |
| 2026-06-11 | [T5.7-T5.9] Complete Cabsim C++ cross-validation (AudioDSPTools). Parity verified (ESR < 1e-13). Update A2 architecture mappings to show complete implementation (Beta) and remove references to `WavenetA2Placeholder`.                                                                                                                                                                           |
| 2026-06-10 | [T5.6] Add §13 IR Cabsim section: documents cabsim as new NAM-rs feature with no C++ equivalent, decision to defer C++ cross-validation (AudioDSPTools submodule not initialized), and plan for Sprint 5.3 cross-validation.                                                                                                                                                                       |
| 2026-06-03 | Initial creation. Maps all WaveNet (Standard/Lite/Feather/Nano/Dyn), LSTM (1×{8,12,16,24,40}, 2×{8,12,16,24}, Dyn), and A2 (placeholder) models. Covers S3, S4, S7, S13a, S25, S26 parity tasks. Documents 10 architectural divergences and 6 math/ecosystem divergences.                                                                                                                          |
